// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/envoyproxy/ai-gateway/internal/apischema/dashscope"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// dashScopeSpeechPath is DashScope's native multimodal generation endpoint that hosts the
// Qwen-TTS / Qwen3-TTS model family. Alibaba's OpenAI-compatible mode does not expose
// /audio/speech, so we route to the native path and translate the body.
//
// https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api
const dashScopeSpeechPath = "/api/v1/services/aigc/multimodal-generation/generation"

// dashScopeAudioContentType is the default MIME type DashScope returns for the signed audio URL.
// Qwen-TTS delivers WAV by default; response_format from the OpenAI request is currently ignored.
const dashScopeAudioContentType = "audio/wav"

// dashScopeAllowedAudioHostSuffix restricts the hosts the audio-URL fetcher will follow. The signed
// URL comes straight from an upstream JSON body, so treat it as untrusted input: anchor it to
// Aliyun-owned hostnames to prevent a spoofed / hijacked response from redirecting the fetch at an
// arbitrary host (SSRF).
const dashScopeAllowedAudioHostSuffix = ".aliyuncs.com"

// validateDashScopeAudioURL enforces the SSRF-prevention rules on the signed audio URL returned by
// DashScope. The primary defence is the host allowlist — the URL must resolve to aliyuncs.com or
// a subdomain of it, anchored with a leading dot so lookalikes such as `evil-aliyuncs.com` do not
// pass. Both http and https are accepted because DashScope's signed URLs are served over plain
// http in practice; the TTS audio payload is not confidentiality-sensitive, and the host
// allowlist prevents the fetch from being redirected off Aliyun infrastructure.
func validateDashScopeAudioURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("dashscope speech: invalid audio URL: %w", err)
	}
	if u.Scheme != "https" && u.Scheme != "http" {
		return fmt.Errorf("dashscope speech: audio URL scheme must be http or https, got %q", u.Scheme)
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("dashscope speech: audio URL is missing host: %s", raw)
	}
	if host != "aliyuncs.com" && !strings.HasSuffix(host, dashScopeAllowedAudioHostSuffix) {
		return fmt.Errorf("dashscope speech: audio URL host %q not in allowed *%s", host, dashScopeAllowedAudioHostSuffix)
	}
	return nil
}

// dashScopeAudioFetcher downloads the audio pointed to by DashScope's signed URL. Overridable
// in tests to avoid real network calls. Reads the full body — TTS payloads are small (tens of
// KB up to a few MB), so streaming is unnecessary at this layer.
var dashScopeAudioFetcher = func(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("dashscope audio fetch: build request: %w", err)
	}
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("dashscope audio fetch: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("dashscope audio fetch: status %d: %s", resp.StatusCode, string(body))
	}
	return io.ReadAll(resp.Body)
}

// NewSpeechOpenAIToDashScopeTranslator translates OpenAI /v1/audio/speech requests into
// DashScope's native Qwen-TTS multimodal-generation shape, and re-emits the audio bytes
// pointed to by the signed URL in DashScope's response so that the caller gets the same
// audio-bytes-in-body contract that OpenAI's endpoint offers.
func NewSpeechOpenAIToDashScopeTranslator(modelNameOverride internalapi.ModelNameOverride) OpenAISpeechTranslator {
	return &openAIToDashScopeSpeechTranslator{
		modelNameOverride: modelNameOverride,
		path:              dashScopeSpeechPath,
	}
}

type openAIToDashScopeSpeechTranslator struct {
	modelNameOverride internalapi.ModelNameOverride
	// path is the upstream request path that ends up as :path on the outbound HTTP request.
	path string
	// requestModel is the model reported back through the response metadata. Kept because
	// DashScope's response only echoes usage metadata, not the model name.
	requestModel internalapi.RequestModel
}

// RequestBody implements [OpenAISpeechTranslator.RequestBody].
//
// OpenAI shape:
//
//	{"model":"tts-1","input":"hello","voice":"alloy","response_format":"mp3",...}
//
// DashScope shape:
//
//	{"model":"qwen3-tts-flash","input":{"text":"hello","voice":"Cherry","language_type":"Auto"}}
//
// Voice, instructions, response_format, speed, and stream_format are not translated:
//   - voice: passed through verbatim — the AIGatewayRoute owner is expected to supply a
//     DashScope-side voice name (e.g. "Cherry"); OpenAI voice presets do not have a natural
//     mapping and forcing one here would silently pick the wrong timbre.
//   - instructions/speed: DashScope has no equivalent.
//   - response_format: DashScope's non-streaming response is always WAV via URL; ignored here
//     and the outbound content-type is set to audio/wav in ResponseHeaders.
//   - stream_format: only non-streaming is implemented in this first cut; SSE support would
//     require calling DashScope's streaming synthesis API (different endpoint) and is out of
//     scope until we see demand.
//
// The `original` byte slice is intentionally ignored — the DashScope wire format differs
// enough from the OpenAI request that we always re-marshal from the parsed struct rather
// than patching the original body. Same reason forceBodyMutation is ignored: the response is
// always a body replacement, never a passthrough.
func (o *openAIToDashScopeSpeechTranslator) RequestBody(_ []byte, req *openai.SpeechRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	if req == nil {
		return nil, nil, fmt.Errorf("dashscope speech: nil request")
	}

	model := req.Model
	if o.modelNameOverride != "" {
		model = o.modelNameOverride
	}
	o.requestModel = model

	input := dashscope.SpeechInput{
		Text:  req.Input,
		Voice: req.Voice,
	}
	// language_type: Qwen-TTS accepts a discrete language enum; auto-derive it from the voice
	// when the caller doesn't override, so language-specific voices (Sohee/Sonrisa/etc.)
	// pronounce their native language without depending on the model's Auto detection heuristic.
	input.LanguageType = qwenTTSLanguageForVoice(req.Voice)
	// OpenAI SpeechRequest has an Instructions field even though it's only meaningful on
	// qwen3-tts-instruct-flash. Passing it unconditionally is safe — the multimodal-generation
	// endpoint ignores it for non-instruct models per Alibaba's docs.
	if req.Instructions != nil && *req.Instructions != "" {
		input.Instructions = *req.Instructions
	}

	dsReq := dashscope.SpeechRequest{Model: model, Input: input}
	newBody, err = json.Marshal(dsReq)
	if err != nil {
		return nil, nil, fmt.Errorf("dashscope speech: marshal request: %w", err)
	}

	newHeaders = []internalapi.Header{
		{pathHeaderName, o.path},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
		// OpenAI SDKs (Spring AI, python-openai, etc.) default Accept to
		// application/octet-stream for /v1/audio/speech, but DashScope rejects that with
		// "Accept type just supports application/json, application/*+json, ...". Force JSON so
		// the SDK's request survives — the audio bytes still come back through ResponseBody.
		{"accept", "application/json"},
	}
	return
}

// qwenTTSLanguageForVoice maps a DashScope built-in voice name to the language_type enum that
// Qwen-TTS expects. Language-scoped voices (Sohee, Sonrisa, Emilien, …) get their native locale;
// everything else stays on Auto so multilingual voices (Cherry, Ethan, …) keep detecting.
//
// Custom-cloned voice IDs (typically `custom_...` or `qwen-tts-vc-...`) fall through to Auto
// because their language isn't inferable from the ID alone. Callers can still override by
// pre-selecting language_type client-side; adding a bypass field on SpeechRequest is out of
// scope until we see demand.
func qwenTTSLanguageForVoice(voice string) string {
	switch voice {
	case "Sohee":
		return "Korean"
	case "Ono Anna":
		return "Japanese"
	case "Bodega", "Sonrisa":
		return "Spanish"
	case "Alek":
		return "Russian"
	case "Dolce":
		return "Italian"
	case "Lenn":
		return "German"
	case "Emilien":
		return "French"
	}
	return "Auto"
}

// ResponseHeaders implements [OpenAISpeechTranslator.ResponseHeaders].
//
// The upstream reply is application/json (the signed-URL envelope) but the client contract
// is binary audio, so overwrite content-type here. content-length is unknown until we
// download the file in ResponseBody; drop it and let Envoy chunk / re-set as needed.
func (o *openAIToDashScopeSpeechTranslator) ResponseHeaders(_ map[string]string) (newHeaders []internalapi.Header, err error) {
	return []internalapi.Header{
		{"content-type", dashScopeAudioContentType},
	}, nil
}

// ResponseBody implements [OpenAISpeechTranslator.ResponseBody]. It parses DashScope's JSON
// envelope, follows the signed URL, and returns the raw audio bytes so the caller sees the
// same audio-bytes-in-body response shape OpenAI's /v1/audio/speech provides.
func (o *openAIToDashScopeSpeechTranslator) ResponseBody(_ map[string]string, body io.Reader, _ bool, span tracingapi.SpeechSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	raw, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("dashscope speech: read response: %w", err)
	}
	var envelope dashscope.SpeechResponse
	if err = json.Unmarshal(raw, &envelope); err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("dashscope speech: parse response JSON: %w", err)
	}
	if envelope.Output.Audio.URL == "" {
		return nil, nil, tokenUsage, "", fmt.Errorf("dashscope speech: response missing output.audio.url; body=%s", truncate(raw, 512))
	}
	if err = validateDashScopeAudioURL(envelope.Output.Audio.URL); err != nil {
		return nil, nil, tokenUsage, "", err
	}

	audio, err := dashScopeAudioFetcher(context.Background(), envelope.Output.Audio.URL)
	if err != nil {
		return nil, nil, tokenUsage, "", err
	}

	if span != nil {
		span.RecordResponse(&audio)
	}

	newBody = audio
	newHeaders = []internalapi.Header{
		{contentLengthHeaderName, strconv.Itoa(len(audio))},
	}
	responseModel = o.requestModel
	return
}

// ResponseError implements [OpenAISpeechTranslator.ResponseError]. Passes DashScope's error
// body straight through — DashScope uses its own shape (`code`/`message`), and reshaping into
// OpenAI's error envelope is out of scope for the TTS path. If the caller needs a uniform
// error surface we can layer that on later.
func (o *openAIToDashScopeSpeechTranslator) ResponseError(_ map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	newBody, err = io.ReadAll(body)
	if err != nil {
		return nil, nil, fmt.Errorf("dashscope speech: read error response: %w", err)
	}
	newHeaders = []internalapi.Header{
		{"content-type", "application/json"},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// truncate keeps error messages readable when the failing payload is huge.
func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "...(truncated)"
}
