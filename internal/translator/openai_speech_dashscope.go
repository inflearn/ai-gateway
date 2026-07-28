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
	"strconv"
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
// - voice: passed through verbatim — the AIGatewayRoute owner is expected to supply a
//   DashScope-side voice name (e.g. "Cherry"); OpenAI voice presets do not have a natural
//   mapping and forcing one here would silently pick the wrong timbre.
// - instructions/speed: DashScope has no equivalent.
// - response_format: DashScope's non-streaming response is always WAV via URL; ignored here
//   and the outbound content-type is set to audio/wav in ResponseHeaders.
// - stream_format: only non-streaming is implemented in this first cut; SSE support would
//   require calling DashScope's streaming synthesis API (different endpoint) and is out of
//   scope until we see demand.
func (o *openAIToDashScopeSpeechTranslator) RequestBody(original []byte, req *openai.SpeechRequest, forceBodyMutation bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	if req == nil {
		return nil, nil, fmt.Errorf("dashscope speech: nil request")
	}

	model := req.Model
	if o.modelNameOverride != "" {
		model = string(o.modelNameOverride)
	}
	o.requestModel = model

	dsReq := dashscope.SpeechRequest{
		Model: model,
		Input: dashscope.SpeechInput{
			Text:  req.Input,
			Voice: req.Voice,
		},
	}
	newBody, err = json.Marshal(dsReq)
	if err != nil {
		return nil, nil, fmt.Errorf("dashscope speech: marshal request: %w", err)
	}

	newHeaders = []internalapi.Header{
		{pathHeaderName, o.path},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	_ = forceBodyMutation // always a shape change → always mutate.
	return
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
