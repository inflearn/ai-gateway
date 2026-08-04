// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"context"
	"fmt"
	"io"
	"strconv"

	"github.com/envoyproxy/ai-gateway/internal/apischema/dashscope"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// cosyVoiceSpeechPath is CosyVoice / Qwen-Audio-TTS HTTP endpoint. Distinct from Qwen-TTS's
// multimodal-generation path — same workspace, different URL.
//
// https://help.aliyun.com/en/model-studio/cosyvoice-tts-http-api
const cosyVoiceSpeechPath = "/api/v1/services/audio/tts/SpeechSynthesizer"

// cosyVoiceDefaultFormat mirrors the vendor default so we can reason about the outbound
// Content-Type in ResponseHeaders without waiting for the request body.
const cosyVoiceDefaultFormat = "mp3"

// cosyVoiceFormatContentType maps CosyVoice's `format` enum to the audio MIME type the caller
// should see. Every value corresponds to a supported CosyVoice output format per Alibaba's docs.
var cosyVoiceFormatContentType = map[string]string{
	"mp3":  "audio/mpeg",
	"wav":  "audio/wav",
	"pcm":  "audio/pcm",
	"opus": "audio/opus",
}

// NewSpeechOpenAIToCosyVoiceTranslator translates OpenAI /v1/audio/speech requests into the
// CosyVoice / Qwen-Audio-TTS HTTP endpoint (`/api/v1/services/audio/tts/SpeechSynthesizer`) so
// clients keep the OpenAI contract while getting CosyVoice's richer surface (mp3/wav/pcm/opus,
// selectable sample rate, speech rate).
func NewSpeechOpenAIToCosyVoiceTranslator(modelNameOverride internalapi.ModelNameOverride) OpenAISpeechTranslator {
	return &openAIToCosyVoiceSpeechTranslator{modelNameOverride: modelNameOverride}
}

type openAIToCosyVoiceSpeechTranslator struct {
	modelNameOverride internalapi.ModelNameOverride
	// requestModel is echoed as the response's `responseModel` — CosyVoice's response envelope
	// does not name the model.
	requestModel internalapi.RequestModel
	// format captures the caller's response_format (defaulted to mp3) so ResponseHeaders knows
	// which Content-Type to emit for the downloaded audio.
	format string
}

// RequestBody implements [OpenAISpeechTranslator.RequestBody].
//
// OpenAI shape → CosyVoice shape mapping:
//   - model             → model                                          (modelNameOverride wins)
//   - input   (string)  → input.text
//   - voice             → input.voice                                    (passthrough — DashScope voice IDs)
//   - response_format   → input.format   (mp3/wav/pcm/opus; default mp3)
//   - speed             → input.rate     (both are 0.5–2.0 multipliers)
//   - instructions      → input.instruction   (singular — CosyVoice differs from Qwen-TTS)
//   - stream_format     → currently ignored (SSE would need X-DashScope-SSE header handling)
//
// The Accept header is forced to application/json because DashScope rejects the
// application/octet-stream that OpenAI SDKs send by default; caller sees audio bytes via the
// ResponseBody path anyway.
func (o *openAIToCosyVoiceSpeechTranslator) RequestBody(_ []byte, req *openai.SpeechRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	if req == nil {
		return nil, nil, fmt.Errorf("cosyvoice speech: nil request")
	}

	model := req.Model
	if o.modelNameOverride != "" {
		model = o.modelNameOverride
	}
	o.requestModel = model

	format := cosyVoiceDefaultFormat
	if req.ResponseFormat != nil && *req.ResponseFormat != "" {
		// The OpenAI enum is a superset of CosyVoice's; unknown values (e.g. flac, aac) fall
		// through untouched so DashScope can reject them explicitly rather than us guessing.
		format = *req.ResponseFormat
	}
	o.format = format

	input := dashscope.CosyVoiceSpeechInput{
		Text:   req.Input,
		Voice:  req.Voice,
		Format: format,
	}
	if req.Speed != nil {
		input.Rate = *req.Speed
	}
	if req.Instructions != nil && *req.Instructions != "" {
		input.Instruction = *req.Instructions
	}

	newBody, err = json.Marshal(dashscope.CosyVoiceSpeechRequest{Model: model, Input: input})
	if err != nil {
		return nil, nil, fmt.Errorf("cosyvoice speech: marshal request: %w", err)
	}

	newHeaders = []internalapi.Header{
		{pathHeaderName, cosyVoiceSpeechPath},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
		// Force the DashScope-supported Accept. See openai_speech_dashscope.go for the same
		// override rationale — OpenAI SDKs send application/octet-stream which upstream rejects.
		{"accept", "application/json"},
	}
	return
}

// ResponseHeaders implements [OpenAISpeechTranslator.ResponseHeaders].
//
// Upstream returns application/json (the signed-URL envelope), but the client contract is
// binary audio. Rewrite Content-Type to whatever format we asked for. Falls back to
// audio/mpeg if the caller supplied an unknown format string — same permissive behaviour as
// RequestBody.
func (o *openAIToCosyVoiceSpeechTranslator) ResponseHeaders(_ map[string]string) (newHeaders []internalapi.Header, err error) {
	ct, ok := cosyVoiceFormatContentType[o.format]
	if !ok {
		ct = cosyVoiceFormatContentType[cosyVoiceDefaultFormat]
	}
	return []internalapi.Header{{"content-type", ct}}, nil
}

// ResponseBody implements [OpenAISpeechTranslator.ResponseBody]. Same JSON-envelope → audio URL
// → binary bytes pattern as the Qwen-TTS translator. The signed URL host allowlist and fetcher
// are shared (see openai_speech_dashscope.go) — both endpoints serve audio from the same
// aliyuncs.com infrastructure.
func (o *openAIToCosyVoiceSpeechTranslator) ResponseBody(_ map[string]string, body io.Reader, _ bool, span tracingapi.SpeechSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	raw, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("cosyvoice speech: read response: %w", err)
	}
	var envelope dashscope.SpeechResponse
	if err = json.Unmarshal(raw, &envelope); err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("cosyvoice speech: parse response JSON: %w", err)
	}
	if envelope.Output.Audio.URL == "" {
		return nil, nil, tokenUsage, "", fmt.Errorf("cosyvoice speech: response missing output.audio.url; body=%s", truncate(raw, 512))
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
	newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(audio))}}
	responseModel = o.requestModel
	return
}

// ResponseError implements [OpenAISpeechTranslator.ResponseError]. Passes DashScope's error
// envelope through unchanged — same trade-off as the Qwen-TTS translator.
func (o *openAIToCosyVoiceSpeechTranslator) ResponseError(_ map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	newBody, err = io.ReadAll(body)
	if err != nil {
		return nil, nil, fmt.Errorf("cosyvoice speech: read error response: %w", err)
	}
	newHeaders = []internalapi.Header{
		{"content-type", "application/json"},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}
