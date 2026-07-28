// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

// Package dashscope contains request/response types for Alibaba Cloud Model
// Studio (DashScope) native APIs. Only the surfaces the gateway currently
// translates against are modelled here; extend as new endpoints are added.
package dashscope

// SpeechRequest is the request body for POST /api/v1/services/aigc/multimodal-generation/generation
// when synthesising speech with the Qwen-TTS / Qwen3-TTS family of models.
//
// Alibaba's OpenAI-compatible mode (/compatible-mode/v1) does not expose /audio/speech,
// so callers of the gateway's /v1/audio/speech endpoint have their body translated into
// this native shape before being forwarded.
//
// https://www.alibabacloud.com/help/en/model-studio/qwen-tts-api
type SpeechRequest struct {
	// Model is the Qwen-TTS model identifier (e.g. "qwen3-tts-flash", "qwen-tts").
	Model string `json:"model"`
	// Input holds the prompt/voice payload.
	Input SpeechInput `json:"input"`
	// Parameters holds optional generation parameters. Currently unused by the translator,
	// present for forward compatibility with fields like sample_rate.
	Parameters *SpeechParameters `json:"parameters,omitempty"`
}

// SpeechInput is the `input` object of a DashScope Qwen-TTS request.
type SpeechInput struct {
	// Text is the text to synthesise. Max 512 tokens per DashScope docs.
	Text string `json:"text"`
	// Voice selects the timbre (e.g. "Cherry", "Ethan"). Required.
	Voice string `json:"voice"`
	// LanguageType is an optional locale hint ("Chinese", "English", "Auto"). Defaults to Auto.
	LanguageType string `json:"language_type,omitempty"`
}

// SpeechParameters is the `parameters` object of a DashScope Qwen-TTS request.
type SpeechParameters struct {
	// SampleRate is the desired audio sample rate. Optional.
	SampleRate int `json:"sample_rate,omitempty"`
}

// SpeechResponse is the non-streaming response body from DashScope's Qwen-TTS endpoint.
//
// The audio itself is not embedded — instead `output.audio.url` points to a signed HTTP
// URL that is valid for ~24 hours. The gateway follows that URL, downloads the bytes,
// and re-emits them to the caller as an OpenAI-shaped binary audio response.
type SpeechResponse struct {
	// RequestID echoes DashScope's server-side request identifier.
	RequestID string `json:"request_id,omitempty"`
	// Output carries the audio pointer.
	Output SpeechOutput `json:"output"`
	// Usage is the token / character usage metadata (structure varies per model family).
	Usage map[string]any `json:"usage,omitempty"`
}

// SpeechOutput is the `output` object of a DashScope Qwen-TTS response.
type SpeechOutput struct {
	// Audio holds the signed audio URL (and an unused base64 payload for non-streaming calls).
	Audio SpeechOutputAudio `json:"audio"`
}

// SpeechOutputAudio is the audio pointer within a DashScope response.
type SpeechOutputAudio struct {
	// URL is a signed HTTP(S) URL to the synthesised audio file. Valid ~24h.
	URL string `json:"url"`
	// Data is a base64-encoded payload. Always empty for non-streaming responses;
	// present only when the client used a streaming form.
	Data string `json:"data,omitempty"`
	// ID is DashScope's audio object identifier.
	ID string `json:"id,omitempty"`
	// ExpiresAt is an optional expiry timestamp on the signed URL (RFC3339).
	ExpiresAt string `json:"expires_at,omitempty"`
}
