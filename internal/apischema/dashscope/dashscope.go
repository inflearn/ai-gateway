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
	// LanguageType is an optional locale hint. Valid: Auto, Chinese, English, German, Italian,
	// Portuguese, Spanish, Japanese, Korean, French, Russian. Defaults to Auto server-side.
	LanguageType string `json:"language_type,omitempty"`
	// Instructions is a natural-language behaviour hint, ≤ 1600 tokens.
	// Only qwen3-tts-instruct-flash actually consumes this; other models ignore it.
	Instructions string `json:"instructions,omitempty"`
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

// CosyVoiceSpeechRequest is the request body for POST /api/v1/services/audio/tts/SpeechSynthesizer,
// the CosyVoice / Qwen-Audio-TTS HTTP endpoint. This is a *different* URL from Qwen-TTS —
// supports mp3/wav/pcm/opus output, selectable sample rate, and speech rate/pitch controls.
//
// https://help.aliyun.com/en/model-studio/cosyvoice-tts-http-api
type CosyVoiceSpeechRequest struct {
	// Model — e.g. qwen-audio-3.0-tts-flash, cosyvoice-v3-flash, cosyvoice-v2.
	Model string `json:"model"`
	// Input carries the text/voice/format payload.
	Input CosyVoiceSpeechInput `json:"input"`
}

// CosyVoiceSpeechInput is the `input` object of a CosyVoice / Qwen-Audio-TTS request.
//
// The gateway forwards a modest subset of the vendor's fields — only those we actually map from
// OpenAI's SpeechRequest today. Add fields here as clients start needing them; the wire format
// is stable per Alibaba's docs.
type CosyVoiceSpeechInput struct {
	// Text is the content to synthesise.
	Text string `json:"text"`
	// Voice selects the timbre (built-in name like "longxiaochun" or a cloned voice_id).
	Voice string `json:"voice"`
	// Format is the output audio format. Valid: "mp3" (default), "pcm", "wav", "opus".
	Format string `json:"format,omitempty"`
	// SampleRate in Hz. Valid: 8000, 16000, 22050 (default), 24000, 44100, 48000.
	SampleRate int `json:"sample_rate,omitempty"`
	// Rate is the speech rate. Range 0.5–2.0.
	Rate float64 `json:"rate,omitempty"`
	// Pitch. Range 0.5–2.0.
	Pitch float64 `json:"pitch,omitempty"`
	// LanguageHints is an optional array of language codes to bias pronunciation.
	LanguageHints []string `json:"language_hints,omitempty"`
	// Instruction is a natural-language behaviour hint (singular — Qwen-TTS uses `instructions`).
	Instruction string `json:"instruction,omitempty"`
}

// VoiceEnrollmentRequest is the request body for POST /api/v1/services/audio/tts/customization —
// the DashScope voice-cloning management endpoint. A single URL hosts all five actions
// (create_voice, list_voice, query_voice, update_voice, delete_voice); which one is executed
// is decided by `input.action` in the body.
//
// The translator only needs `model` (routing) and `input.action` (observability/metrics); every
// other field flows through as raw JSON. Callers speak DashScope's native schema — this
// endpoint has no OpenAI counterpart to translate against.
//
// https://www.alibabacloud.com/help/en/model-studio/voice-clone-design-http-api
type VoiceEnrollmentRequest struct {
	// Model is always "voice-enrollment" for this endpoint; kept in the parsed request so the
	// gateway can populate x-ai-eg-model for route matching.
	Model string `json:"model"`
	// Input carries the per-action payload. Only `action` is inspected here for logging /
	// metrics tagging; the remaining fields (target_model, prefix, url, voice_id, …) are left
	// to flow through as raw JSON.
	Input VoiceEnrollmentInput `json:"input"`
}

// VoiceEnrollmentInput mirrors the `input` object of a voice-enrollment request. Only the
// action tag is modelled — everything else is per-action noise that the gateway forwards
// unchanged.
type VoiceEnrollmentInput struct {
	// Action selects the operation: create_voice | list_voice | query_voice | update_voice | delete_voice.
	Action string `json:"action"`
}

// SpeechOutputAudio is the audio pointer within a DashScope response.
//
// Note: DashScope's non-streaming reply also carries an `expires_at` field encoded as a Unix
// timestamp (number, e.g. 1785313180). We intentionally do NOT model it here — the translator
// only follows `url` and does not care when the signed URL expires (the client either fetches
// via the gateway immediately or the request errors out). Modelling it as string would fail
// json.Unmarshal against the actual number value.
type SpeechOutputAudio struct {
	// URL is a signed HTTP(S) URL to the synthesised audio file. Valid ~24h.
	URL string `json:"url"`
	// Data is a base64-encoded payload. Always empty for non-streaming responses;
	// present only when the client used a streaming form.
	Data string `json:"data,omitempty"`
	// ID is DashScope's audio object identifier.
	ID string `json:"id,omitempty"`
}
