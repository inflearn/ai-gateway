// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

func TestSpeechOpenAIToCosyVoice_RequestBody(t *testing.T) {
	t.Run("shape / headers / format mapping", func(t *testing.T) {
		tr := NewSpeechOpenAIToCosyVoiceTranslator("")
		fmtWav := "wav"
		speed := 1.25
		instr := "천천히 또박또박"
		req := &openai.SpeechRequest{
			Model:          "qwen-audio-3.0-tts-flash",
			Input:          "안녕",
			Voice:          "longxiaochun",
			ResponseFormat: &fmtWav,
			Speed:          &speed,
			Instructions:   &instr,
		}
		hm, body, err := tr.RequestBody(nil, req, false)
		require.NoError(t, err)

		var pathVal, cl, accept string
		for _, h := range hm {
			switch h.Key() {
			case ":path":
				pathVal = h.Value()
			case "content-length":
				cl = h.Value()
			case "accept":
				accept = h.Value()
			}
		}
		require.Equal(t, cosyVoiceSpeechPath, pathVal)
		require.NotEmpty(t, cl)
		require.Equal(t, "application/json", accept, "Accept must be overridden to satisfy DashScope")

		var out map[string]any
		require.NoError(t, json.Unmarshal(body, &out))
		require.Equal(t, "qwen-audio-3.0-tts-flash", out["model"])
		in := out["input"].(map[string]any)
		require.Equal(t, "안녕", in["text"])
		require.Equal(t, "longxiaochun", in["voice"])
		require.Equal(t, "wav", in["format"])
		require.InDelta(t, 1.25, in["rate"], 0.0001)
		require.Equal(t, "천천히 또박또박", in["instruction"])
	})

	t.Run("default format is mp3 when response_format omitted", func(t *testing.T) {
		tr := NewSpeechOpenAIToCosyVoiceTranslator("")
		_, body, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "cosyvoice-v3-flash", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		var out map[string]any
		require.NoError(t, json.Unmarshal(body, &out))
		require.Equal(t, "mp3", out["input"].(map[string]any)["format"], "default must match Alibaba's default")
	})

	t.Run("modelNameOverride wins over request.model", func(t *testing.T) {
		tr := NewSpeechOpenAIToCosyVoiceTranslator("cosyvoice-v3-flash")
		_, body, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "tts-1", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		var out map[string]any
		require.NoError(t, json.Unmarshal(body, &out))
		require.Equal(t, "cosyvoice-v3-flash", out["model"])
	})

	t.Run("nil request errors", func(t *testing.T) {
		tr := NewSpeechOpenAIToCosyVoiceTranslator("")
		_, _, err := tr.RequestBody(nil, nil, false)
		require.Error(t, err)
	})
}

func TestSpeechOpenAIToCosyVoice_ResponseHeaders(t *testing.T) {
	// content-type follows the caller's requested format.
	cases := map[string]string{
		"mp3":  "audio/mpeg",
		"wav":  "audio/wav",
		"pcm":  "audio/pcm",
		"opus": "audio/opus",
		"":     "audio/mpeg", // uninitialised → default
	}
	for reqFmt, wantCT := range cases {
		t.Run("format="+reqFmt, func(t *testing.T) {
			tr := NewSpeechOpenAIToCosyVoiceTranslator("").(*openAIToCosyVoiceSpeechTranslator)
			tr.format = reqFmt
			hm, err := tr.ResponseHeaders(nil)
			require.NoError(t, err)
			require.Len(t, hm, 1)
			require.Equal(t, "content-type", hm[0].Key())
			require.Equal(t, wantCT, hm[0].Value())
		})
	}
}

func TestSpeechOpenAIToCosyVoice_ResponseBody(t *testing.T) {
	orig := dashScopeAudioFetcher
	t.Cleanup(func() { dashScopeAudioFetcher = orig })

	t.Run("JSON envelope → audio bytes", func(t *testing.T) {
		payload := []byte("ID3fake_mp3")
		dashScopeAudioFetcher = func(_ context.Context, _ string) ([]byte, error) { return payload, nil }

		tr := NewSpeechOpenAIToCosyVoiceTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "cosyvoice-v3-flash", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		envelope := `{"request_id":"rid","output":{"audio":{"url":"https://dashscope-intl.aliyuncs.com/a.mp3","id":"aid"}},"usage":{"characters":1}}`
		hm, body, _, respModel, err := tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.NoError(t, err)
		require.Equal(t, payload, body)
		require.Equal(t, "cosyvoice-v3-flash", respModel)

		var cl string
		for _, h := range hm {
			if h.Key() == "content-length" {
				cl = h.Value()
			}
		}
		require.Equal(t, "11", cl)
	})

	t.Run("missing audio URL surfaces error", func(t *testing.T) {
		tr := NewSpeechOpenAIToCosyVoiceTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		envelope := `{"output":{"audio":{"url":""}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "missing output.audio.url")
	})

	t.Run("SSRF guard rejects non-aliyuncs host", func(t *testing.T) {
		tr := NewSpeechOpenAIToCosyVoiceTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		envelope := `{"output":{"audio":{"url":"https://evil.example.com/a.mp3"}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "not in allowed")
	})
}
