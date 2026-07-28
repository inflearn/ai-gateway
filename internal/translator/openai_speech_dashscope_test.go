// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

func TestSpeechOpenAIToDashScope_RequestBody(t *testing.T) {
	t.Run("shape and headers", func(t *testing.T) {
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		req := &openai.SpeechRequest{
			Model: "qwen3-tts-flash",
			Input: "안녕",
			Voice: "Cherry",
		}
		hm, body, err := tr.RequestBody(nil, req, false)
		require.NoError(t, err)

		// Path must be rewritten to DashScope's native endpoint; content-length must match body.
		var pathVal, cl string
		for _, h := range hm {
			switch h.Key() {
			case ":path":
				pathVal = h.Value()
			case "content-length":
				cl = h.Value()
			}
		}
		require.Equal(t, dashScopeSpeechPath, pathVal)
		require.NotEmpty(t, cl)

		var out map[string]any
		require.NoError(t, json.Unmarshal(body, &out))
		require.Equal(t, "qwen3-tts-flash", out["model"])
		input := out["input"].(map[string]any)
		require.Equal(t, "안녕", input["text"])
		require.Equal(t, "Cherry", input["voice"])
	})

	t.Run("modelNameOverride wins over request.model", func(t *testing.T) {
		tr := NewSpeechOpenAIToDashScopeTranslator("qwen3-tts-flash")
		req := &openai.SpeechRequest{Model: "tts-1", Input: "hi", Voice: "Ethan"}
		_, body, err := tr.RequestBody(nil, req, false)
		require.NoError(t, err)

		var out map[string]any
		require.NoError(t, json.Unmarshal(body, &out))
		require.Equal(t, "qwen3-tts-flash", out["model"], "override must replace the client-supplied model")
	})

	t.Run("nil request errors", func(t *testing.T) {
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, err := tr.RequestBody(nil, nil, false)
		require.Error(t, err)
	})
}

func TestSpeechOpenAIToDashScope_ResponseHeaders(t *testing.T) {
	tr := NewSpeechOpenAIToDashScopeTranslator("")
	hm, err := tr.ResponseHeaders(nil)
	require.NoError(t, err)
	require.Len(t, hm, 1)
	require.Equal(t, "content-type", hm[0].Key())
	require.Equal(t, dashScopeAudioContentType, hm[0].Value())
}

func TestSpeechOpenAIToDashScope_ResponseBody(t *testing.T) {
	// Stub the URL fetcher so tests don't hit the network. Restore at end.
	orig := dashScopeAudioFetcher
	t.Cleanup(func() { dashScopeAudioFetcher = orig })

	t.Run("happy path: JSON envelope → audio bytes", func(t *testing.T) {
		audioPayload := []byte("RIFF....WAVE.fakeaudio.")
		var fetchedURL string
		dashScopeAudioFetcher = func(_ context.Context, url string) ([]byte, error) {
			fetchedURL = url
			return audioPayload, nil
		}

		tr := NewSpeechOpenAIToDashScopeTranslator("")
		// Fake the RequestBody call so requestModel is populated.
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "qwen3-tts-flash", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		envelope := `{"request_id":"rid","output":{"audio":{"url":"https://dashscope-intl.aliyuncs.com/audio/abc.wav","id":"aid"}},"usage":{"characters":3}}`
		hm, body, _, respModel, err := tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.NoError(t, err)
		require.Equal(t, audioPayload, body)
		require.Equal(t, "qwen3-tts-flash", string(respModel))
		require.Equal(t, "https://dashscope-intl.aliyuncs.com/audio/abc.wav", fetchedURL)

		// content-length must match downloaded audio byte length.
		var cl string
		for _, h := range hm {
			if h.Key() == "content-length" {
				cl = h.Value()
			}
		}
		require.Equal(t, "23", cl)
	})

	t.Run("missing audio URL surfaces error", func(t *testing.T) {
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		envelope := `{"request_id":"rid","output":{"audio":{"url":""}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "missing output.audio.url")
	})

	t.Run("fetcher error propagates", func(t *testing.T) {
		dashScopeAudioFetcher = func(_ context.Context, _ string) ([]byte, error) {
			return nil, errors.New("boom")
		}
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)

		envelope := `{"output":{"audio":{"url":"https://dashscope-intl.aliyuncs.com/x"}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "boom")
	})

	t.Run("malformed JSON surfaces error", func(t *testing.T) {
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, _, _, err := tr.ResponseBody(nil, strings.NewReader("not json"), true, nil)
		require.Error(t, err)
	})

	// SSRF guard: reject anything that isn't https on an *.aliyuncs.com host. Runs before
	// the fetcher, so the stub isn't consulted.
	t.Run("rejects non-https audio URL", func(t *testing.T) {
		called := false
		dashScopeAudioFetcher = func(_ context.Context, _ string) ([]byte, error) {
			called = true
			return nil, nil
		}
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)
		envelope := `{"output":{"audio":{"url":"http://dashscope-intl.aliyuncs.com/x"}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "scheme must be https")
		require.False(t, called, "fetcher must not run when URL fails validation")
	})

	t.Run("rejects non-aliyuncs.com host", func(t *testing.T) {
		called := false
		dashScopeAudioFetcher = func(_ context.Context, _ string) ([]byte, error) {
			called = true
			return nil, nil
		}
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)
		envelope := `{"output":{"audio":{"url":"https://evil.example.com/x"}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "not in allowed")
		require.False(t, called, "fetcher must not run when URL fails validation")
	})

	t.Run("rejects lookalike aliyuncs.com host", func(t *testing.T) {
		// The suffix check must anchor with the leading dot so a host like
		// `evil-aliyuncs.com` is not accepted.
		tr := NewSpeechOpenAIToDashScopeTranslator("")
		_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
		require.NoError(t, err)
		envelope := `{"output":{"audio":{"url":"https://evil-aliyuncs.com/x"}}}`
		_, _, _, _, err = tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
		require.Error(t, err)
		require.Contains(t, err.Error(), "not in allowed")
	})
}

func TestSpeechOpenAIToDashScope_ResponseError(t *testing.T) {
	tr := NewSpeechOpenAIToDashScopeTranslator("")
	body := bytes.NewBufferString(`{"code":"InvalidParameter","message":"voice missing"}`)
	hm, newBody, err := tr.ResponseError(nil, body)
	require.NoError(t, err)
	require.Contains(t, string(newBody), "InvalidParameter")

	// content-type stays JSON, content-length matches.
	found := map[string]string{}
	for _, h := range hm {
		found[h.Key()] = h.Value()
	}
	require.Equal(t, "application/json", found["content-type"])
	require.Equal(t, "53", found["content-length"])
}

// Sanity: ensure the io.Reader path handles a body slightly larger than usual streaming boundaries.
func TestSpeechOpenAIToDashScope_ResponseBody_LargeReader(t *testing.T) {
	orig := dashScopeAudioFetcher
	t.Cleanup(func() { dashScopeAudioFetcher = orig })
	dashScopeAudioFetcher = func(_ context.Context, _ string) ([]byte, error) {
		return bytes.Repeat([]byte{0x0a}, 128*1024), nil
	}

	tr := NewSpeechOpenAIToDashScopeTranslator("")
	_, _, err := tr.RequestBody(nil, &openai.SpeechRequest{Model: "m", Input: "x", Voice: "v"}, false)
	require.NoError(t, err)

	envelope := `{"output":{"audio":{"url":"https://dashscope-intl.aliyuncs.com/ex"}}}`
	_, body, _, _, err := tr.ResponseBody(nil, io.NopCloser(strings.NewReader(envelope)), true, nil)
	require.NoError(t, err)
	require.Len(t, body, 128*1024)
}
