// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/dashscope"
)

func TestDashScopeVoiceEnrollment_RequestBody(t *testing.T) {
	tr := NewDashScopeVoiceEnrollmentTranslator()
	req := &dashscope.VoiceEnrollmentRequest{
		Model: "voice-enrollment",
		Input: dashscope.VoiceEnrollmentInput{Action: "create_voice"},
	}
	original := []byte(`{"model":"voice-enrollment","input":{"action":"create_voice","target_model":"qwen3-tts-flash","prefix":"pk","url":"https://example.com/a.wav"}}`)

	t.Run("path is rewritten, body is not replayed by default", func(t *testing.T) {
		hm, body, err := tr.RequestBody(original, req, false)
		require.NoError(t, err)
		require.Empty(t, body, "no forced mutation → translator should not carry the body itself")

		var pathVal string
		for _, h := range hm {
			if h.Key() == ":path" {
				pathVal = h.Value()
			}
		}
		require.Equal(t, dashScopeVoiceEnrollmentPath, pathVal)
	})

	t.Run("forceBodyMutation replays the original body verbatim", func(t *testing.T) {
		hm, body, err := tr.RequestBody(original, req, true)
		require.NoError(t, err)
		require.Equal(t, original, body, "forced replay must preserve caller's payload byte-for-byte")

		var cl string
		for _, h := range hm {
			if h.Key() == "content-length" {
				cl = h.Value()
			}
		}
		require.NotEmpty(t, cl)
	})
}

func TestDashScopeVoiceEnrollment_ResponseHeaders(t *testing.T) {
	tr := NewDashScopeVoiceEnrollmentTranslator()
	hm, err := tr.ResponseHeaders(nil)
	require.NoError(t, err)
	require.Empty(t, hm, "response headers pass through untouched")
}

func TestDashScopeVoiceEnrollment_ResponseBody(t *testing.T) {
	tr := NewDashScopeVoiceEnrollmentTranslator()
	envelope := `{"request_id":"rid","output":{"voice_id":"custom-abc"},"usage":{"count":1}}`
	hm, body, tokens, respModel, err := tr.ResponseBody(nil, strings.NewReader(envelope), true, nil)
	require.NoError(t, err)
	require.Empty(t, hm)
	require.Empty(t, body, "pass-through: envoy delivers upstream body directly, no mutation")
	require.Equal(t, "voice-enrollment", respModel)
	_, set := tokens.OutputTokens()
	require.False(t, set, "voice-enrollment carries no token accounting")
}

func TestDashScopeVoiceEnrollment_ResponseError(t *testing.T) {
	tr := NewDashScopeVoiceEnrollmentTranslator()
	hm, body, err := tr.ResponseError(nil, strings.NewReader(`{"code":"InvalidParameter","message":"missing url"}`))
	require.NoError(t, err)
	require.Empty(t, hm)
	require.Empty(t, body, "error envelope passes through unchanged")
}
