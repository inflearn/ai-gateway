// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
)

func TestGeminiCachedContents_RequestBody(t *testing.T) {
	t.Run("no override, no forceBodyMutation: passthrough", func(t *testing.T) {
		tr := NewGeminiCachedContentsToGCPVertexAITranslator("")
		raw := []byte(`{"model":"gemini-2.5-flash"}`)
		headers, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{Model: "gemini-2.5-flash"}, false)
		require.NoError(t, err)
		require.Nil(t, headers)
		require.Nil(t, body)
	})

	t.Run("no override, forceBodyMutation=true: echoes original body", func(t *testing.T) {
		tr := NewGeminiCachedContentsToGCPVertexAITranslator("")
		raw := []byte(`{"ttl":"3600s"}`)
		headers, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{TTL: "3600s"}, true)
		require.NoError(t, err)
		require.Equal(t, raw, body)
		require.Len(t, headers, 1)
		require.Equal(t, contentLengthHeaderName, headers[0][0])
	})

	t.Run("modelNameOverride rewrites short model in full Vertex path", func(t *testing.T) {
		tr := NewGeminiCachedContentsToGCPVertexAITranslator("gemini-3.1-flash-lite-preview")
		raw := []byte(`{"model":"projects/p/locations/global/publishers/google/models/gemini-3.1-flash-lite","ttl":"60s"}`)
		_, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{Model: "projects/p/locations/global/publishers/google/models/gemini-3.1-flash-lite", TTL: "60s"}, false)
		require.NoError(t, err)
		require.Contains(t, string(body), `"model":"projects/p/locations/global/publishers/google/models/gemini-3.1-flash-lite-preview"`)
	})

	t.Run("modelNameOverride replaces bare short model", func(t *testing.T) {
		tr := NewGeminiCachedContentsToGCPVertexAITranslator("gemini-3.1-flash-lite-preview")
		raw := []byte(`{"model":"gemini-3.1-flash-lite"}`)
		_, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{Model: "gemini-3.1-flash-lite"}, false)
		require.NoError(t, err)
		require.Contains(t, string(body), `"model":"gemini-3.1-flash-lite-preview"`)
	})

	t.Run("override equal to original is a no-op", func(t *testing.T) {
		tr := NewGeminiCachedContentsToGCPVertexAITranslator("gemini-2.5-flash")
		raw := []byte(`{"model":"gemini-2.5-flash"}`)
		headers, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{Model: "gemini-2.5-flash"}, false)
		require.NoError(t, err)
		require.Nil(t, headers)
		require.Nil(t, body)
	})

	t.Run("empty body with no model: no rewrite", func(t *testing.T) {
		tr := NewGeminiCachedContentsToGCPVertexAITranslator("gemini-2.5-flash-lite-preview")
		headers, body, err := tr.RequestBody(nil, &gcp.CachedContentRequest{}, false)
		require.NoError(t, err)
		require.Nil(t, headers)
		require.Nil(t, body)
	})
}

func TestRewriteVertexModelName(t *testing.T) {
	tests := []struct {
		name     string
		original string
		override string
		want     string
	}{
		{"short to short", "gemini-2.5-flash", "gemini-2.5-flash-lite-preview", "gemini-2.5-flash-lite-preview"},
		{"vertex path keeps prefix", "projects/p/locations/global/publishers/google/models/gemini-2.5-flash", "gemini-2.5-flash-preview", "projects/p/locations/global/publishers/google/models/gemini-2.5-flash-preview"},
		{"override is full path: passthrough", "gemini-2.5-flash", "projects/p/locations/global/publishers/google/models/gemini-2.5-flash-preview", "projects/p/locations/global/publishers/google/models/gemini-2.5-flash-preview"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, rewriteVertexModelName(tc.original, tc.override))
		})
	}
}

func TestGeminiCachedContents_ResponseHeaders(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator("")
	headers, err := tr.ResponseHeaders(map[string]string{"content-type": "application/json"})
	require.NoError(t, err)
	require.Nil(t, headers)
}

func TestGeminiCachedContents_ResponseBody(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator("")
	body := strings.NewReader(`{"name":"projects/p/locations/us-central1/cachedContents/abc123"}`)
	headers, mutated, usage, model, err := tr.ResponseBody(nil, body, true, nil)
	require.NoError(t, err)
	require.Nil(t, headers)
	require.Nil(t, mutated)
	_, ok := usage.InputTokens()
	require.False(t, ok)
	require.Empty(t, model)
}

func TestGeminiCachedContents_ResponseError(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator("")
	headers, body, err := tr.ResponseError(
		map[string]string{"content-type": "application/json"},
		bytes.NewReader([]byte(`{"error":{"code":404,"message":"not found","status":"NOT_FOUND"}}`)),
	)
	require.NoError(t, err)
	require.NotNil(t, headers)
	require.NotEmpty(t, body)
}
