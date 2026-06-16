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
	tr := NewGeminiCachedContentsToGCPVertexAITranslator()

	t.Run("forceBodyMutation=false returns no headers/body", func(t *testing.T) {
		// The processor will forward the original bytes when the translator returns nil.
		headers, body, err := tr.RequestBody([]byte(`{"model":"gemini-1.5-pro"}`), &gcp.CachedContentRequest{Model: "gemini-1.5-pro"}, false)
		require.NoError(t, err)
		require.Nil(t, headers)
		require.Nil(t, body)
	})

	t.Run("forceBodyMutation=true returns original body with content-length", func(t *testing.T) {
		raw := []byte(`{"model":"gemini-1.5-pro","ttl":"3600s"}`)
		headers, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{Model: "gemini-1.5-pro", TTL: "3600s"}, true)
		require.NoError(t, err)
		require.Equal(t, raw, body)
		require.Len(t, headers, 1)
		require.Equal(t, contentLengthHeaderName, headers[0][0])
	})

	t.Run("forceBodyMutation=true with empty body returns nothing", func(t *testing.T) {
		// GET/DELETE have empty bodies; nothing to mutate even with forceBodyMutation.
		headers, body, err := tr.RequestBody(nil, &gcp.CachedContentRequest{}, true)
		require.NoError(t, err)
		require.Nil(t, headers)
		require.Nil(t, body)
	})
}

func TestGeminiCachedContents_ResponseHeaders(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator()
	headers, err := tr.ResponseHeaders(map[string]string{"content-type": "application/json"})
	require.NoError(t, err)
	require.Nil(t, headers)
}

func TestGeminiCachedContents_ResponseBody(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator()
	body := strings.NewReader(`{"name":"projects/p/locations/us-central1/cachedContents/abc123","model":"projects/p/locations/us-central1/publishers/google/models/gemini-1.5-pro"}`)
	headers, mutated, usage, model, err := tr.ResponseBody(nil, body, true, nil)
	require.NoError(t, err)
	require.Nil(t, headers)
	require.Nil(t, mutated)
	require.Equal(t, uint32(0), func() uint32 { v, _ := usage.InputTokens(); return v }())
	require.Empty(t, model)
}

func TestGeminiCachedContents_ResponseError(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator()
	// Error path delegates to convertGCPVertexAIErrorToOpenAI; smoke-test it does not panic
	// and returns a non-nil body.
	headers, body, err := tr.ResponseError(
		map[string]string{"content-type": "application/json"},
		bytes.NewReader([]byte(`{"error":{"code":404,"message":"not found","status":"NOT_FOUND"}}`)),
	)
	require.NoError(t, err)
	require.NotNil(t, headers)
	require.NotEmpty(t, body)
}
