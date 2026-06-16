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
	newTr := func() *geminiCachedContentsToGCPVertexAITranslator {
		return NewGeminiCachedContentsToGCPVertexAITranslator().(*geminiCachedContentsToGCPVertexAITranslator)
	}

	t.Run("rewrites path to suffix that backend will prefix", func(t *testing.T) {
		tr := newTr()
		tr.SetRequestHeaders(map[string]string{
			":path": "/v1/projects/my-project/locations/global/cachedContents",
		})
		headers, body, err := tr.RequestBody([]byte(`{"model":"gemini-2.5-flash"}`), &gcp.CachedContentRequest{Model: "gemini-2.5-flash"}, true)
		require.NoError(t, err)
		// First header must be the rewritten :path with project/location stripped.
		require.Equal(t, pathHeaderName, headers[0][0])
		require.Equal(t, "cachedContents", headers[0][1])
		require.Equal(t, []byte(`{"model":"gemini-2.5-flash"}`), body)
	})

	t.Run("preserves trailing cache id segment", func(t *testing.T) {
		tr := newTr()
		tr.SetRequestHeaders(map[string]string{
			":path": "/v1/projects/p/locations/us-central1/cachedContents/abc123",
		})
		headers, _, err := tr.RequestBody(nil, &gcp.CachedContentRequest{}, false)
		require.NoError(t, err)
		require.Equal(t, "cachedContents/abc123", headers[0][1])
	})

	t.Run("preserves query parameters (updateMask, pageSize)", func(t *testing.T) {
		tr := newTr()
		tr.SetRequestHeaders(map[string]string{
			":path": "/v1/projects/p/locations/global/cachedContents/abc?updateMask=ttl",
		})
		headers, _, err := tr.RequestBody(nil, &gcp.CachedContentRequest{}, false)
		require.NoError(t, err)
		require.Equal(t, "cachedContents/abc?updateMask=ttl", headers[0][1])
	})

	t.Run("forceBodyMutation=true with body includes content-length header", func(t *testing.T) {
		tr := newTr()
		tr.SetRequestHeaders(map[string]string{
			":path": "/v1/projects/p/locations/global/cachedContents",
		})
		raw := []byte(`{"ttl":"3600s"}`)
		headers, body, err := tr.RequestBody(raw, &gcp.CachedContentRequest{TTL: "3600s"}, true)
		require.NoError(t, err)
		require.Equal(t, raw, body)
		require.Len(t, headers, 2)
		require.Equal(t, pathHeaderName, headers[0][0])
		require.Equal(t, contentLengthHeaderName, headers[1][0])
	})

	t.Run("forceBodyMutation=false omits body and content-length", func(t *testing.T) {
		tr := newTr()
		tr.SetRequestHeaders(map[string]string{
			":path": "/v1/projects/p/locations/global/cachedContents",
		})
		headers, body, err := tr.RequestBody([]byte(`{"x":1}`), &gcp.CachedContentRequest{}, false)
		require.NoError(t, err)
		require.Nil(t, body)
		require.Len(t, headers, 1)
		require.Equal(t, pathHeaderName, headers[0][0])
	})

	t.Run("missing :path returns error", func(t *testing.T) {
		tr := newTr()
		_, _, err := tr.RequestBody(nil, &gcp.CachedContentRequest{}, false)
		require.ErrorContains(t, err, "missing request path")
	})

	t.Run("path without /cachedContents returns error", func(t *testing.T) {
		tr := newTr()
		tr.SetRequestHeaders(map[string]string{":path": "/v1/projects/p/locations/global/something-else"})
		_, _, err := tr.RequestBody(nil, &gcp.CachedContentRequest{}, false)
		require.ErrorContains(t, err, "unexpected cachedContents path")
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
	body := strings.NewReader(`{"name":"projects/p/locations/us-central1/cachedContents/abc123"}`)
	headers, mutated, usage, model, err := tr.ResponseBody(nil, body, true, nil)
	require.NoError(t, err)
	require.Nil(t, headers)
	require.Nil(t, mutated)
	_, ok := usage.InputTokens()
	require.False(t, ok, "passthrough must not set input tokens")
	require.Empty(t, model)
}

func TestGeminiCachedContents_ResponseError(t *testing.T) {
	tr := NewGeminiCachedContentsToGCPVertexAITranslator()
	headers, body, err := tr.ResponseError(
		map[string]string{"content-type": "application/json"},
		bytes.NewReader([]byte(`{"error":{"code":404,"message":"not found","status":"NOT_FOUND"}}`)),
	)
	require.NoError(t, err)
	require.NotNil(t, headers)
	require.NotEmpty(t, body)
}
