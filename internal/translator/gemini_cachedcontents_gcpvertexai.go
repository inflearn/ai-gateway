// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"io"
	"strconv"

	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// GeminiCachedContentsSpan is the span type for Gemini cachedContents passthrough.
// struct{} is used because the translator passes responses through without conversion.
type GeminiCachedContentsSpan = tracingapi.Span[struct{}, struct{}]

// GeminiCachedContentsTranslator translates Gemini cachedContents API requests to GCP Vertex AI.
type GeminiCachedContentsTranslator = Translator[gcp.CachedContentRequest, GeminiCachedContentsSpan]

// geminiCachedContentsToGCPVertexAITranslator forwards cachedContents management calls (CRUD) to
// GCP Vertex AI without modifying the request. The full URL — including project, location, and
// the cachedContents resource path — is sent by the client and preserved end-to-end.
//
// Path prefix handling: cachedContents requests arrive with a full
// "/v1/projects/{p}/locations/{r}/cachedContents/..." path. The shared GCP backend auth handler
// (internal/backendauth/gcp.go) detects this prefix and skips prepending its configured
// project/location, so the upstream URL stays exactly as the client sent it. This keeps
// generateContent's short-suffix convention working untouched while letting cachedContents
// use full Google-style paths.
//
// GET/PATCH/DELETE have no request body — Envoy will not invoke ProcessRequestBody for them, so
// no translator method runs at all in that case. The unchanged path is enough on its own.
type geminiCachedContentsToGCPVertexAITranslator struct{}

// NewGeminiCachedContentsToGCPVertexAITranslator creates a passthrough translator for cachedContents.
func NewGeminiCachedContentsToGCPVertexAITranslator() GeminiCachedContentsTranslator {
	return &geminiCachedContentsToGCPVertexAITranslator{}
}

// RequestBody implements [GeminiCachedContentsTranslator.RequestBody]. Pure passthrough — no
// path or body mutation. When forceBodyMutation is set (retry path) we echo the original body
// back so the upstream filter forwards it.
func (g *geminiCachedContentsToGCPVertexAITranslator) RequestBody(
	original []byte, _ *gcp.CachedContentRequest, forceBodyMutation bool,
) (newHeaders []internalapi.Header, newBody []byte, err error) {
	if forceBodyMutation && len(original) > 0 {
		newBody = original
		newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(newBody))}}
	}
	return
}

// ResponseHeaders implements [GeminiCachedContentsTranslator.ResponseHeaders].
func (g *geminiCachedContentsToGCPVertexAITranslator) ResponseHeaders(_ map[string]string) (
	newHeaders []internalapi.Header, err error,
) {
	return nil, nil
}

// ResponseBody implements [GeminiCachedContentsTranslator.ResponseBody]. cachedContents responses
// are not streamed and carry no usage metadata that maps onto LLM token accounting; the response
// is returned to the client unmodified.
func (g *geminiCachedContentsToGCPVertexAITranslator) ResponseBody(
	_ map[string]string, _ io.Reader, _ bool, _ GeminiCachedContentsSpan,
) (newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error) {
	return nil, nil, metrics.TokenUsage{}, "", nil
}

// ResponseError implements [GeminiCachedContentsTranslator.ResponseError]. Converts GCP Vertex AI
// error responses to OpenAI-compatible error format for consistency with other Gemini paths.
func (g *geminiCachedContentsToGCPVertexAITranslator) ResponseError(
	respHeaders map[string]string, body io.Reader,
) (newHeaders []internalapi.Header, newBody []byte, err error) {
	return convertGCPVertexAIErrorToOpenAI(respHeaders, body)
}
