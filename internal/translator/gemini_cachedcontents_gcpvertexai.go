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

// geminiCachedContentsToGCPVertexAITranslator passes Gemini cachedContents API requests through
// to GCP Vertex AI unchanged. It does not rewrite the request path or mutate the body — the
// processor forwards the original request, only attaching upstream auth.
type geminiCachedContentsToGCPVertexAITranslator struct{}

// NewGeminiCachedContentsToGCPVertexAITranslator creates a passthrough translator for cachedContents.
func NewGeminiCachedContentsToGCPVertexAITranslator() GeminiCachedContentsTranslator {
	return &geminiCachedContentsToGCPVertexAITranslator{}
}

// RequestBody implements [GeminiCachedContentsTranslator.RequestBody]. It forwards the original
// body unchanged. The processor leaves the request path alone since cachedContents URLs do not
// embed a model name (the model is in the body for POST, and irrelevant for GET/PATCH/DELETE).
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
