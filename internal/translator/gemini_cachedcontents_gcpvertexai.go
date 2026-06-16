// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/tidwall/sjson"

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
// GCP Vertex AI. The path is preserved as-is — backendauth.gcpHandler.Do recognises the
// "/v1/projects/" prefix and skips its own prepend, so the upstream URL stays exactly as the
// client sent it.
//
// When modelNameOverride is set (e.g. via AIGatewayRoute aliasing "gemini-1.5-pro" to a specific
// preview release), the body's "model" field on a POST is rewritten to the override before
// forwarding. This keeps cache-create symmetric with generateContent: the model that lands in
// Vertex's stored cache matches the model later sent on generateContent calls, so the cache is
// usable.
//
// GET / PATCH / DELETE have no body; the translator forwards them unchanged.
type geminiCachedContentsToGCPVertexAITranslator struct {
	modelNameOverride internalapi.ModelNameOverride
}

// NewGeminiCachedContentsToGCPVertexAITranslator creates a passthrough translator for cachedContents.
func NewGeminiCachedContentsToGCPVertexAITranslator(modelNameOverride internalapi.ModelNameOverride) GeminiCachedContentsTranslator {
	return &geminiCachedContentsToGCPVertexAITranslator{modelNameOverride: modelNameOverride}
}

// RequestBody implements [GeminiCachedContentsTranslator.RequestBody]. When modelNameOverride is
// set and the inbound body carries a model field, rewrite it (preserving the Vertex resource
// prefix if present) so the cached content is created against the same backend model
// generateContent will use.
func (g *geminiCachedContentsToGCPVertexAITranslator) RequestBody(
	original []byte, body *gcp.CachedContentRequest, forceBodyMutation bool,
) (newHeaders []internalapi.Header, newBody []byte, err error) {
	if g.modelNameOverride != "" && body != nil && body.Model != "" {
		rewritten := rewriteVertexModelName(body.Model, string(g.modelNameOverride))
		if rewritten != body.Model {
			newBody, err = sjson.SetBytesOptions(original, "model", rewritten, sjsonOptions)
			if err != nil {
				return nil, nil, fmt.Errorf("failed to apply modelNameOverride to cachedContents body: %w", err)
			}
		}
	}
	if forceBodyMutation && len(newBody) == 0 && len(original) > 0 {
		newBody = original
	}
	if len(newBody) > 0 {
		newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(newBody))}}
	}
	return
}

// rewriteVertexModelName replaces the short model name segment in a Vertex resource path while
// preserving the "projects/.../publishers/google/models/" prefix. If the input has no such
// prefix, it returns the override directly. The override itself is also accepted as a full
// resource path, in which case it is returned as-is.
func rewriteVertexModelName(original, override string) string {
	if strings.Contains(override, "/models/") {
		// Caller already supplied a fully-qualified resource name; use it verbatim.
		return override
	}
	const seg = "/models/"
	idx := strings.LastIndex(original, seg)
	if idx == -1 {
		return override
	}
	return original[:idx+len(seg)] + override
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
