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

// geminiCachedContentsToGCPVertexAITranslator forwards cachedContents management calls to GCP
// Vertex AI. The body is never re-marshalled — only the request path is reduced to the suffix
// that gcpHandler.Do (in internal/backendauth/gcp.go) will then prepend with
// "/v1/projects/{project}/locations/{region}".
//
// Without this rewrite the full path the client sent would be prepended again, producing
// a doubled URL like "/v1/projects/A/locations/B//v1/projects/X/locations/Y/cachedContents".
type geminiCachedContentsToGCPVertexAITranslator struct {
	// originalPath is captured from the incoming :path header via SetRequestHeaders so RequestBody
	// can strip the project/location prefix before forwarding to the backend.
	originalPath string
}

// NewGeminiCachedContentsToGCPVertexAITranslator creates a passthrough translator for cachedContents.
func NewGeminiCachedContentsToGCPVertexAITranslator() GeminiCachedContentsTranslator {
	return &geminiCachedContentsToGCPVertexAITranslator{}
}

// SetRequestHeaders captures the original request path. Implements [RequestHeadersSetter].
func (g *geminiCachedContentsToGCPVertexAITranslator) SetRequestHeaders(headers map[string]string) {
	g.originalPath = headers[":path"]
}

// RequestBody implements [GeminiCachedContentsTranslator.RequestBody]. It rewrites the request
// path to the cachedContents suffix (everything from "/cachedContents" onward) so the backend
// auth handler can prepend the configured project/location prefix.
func (g *geminiCachedContentsToGCPVertexAITranslator) RequestBody(
	original []byte, _ *gcp.CachedContentRequest, forceBodyMutation bool,
) (newHeaders []internalapi.Header, newBody []byte, err error) {
	suffix, sErr := extractCachedContentsPathSuffix(g.originalPath)
	if sErr != nil {
		return nil, nil, sErr
	}

	newHeaders = []internalapi.Header{{pathHeaderName, suffix}}
	if forceBodyMutation && len(original) > 0 {
		newBody = original
		newHeaders = append(newHeaders, internalapi.Header{contentLengthHeaderName, strconv.Itoa(len(newBody))})
	}
	return
}

// extractCachedContentsPathSuffix strips the "/v1/projects/{p}/locations/{l}" prefix from a
// Vertex AI cachedContents path and returns the remainder (e.g. "cachedContents/abc?updateMask=ttl").
// gcpHandler.Do will prepend "/v1/projects/{configured-project}/locations/{configured-region}" to
// produce the final upstream path.
func extractCachedContentsPathSuffix(rawPath string) (string, error) {
	if rawPath == "" {
		return "", fmt.Errorf("missing request path for cachedContents")
	}
	idx := strings.Index(rawPath, "/cachedContents")
	if idx == -1 {
		return "", fmt.Errorf("unexpected cachedContents path: %q", rawPath)
	}
	// Drop the leading slash so the backend's prefix join yields exactly one slash.
	return rawPath[idx+1:], nil
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
