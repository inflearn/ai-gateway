// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"fmt"
	"io"
	"strconv"

	"google.golang.org/genai"

	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// GeminiGenerateContentSpan is the span type used for Gemini native passthrough requests.
// struct{} is used because the translator passes responses through without OpenAI conversion,
// so there is no response object to record in a span.
type GeminiGenerateContentSpan = tracingapi.Span[struct{}, struct{}]

// GeminiGenerateContentTranslator translates Gemini native API requests to GCP Vertex AI.
type GeminiGenerateContentTranslator = Translator[gcp.GenerateContentRequest, GeminiGenerateContentSpan]

// geminiToGCPVertexAITranslator passes Gemini native API requests through to GCP Vertex AI,
// preserving the request body as-is while fixing up the request path and extracting
// token usage from responses for metrics.
type geminiToGCPVertexAITranslator struct {
	modelNameOverride internalapi.ModelNameOverride
	requestModel      internalapi.RequestModel
	stream            bool
	streamDelimiter   []byte
	bufferedBody      []byte
}

// NewGeminiToGCPVertexAITranslator creates a new Gemini native API passthrough translator.
func NewGeminiToGCPVertexAITranslator(modelNameOverride internalapi.ModelNameOverride) GeminiGenerateContentTranslator {
	return &geminiToGCPVertexAITranslator{modelNameOverride: modelNameOverride}
}

// RequestBody implements [GeminiGenerateContentTranslator.RequestBody].
// It sets the correct Vertex AI path and re-marshals the body (json:"-" internal fields are excluded).
func (g *geminiToGCPVertexAITranslator) RequestBody(_ []byte, body *gcp.GenerateContentRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	g.requestModel = internalapi.RequestModel(body.Model)
	if g.modelNameOverride != "" {
		g.requestModel = g.modelNameOverride
	}
	g.stream = body.Stream

	var path string
	if g.stream {
		path = buildGCPModelPathSuffix(gcpModelPublisherGoogle, string(g.requestModel), gcpMethodStreamGenerateContent, "alt=sse")
	} else {
		path = buildGCPModelPathSuffix(gcpModelPublisherGoogle, string(g.requestModel), gcpMethodGenerateContent)
	}

	// Re-marshal: json:"-" fields (Model, Stream) are not serialised.
	newBody, err = json.Marshal(body)
	if err != nil {
		return nil, nil, fmt.Errorf("error marshaling Gemini request: %w", err)
	}
	newHeaders = []internalapi.Header{
		{pathHeaderName, path},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// ResponseHeaders implements [GeminiGenerateContentTranslator.ResponseHeaders].
func (g *geminiToGCPVertexAITranslator) ResponseHeaders(_ map[string]string) (
	newHeaders []internalapi.Header, err error,
) {
	if g.stream {
		newHeaders = []internalapi.Header{{contentTypeHeaderName, eventStreamContentType}}
	}
	return
}

// ResponseBody implements [GeminiGenerateContentTranslator.ResponseBody].
// For non-streaming responses, it parses the response to extract token usage and returns the body as-is.
// For streaming responses, it passes through raw SSE bytes and extracts token usage from the last chunk.
func (g *geminiToGCPVertexAITranslator) ResponseBody(_ map[string]string, body io.Reader, endOfStream bool, _ GeminiGenerateContentSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	responseModel = g.requestModel
	if g.stream {
		newHeaders, newBody, tokenUsage, err = g.handleStreamingResponse(body, endOfStream)
		return
	}

	// Non-streaming: read full body, parse for token usage, return as-is.
	rawBody, err := io.ReadAll(body)
	if err != nil {
		return nil, nil, metrics.TokenUsage{}, "", fmt.Errorf("error reading GCP response: %w", err)
	}

	var gcpResp genai.GenerateContentResponse
	if err = json.Unmarshal(rawBody, &gcpResp); err != nil {
		return nil, nil, metrics.TokenUsage{}, "", fmt.Errorf("error decoding GCP response: %w", err)
	}

	if gcpResp.ModelVersion != "" {
		responseModel = internalapi.ResponseModel(gcpResp.ModelVersion)
	}

	if u := gcpResp.UsageMetadata; u != nil {
		if u.PromptTokenCount >= 0 {
			tokenUsage.SetInputTokens(uint32(u.PromptTokenCount)) //nolint:gosec
		}
		if u.CandidatesTokenCount >= 0 {
			tokenUsage.SetOutputTokens(uint32(u.CandidatesTokenCount)) //nolint:gosec
		}
		if u.TotalTokenCount >= 0 {
			tokenUsage.SetTotalTokens(uint32(u.TotalTokenCount)) //nolint:gosec
		}
		if u.CachedContentTokenCount >= 0 {
			tokenUsage.SetCachedInputTokens(uint32(u.CachedContentTokenCount)) //nolint:gosec
		}
		if u.ThoughtsTokenCount >= 0 {
			tokenUsage.SetReasoningTokens(uint32(u.ThoughtsTokenCount)) //nolint:gosec
		}
	}

	newBody = rawBody
	newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(newBody))}}
	return
}

// handleStreamingResponse passes through raw SSE chunks and extracts token usage from the last chunk.
func (g *geminiToGCPVertexAITranslator) handleStreamingResponse(body io.Reader, endOfStream bool) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, err error,
) {
	// Combine buffered data with new input.
	bodyReader := io.MultiReader(bytes.NewReader(g.bufferedBody), body)
	allData, err := io.ReadAll(bodyReader)
	if err != nil {
		return nil, nil, metrics.TokenUsage{}, fmt.Errorf("failed to read streaming body: %w", err)
	}

	if len(allData) == 0 {
		if endOfStream {
			return nil, []byte{}, metrics.TokenUsage{}, nil
		}
		return nil, []byte{}, metrics.TokenUsage{}, nil
	}

	// Detect SSE delimiter on first chunk.
	if g.streamDelimiter == nil {
		g.streamDelimiter = detectSSEDelimiter(allData)
	}

	var parts [][]byte
	if g.streamDelimiter != nil {
		parts = bytes.Split(allData, g.streamDelimiter)
	} else {
		parts = [][]byte{allData}
	}

	g.bufferedBody = nil

	for _, part := range parts {
		trimmed := bytes.TrimSpace(part)
		if len(trimmed) == 0 {
			continue
		}

		// Try to parse the JSON payload (strip "data: " prefix if present).
		line := bytes.TrimPrefix(trimmed, sseDataPrefix)
		var chunk genai.GenerateContentResponse
		if err2 := json.Unmarshal(line, &chunk); err2 != nil {
			// Incomplete chunk – buffer for next call.
			g.bufferedBody = trimmed
			continue
		}

		// Extract token usage from every chunk that carries it (last chunk typically).
		if u := chunk.UsageMetadata; u != nil && u.PromptTokenCount > 0 {
			if u.PromptTokenCount >= 0 {
				tokenUsage.SetInputTokens(uint32(u.PromptTokenCount)) //nolint:gosec
			}
			if u.CandidatesTokenCount >= 0 {
				tokenUsage.SetOutputTokens(uint32(u.CandidatesTokenCount)) //nolint:gosec
			}
			if u.TotalTokenCount >= 0 {
				tokenUsage.SetTotalTokens(uint32(u.TotalTokenCount)) //nolint:gosec
			}
			if u.CachedContentTokenCount >= 0 {
				tokenUsage.SetCachedInputTokens(uint32(u.CachedContentTokenCount)) //nolint:gosec
			}
			if u.ThoughtsTokenCount >= 0 {
				tokenUsage.SetReasoningTokens(uint32(u.ThoughtsTokenCount)) //nolint:gosec
			}
		}

		// Pass through the original SSE event bytes (with delimiter).
		if g.streamDelimiter != nil {
			newBody = append(newBody, trimmed...)
			newBody = append(newBody, g.streamDelimiter...)
		} else {
			newBody = append(newBody, trimmed...)
		}
	}

	if newBody == nil {
		newBody = []byte{}
	}
	return
}

// ResponseError implements [GeminiGenerateContentTranslator.ResponseError].
// Converts GCP Vertex AI error responses to OpenAI-compatible error format.
func (g *geminiToGCPVertexAITranslator) ResponseError(respHeaders map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	return convertGCPVertexAIErrorToOpenAI(respHeaders, body)
}
