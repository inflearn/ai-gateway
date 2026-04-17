// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/envoyproxy/ai-gateway/internal/apischema/gcp"
	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// TestGeminiToGCPVertexAITranslator_RequestBody tests the RequestBody method.
func TestGeminiToGCPVertexAITranslator_RequestBody(t *testing.T) {
	t.Run("non-streaming: correct path and body", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		req := &gcp.GenerateContentRequest{
			Model:  "gemini-2.0-flash",
			Stream: false,
			Contents: []genai.Content{
				{
					Parts: []*genai.Part{{Text: "Hello"}},
					Role:  "user",
				},
			},
		}
		rawBody, err := json.Marshal(req)
		require.NoError(t, err)

		headers, body, err := tr.RequestBody(rawBody, req, false)
		require.NoError(t, err)
		require.NotNil(t, body)

		// Verify headers: [:path, content-length]
		require.Len(t, headers, 2)
		require.Equal(t, pathHeaderName, headers[0].Key())
		require.Equal(t, "publishers/google/models/gemini-2.0-flash:generateContent", headers[0].Value())
		require.Equal(t, contentLengthHeaderName, headers[1].Key())
		require.Equal(t, fmt.Sprintf("%d", len(body)), headers[1].Value())

		// Verify body is valid JSON.
		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(body, &parsed))
	})

	t.Run("streaming: path ends with :streamGenerateContent?alt=sse", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		req := &gcp.GenerateContentRequest{
			Model:  "gemini-2.0-flash",
			Stream: true,
			Contents: []genai.Content{
				{
					Parts: []*genai.Part{{Text: "Hello"}},
					Role:  "user",
				},
			},
		}
		rawBody, err := json.Marshal(req)
		require.NoError(t, err)

		headers, body, err := tr.RequestBody(rawBody, req, false)
		require.NoError(t, err)
		require.NotNil(t, body)

		require.Len(t, headers, 2)
		require.Equal(t, pathHeaderName, headers[0].Key())
		wantPath := "publishers/google/models/gemini-2.0-flash:streamGenerateContent?alt=sse"
		require.Equal(t, wantPath, headers[0].Value())
	})

	t.Run("with modelNameOverride: override appears in path", func(t *testing.T) {
		const overrideModel = "gemini-1.5-pro-001"
		tr := NewGeminiToGCPVertexAITranslator(overrideModel).(*geminiToGCPVertexAITranslator)
		req := &gcp.GenerateContentRequest{
			Model:  "gemini-2.0-flash",
			Stream: false,
			Contents: []genai.Content{
				{
					Parts: []*genai.Part{{Text: "Hello"}},
					Role:  "user",
				},
			},
		}
		rawBody, err := json.Marshal(req)
		require.NoError(t, err)

		headers, _, err := tr.RequestBody(rawBody, req, false)
		require.NoError(t, err)

		require.Equal(t, pathHeaderName, headers[0].Key())
		wantPath := fmt.Sprintf("publishers/google/models/%s:generateContent", overrideModel)
		require.Equal(t, wantPath, headers[0].Value())
	})

	t.Run("Model and Stream fields are NOT serialized in body JSON", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		req := &gcp.GenerateContentRequest{
			Model:  "gemini-2.0-flash",
			Stream: true,
			Contents: []genai.Content{
				{
					Parts: []*genai.Part{{Text: "Hello"}},
					Role:  "user",
				},
			},
		}
		rawBody, err := json.Marshal(req)
		require.NoError(t, err)

		_, body, err := tr.RequestBody(rawBody, req, false)
		require.NoError(t, err)

		bodyStr := string(body)
		require.NotContains(t, bodyStr, `"model"`)
		require.NotContains(t, bodyStr, `"stream"`)
	})

	t.Run("content-length matches body length", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		req := &gcp.GenerateContentRequest{
			Model:  "gemini-2.0-flash",
			Stream: false,
			Contents: []genai.Content{
				{
					Parts: []*genai.Part{{Text: "A longer message for content-length verification"}},
					Role:  "user",
				},
			},
		}
		rawBody, err := json.Marshal(req)
		require.NoError(t, err)

		headers, body, err := tr.RequestBody(rawBody, req, false)
		require.NoError(t, err)

		require.Equal(t, contentLengthHeaderName, headers[1].Key())
		require.Equal(t, fmt.Sprintf("%d", len(body)), headers[1].Value())
	})
}

// TestGeminiToGCPVertexAITranslator_ResponseHeaders tests the ResponseHeaders method.
func TestGeminiToGCPVertexAITranslator_ResponseHeaders(t *testing.T) {
	t.Run("non-streaming: no headers returned", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = false

		headers, err := tr.ResponseHeaders(nil)
		require.NoError(t, err)
		require.Nil(t, headers)
	})

	t.Run("streaming: content-type text/event-stream returned", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true

		headers, err := tr.ResponseHeaders(nil)
		require.NoError(t, err)
		require.Len(t, headers, 1)
		require.Equal(t, contentTypeHeaderName, headers[0].Key())
		require.Equal(t, eventStreamContentType, headers[0].Value())
	})
}

// TestGeminiToGCPVertexAITranslator_ResponseBody_NonStreaming tests non-streaming ResponseBody.
func TestGeminiToGCPVertexAITranslator_ResponseBody_NonStreaming(t *testing.T) {
	t.Run("normal response: body returned as-is, content-length set", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = false
		tr.requestModel = "gemini-2.0-flash"

		gcpResp := genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					Content: &genai.Content{
						Parts: []*genai.Part{{Text: "Hello there"}},
						Role:  "model",
					},
				},
			},
		}
		respBody, err := json.Marshal(gcpResp)
		require.NoError(t, err)

		headers, body, tokenUsage, responseModel, err := tr.ResponseBody(nil, bytes.NewReader(respBody), true, nil)
		require.NoError(t, err)
		require.Equal(t, respBody, body)
		require.Equal(t, "gemini-2.0-flash", responseModel)
		require.Len(t, headers, 1)
		require.Equal(t, contentLengthHeaderName, headers[0].Key())
		require.Equal(t, fmt.Sprintf("%d", len(body)), headers[0].Value())

		// No usage metadata: tokens not set.
		_, inputSet := tokenUsage.InputTokens()
		require.False(t, inputSet)
	})

	t.Run("response with modelVersion: responseModel updated", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = false
		tr.requestModel = "gemini-2.0-flash"

		gcpResp := genai.GenerateContentResponse{
			ModelVersion: "gemini-2.0-flash-001",
		}
		respBody, err := json.Marshal(gcpResp)
		require.NoError(t, err)

		_, _, _, responseModel, err := tr.ResponseBody(nil, bytes.NewReader(respBody), true, nil)
		require.NoError(t, err)
		require.Equal(t, "gemini-2.0-flash-001", responseModel)
	})

	t.Run("response with usageMetadata: all token counts extracted correctly", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = false
		tr.requestModel = "gemini-2.0-flash"

		gcpResp := genai.GenerateContentResponse{
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:        100,
				CandidatesTokenCount:    50,
				TotalTokenCount:         150,
				CachedContentTokenCount: 10,
				ThoughtsTokenCount:      5,
			},
		}
		respBody, err := json.Marshal(gcpResp)
		require.NoError(t, err)

		_, _, tokenUsage, _, err := tr.ResponseBody(nil, bytes.NewReader(respBody), true, nil)
		require.NoError(t, err)

		inputTokens, ok := tokenUsage.InputTokens()
		require.True(t, ok)
		require.Equal(t, uint32(100), inputTokens)

		outputTokens, ok := tokenUsage.OutputTokens()
		require.True(t, ok)
		require.Equal(t, uint32(50), outputTokens)

		totalTokens, ok := tokenUsage.TotalTokens()
		require.True(t, ok)
		require.Equal(t, uint32(150), totalTokens)

		cachedTokens, ok := tokenUsage.CachedInputTokens()
		require.True(t, ok)
		require.Equal(t, uint32(10), cachedTokens)

		reasoningTokens, ok := tokenUsage.ReasoningTokens()
		require.True(t, ok)
		require.Equal(t, uint32(5), reasoningTokens)
	})

	t.Run("invalid JSON: error returned", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = false

		_, _, _, _, err := tr.ResponseBody(nil, bytes.NewReader([]byte("{not valid json")), true, nil)
		require.Error(t, err)
	})
}

// TestGeminiToGCPVertexAITranslator_ResponseBody_Streaming tests streaming ResponseBody.
func TestGeminiToGCPVertexAITranslator_ResponseBody_Streaming(t *testing.T) {
	buildSSEChunk := func(resp genai.GenerateContentResponse) string {
		b, err := json.Marshal(resp)
		if err != nil {
			panic(err)
		}
		return "data: " + string(b) + "\n\n"
	}

	t.Run("SSE chunks passed through as-is", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true
		tr.requestModel = "gemini-2.0-flash"

		chunk := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "Hello"}}, Role: "model"}},
			},
		})

		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(chunk), false, nil)
		require.NoError(t, err)
		require.NotEmpty(t, body)
	})

	t.Run("token usage extracted from last chunk with usageMetadata", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true
		tr.requestModel = "gemini-2.0-flash"

		chunk1 := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "Hello"}}, Role: "model"}},
			},
		})
		chunk2 := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: " World"}}, Role: "model"}},
			},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     20,
				CandidatesTokenCount: 10,
				TotalTokenCount:      30,
			},
		})

		allChunks := chunk1 + chunk2
		_, _, tokenUsage, _, err := tr.ResponseBody(nil, strings.NewReader(allChunks), true, nil)
		require.NoError(t, err)

		inputTokens, ok := tokenUsage.InputTokens()
		require.True(t, ok)
		require.Equal(t, uint32(20), inputTokens)

		outputTokens, ok := tokenUsage.OutputTokens()
		require.True(t, ok)
		require.Equal(t, uint32(10), outputTokens)

		totalTokens, ok := tokenUsage.TotalTokens()
		require.True(t, ok)
		require.Equal(t, uint32(30), totalTokens)
	})

	t.Run("multiple chunks buffered correctly", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true
		tr.requestModel = "gemini-2.0-flash"

		chunk1 := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "A"}}, Role: "model"}},
			},
		})
		chunk2 := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "B"}}, Role: "model"}},
			},
		})
		chunk3 := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "C"}}, Role: "model"}},
			},
			UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     5,
				CandidatesTokenCount: 3,
				TotalTokenCount:      8,
			},
		})

		allChunks := chunk1 + chunk2 + chunk3
		_, body, tokenUsage, _, err := tr.ResponseBody(nil, strings.NewReader(allChunks), true, nil)
		require.NoError(t, err)
		require.NotEmpty(t, body)

		inputTokens, ok := tokenUsage.InputTokens()
		require.True(t, ok)
		require.Equal(t, uint32(5), inputTokens)
	})

	t.Run("endOfStream=false: partial SSE data buffered, non-empty body for complete events", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true
		tr.requestModel = "gemini-2.0-flash"

		// Send a complete chunk.
		chunk := buildSSEChunk(genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{Content: &genai.Content{Parts: []*genai.Part{{Text: "Hello"}}, Role: "model"}},
			},
		})
		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(chunk), false, nil)
		require.NoError(t, err)
		// Body should be returned for complete events even when endOfStream=false.
		require.NotNil(t, body)
	})

	t.Run("endOfStream=true with no data: empty body returned", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true
		tr.requestModel = "gemini-2.0-flash"

		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(""), true, nil)
		require.NoError(t, err)
		require.Empty(t, body)
	})

	t.Run("endOfStream=false with no data: empty body returned", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)
		tr.stream = true
		tr.requestModel = "gemini-2.0-flash"

		_, body, _, _, err := tr.ResponseBody(nil, strings.NewReader(""), false, nil)
		require.NoError(t, err)
		require.Empty(t, body)
	})
}

// TestGeminiToGCPVertexAITranslator_ResponseError tests the ResponseError method.
func TestGeminiToGCPVertexAITranslator_ResponseError(t *testing.T) {
	t.Run("valid GCP JSON error converted to OpenAI error format", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)

		gcpErrBody := `{
  "error": {
    "code": 400,
    "message": "Invalid request payload.",
    "status": "INVALID_ARGUMENT",
    "details": [
      {
        "@type": "type.googleapis.com/google.rpc.BadRequest",
        "fieldViolations": [
          {
            "description": "Invalid request payload."
          }
        ]
      }
    ]
  }
}`
		headers := map[string]string{statusHeaderName: "400"}
		newHeaders, newBody, err := tr.ResponseError(headers, strings.NewReader(gcpErrBody))
		require.NoError(t, err)
		require.NotNil(t, newBody)
		require.NotEmpty(t, newHeaders)

		// Content-type header should be set.
		require.Equal(t, contentTypeHeaderName, newHeaders[0].Key())

		// Body should be a valid OpenAI error.
		var openaiErr openai.Error
		require.NoError(t, json.Unmarshal(newBody, &openaiErr))
		require.Equal(t, "error", openaiErr.Type)
		require.Equal(t, "INVALID_ARGUMENT", openaiErr.Error.Type)
		require.Contains(t, openaiErr.Error.Message, "Invalid request payload.")
		require.NotNil(t, openaiErr.Error.Code)
		require.Equal(t, "400", *openaiErr.Error.Code)
	})

	t.Run("plain text error body handled", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)

		headers := map[string]string{statusHeaderName: "503"}
		newHeaders, newBody, err := tr.ResponseError(headers, strings.NewReader("Service temporarily unavailable"))
		require.NoError(t, err)
		require.NotNil(t, newBody)
		require.NotEmpty(t, newHeaders)

		var openaiErr openai.Error
		require.NoError(t, json.Unmarshal(newBody, &openaiErr))
		require.Equal(t, "error", openaiErr.Type)
		require.Equal(t, gcpVertexAIBackendError, openaiErr.Error.Type)
		require.Equal(t, "Service temporarily unavailable", openaiErr.Error.Message)
		require.NotNil(t, openaiErr.Error.Code)
		require.Equal(t, "503", *openaiErr.Error.Code)
	})

	t.Run("invalid JSON error body treated as plain text", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)

		headers := map[string]string{statusHeaderName: "400"}
		newHeaders, newBody, err := tr.ResponseError(headers, strings.NewReader(`{"error": invalid json}`))
		require.NoError(t, err)
		require.NotNil(t, newBody)
		require.NotEmpty(t, newHeaders)

		var openaiErr openai.Error
		require.NoError(t, json.Unmarshal(newBody, &openaiErr))
		require.Equal(t, gcpVertexAIBackendError, openaiErr.Error.Type)
		require.Equal(t, `{"error": invalid json}`, openaiErr.Error.Message)
	})

	t.Run("empty body handled gracefully", func(t *testing.T) {
		tr := NewGeminiToGCPVertexAITranslator("").(*geminiToGCPVertexAITranslator)

		headers := map[string]string{statusHeaderName: "500"}
		newHeaders, newBody, err := tr.ResponseError(headers, strings.NewReader(""))
		require.NoError(t, err)
		require.NotNil(t, newBody)
		require.NotEmpty(t, newHeaders)

		var openaiErr openai.Error
		require.NoError(t, json.Unmarshal(newBody, &openaiErr))
		require.Equal(t, "error", openaiErr.Type)
		require.Equal(t, gcpVertexAIBackendError, openaiErr.Error.Type)
		require.Equal(t, "500", *openaiErr.Error.Code)
	})
}
