// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
	"k8s.io/utils/ptr"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

func TestOpenAIToGCPVertexAIImageGeneration_RequestBody(t *testing.T) {
	for _, tc := range []struct {
		name              string
		modelNameOverride string
		req               *openai.ImageGenerationRequest
		expPath           string
		expConfig         *geminiImageGenerationConfig
	}{
		{
			name:    "minimal request",
			req:     &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image", Prompt: "a cat"},
			expPath: "publishers/google/models/gemini-3.1-flash-image:generateContent",
			expConfig: &geminiImageGenerationConfig{
				ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
			},
		},
		{
			name:              "model name override",
			modelNameOverride: "gemini-3.1-flash-image-preview",
			req:               &openai.ImageGenerationRequest{Model: "gemini-flash-image", Prompt: "a cat"},
			expPath:           "publishers/google/models/gemini-3.1-flash-image-preview:generateContent",
			expConfig: &geminiImageGenerationConfig{
				ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
			},
		},
		{
			name: "size, quality, n and output options",
			req: &openai.ImageGenerationRequest{
				Model: "gemini-3.1-flash-image", Prompt: "a cat", N: 2,
				Size: "1792x1024", Quality: "high", OutputFormat: "jpeg", OutputCompression: ptr.To(80),
			},
			expPath: "publishers/google/models/gemini-3.1-flash-image:generateContent",
			expConfig: &geminiImageGenerationConfig{
				ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
				CandidateCount:     2,
				ImageConfig: &genai.ImageConfig{
					AspectRatio:              "16:9",
					ImageSize:                "4K",
					OutputMIMEType:           "image/jpeg",
					OutputCompressionQuality: ptr.To(int32(80)),
				},
			},
		},
		{
			name:    "gemini native image size passed through size",
			req:     &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image", Prompt: "a cat", Size: "2K", Quality: "low"},
			expPath: "publishers/google/models/gemini-3.1-flash-image:generateContent",
			expConfig: &geminiImageGenerationConfig{
				ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
				ImageConfig:        &genai.ImageConfig{ImageSize: "2K"},
			},
		},
		{
			name:    "auto size is left to the model",
			req:     &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image", Prompt: "a cat", Size: "auto"},
			expPath: "publishers/google/models/gemini-3.1-flash-image:generateContent",
			expConfig: &geminiImageGenerationConfig{
				ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tr := NewImageGenerationOpenAIToGCPVertexAITranslator(tc.modelNameOverride)
			original, err := json.Marshal(tc.req)
			require.NoError(t, err)

			headers, body, err := tr.RequestBody(original, tc.req, false)
			require.NoError(t, err)
			require.Len(t, headers, 2)
			require.Equal(t, pathHeaderName, headers[0].Key())
			require.Equal(t, tc.expPath, headers[0].Value())
			require.Equal(t, contentLengthHeaderName, headers[1].Key())

			var got geminiImageGenerationRequest
			require.NoError(t, json.Unmarshal(body, &got))
			require.Equal(t, []genai.Content{{Role: "user", Parts: []*genai.Part{{Text: tc.req.Prompt}}}}, got.Contents)
			require.Equal(t, tc.expConfig, got.GenerationConfig)
		})
	}
}

func TestOpenAIToGCPVertexAIImageGeneration_RequestBody_Errors(t *testing.T) {
	t.Run("empty prompt", func(t *testing.T) {
		tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(nil, &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image"}, false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
	})
	t.Run("streaming is unsupported", func(t *testing.T) {
		tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
		req := &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image", Prompt: "a cat", Stream: true}
		_, _, err := tr.RequestBody(nil, req, false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
	})
}

func TestAspectRatioFromOpenAISize(t *testing.T) {
	for _, tc := range []struct{ size, exp string }{
		{"1024x1024", "1:1"},
		{"256x256", "1:1"},
		{"1536x1024", "3:2"},
		{"1024x1536", "2:3"},
		{"1792x1024", "16:9"},
		{"1024x1792", "9:16"},
		{"2560x1080", "21:9"},
		{"not-a-size", ""},
		{"1024x", ""},
		{"0x1024", ""},
	} {
		t.Run(tc.size, func(t *testing.T) {
			require.Equal(t, tc.exp, aspectRatioFromOpenAISize(tc.size))
		})
	}
}

func TestOpenAIToGCPVertexAIImageGeneration_ResponseBody(t *testing.T) {
	png := []byte{0x89, 'P', 'N', 'G'}
	gcpResp := genai.GenerateContentResponse{
		CreateTime:   time.Unix(1700000000, 0).UTC(),
		ModelVersion: "gemini-3.1-flash-image",
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{
				{Text: "thinking out loud", Thought: true},
				{Text: "A photorealistic cat."},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: png}},
			}},
		}},
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     11,
			CandidatesTokenCount: 1290,
			ThoughtsTokenCount:   10,
			TotalTokenCount:      1311,
			PromptTokensDetails: []*genai.ModalityTokenCount{
				{Modality: genai.MediaModalityText, TokenCount: 8},
				{Modality: genai.MediaModalityImage, TokenCount: 3},
			},
		},
	}
	raw, err := json.Marshal(&gcpResp)
	require.NoError(t, err)

	tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
	_, _, err = tr.RequestBody(nil, &openai.ImageGenerationRequest{Model: "gemini-flash-image", Prompt: "a cat"}, false)
	require.NoError(t, err)

	headers, body, usage, responseModel, err := tr.ResponseBody(nil, bytes.NewReader(raw), true, nil)
	require.NoError(t, err)
	require.Len(t, headers, 1)
	require.Equal(t, contentLengthHeaderName, headers[0].Key())
	require.Equal(t, internalapi.ResponseModel("gemini-3.1-flash-image"), responseModel)
	requireTokenCount(t, 11, usage.InputTokens)
	requireTokenCount(t, 1300, usage.OutputTokens)
	requireTokenCount(t, 1311, usage.TotalTokens)

	var got openai.ImageGenerationResponse
	require.NoError(t, json.Unmarshal(body, &got))
	require.Equal(t, int64(1700000000), got.Created)
	require.Equal(t, "png", got.OutputFormat)
	require.Len(t, got.Data, 1)
	require.Equal(t, base64.StdEncoding.EncodeToString(png), got.Data[0].B64JSON)
	require.Equal(t, "A photorealistic cat.", got.Data[0].RevisedPrompt)
	require.Equal(t, &openai.ImageGenerationUsage{
		InputTokens:        11,
		OutputTokens:       1300,
		TotalTokens:        1311,
		InputTokensDetails: &openai.ImageGenerationInputTokensDetails{TextTokens: 8, ImageTokens: 3},
	}, got.Usage)
}

// requireTokenCount asserts on one of the optional token counters of [metrics.TokenUsage].
func requireTokenCount(t *testing.T, exp uint32, get func() (uint32, bool)) {
	t.Helper()
	got, ok := get()
	require.True(t, ok)
	require.Equal(t, exp, got)
}

func TestOpenAIToGCPVertexAIImageGeneration_ResponseBody_MultipleCandidates(t *testing.T) {
	first, second := []byte("first"), []byte("second")
	gcpResp := genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{
			{Content: &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/png", Data: first}}}}},
			nil,
			{Content: nil},
			{Content: &genai.Content{Parts: []*genai.Part{{InlineData: &genai.Blob{MIMEType: "image/jpeg", Data: second}}}}},
		},
	}
	raw, err := json.Marshal(&gcpResp)
	require.NoError(t, err)

	tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
	_, _, err = tr.RequestBody(nil, &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image", Prompt: "a cat", N: 2}, false)
	require.NoError(t, err)

	_, body, _, responseModel, err := tr.ResponseBody(nil, bytes.NewReader(raw), true, nil)
	require.NoError(t, err)
	// No modelVersion in the response: fall back to the request model.
	require.Equal(t, internalapi.ResponseModel("gemini-3.1-flash-image"), responseModel)

	var got openai.ImageGenerationResponse
	require.NoError(t, json.Unmarshal(body, &got))
	require.Len(t, got.Data, 2)
	require.Equal(t, base64.StdEncoding.EncodeToString(first), got.Data[0].B64JSON)
	require.Equal(t, base64.StdEncoding.EncodeToString(second), got.Data[1].B64JSON)
	require.NotZero(t, got.Created)
	require.Nil(t, got.Usage)
}

func TestOpenAIToGCPVertexAIImageGeneration_ResponseBody_NoImage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		resp   genai.GenerateContentResponse
		expErr string
	}{
		{
			name: "prompt blocked",
			resp: genai.GenerateContentResponse{
				PromptFeedback: &genai.GenerateContentResponsePromptFeedback{
					BlockReason: genai.BlockedReasonSafety, BlockReasonMessage: "unsafe prompt",
				},
			},
			expErr: "prompt blocked (SAFETY: unsafe prompt)",
		},
		{
			name: "candidate stopped early",
			resp: genai.GenerateContentResponse{
				Candidates: []*genai.Candidate{{FinishReason: genai.FinishReasonImageSafety}},
			},
			expErr: "finishReason=IMAGE_SAFETY",
		},
		{
			name:   "no candidates at all",
			resp:   genai.GenerateContentResponse{},
			expErr: "returned no image in the response",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(&tc.resp)
			require.NoError(t, err)

			tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
			_, _, err = tr.RequestBody(nil, &openai.ImageGenerationRequest{Model: "gemini-3.1-flash-image", Prompt: "a cat"}, false)
			require.NoError(t, err)

			_, _, _, _, err = tr.ResponseBody(nil, bytes.NewReader(raw), true, nil)
			require.ErrorContains(t, err, tc.expErr)
		})
	}
}

func TestOpenAIToGCPVertexAIImageGeneration_ResponseBody_MalformedJSON(t *testing.T) {
	tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
	_, _, _, _, err := tr.ResponseBody(nil, strings.NewReader("not json"), true, nil)
	require.ErrorContains(t, err, "error decoding GCP response")
}

func TestOpenAIToGCPVertexAIImageGeneration_ResponseHeadersAndError(t *testing.T) {
	tr := NewImageGenerationOpenAIToGCPVertexAITranslator("")
	headers, err := tr.ResponseHeaders(nil)
	require.NoError(t, err)
	require.Nil(t, headers)

	respHeaders := map[string]string{contentTypeHeaderName: jsonContentType, statusHeaderName: "429"}
	gcpErr := `{"error":{"code":429,"message":"quota exceeded","status":"RESOURCE_EXHAUSTED"}}`
	_, body, err := tr.ResponseError(respHeaders, strings.NewReader(gcpErr))
	require.NoError(t, err)

	var openAIError openai.Error
	require.NoError(t, json.Unmarshal(body, &openAIError))
	require.Equal(t, "RESOURCE_EXHAUSTED", openAIError.Error.Type)
	require.Contains(t, openAIError.Error.Message, "quota exceeded")
}
