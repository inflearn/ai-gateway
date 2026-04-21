// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"fmt"
	"mime/multipart"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// buildImageEditMultipart creates a multipart/form-data body for /v1/images/edits testing.
func buildImageEditMultipart(boundary, model, prompt string, includeFile bool) []byte {
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	_ = w.SetBoundary(boundary)

	if model != "" {
		fw, _ := w.CreateFormField("model")
		_, _ = fmt.Fprint(fw, model)
	}
	if prompt != "" {
		fw, _ := w.CreateFormField("prompt")
		_, _ = fmt.Fprint(fw, prompt)
	}
	if includeFile {
		fw, _ := w.CreateFormFile("image", "image.png")
		_, _ = fw.Write([]byte("fake-png-data"))
	}

	_ = w.Close()
	return buf.Bytes()
}

func TestExtractMultipartBoundary(t *testing.T) {
	t.Run("valid boundary", func(t *testing.T) {
		body := buildImageEditMultipart("testboundary123", "gpt-image-1", "a cat", false)
		boundary, err := extractMultipartBoundary(body)
		require.NoError(t, err)
		require.Equal(t, "testboundary123", boundary)
	})

	t.Run("invalid body", func(t *testing.T) {
		_, err := extractMultipartBoundary([]byte("not multipart"))
		require.Error(t, err)
	})

	t.Run("empty body", func(t *testing.T) {
		_, err := extractMultipartBoundary([]byte{})
		require.Error(t, err)
	})
}

func TestRebuildMultipartWithModelOverride(t *testing.T) {
	boundary := "boundary123"
	body := buildImageEditMultipart(boundary, "original-model", "a cat", true)

	rebuilt, err := rebuildMultipartWithModelOverride(body, boundary, "gpt-image-1")
	require.NoError(t, err)
	require.NotEmpty(t, rebuilt)

	// Parse rebuilt body and verify model is updated.
	mr := multipart.NewReader(bytes.NewReader(rebuilt), boundary)
	found := false
	for {
		part, err := mr.NextPart()
		if err != nil {
			break
		}
		if part.FormName() == "model" {
			var sb strings.Builder
			buf := make([]byte, 64)
			for {
				n, e := part.Read(buf)
				sb.Write(buf[:n])
				if e != nil {
					break
				}
			}
			require.Equal(t, "gpt-image-1", sb.String())
			found = true
		}
	}
	require.True(t, found, "model field not found in rebuilt multipart")
}

func TestImageEditsTranslator_RequestBody_NoOverride(t *testing.T) {
	tr := NewImageEditsOpenAIToOpenAITranslator("v1", "")
	body := buildImageEditMultipart("boundary123", "gpt-image-1", "a cat", true)
	req := &openai.ImageEditRequest{Model: "gpt-image-1", Prompt: "a cat"}

	headers, newBody, err := tr.RequestBody(body, req, false)
	require.NoError(t, err)
	require.Equal(t, "/v1/images/edits", headers[0].Value())
	require.Nil(t, newBody) // body unchanged, no mutation
}

func TestImageEditsTranslator_RequestBody_ModelOverride(t *testing.T) {
	tr := NewImageEditsOpenAIToOpenAITranslator("v1", "gpt-image-1")
	body := buildImageEditMultipart("boundary123", "dall-e-2", "a cat", true)
	req := &openai.ImageEditRequest{Model: "dall-e-2", Prompt: "a cat"}

	headers, newBody, err := tr.RequestBody(body, req, false)
	require.NoError(t, err)
	require.Equal(t, "/v1/images/edits", headers[0].Value())
	require.NotNil(t, newBody) // body mutated with new model

	// Verify model is updated in rebuilt multipart.
	mr := multipart.NewReader(bytes.NewReader(newBody), "boundary123")
	for {
		part, err := mr.NextPart()
		if err != nil {
			break
		}
		if part.FormName() == "model" {
			var sb strings.Builder
			buf := make([]byte, 64)
			for {
				n, e := part.Read(buf)
				sb.Write(buf[:n])
				if e != nil {
					break
				}
			}
			require.Equal(t, "gpt-image-1", sb.String())
		}
	}
}

func TestImageEditsTranslator_RequestBody_ForceMutation(t *testing.T) {
	tr := NewImageEditsOpenAIToOpenAITranslator("v1", "")
	body := buildImageEditMultipart("boundary123", "gpt-image-1", "a cat", false)
	req := &openai.ImageEditRequest{Model: "gpt-image-1", Prompt: "a cat"}

	headers, newBody, err := tr.RequestBody(body, req, true)
	require.NoError(t, err)
	require.Equal(t, "/v1/images/edits", headers[0].Value())
	require.Equal(t, body, newBody) // force mutation returns original body
}

func TestImageEditsTranslator_ResponseBody_OK(t *testing.T) {
	tr := NewImageEditsOpenAIToOpenAITranslator("v1", "gpt-image-1")
	// RequestBody must be called first to populate requestModel.
	reqBody := buildImageEditMultipart("boundary123", "dall-e-2", "a cat", false)
	_, _, err := tr.RequestBody(reqBody, &openai.ImageEditRequest{Model: "dall-e-2"}, false)
	require.NoError(t, err)

	resp := &openai.ImageGenerationResponse{
		Created: 1234567890,
		Data:    []openai.ImageGenerationResponseData{{URL: "https://example.com/image.png"}},
	}
	buf, _ := json.Marshal(resp)

	_, _, usage, responseModel, err := tr.ResponseBody(nil, bytes.NewReader(buf), false, nil)
	require.NoError(t, err)
	require.Equal(t, "gpt-image-1", responseModel)
	require.Equal(t, tokenUsageFrom(-1, -1, -1, -1, -1, -1), usage)
}

func TestImageEditsTranslator_ResponseBody_WithUsage(t *testing.T) {
	tr := NewImageEditsOpenAIToOpenAITranslator("v1", "gpt-image-1")
	resp := &openai.ImageGenerationResponse{
		Usage: &openai.ImageGenerationUsage{
			InputTokens:  100,
			OutputTokens: 50,
			TotalTokens:  150,
		},
	}
	buf, _ := json.Marshal(resp)

	_, _, usage, _, err := tr.ResponseBody(nil, bytes.NewReader(buf), false, nil)
	require.NoError(t, err)
	require.Equal(t, tokenUsageFrom(100, -1, -1, 50, 150, -1), usage)
}
