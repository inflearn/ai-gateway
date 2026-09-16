// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"mime/multipart"
	"net/textproto"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// pngBytes is a valid 1x1 PNG, used so content sniffing has something real to work with.
var pngBytes = func() []byte {
	b, err := base64.StdEncoding.DecodeString(
		"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
	if err != nil {
		panic(err)
	}
	return b
}()

type editFormFile struct {
	field, filename, contentType string
	content                      []byte
}

// buildGeminiEditMultipart renders a /v1/images/edits body the way an OpenAI SDK would.
func buildGeminiEditMultipart(t *testing.T, fields map[string]string, files ...editFormFile) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	for k, v := range fields {
		require.NoError(t, w.WriteField(k, v))
	}
	for _, f := range files {
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q; filename=%q`, f.field, f.filename))
		if f.contentType != "" {
			h.Set("Content-Type", f.contentType)
		}
		pw, err := w.CreatePart(h)
		require.NoError(t, err)
		_, err = pw.Write(f.content)
		require.NoError(t, err)
	}
	require.NoError(t, w.Close())
	return buf.Bytes()
}

func TestOpenAIToGCPVertexAIImageEdits_RequestBody(t *testing.T) {
	body := buildGeminiEditMultipart(t,
		map[string]string{"model": "gemini-3.1-flash-image", "prompt": "make the sky purple", "size": "1536x1024", "quality": "medium", "n": "1"},
		editFormFile{field: "image", filename: "photo.png", contentType: "image/png", content: pngBytes},
	)

	tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
	headers, newBody, err := tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-3.1-flash-image", Prompt: "make the sky purple"}, false)
	require.NoError(t, err)

	require.Len(t, headers, 3)
	require.Equal(t, pathHeaderName, headers[0].Key())
	require.Equal(t, "publishers/google/models/gemini-3.1-flash-image:generateContent", headers[0].Value())
	// The body stops being multipart at this point, so the content type must be rewritten.
	require.Equal(t, contentTypeHeaderName, headers[1].Key())
	require.Equal(t, jsonContentType, headers[1].Value())
	require.Equal(t, contentLengthHeaderName, headers[2].Key())

	var got geminiImageGenerationRequest
	require.NoError(t, json.Unmarshal(newBody, &got))
	require.Len(t, got.Contents, 1)
	require.Equal(t, "user", got.Contents[0].Role)
	// Image first, instruction last.
	require.Len(t, got.Contents[0].Parts, 2)
	require.Equal(t, "image/png", got.Contents[0].Parts[0].InlineData.MIMEType)
	require.Equal(t, pngBytes, got.Contents[0].Parts[0].InlineData.Data)
	require.Equal(t, "make the sky purple", got.Contents[0].Parts[1].Text)
	require.Equal(t, &geminiImageGenerationConfig{
		ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
		CandidateCount:     1,
		ImageConfig:        &genai.ImageConfig{AspectRatio: "3:2", ImageSize: "2K"},
	}, got.GenerationConfig)
}

func TestOpenAIToGCPVertexAIImageEdits_RequestBody_MultipleImagesAndOverride(t *testing.T) {
	second := []byte("\xff\xd8\xff\xe0 jpeg-ish")
	body := buildGeminiEditMultipart(t,
		map[string]string{"model": "gemini-flash-image", "prompt": "combine these"},
		editFormFile{field: "image[]", filename: "a.png", contentType: "image/png", content: pngBytes},
		editFormFile{field: "image[]", filename: "b.jpg", contentType: "image/jpeg", content: second},
	)

	tr := NewImageEditsOpenAIToGCPVertexAITranslator("gemini-3.1-flash-image")
	headers, newBody, err := tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-flash-image", Prompt: "combine these"}, false)
	require.NoError(t, err)
	require.Equal(t, "publishers/google/models/gemini-3.1-flash-image:generateContent", headers[0].Value())

	var got geminiImageGenerationRequest
	require.NoError(t, json.Unmarshal(newBody, &got))
	require.Len(t, got.Contents[0].Parts, 3)
	require.Equal(t, "image/png", got.Contents[0].Parts[0].InlineData.MIMEType)
	require.Equal(t, "image/jpeg", got.Contents[0].Parts[1].InlineData.MIMEType)
	require.Equal(t, second, got.Contents[0].Parts[1].InlineData.Data)
	require.Equal(t, "combine these", got.Contents[0].Parts[2].Text)
	// No size/quality given: let the model decide.
	require.Nil(t, got.GenerationConfig.ImageConfig)
}

func TestOpenAIToGCPVertexAIImageEdits_RequestBody_MIMEDetection(t *testing.T) {
	for _, tc := range []struct {
		name        string
		file        editFormFile
		expMIMEType string
	}{
		{
			name:        "declared content type wins",
			file:        editFormFile{field: "image", filename: "photo.bin", contentType: "image/webp", content: pngBytes},
			expMIMEType: "image/webp",
		},
		{
			name:        "falls back to the filename extension",
			file:        editFormFile{field: "image", filename: "photo.jpeg", content: pngBytes},
			expMIMEType: "image/jpeg",
		},
		{
			name:        "sniffs the bytes when nothing else says",
			file:        editFormFile{field: "image", filename: "photo", content: pngBytes},
			expMIMEType: "image/png",
		},
		{
			name:        "non-image declared type is not trusted",
			file:        editFormFile{field: "image", filename: "photo", contentType: "application/octet-stream", content: pngBytes},
			expMIMEType: "image/png",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := buildGeminiEditMultipart(t, map[string]string{"prompt": "edit"}, tc.file)
			tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
			_, newBody, err := tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-3.1-flash-image", Prompt: "edit"}, false)
			require.NoError(t, err)

			var got geminiImageGenerationRequest
			require.NoError(t, json.Unmarshal(newBody, &got))
			require.Equal(t, tc.expMIMEType, got.Contents[0].Parts[0].InlineData.MIMEType)
		})
	}
}

func TestOpenAIToGCPVertexAIImageEdits_RequestBody_Errors(t *testing.T) {
	image := editFormFile{field: "image", filename: "photo.png", contentType: "image/png", content: pngBytes}

	t.Run("mask is rejected", func(t *testing.T) {
		body := buildGeminiEditMultipart(t, map[string]string{"prompt": "edit"}, image,
			editFormFile{field: "mask", filename: "mask.png", contentType: "image/png", content: pngBytes})
		tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-3.1-flash-image", Prompt: "edit"}, false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
		require.ErrorContains(t, err, "mask is not supported")
	})

	t.Run("image is required", func(t *testing.T) {
		body := buildGeminiEditMultipart(t, map[string]string{"prompt": "edit"})
		tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-3.1-flash-image", Prompt: "edit"}, false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
		require.ErrorContains(t, err, "at least one image")
	})

	t.Run("prompt is required", func(t *testing.T) {
		body := buildGeminiEditMultipart(t, nil, image)
		tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-3.1-flash-image"}, false)
		require.ErrorIs(t, err, internalapi.ErrInvalidRequestBody)
		require.ErrorContains(t, err, "prompt is required")
	})

	t.Run("malformed multipart", func(t *testing.T) {
		tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
		_, _, err := tr.RequestBody([]byte("not multipart"), &openai.ImageEditRequest{Model: "gemini-3.1-flash-image", Prompt: "edit"}, false)
		require.ErrorIs(t, err, internalapi.ErrMalformedRequest)
	})
}

// TestOpenAIToGCPVertexAIImageEdits_ResponseBody pins that the edits translator reuses the image
// generation response handling: /v1/images/edits and /v1/images/generations return the same shape.
func TestOpenAIToGCPVertexAIImageEdits_ResponseBody(t *testing.T) {
	gcpResp := genai.GenerateContentResponse{
		ModelVersion: "gemini-3.1-flash-image",
		Candidates: []*genai.Candidate{{
			Content: &genai.Content{Parts: []*genai.Part{
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: pngBytes}},
			}},
		}},
	}
	raw, err := json.Marshal(&gcpResp)
	require.NoError(t, err)

	body := buildGeminiEditMultipart(t, map[string]string{"prompt": "edit"},
		editFormFile{field: "image", filename: "photo.png", contentType: "image/png", content: pngBytes})
	tr := NewImageEditsOpenAIToGCPVertexAITranslator("")
	_, _, err = tr.RequestBody(body, &openai.ImageEditRequest{Model: "gemini-3.1-flash-image", Prompt: "edit"}, false)
	require.NoError(t, err)

	_, newBody, _, responseModel, err := tr.ResponseBody(nil, bytes.NewReader(raw), true, nil)
	require.NoError(t, err)
	require.Equal(t, internalapi.ResponseModel("gemini-3.1-flash-image"), responseModel)

	var got openai.ImageGenerationResponse
	require.NoError(t, json.Unmarshal(newBody, &got))
	require.Len(t, got.Data, 1)
	require.Equal(t, base64.StdEncoding.EncodeToString(pngBytes), got.Data[0].B64JSON)
	require.Equal(t, "png", got.OutputFormat)
}
