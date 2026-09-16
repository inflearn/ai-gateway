// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
)

// NewImageEditsOpenAIToGCPVertexAITranslator implements [OpenAIImageEditsTranslator] for OpenAI
// /v1/images/edits to GCP Vertex AI Gemini image models.
//
// Editing is the same generateContent call as generation, with the source images supplied as
// inlineData parts ahead of the instruction, so the response handling is shared with
// [openAIToGCPVertexAIImageGenerationTranslator].
func NewImageEditsOpenAIToGCPVertexAITranslator(modelNameOverride internalapi.ModelNameOverride) OpenAIImageEditsTranslator {
	return &openAIToGCPVertexAIImageEditsTranslator{
		openAIToGCPVertexAIImageGenerationTranslator: openAIToGCPVertexAIImageGenerationTranslator{
			modelNameOverride: modelNameOverride,
		},
	}
}

// openAIToGCPVertexAIImageEditsTranslator translates the multipart /v1/images/edits request into a
// generateContent call. Only RequestBody differs from image generation: the response shape of the
// two endpoints is identical, so ResponseBody, ResponseHeaders and ResponseError are inherited.
type openAIToGCPVertexAIImageEditsTranslator struct {
	openAIToGCPVertexAIImageGenerationTranslator
}

// imageEditForm holds the parts of a /v1/images/edits multipart body that map onto Gemini.
type imageEditForm struct {
	prompt string
	images []*genai.Blob
	// options carries the scalar form fields that also exist on /v1/images/generations, so the
	// same size/quality/output mapping applies.
	options openai.ImageGenerationRequest
	hasMask bool
}

// RequestBody implements [OpenAIImageEditsTranslator.RequestBody].
//
// The incoming body is multipart/form-data; the outgoing one is JSON, so the content type is
// rewritten along with the path.
func (o *openAIToGCPVertexAIImageEditsTranslator) RequestBody(original []byte, p *openai.ImageEditRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	o.requestModel = p.Model
	if o.modelNameOverride != "" {
		o.requestModel = o.modelNameOverride
	}

	form, err := parseImageEditForm(original)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", internalapi.ErrMalformedRequest, err)
	}
	if len(form.images) == 0 {
		return nil, nil, fmt.Errorf("%w: at least one image is required for image edits", internalapi.ErrInvalidRequestBody)
	}
	if form.prompt == "" {
		return nil, nil, fmt.Errorf("%w: prompt is required for image edits", internalapi.ErrInvalidRequestBody)
	}
	if form.hasMask {
		// Gemini has no masked-edit primitive: the edit region is described in the prompt. Silently
		// dropping the mask would edit the whole image and quietly return something the caller did
		// not ask for, so refuse instead.
		return nil, nil, fmt.Errorf("%w: mask is not supported by GCP Vertex AI image models; describe the region to change in the prompt instead", internalapi.ErrInvalidRequestBody)
	}

	// Images first, instruction last: Gemini reads the trailing text as the edit to apply.
	parts := make([]*genai.Part, 0, len(form.images)+1)
	for _, image := range form.images {
		parts = append(parts, &genai.Part{InlineData: image})
	}
	parts = append(parts, &genai.Part{Text: form.prompt})

	gcpReq := &geminiImageGenerationRequest{
		Contents: []genai.Content{{Role: "user", Parts: parts}},
		GenerationConfig: &geminiImageGenerationConfig{
			ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
			ImageConfig:        openAIImageOptionsToGeminiImageConfig(&form.options),
		},
	}
	if form.options.N > 0 {
		gcpReq.GenerationConfig.CandidateCount = int32(form.options.N) //nolint:gosec // n is a small request-supplied count.
	}

	newBody, err = json.Marshal(gcpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("error marshaling Gemini image edit request: %w", err)
	}
	newHeaders = []internalapi.Header{
		{pathHeaderName, buildGCPModelPathSuffix(gcpModelPublisherGoogle, o.requestModel, gcpMethodGenerateContent)},
		{contentTypeHeaderName, jsonContentType},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// parseImageEditForm reads the whole multipart body, including the binary image parts that
// [parseImageEditRequest] deliberately skips when it only needs routing metadata.
func parseImageEditForm(body []byte) (*imageEditForm, error) {
	boundary, err := extractMultipartBoundary(body)
	if err != nil {
		return nil, err
	}

	form := &imageEditForm{}
	mr := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read multipart part: %w", err)
		}

		name := part.FormName()
		if name == "mask" {
			form.hasMask = true
			continue
		}

		content, err := io.ReadAll(part)
		if err != nil {
			return nil, fmt.Errorf("failed to read multipart part %q: %w", name, err)
		}

		switch name {
		case "image", "image[]":
			if len(content) == 0 {
				continue
			}
			form.images = append(form.images, &genai.Blob{
				MIMEType: imagePartMIMEType(part, content),
				Data:     content,
			})
		case "prompt":
			form.prompt = string(content)
		case "n":
			form.options.N, _ = strconv.Atoi(string(content))
		case "size":
			form.options.Size = string(content)
		case "quality":
			form.options.Quality = string(content)
		case "output_format":
			form.options.OutputFormat = string(content)
		case "output_compression":
			if v, convErr := strconv.Atoi(string(content)); convErr == nil {
				form.options.OutputCompression = &v
			}
		default:
			// model, response_format, background, user, input_fidelity and anything else Gemini
			// has no equivalent for are ignored; the model applies its own defaults.
		}
	}
	return form, nil
}

// imagePartMIMEType resolves the media type of an uploaded image part, preferring what the client
// declared, then the filename extension, and finally sniffing the bytes.
func imagePartMIMEType(part *multipart.Part, content []byte) string {
	if ct := part.Header.Get("Content-Type"); ct != "" {
		if parsed, _, err := mime.ParseMediaType(ct); err == nil && strings.HasPrefix(parsed, "image/") {
			return parsed
		}
	}
	if ext := filepath.Ext(part.FileName()); ext != "" {
		if byExt := mime.TypeByExtension(ext); strings.HasPrefix(byExt, "image/") {
			// TypeByExtension can append parameters such as "; charset=utf-8".
			if parsed, _, err := mime.ParseMediaType(byExt); err == nil {
				return parsed
			}
		}
	}
	if sniffed := http.DetectContentType(content); strings.HasPrefix(sniffed, "image/") {
		return sniffed
	}
	return "image/png"
}
