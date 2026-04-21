// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"bytes"
	"cmp"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"path"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// NewImageEditsOpenAIToOpenAITranslator implements [OpenAIImageEditsTranslator] for OpenAI to OpenAI image edits translation.
func NewImageEditsOpenAIToOpenAITranslator(apiVersion string, modelNameOverride internalapi.ModelNameOverride) OpenAIImageEditsTranslator {
	return &openAIToOpenAIImageEditsTranslator{
		modelNameOverride: modelNameOverride,
		path:              path.Join("/", apiVersion, "images/edits"),
	}
}

// openAIToOpenAIImageEditsTranslator implements [OpenAIImageEditsTranslator] for /v1/images/edits.
type openAIToOpenAIImageEditsTranslator struct {
	modelNameOverride internalapi.ModelNameOverride
	// path is the images/edits endpoint path, prefixed with the OpenAI path prefix.
	path string
	// requestModel stores the effective model for this request (override or parsed from form).
	requestModel internalapi.RequestModel
}

// RequestBody implements [OpenAIImageEditsTranslator.RequestBody].
// The request body is multipart/form-data. If modelNameOverride is set, the multipart is rebuilt
// with the updated model field; otherwise the raw body is forwarded unchanged.
func (o *openAIToOpenAIImageEditsTranslator) RequestBody(original []byte, p *openai.ImageEditRequest, forceBodyMutation bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	o.requestModel = cmp.Or(o.modelNameOverride, p.Model)

	if o.modelNameOverride != "" && o.modelNameOverride != p.Model {
		boundary, bErr := extractMultipartBoundary(original)
		if bErr != nil {
			return nil, nil, fmt.Errorf("failed to extract multipart boundary for model override: %w", bErr)
		}
		newBody, err = rebuildMultipartWithModelOverride(original, boundary, string(o.modelNameOverride))
		if err != nil {
			return nil, nil, fmt.Errorf("failed to rebuild multipart with model override: %w", err)
		}
	}

	if forceBodyMutation && len(newBody) == 0 {
		newBody = original
	}

	newHeaders = []internalapi.Header{{pathHeaderName, o.path}}
	if len(newBody) > 0 {
		newHeaders = append(newHeaders, internalapi.Header{contentLengthHeaderName, fmt.Sprintf("%d", len(newBody))})
	}
	return
}

// ResponseError implements [OpenAIImageEditsTranslator.ResponseError].
func (o *openAIToOpenAIImageEditsTranslator) ResponseError(respHeaders map[string]string, body io.Reader) ([]internalapi.Header, []byte, error) {
	return convertErrorOpenAIToOpenAIError(respHeaders, body)
}

// ResponseHeaders implements [OpenAIImageEditsTranslator.ResponseHeaders].
func (o *openAIToOpenAIImageEditsTranslator) ResponseHeaders(map[string]string) ([]internalapi.Header, error) {
	return nil, nil
}

// ResponseBody implements [OpenAIImageEditsTranslator.ResponseBody].
// The response format for /v1/images/edits is identical to /v1/images/generations.
func (o *openAIToOpenAIImageEditsTranslator) ResponseBody(_ map[string]string, body io.Reader, _ bool, span tracingapi.ImageEditsSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	resp := &openai.ImageGenerationResponse{}
	if err := json.NewDecoder(body).Decode(&resp); err != nil {
		return nil, nil, tokenUsage, responseModel, fmt.Errorf("failed to decode image edits response body: %w", err)
	}

	if resp.Usage != nil {
		tokenUsage.SetInputTokens(uint32(resp.Usage.InputTokens))   //nolint:gosec
		tokenUsage.SetOutputTokens(uint32(resp.Usage.OutputTokens)) //nolint:gosec
		tokenUsage.SetTotalTokens(uint32(resp.Usage.TotalTokens))   //nolint:gosec
	}

	responseModel = o.requestModel

	if span != nil {
		span.RecordResponse(resp)
	}

	return
}

// extractMultipartBoundary extracts the boundary from a raw multipart/form-data body.
// The boundary is the first line of the body (after the leading "--").
func extractMultipartBoundary(body []byte) (string, error) {
	eol := bytes.Index(body, []byte("\r\n"))
	if eol < 3 || !bytes.HasPrefix(body, []byte("--")) {
		return "", fmt.Errorf("invalid multipart body: missing boundary marker")
	}
	return string(body[2:eol]), nil
}

// rebuildMultipartWithModelOverride rebuilds the multipart body with the given model value
// replacing the original "model" form field, preserving all other parts unchanged.
func rebuildMultipartWithModelOverride(original []byte, boundary, newModel string) ([]byte, error) {
	mr := multipart.NewReader(bytes.NewReader(original), boundary)

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if err := mw.SetBoundary(boundary); err != nil {
		return nil, fmt.Errorf("failed to set multipart boundary: %w", err)
	}

	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read multipart part: %w", err)
		}

		pw, err := mw.CreatePart(part.Header)
		if err != nil {
			return nil, fmt.Errorf("failed to create output multipart part: %w", err)
		}

		if part.FormName() == "model" {
			if _, err = io.WriteString(pw, newModel); err != nil {
				return nil, fmt.Errorf("failed to write model field: %w", err)
			}
			// Discard the original model value.
			if _, err = io.Copy(io.Discard, part); err != nil {
				return nil, fmt.Errorf("failed to discard original model field: %w", err)
			}
		} else {
			if _, err = io.Copy(pw, part); err != nil {
				return nil, fmt.Errorf("failed to copy multipart part: %w", err)
			}
		}
	}

	if err := mw.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	return buf.Bytes(), nil
}
