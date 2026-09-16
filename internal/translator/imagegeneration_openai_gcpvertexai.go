// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"encoding/base64"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"time"

	"google.golang.org/genai"
	"k8s.io/utils/ptr"

	"github.com/envoyproxy/ai-gateway/internal/apischema/openai"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/json"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// NewImageGenerationOpenAIToGCPVertexAITranslator implements [Factory] for OpenAI
// /v1/images/generations to GCP Vertex AI Gemini image models (e.g. gemini-3.1-flash-image).
//
// Gemini image models have no dedicated image endpoint on Vertex: images come back as
// inlineData parts of an ordinary generateContent response when IMAGE is among the
// requested response modalities.
func NewImageGenerationOpenAIToGCPVertexAITranslator(modelNameOverride internalapi.ModelNameOverride) OpenAIImageGenerationTranslator {
	return &openAIToGCPVertexAIImageGenerationTranslator{modelNameOverride: modelNameOverride}
}

// openAIToGCPVertexAIImageGenerationTranslator translates OpenAI's /v1/images/generations to the
// Gemini generateContent API.
type openAIToGCPVertexAIImageGenerationTranslator struct {
	modelNameOverride internalapi.ModelNameOverride
	// requestModel is the effective model for this request (override or provided). The Gemini
	// response carries modelVersion, but we keep this for the case where it is absent.
	requestModel internalapi.RequestModel
}

// geminiImageGenerationRequest is the subset of the Vertex AI generateContent request body used
// for image generation.
//
// It deliberately does not reuse [gcp.GenerateContentRequest]: that type's generationConfig is
// genai.GenerationConfig, which has no imageConfig field, so aspect ratio and image size could
// not be expressed.
type geminiImageGenerationRequest struct {
	Contents         []genai.Content              `json:"contents"`
	GenerationConfig *geminiImageGenerationConfig `json:"generationConfig,omitempty"`
}

// geminiImageGenerationConfig is the generationConfig subset relevant to image generation.
type geminiImageGenerationConfig struct {
	ResponseModalities []genai.Modality   `json:"responseModalities,omitempty"`
	CandidateCount     int32              `json:"candidateCount,omitempty"`
	ImageConfig        *genai.ImageConfig `json:"imageConfig,omitempty"`
}

// geminiSupportedAspectRatios are the aspect ratios accepted by imageConfig.aspectRatio, paired
// with their numeric value so an arbitrary OpenAI "WxH" size can be snapped onto the closest one.
var geminiSupportedAspectRatios = []struct {
	name  string
	ratio float64
}{
	{"1:1", 1.0},
	{"2:3", 2.0 / 3.0},
	{"3:2", 3.0 / 2.0},
	{"3:4", 3.0 / 4.0},
	{"4:3", 4.0 / 3.0},
	{"9:16", 9.0 / 16.0},
	{"16:9", 16.0 / 9.0},
	{"21:9", 21.0 / 9.0},
}

// RequestBody implements [OpenAIImageGenerationTranslator.RequestBody].
func (o *openAIToGCPVertexAIImageGenerationTranslator) RequestBody(_ []byte, req *openai.ImageGenerationRequest, _ bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	o.requestModel = req.Model
	if o.modelNameOverride != "" {
		o.requestModel = o.modelNameOverride
	}

	if req.Prompt == "" {
		return nil, nil, fmt.Errorf("%w: prompt is required for image generation", internalapi.ErrInvalidRequestBody)
	}
	if req.Stream {
		// OpenAI streams partial images as SSE events; Gemini's generateContent has no equivalent
		// for image output, so fail loudly instead of silently returning a non-streamed body.
		return nil, nil, fmt.Errorf("%w: streaming image generation is not supported by GCP Vertex AI", internalapi.ErrInvalidRequestBody)
	}

	gcpReq := &geminiImageGenerationRequest{
		Contents: []genai.Content{{
			Role:  "user",
			Parts: []*genai.Part{{Text: req.Prompt}},
		}},
		GenerationConfig: &geminiImageGenerationConfig{
			// Gemini image models require TEXT alongside IMAGE; text parts become revised_prompt.
			ResponseModalities: []genai.Modality{genai.ModalityText, genai.ModalityImage},
		},
	}
	if req.N > 0 {
		gcpReq.GenerationConfig.CandidateCount = int32(req.N) //nolint:gosec // n is a small request-supplied count.
	}
	gcpReq.GenerationConfig.ImageConfig = openAIImageOptionsToGeminiImageConfig(req)

	newBody, err = json.Marshal(gcpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("error marshaling Gemini image generation request: %w", err)
	}
	newHeaders = []internalapi.Header{
		{pathHeaderName, buildGCPModelPathSuffix(gcpModelPublisherGoogle, o.requestModel, gcpMethodGenerateContent)},
		{contentLengthHeaderName, strconv.Itoa(len(newBody))},
	}
	return
}

// openAIImageOptionsToGeminiImageConfig maps the OpenAI image options onto imageConfig.
// It returns nil when the request carries nothing that maps, so the model's defaults apply.
func openAIImageOptionsToGeminiImageConfig(req *openai.ImageGenerationRequest) *genai.ImageConfig {
	var config genai.ImageConfig
	var set bool

	switch size := strings.ToLower(strings.TrimSpace(req.Size)); size {
	case "", "auto":
		// Model default.
	case "1k", "2k", "4k":
		// Gemini's own vocabulary, passed through for clients that want to ask for it directly.
		config.ImageSize = strings.ToUpper(size)
		set = true
	default:
		if ratio := aspectRatioFromOpenAISize(size); ratio != "" {
			config.AspectRatio = ratio
			set = true
		}
	}

	// quality selects the rendered resolution; size above only selects the shape.
	if config.ImageSize == "" {
		switch strings.ToLower(req.Quality) {
		case "low":
			config.ImageSize = "1K"
			set = true
		case "medium":
			config.ImageSize = "2K"
			set = true
		case "high":
			config.ImageSize = "4K"
			set = true
		}
	}

	if req.OutputFormat != "" {
		config.OutputMIMEType = "image/" + strings.ToLower(req.OutputFormat)
		set = true
	}
	if req.OutputCompression != nil {
		config.OutputCompressionQuality = ptr.To(int32(*req.OutputCompression)) //nolint:gosec // percentage 0-100.
		set = true
	}

	if !set {
		return nil
	}
	return &config
}

// aspectRatioFromOpenAISize converts an OpenAI "WxH" size to the closest aspect ratio Gemini
// supports. It returns "" when the value is not a WxH pair.
func aspectRatioFromOpenAISize(size string) string {
	w, h, ok := strings.Cut(size, "x")
	if !ok {
		return ""
	}
	width, err := strconv.ParseFloat(strings.TrimSpace(w), 64)
	if err != nil || width <= 0 {
		return ""
	}
	height, err := strconv.ParseFloat(strings.TrimSpace(h), 64)
	if err != nil || height <= 0 {
		return ""
	}

	// Compare in log space so 16:9 and 9:16 are equidistant from 1:1.
	target := math.Log(width / height)
	best, bestDelta := "", math.MaxFloat64
	for _, candidate := range geminiSupportedAspectRatios {
		if delta := math.Abs(math.Log(candidate.ratio) - target); delta < bestDelta {
			best, bestDelta = candidate.name, delta
		}
	}
	return best
}

// ResponseHeaders implements [OpenAIImageGenerationTranslator.ResponseHeaders].
func (o *openAIToGCPVertexAIImageGenerationTranslator) ResponseHeaders(map[string]string) (newHeaders []internalapi.Header, err error) {
	return nil, nil
}

// ResponseBody implements [OpenAIImageGenerationTranslator.ResponseBody].
// Images arrive as inlineData parts on the candidates; each becomes one b64_json entry.
func (o *openAIToGCPVertexAIImageGenerationTranslator) ResponseBody(_ map[string]string, body io.Reader, _ bool, span tracingapi.ImageGenerationSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	gcpResp := &genai.GenerateContentResponse{}
	if err = json.NewDecoder(body).Decode(gcpResp); err != nil {
		return nil, nil, tokenUsage, "", fmt.Errorf("error decoding GCP response: %w", err)
	}

	responseModel = o.requestModel
	if gcpResp.ModelVersion != "" {
		responseModel = gcpResp.ModelVersion
	}

	openAIResp, err := geminiResponseToOpenAIImageGeneration(gcpResp)
	if err != nil {
		return nil, nil, tokenUsage, responseModel, err
	}

	newBody, err = json.Marshal(openAIResp)
	if err != nil {
		return nil, nil, tokenUsage, responseModel, fmt.Errorf("error marshaling OpenAI image generation response: %w", err)
	}

	if openAIResp.Usage != nil {
		tokenUsage.SetInputTokens(uint32(openAIResp.Usage.InputTokens))   //nolint:gosec
		tokenUsage.SetOutputTokens(uint32(openAIResp.Usage.OutputTokens)) //nolint:gosec
		tokenUsage.SetTotalTokens(uint32(openAIResp.Usage.TotalTokens))   //nolint:gosec
	}

	if span != nil {
		span.RecordResponse(openAIResp)
	}
	newHeaders = []internalapi.Header{{contentLengthHeaderName, strconv.Itoa(len(newBody))}}
	return
}

// geminiResponseToOpenAIImageGeneration converts a generateContent response into the OpenAI
// images response shape.
func geminiResponseToOpenAIImageGeneration(gcpResp *genai.GenerateContentResponse) (*openai.ImageGenerationResponse, error) {
	created := gcpResp.CreateTime.Unix()
	if gcpResp.CreateTime.IsZero() {
		created = time.Now().Unix()
	}
	resp := &openai.ImageGenerationResponse{Created: created}

	for _, candidate := range gcpResp.Candidates {
		if candidate == nil || candidate.Content == nil {
			continue
		}
		// Gemini interleaves its commentary with the image; attach it as revised_prompt on the
		// images produced by the same candidate, mirroring what DALL-E 3 returns.
		var text strings.Builder
		var images []*openai.ImageGenerationResponseData
		for _, part := range candidate.Content.Parts {
			switch {
			case part == nil:
			case part.InlineData != nil && strings.HasPrefix(part.InlineData.MIMEType, "image/"):
				images = append(images, &openai.ImageGenerationResponseData{
					B64JSON: base64.StdEncoding.EncodeToString(part.InlineData.Data),
				})
				if resp.OutputFormat == "" {
					resp.OutputFormat = strings.TrimPrefix(part.InlineData.MIMEType, "image/")
				}
			case part.Text != "" && !part.Thought:
				text.WriteString(part.Text)
			}
		}
		for _, image := range images {
			image.RevisedPrompt = text.String()
			resp.Data = append(resp.Data, *image)
		}
	}

	if len(resp.Data) == 0 {
		return nil, imageGenerationNoImageError(gcpResp)
	}

	if usage := gcpResp.UsageMetadata; usage != nil {
		resp.Usage = &openai.ImageGenerationUsage{
			InputTokens:  int(usage.PromptTokenCount),
			OutputTokens: int(usage.CandidatesTokenCount) + int(usage.ThoughtsTokenCount),
			TotalTokens:  int(usage.TotalTokenCount),
		}
		if details := geminiPromptTokensDetails(usage.PromptTokensDetails); details != nil {
			resp.Usage.InputTokensDetails = details
		}
	}
	return resp, nil
}

// geminiPromptTokensDetails converts the per-modality prompt token breakdown, or returns nil when
// the response carries none.
func geminiPromptTokensDetails(modalities []*genai.ModalityTokenCount) *openai.ImageGenerationInputTokensDetails {
	var details openai.ImageGenerationInputTokensDetails
	var set bool
	for _, modality := range modalities {
		if modality == nil {
			continue
		}
		switch modality.Modality {
		case genai.MediaModalityText:
			details.TextTokens += int(modality.TokenCount)
			set = true
		case genai.MediaModalityImage:
			details.ImageTokens += int(modality.TokenCount)
			set = true
		default:
		}
	}
	if !set {
		return nil
	}
	return &details
}

// imageGenerationNoImageError explains a 200 response that carried no image, which is how Gemini
// reports a blocked prompt or a candidate that stopped early.
func imageGenerationNoImageError(gcpResp *genai.GenerateContentResponse) error {
	if feedback := gcpResp.PromptFeedback; feedback != nil && feedback.BlockReason != "" {
		msg := string(feedback.BlockReason)
		if feedback.BlockReasonMessage != "" {
			msg = fmt.Sprintf("%s: %s", msg, feedback.BlockReasonMessage)
		}
		return fmt.Errorf("GCP Vertex AI returned no image: prompt blocked (%s)", msg)
	}
	for _, candidate := range gcpResp.Candidates {
		if candidate != nil && candidate.FinishReason != "" {
			return fmt.Errorf("GCP Vertex AI returned no image: finishReason=%s", candidate.FinishReason)
		}
	}
	return fmt.Errorf("GCP Vertex AI returned no image in the response")
}

// ResponseError implements [OpenAIImageGenerationTranslator.ResponseError].
func (o *openAIToGCPVertexAIImageGenerationTranslator) ResponseError(respHeaders map[string]string, body io.Reader) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	return convertGCPVertexAIErrorToOpenAI(respHeaders, body)
}
