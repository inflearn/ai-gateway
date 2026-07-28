// Copyright Envoy AI Gateway Authors
// SPDX-License-Identifier: Apache-2.0
// The full text of the Apache license is available in the LICENSE file at
// the root of the repo.

package translator

import (
	"io"
	"strconv"

	"github.com/envoyproxy/ai-gateway/internal/apischema/dashscope"
	"github.com/envoyproxy/ai-gateway/internal/internalapi"
	"github.com/envoyproxy/ai-gateway/internal/metrics"
	"github.com/envoyproxy/ai-gateway/internal/tracing/tracingapi"
)

// dashScopeVoiceEnrollmentPath is DashScope's native voice-cloning endpoint. A single URL
// hosts every action (create_voice / list_voice / query_voice / update_voice / delete_voice);
// the action is chosen in the request body.
//
// https://www.alibabacloud.com/help/en/model-studio/voice-clone-design-http-api
const dashScopeVoiceEnrollmentPath = "/api/v1/services/audio/tts/customization"

// DashScopeVoiceEnrollmentSpan is a zero-value span for voice-enrollment. There is no
// per-request response object to record — the endpoint returns short JSON envelopes
// (voice_id, status, page_index, …) that we forward untouched.
type DashScopeVoiceEnrollmentSpan = tracingapi.Span[struct{}, struct{}]

// DashScopeVoiceEnrollmentTranslator translates DashScope voice-enrollment requests as a
// pass-through. It exists so the gateway can attach auth/logging/RBAC to the DashScope
// management surface without reshaping the payload.
type DashScopeVoiceEnrollmentTranslator = Translator[dashscope.VoiceEnrollmentRequest, DashScopeVoiceEnrollmentSpan]

// dashScopeVoiceEnrollmentTranslator forwards voice-enrollment CRUD calls to DashScope
// unchanged. The body varies per action (create takes `url`+`prefix`, delete takes only
// `voice_id`, list is paginated, etc.) so any translator-side reshaping would just add
// friction; we forward the caller's payload verbatim and only rewrite `:path`.
type dashScopeVoiceEnrollmentTranslator struct{}

// NewDashScopeVoiceEnrollmentTranslator creates a pass-through translator for voice-enrollment.
func NewDashScopeVoiceEnrollmentTranslator() DashScopeVoiceEnrollmentTranslator {
	return &dashScopeVoiceEnrollmentTranslator{}
}

// RequestBody implements [DashScopeVoiceEnrollmentTranslator.RequestBody].
//
// Body flows through untouched — DashScope owns the schema per action, and modelling every
// variant here would just track upstream drift. The router already parsed body.model /
// body.input.action for logging + routing, so nothing else is required. We only need to
// re-emit the original bytes when Envoy's upstream stage forces a body replacement (retry /
// streaming), and to rewrite `:path` to the DashScope native URL because Envoy is otherwise
// still sitting on the client's inbound path.
func (dashScopeVoiceEnrollmentTranslator) RequestBody(original []byte, _ *dashscope.VoiceEnrollmentRequest, forceBodyMutation bool) (
	newHeaders []internalapi.Header, newBody []byte, err error,
) {
	newHeaders = []internalapi.Header{{pathHeaderName, dashScopeVoiceEnrollmentPath}}
	if forceBodyMutation && len(original) > 0 {
		newBody = original
		newHeaders = append(newHeaders, internalapi.Header{contentLengthHeaderName, strconv.Itoa(len(newBody))})
	}
	return
}

// ResponseHeaders implements [DashScopeVoiceEnrollmentTranslator.ResponseHeaders].
// DashScope returns application/json envelopes and the caller expects that shape, so we do
// not touch response headers.
func (dashScopeVoiceEnrollmentTranslator) ResponseHeaders(_ map[string]string) (newHeaders []internalapi.Header, err error) {
	return nil, nil
}

// ResponseBody implements [DashScopeVoiceEnrollmentTranslator.ResponseBody]. Passes DashScope's
// JSON response through unchanged; the action-specific envelope (`voice_id`, `voice_list`,
// resource_link, status, …) is the caller's contract, not the gateway's to reshape.
func (dashScopeVoiceEnrollmentTranslator) ResponseBody(_ map[string]string, _ io.Reader, _ bool, _ DashScopeVoiceEnrollmentSpan) (
	newHeaders []internalapi.Header, newBody []byte, tokenUsage metrics.TokenUsage, responseModel internalapi.ResponseModel, err error,
) {
	return nil, nil, metrics.TokenUsage{}, "voice-enrollment", nil
}

// ResponseError implements [DashScopeVoiceEnrollmentTranslator.ResponseError]. DashScope's
// error envelope (`code`, `message`) is passed through as-is — no OpenAI-shaped rewriting is
// meaningful for a fork-only native endpoint.
func (dashScopeVoiceEnrollmentTranslator) ResponseError(_ map[string]string, _ io.Reader) (newHeaders []internalapi.Header, newBody []byte, err error) {
	return nil, nil, nil
}
