#!/usr/bin/env bash
# Smoke-test Gemini image models through the OpenAI /v1/images/generations endpoint.
# The gateway translates the request to Vertex generateContent with IMAGE response modality and
# converts the inline image data back into OpenAI's b64_json shape.
#
# Required env vars:
#   TOKEN    - bearer token (e.g. your.name@inflab.com)
#
# Optional:
#   GW       - gateway URL (default: https://ai-gateway.devinflab.com)
#   MODELS   - comma-separated override list (default: gemini-3.1-flash-image)
#   PROMPT   - prompt to send
#   SIZE     - OpenAI size, mapped to the closest Gemini aspect ratio (default: 1024x1024)
#   QUALITY  - low|medium|high, selects 1K/2K/4K (default: unset, model default)
#   OUTDIR   - where to write the decoded images (default: ./out/gemini-image)
#
# Usage:
#   TOKEN=you@inflab.com ./scripts/test-gemini-image.sh

set -uo pipefail   # NOTE: no -e — keep going after individual model failures.

GW="${GW:-https://ai-gateway.devinflab.com}"
TOKEN="${TOKEN:?TOKEN env var required (e.g. your.name@inflab.com)}"
PROMPT="${PROMPT:-A watercolor painting of a cat sitting on a Seoul rooftop at sunrise.}"
SIZE="${SIZE:-1024x1024}"
OUTDIR="${OUTDIR:-./out/gemini-image}"

DEFAULT_MODELS=("gemini-3.1-flash-image")
if [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -r -a TARGET_MODELS <<< "$MODELS"
else
  TARGET_MODELS=("${DEFAULT_MODELS[@]}")
fi

BLUE=$'\033[1;34m'; GREEN=$'\033[1;32m'; RED=$'\033[1;31m'; NC=$'\033[0m'
ok()  { printf '%s%s%s\n' "$GREEN" "$*" "$NC"; }
bad() { printf '%s%s%s\n' "$RED" "$*" "$NC"; }

command -v jq >/dev/null || { bad "jq not installed"; exit 1; }
mkdir -p "$OUTDIR"

printf '%sGateway%s : %s\n' "$BLUE" "$NC" "$GW"
printf '%sSize   %s : %s%s\n' "$BLUE" "$NC" "$SIZE" "${QUALITY:+ (quality=$QUALITY)}"
printf '%sOutput %s : %s\n\n' "$BLUE" "$NC" "$OUTDIR"

failures=0
for model in "${TARGET_MODELS[@]}"; do
  printf '%s==> %s%s\n' "$BLUE" "$model" "$NC"

  body=$(jq -n \
    --arg model "$model" --arg prompt "$PROMPT" --arg size "$SIZE" --arg quality "${QUALITY:-}" \
    '{model: $model, prompt: $prompt, size: $size, n: 1}
     + (if $quality == "" then {} else {quality: $quality} end)')

  resp=$(curl -sS -w '\n%{http_code}' -X POST "$GW/v1/images/generations" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$body")
  status=$(tail -n1 <<< "$resp")
  payload=$(sed '$d' <<< "$resp")

  if [[ "$status" != "200" ]]; then
    bad "  HTTP $status"
    jq -C . <<< "$payload" 2>/dev/null || printf '  %s\n' "$payload"
    ((failures++))
    continue
  fi

  b64=$(jq -r '.data[0].b64_json // empty' <<< "$payload")
  if [[ -z "$b64" ]]; then
    bad "  HTTP 200 but no image in response"
    jq -C 'del(.data)' <<< "$payload"
    ((failures++))
    continue
  fi

  ext=$(jq -r '.output_format // "png"' <<< "$payload")
  file="$OUTDIR/${model}.${ext}"
  base64 -d <<< "$b64" > "$file" 2>/dev/null || base64 -D <<< "$b64" > "$file"
  bytes=$(wc -c < "$file" | tr -d ' ')

  ok "  HTTP 200 — $file (${bytes} bytes)"
  jq -r '"  tokens: in=\(.usage.input_tokens // 0) out=\(.usage.output_tokens // 0) total=\(.usage.total_tokens // 0)"' <<< "$payload"
  revised=$(jq -r '.data[0].revised_prompt // empty' <<< "$payload")
  [[ -n "$revised" ]] && printf '  model note: %s\n' "$(cut -c1-120 <<< "$revised")"
done

echo
if (( failures > 0 )); then
  bad "$failures/${#TARGET_MODELS[@]} model(s) failed"
  exit 1
fi
ok "all ${#TARGET_MODELS[@]} model(s) OK"
