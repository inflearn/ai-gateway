#!/usr/bin/env bash
# Test the Gemini cachedContents passthrough end-to-end.
#
# Required env vars (or edit defaults below):
#   GW       - gateway URL (e.g. https://ai-gateway.devinflab.com)
#   TOKEN    - bearer token (email@inflab.com or allowed service prefix)
#   PROJECT  - GCP project name
#   REGION   - vertex region (global / us-central1 / asia-northeast3)
#   MODEL    - model id matching the region (e.g. gemini-2.5-flash for us-central1)
#
# Usage:
#   ./scripts/test-gemini-cache.sh                # run full flow (create → use → patch → delete)
#   ./scripts/test-gemini-cache.sh create         # just create a cache, print its name
#   ./scripts/test-gemini-cache.sh list           # list caches
#   ./scripts/test-gemini-cache.sh get NAME       # get one cache by full resource name
#   ./scripts/test-gemini-cache.sh use NAME       # generateContent referencing the cache
#   ./scripts/test-gemini-cache.sh patch NAME TTL # update ttl (e.g. patch NAME 7200s)
#   ./scripts/test-gemini-cache.sh delete NAME    # delete a cache

set -euo pipefail

GW="${GW:-https://ai-gateway.devinflab.com}"
TOKEN="${TOKEN:?TOKEN env var required (e.g. your.name@inflab.com)}"
PROJECT="${PROJECT:?PROJECT env var required (GCP project id)}"
REGION="${REGION:-asia-northeast3}"
MODEL="${MODEL:-gemini-2.5-flash}"

BLUE=$'\033[1;34m'; GREEN=$'\033[1;32m'; RED=$'\033[1;31m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'

step()  { printf '\n%s==>%s %s\n' "$BLUE" "$NC" "$*"; }
ok()    { printf '%s%s%s\n' "$GREEN" "$*" "$NC"; }
warn()  { printf '%s%s%s\n' "$YELLOW" "$*" "$NC"; }
fail()  { printf '%s%s%s\n' "$RED" "$*" "$NC"; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || fail "$1 not installed"; }
need curl
need jq

show_env() {
  echo "Gateway : $GW"
  echo "Project : $PROJECT"
  echo "Region  : $REGION"
  echo "Model   : $MODEL"
  echo "Token   : ${TOKEN:0:20}..."
}

# ---- 1. create ----
do_create() {
  step "Create cachedContent (region=$REGION, model=$MODEL)"

  # Vertex AI requires at least 1024 cached tokens. Repeat a sentence to be safe.
  local prompt
  prompt=$(printf 'You are a helpful assistant that explains the Envoy AI Gateway architecture in detail, including ext_proc, filter chain, and AIGatewayRoute routing. %.0s' {1..120})

  local body
  body=$(jq -n \
    --arg model "projects/$PROJECT/locations/$REGION/publishers/google/models/$MODEL" \
    --arg text "$prompt" \
    --arg ttl "3600s" \
    --arg name "test-cache-$(date +%s)" \
    '{
      model: $model,
      contents: [{role: "user", parts: [{text: $text}]}],
      ttl: $ttl,
      displayName: $name
    }')

  local resp
  resp=$(curl -sS -X POST "$GW/v1/projects/$PROJECT/locations/$REGION/cachedContents" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$body")

  if echo "$resp" | jq -e '.name' >/dev/null 2>&1; then
    local cache_name
    cache_name=$(echo "$resp" | jq -r '.name')
    ok "Created: $cache_name"
    echo "$resp" | jq '{name, model, expireTime, usageMetadata}'
    echo "$cache_name" > /tmp/last-cache-name
  else
    fail "Create failed:\n$resp"
  fi
}

# ---- 2. list ----
do_list() {
  step "List cachedContents (region=$REGION)"
  curl -sS -X GET "$GW/v1/projects/$PROJECT/locations/$REGION/cachedContents?pageSize=20" \
    -H "Authorization: Bearer $TOKEN" \
    | jq '.cachedContents[]? | {name, displayName, expireTime}'
}

# ---- 3. get ----
do_get() {
  local name="${1:?usage: get <cache-resource-name>}"
  step "Get $name"
  curl -sS -X GET "$GW/$name" \
    -H "Authorization: Bearer $TOKEN" | jq .
}

# ---- 4. use (generateContent with cachedContent) ----
do_use() {
  local name="${1:?usage: use <cache-resource-name>}"
  step "generateContent referencing cache"

  local body
  body=$(jq -n \
    --arg cache "$name" \
    '{
      cachedContent: $cache,
      contents: [{role: "user", parts: [{text: "Summarise the cached context in one sentence."}]}]
    }')

  local resp
  resp=$(curl -sS -X POST "$GW/v1/projects/$PROJECT/locations/$REGION/publishers/google/models/$MODEL:generateContent" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$body")

  if echo "$resp" | jq -e '.candidates' >/dev/null 2>&1; then
    ok "Response:"
    echo "$resp" | jq '{usageMetadata, text: .candidates[0].content.parts[0].text}'

    local cached_tokens
    cached_tokens=$(echo "$resp" | jq -r '.usageMetadata.cachedContentTokenCount // 0')
    if [[ "$cached_tokens" -gt 0 ]]; then
      ok "✓ Cache hit: $cached_tokens cached tokens"
    else
      warn "Cache miss — cachedContentTokenCount=0. Check ext_proc routing and cache age."
    fi
  else
    fail "generateContent failed:\n$resp"
  fi
}

# ---- 5. patch (update TTL) ----
do_patch() {
  local name="${1:?usage: patch <cache-resource-name> <new-ttl>}"
  local ttl="${2:?usage: patch <cache-resource-name> <new-ttl>}"
  step "Patch ttl=$ttl on $name"
  curl -sS -X PATCH "$GW/$name?updateMask=ttl" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"ttl\": \"$ttl\"}" | jq '{name, ttl, expireTime}'
}

# ---- 6. delete ----
do_delete() {
  local name="${1:?usage: delete <cache-resource-name>}"
  step "Delete $name"
  local code
  code=$(curl -sS -o /tmp/del-body -w "%{http_code}" -X DELETE "$GW/$name" \
    -H "Authorization: Bearer $TOKEN")
  if [[ "$code" == "200" || "$code" == "204" ]]; then
    ok "Deleted (HTTP $code)"
  else
    fail "Delete failed HTTP $code:\n$(cat /tmp/del-body)"
  fi
}

# ---- full flow ----
do_all() {
  show_env

  do_create
  local cache_name
  cache_name=$(cat /tmp/last-cache-name)

  do_list

  do_get "$cache_name"

  do_use "$cache_name"

  do_patch "$cache_name" "7200s"

  do_delete "$cache_name"

  ok "✓ All steps passed."
}

cmd="${1:-all}"
shift || true
case "$cmd" in
  all)    do_all ;;
  create) do_create ;;
  list)   do_list ;;
  get)    do_get "$@" ;;
  use)    do_use "$@" ;;
  patch)  do_patch "$@" ;;
  delete) do_delete "$@" ;;
  env)    show_env ;;
  *)      fail "unknown command: $cmd. Try: all|create|list|get|use|patch|delete|env" ;;
esac
