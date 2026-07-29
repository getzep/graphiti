#!/usr/bin/env bash
# Demo helpers for a deployed Graphiti API. Source it, don't run it:
#
#   export GRAPHITI_URL=https://your-service.onrender.com
#   source examples/render/demo.sh
#
# Then:
#   use_group take1         point everything at a group, one take per group
#   health                  is the service up
#   ingest                  POST the sample episode into the current group
#   watch_ingest            follow the ingestion queue draining
#   ask "who owns X?"       the current answer, one line
#   timeline "leads the"    every version of that fact, oldest first
#
# There is deliberately no clear helper. DELETE /group and POST /clear return
# {"success": true} on this deployment and delete nothing — they query the default
# graph while the data lives in a graph named after the group_id. To actually erase a
# group, open a shell on graphiti-falkordb and run `redis-cli GRAPH.DELETE <group>`;
# to start clean without deleting anything, just `use_group` a new name.
#
# Every command is one word with no quoting or line continuations, because a
# multi-line curl that loses its -H flag mid-paste sends the body without a JSON
# content type and the API replies 422 about a string it can't read as an object.
#
# These wrap POST /search and /messages and project the response down to the
# interesting fields. The raw endpoints return uuid/created_at/expired_at too —
# see $GRAPHITI_URL/docs for the full shape.

# Executing this file would define the functions in a subshell that exits
# immediately, so catch that and say so rather than appearing to do nothing.
# zsh needs its own test: ZSH_VERSION being set says nothing about how the file
# was loaded, so `zsh demo.sh` would slip past a bash-only check. ZSH_EVAL_CONTEXT
# gains a :file component when sourced and stays at toplevel when executed.
_graphiti_sourced=0
if [ -n "$ZSH_VERSION" ]; then
  case "$ZSH_EVAL_CONTEXT" in *:file*) _graphiti_sourced=1 ;; esac
elif [ -n "$BASH_VERSION" ]; then
  [ "${BASH_SOURCE[0]}" != "$0" ] && _graphiti_sourced=1
fi
if [ "$_graphiti_sourced" = 0 ]; then
  echo 'demo.sh defines shell functions, so it has to be sourced:' >&2
  echo '  source examples/render/demo.sh' >&2
  exit 1
fi
unset _graphiti_sourced

: "${GRAPHITI_URL:?set GRAPHITI_URL first, e.g. export GRAPHITI_URL=https://graphiti-api.onrender.com}"
: "${GRAPHITI_GROUP:=demo}"
# The default search string watch_ingest counts facts against.
: "${GRAPHITI_QUERY:=payments team ledger migration}"

command -v jq >/dev/null || echo 'demo.sh: jq not found — brew install jq'

# Where the sample episode lives, resolved once at source time so the helpers
# work from any directory afterwards.
GRAPHITI_EPISODE="${GRAPHITI_EPISODE:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/sample-episode.json}"

# Point every helper at a group. One group per take: re-ingesting into a group
# that already holds facts dedupes into them, so nothing visibly happens.
# Named use_group rather than take because oh-my-zsh already defines take.
use_group() {
  if [ -z "$1" ]; then
    echo '  usage: use_group take2' >&2
    return 1
  fi
  # The name goes into the DELETE /group/{id} path unencoded, so a slash would
  # silently address a different route and a space would break the URL outright.
  case "$1" in
    *[!A-Za-z0-9._-]*)
      echo "  '$1' won't work as a group name — use letters, digits, . _ or - only" >&2
      return 1 ;;
  esac
  GRAPHITI_GROUP="$1"
  echo "  group: $GRAPHITI_GROUP"
}

health() {
  curl -sS -o /dev/null -w '  healthcheck → HTTP %{http_code} in %{time_total}s\n' \
    "$GRAPHITI_URL/healthcheck"
}

# Rewrite the sample episode's group_id to match $GRAPHITI_GROUP and POST it.
# --data-binary rather than -d: -d strips newlines, which is harmless for JSON
# but makes the payload unreadable if you ever need to echo it back.
ingest() {
  local file="${1:-$GRAPHITI_EPISODE}" tmp body code
  [ -f "$file" ] || { echo "  no episode file at $file" >&2; return 1; }
  # An explicit XXXXXX template: macOS accepts a bare prefix after -t, GNU and
  # BusyBox mktemp reject it, so -t alone would break the helper on Linux.
  tmp=$(mktemp "${TMPDIR:-/tmp}/graphiti-episode.XXXXXX") || return 1
  body=$(mktemp "${TMPDIR:-/tmp}/graphiti-ingest.XXXXXX") || { rm -f "$tmp"; return 1; }
  jq --arg g "$GRAPHITI_GROUP" '.group_id = $g' "$file" > "$tmp" || { rm -f "$tmp" "$body"; return 1; }
  code=$(curl -sS -o "$body" -w '%{http_code} %{time_total}' \
    -X POST "$GRAPHITI_URL/messages" \
    -H 'Content-Type: application/json' \
    --data-binary @"$tmp")
  printf '  ingest → HTTP %s in %ss (group %s)\n' "${code% *}" "${code#* }" "$GRAPHITI_GROUP"
  # A 202 body is just an ack, so only surface it when something went wrong.
  case "${code% *}" in
    2*) ;;
    *) jq -c . "$body" 2>/dev/null || cat "$body" ;;
  esac
  rm -f "$tmp" "$body"
}

# POST /search and emit the raw JSON. $1 query, $2 max_facts (default 10).
_graphiti_search() {
  curl -s -X POST "$GRAPHITI_URL/search" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg g "$GRAPHITI_GROUP" --arg q "$1" --argjson n "${2:-10}" \
          '{group_ids: [$g], query: $q, max_facts: $n}')"
}

# How many facts the current group returns for $1, or empty if the service did
# not answer with usable JSON. Callers must handle empty: a cold start or a 502
# yields no output, and feeding that to [ ] produces "integer expression
# expected" instead of anything a reader can act on.
_graphiti_count() {
  _graphiti_search "$1" 50 | jq -e '.facts | length' 2>/dev/null
}

# Emit the search response only if it is JSON with a facts array, so callers can
# tell "the service is down" from "the graph knows nothing" and say which.
# $1 query, $2 max_facts.
_graphiti_facts() {
  local json
  json=$(_graphiti_search "$1" "${2:-10}")
  printf '%s' "$json" | jq -e 'has("facts")' >/dev/null 2>&1 || return 1
  printf '%s' "$json"
}

# The current answer, on one line, rendered as the graph edge it came from so it
# reads the same as timeline output. /search returns facts in relevance order, so
# narrow to the newest valid_at first and then keep the most relevant of those:
# sorting on valid_at alone tie-breaks arbitrarily between same-day facts and can
# answer a different question than the one asked.
ask() {
  local out json
  if [ -z "$1" ]; then
    echo '  usage: ask "who owns the ledger migration?"' >&2
    return 1
  fi
  json=$(_graphiti_facts "$1" 10) || {
    echo "  no usable response from $GRAPHITI_URL/search — is the service up? try: health" >&2
    return 1
  }
  out=$(printf '%s' "$json" | jq -r '
    (.facts | map(select(.valid_at != null))) as $dated
    | if (.facts | length) == 0 then "  (no facts matched)"
      elif ($dated | length) == 0 then "  ──\(.facts[0].name)──▶  \(.facts[0].fact)  (undated)"
      else ($dated | map(.valid_at) | max) as $newest
           | $dated | map(select(.valid_at == $newest)) | .[0]
           | "  ──\(.name)──▶  \(.fact)  (as of \(.valid_at[0:10]))"
      end' 2>/dev/null)
  [ -n "$out" ] || { echo '  (no facts matched)'; return 0; }
  printf '%s\n' "$out"
}

# Every version of a fact, oldest first, as the graph edge behind it: the
# relationship label and the window the edge is valid for. $1 is a regex matched
# against fact text. unique_by guards against the same episode having been
# ingested twice. Named timeline, not history, so it doesn't shadow the builtin.
#
# This stops short of drawing (Alex) ──LEADS──▶ (payments team) because /search
# returns no node names: FactResult has the edge label and text but not the
# edge's two endpoints, and the API exposes no node getter to resolve them. For
# real endpoints, add source/target to FactResult in
# server/graph_service/dto/retrieve.py and resolve the names in
# get_fact_result_from_edge.
timeline() {
  local out json
  if [ -z "$1" ]; then
    echo '  usage: timeline "leads the payments"' >&2
    return 1
  fi
  json=$(_graphiti_facts "$1" 50) || {
    echo "  no usable response from $GRAPHITI_URL/search — is the service up? try: health" >&2
    return 1
  }
  out=$(printf '%s' "$json" | jq -r --arg pat "$1" '
    .facts | map(select(.fact | test($pat; "i"))) | unique_by(.fact) | sort_by(.valid_at)
    | if length == 0 then ["  (no facts match \($pat))"] | .[]
      else .[]
        | (if .valid_at then .valid_at[0:10] else "undated   " end) as $from
        # Only claim an end date when the edge actually carries one. Printing
        # "→ now" for a null invalid_at asserts the superseded fact is still
        # live, which contradicts the fact that replaced it.
        | (if .invalid_at then " → \(.invalid_at[0:10])" else "" end) as $to
        | "  \($from)\($to)  ──\(.name)──▶  \(.fact)"
      end' 2>/dev/null)
  # The service answered, so empty output here means jq itself failed — and the
  # only way that happens is an invalid regex in $1, since a zero-match search
  # takes the length == 0 branch above and prints its own line.
  if [ -z "$out" ]; then
    echo "  '$1' is not a valid regex — try plain words:" >&2
    echo '    timeline "leads the payments"' >&2
    return 1
  fi
  printf '%s\n' "$out"
}

# Poll until the fact count stops changing. Prints only when it moves, so the
# output stays short. Ingestion is serial and LLM-bound, so expect ~30s for the
# three-message sample.
#
# The count has to rise above whatever the group already held, not just above
# zero: re-ingesting into a group that still has the previous run's facts
# dedupes into them, so the count can sit flat at its starting value and the
# watcher would otherwise declare victory on the *old* data. Start each take in
# an empty group.
watch_ingest() {
  local force=''
  # Consume --force before reading the query, or it would become the query.
  [ "$1" = '--force' ] && { force=1; shift; }
  local query="${1:-$GRAPHITI_QUERY}"
  local t0 last count stable=0 elapsed baseline
  t0=$(date +%s)

  baseline=$(_graphiti_count "$query")
  # Without a baseline there is no way to tell new facts from old ones, so stop
  # rather than poll against an unknown starting point.
  if [ -z "$baseline" ]; then
    echo "  no usable response from $GRAPHITI_URL/search — is the service up? try: health" >&2
    return 1
  fi
  # Refuse rather than poll for three minutes and time out: on a populated group
  # the count legitimately may never move, so there is nothing to wait for.
  if [ "$baseline" -gt 0 ] && [ -z "$force" ]; then
    echo "  group '$GRAPHITI_GROUP' already holds $baseline facts — nothing to watch," >&2
    echo "  since a re-ingest dedupes into them and the count never moves." >&2
    echo "  Start a clean take:  use_group take2 && ingest && watch_ingest" >&2
    echo "  (Emptying this group instead needs redis-cli GRAPH.DELETE on" >&2
    echo "   graphiti-falkordb — the HTTP delete endpoints don't work here.)" >&2
    return 1
  fi

  last=-1
  while true; do
    count=$(_graphiti_count "$query")
    # A dropped poll is not a change in the graph. Carry the last known count
    # forward, or a blip would reset the stability counter and stall the watcher.
    [ -n "$count" ] || count=$last
    elapsed=$(( $(date +%s) - t0 ))
    if [ "$count" != "$last" ]; then
      printf '  t=%-6s facts in graph: %s\n' "${elapsed}s" "$count"
      last=$count
      stable=0
    else
      stable=$(( stable + 1 ))
    fi
    # Episodes are processed one at a time, and the gap between one finishing and
    # the next producing facts runs to ~22s, so a short plateau is not the end of
    # the queue — it is the middle of it. 8 polls x 4s = 32s of no movement gives
    # enough margin; anything less declares victory on a half-built graph and the
    # answers come out of date.
    [ "$stable" -ge 8 ] && [ "$count" -gt "$baseline" ] && break
    # Timing out is a failure, not a finish — don't follow it with "done", which
    # would read as a successful drain in the recording.
    if [ "$elapsed" -gt 180 ]; then
      printf '  gave up after 180s at %s facts (started from %s)\n' "$count" "$baseline" >&2
      return 1
    fi
    sleep 4
  done
  printf '  done in %ss\n' "$(( $(date +%s) - t0 ))"
}
