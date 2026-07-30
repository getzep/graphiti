#!/usr/bin/env bash
# Demo helpers for a deployed Graphiti API. Source it, don't run it:
#
#   export GRAPHITI_URL=https://your-service.onrender.com
#   export GRAPHITI_API_KEY=...   # graphiti-api -> Environment in the Render Dashboard
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
# These wrap POST /search and /messages and project the response down to the interesting
# fields; see $GRAPHITI_URL/docs for the full shape.
#
# There is deliberately no clear helper: DELETE /group and POST /clear return
# {"success": true} and delete nothing, because they query the default graph while the data
# lives in a graph named after the group_id. To actually erase a group, run
# `redis-cli GRAPH.DELETE <group>` on graphiti-falkordb; to start clean, use_group a new name.

# Executing this file would define the functions in a subshell that exits immediately, so say
# so rather than appear to do nothing. zsh needs its own test: ZSH_VERSION says nothing about
# how the file was loaded, but ZSH_EVAL_CONTEXT gains a :file component when sourced.
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
# Refuse up front rather than 401 six times. A local compose stack defaults to
# insecure-local-dev-key.
: "${GRAPHITI_API_KEY:?set GRAPHITI_API_KEY first — graphiti-api -> Environment in the Render Dashboard}"
: "${GRAPHITI_GROUP:=demo}"
# The default search string watch_ingest counts facts against.
: "${GRAPHITI_QUERY:=payments team ledger migration}"

command -v jq >/dev/null || echo 'demo.sh: jq not found — brew install jq'

# curl with the bearer header attached, in one place so a new helper can't forget it. Only
# health() bypasses it. Read per call, so a rotated key needs no re-source.
#
# -H @- takes the header from stdin, keeping the key out of argv where `ps` would expose it.
# Needs curl 7.55+ (2017); no helper sends a body on stdin, so stdin is free.
_graphiti_curl() {
  printf 'Authorization: Bearer %s\n' "$GRAPHITI_API_KEY" | curl -H @- "$@"
}

# $1 is an HTTP status. Returns 0 on 2xx, else explains on stderr and returns 1, so a caller can
# end with it or use `|| return 1`. Keyed off the status, not the body, whose wording is auth.py's
# to change.
_graphiti_check_status() {
  case "$1" in
    2*) return 0 ;;
    401)
      echo '  401 — GRAPHITI_API_KEY was rejected. Re-copy it from the Render Dashboard:' >&2
      echo '    graphiti-api -> Environment -> GRAPHITI_API_KEY' >&2 ;;
    429)
      # Only reachable with a wrong key, since the service never limits a right one — so say
      # what a bare "HTTP 429" would not.
      echo '  429 — too many rejected keys; the service is throttling them. The key is' >&2
      echo '  still wrong. Re-copy it, then wait a minute and retry:' >&2
      echo '    graphiti-api -> Environment -> GRAPHITI_API_KEY' >&2 ;;
    000)
      echo "  no response from $GRAPHITI_URL — is the service up? try: health" >&2 ;;
    *)
      echo "  HTTP $1 from $GRAPHITI_URL" >&2 ;;
  esac
  return 1
}

# Resolved once at source time, so the helpers work from any directory afterwards.
GRAPHITI_EPISODE="${GRAPHITI_EPISODE:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/sample-episode.json}"

# Point every helper at a group. One group per take: re-ingesting into a group that already holds
# facts dedupes into them, so nothing visibly happens. Not named `take` — oh-my-zsh defines that.
use_group() {
  if [ -z "$1" ]; then
    echo '  usage: use_group take2' >&2
    return 1
  fi
  # The name goes into the DELETE /group/{id} path unencoded, so a slash would address a
  # different route and a space would break the URL outright.
  case "$1" in
    *[!A-Za-z0-9._-]*)
      echo "  '$1' won't work as a group name — use letters, digits, . _ or - only" >&2
      return 1 ;;
  esac
  GRAPHITI_GROUP="$1"
  echo "  group: $GRAPHITI_GROUP"
}

# Plain curl: /healthcheck takes no key, and calling it without one is what proves that.
health() {
  # http_code, not status: status is read-only in zsh. Same as ingest below.
  local metrics http_code
  metrics=$(curl -sS -o /dev/null -w '%{http_code} %{time_total}' "$GRAPHITI_URL/healthcheck")
  http_code="${metrics% *}"
  printf '  healthcheck → HTTP %s in %ss\n' "$http_code" "${metrics#* }"
  # Last, so its status becomes health's: `health && ingest` used to run against a service
  # that never answered.
  _graphiti_check_status "$http_code"
}

# Rewrite the sample episode's group_id to match $GRAPHITI_GROUP and POST it. --data-binary
# rather than -d, which strips newlines and leaves the payload unreadable if echoed back.
ingest() {
  # http_code, not status: status is read-only in zsh, which most readers are sourcing into.
  local file="${1:-$GRAPHITI_EPISODE}" tmp body metrics http_code
  [ -f "$file" ] || { echo "  no episode file at $file" >&2; return 1; }
  # An explicit XXXXXX template: macOS accepts a bare prefix after -t, GNU and BusyBox
  # mktemp reject it.
  tmp=$(mktemp "${TMPDIR:-/tmp}/graphiti-episode.XXXXXX") || return 1
  body=$(mktemp "${TMPDIR:-/tmp}/graphiti-ingest.XXXXXX") || { rm -f "$tmp"; return 1; }
  jq --arg g "$GRAPHITI_GROUP" '.group_id = $g' "$file" > "$tmp" || { rm -f "$tmp" "$body"; return 1; }
  metrics=$(_graphiti_curl -sS -o "$body" -w '%{http_code} %{time_total}' \
    -X POST "$GRAPHITI_URL/messages" \
    -H 'Content-Type: application/json' \
    --data-binary @"$tmp")
  http_code="${metrics% *}"
  printf '  ingest → HTTP %s in %ss (group %s)\n' "$http_code" "${metrics#* }" "$GRAPHITI_GROUP"
  # A 202 body is just an ack, so surface it only on an error — and not on a 401, where the
  # hint below says the same thing with the fix attached.
  case "$http_code" in
    2*|401) ;;
    *) jq -c . "$body" 2>/dev/null || cat "$body" ;;
  esac
  rm -f "$tmp" "$body"
  # Last, so its status becomes ingest's: a non-2xx is a failure, not a line that scrolls past
  # looking like progress.
  _graphiti_check_status "$http_code"
}

# POST /search and emit the body, then the HTTP status on a final line. $1 query, $2 max_facts
# (default 10). Only _graphiti_facts calls this, and splits the two apart: the status is what
# tells a rejected key from a service that is simply down.
_graphiti_search() {
  _graphiti_curl -s -w '\n%{http_code}' -X POST "$GRAPHITI_URL/search" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg g "$GRAPHITI_GROUP" --arg q "$1" --argjson n "${2:-10}" \
          '{group_ids: [$g], query: $q, max_facts: $n}')"
}

# Emit the search response only if it is JSON with a facts array, so callers can tell "the service
# is down" from "the graph knows nothing". $1 query, $2 max_facts. Diagnoses its own failure on
# stderr, so callers just propagate.
_graphiti_facts() {
  local response json http_code detail
  response=$(_graphiti_search "$1" "${2:-10}")
  # Both anchor on the *last* newline, so a pretty-printed body survives intact.
  http_code="${response##*$'\n'}"
  json="${response%$'\n'*}"
  if printf '%s' "$json" | jq -e 'has("facts")' >/dev/null 2>&1; then
    printf '%s' "$json"
    return 0
  fi
  # A 401 gets the fix rather than its body, which says the same without one.
  [ "$http_code" = 401 ] && { _graphiti_check_status "$http_code"; return 1; }
  # FastAPI reports other errors as {"detail": ...}, so a detail field means the service answered
  # and rejected us — quoting it beats guessing it is down.
  detail=$(printf '%s' "$json" | jq -r 'if (.detail|type) == "string" then .detail else empty end' 2>/dev/null)
  if [ -n "$detail" ]; then
    echo "  $GRAPHITI_URL/search refused the request: $detail" >&2
  else
    _graphiti_check_status "$http_code"
  fi
  # Unconditional: this is the failure path, whatever the status was.
  return 1
}

# How many facts the current group returns for $1, or empty if the service did not answer with
# usable JSON. Callers must handle empty: feeding it to [ ] gives "integer expression expected".
#
# The quiet variant of _graphiti_facts, because watch_ingest's loop carries the last count forward
# on a dropped request and a diagnostic per blip would scroll past.
_graphiti_count() {
  _graphiti_facts "$1" 50 2>/dev/null | jq -e '.facts | length' 2>/dev/null
}

# The current answer, on one line, rendered as the graph edge it came from so it reads like
# timeline output. /search returns facts in relevance order, so narrow to the newest valid_at and
# then keep the most relevant of those: sorting on valid_at alone tie-breaks arbitrarily between
# same-day facts and can answer a different question than the one asked.
ask() {
  local out json
  if [ -z "$1" ]; then
    echo '  usage: ask "who owns the ledger migration?"' >&2
    return 1
  fi
  # _graphiti_facts has already said what went wrong.
  json=$(_graphiti_facts "$1" 10) || return 1
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

# Every version of a fact, oldest first, as the graph edge behind it: the relationship label and
# the window it is valid for. $1 is a regex matched against fact text; unique_by guards against
# the same episode being ingested twice. Not named history, which would shadow the builtin.
#
# Stops short of drawing (Alex) ──LEADS──▶ (payments team) because /search returns no node names:
# FactResult carries the edge label and text but not its endpoints. To get real ones, add
# source/target to FactResult in server/graph_service/dto/retrieve.py and resolve the names in
# get_fact_result_from_edge.
timeline() {
  local out json
  if [ -z "$1" ]; then
    echo '  usage: timeline "leads the payments"' >&2
    return 1
  fi
  # _graphiti_facts has already said what went wrong.
  json=$(_graphiti_facts "$1" 50) || return 1
  out=$(printf '%s' "$json" | jq -r --arg pat "$1" '
    .facts | map(select(.fact | test($pat; "i"))) | unique_by(.fact) | sort_by(.valid_at)
    | if length == 0 then ["  (no facts match \($pat))"] | .[]
      else .[]
        | (if .valid_at then .valid_at[0:10] else "undated   " end) as $from
        # Only claim an end date when the edge carries one: printing "→ now" for a null
        # invalid_at would assert a superseded fact is still live.
        | (if .invalid_at then " → \(.invalid_at[0:10])" else "" end) as $to
        | "  \($from)\($to)  ──\(.name)──▶  \(.fact)"
      end' 2>/dev/null)
  # The service answered, so empty output means jq failed — and the only way that happens is an
  # invalid regex in $1, since a zero-match search takes the length == 0 branch above.
  if [ -z "$out" ]; then
    echo "  '$1' is not a valid regex — try plain words:" >&2
    echo '    timeline "leads the payments"' >&2
    return 1
  fi
  printf '%s\n' "$out"
}

# Poll until the fact count stops changing, printing only when it moves. Ingestion is serial and
# LLM-bound, so expect ~30s for the three-message sample.
#
# The count has to rise above what the group already held, not just above zero: a re-ingest dedupes
# into the previous run's facts, so the count can sit flat and the watcher would declare victory on
# the *old* data. Start each take in an empty group.
watch_ingest() {
  local force=''
  # Consume --force before reading the query, or it would become the query.
  [ "$1" = '--force' ] && { force=1; shift; }
  local query="${1:-$GRAPHITI_QUERY}"
  local t0 last count stable=0 elapsed baseline json
  t0=$(date +%s)

  # Via _graphiti_facts, not the silent _graphiti_count: without a baseline there is no telling
  # new facts from old, so a rejected key surfaces here rather than as three minutes of polling.
  json=$(_graphiti_facts "$query" 50) || return 1
  baseline=$(printf '%s' "$json" | jq '.facts | length')
  # Refuse rather than poll for three minutes and time out: on a populated group the count may
  # legitimately never move, so there is nothing to wait for.
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
    # A dropped poll is not a change in the graph: carry the last count forward, or a blip
    # resets the stability counter and stalls the watcher.
    [ -n "$count" ] || count=$last
    elapsed=$(( $(date +%s) - t0 ))
    if [ "$count" != "$last" ]; then
      printf '  t=%-6s facts in graph: %s\n' "${elapsed}s" "$count"
      last=$count
      stable=0
    else
      stable=$(( stable + 1 ))
    fi
    # Episodes are processed one at a time and the gap between one finishing and the next
    # producing facts runs to ~22s, so a short plateau is the middle of the queue, not the end
    # of it. 8 polls x 4s = 32s of no movement gives enough margin; less declares victory on a
    # half-built graph.
    [ "$stable" -ge 8 ] && [ "$count" -gt "$baseline" ] && break
    # Timing out is a failure, not a finish — don't follow it with "done".
    if [ "$elapsed" -gt 180 ]; then
      printf '  gave up after 180s at %s facts (started from %s)\n' "$count" "$baseline" >&2
      return 1
    fi
    sleep 4
  done
  printf '  done in %ss\n' "$(( $(date +%s) - t0 ))"
}
