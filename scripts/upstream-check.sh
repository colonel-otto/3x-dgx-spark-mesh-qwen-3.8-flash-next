#!/usr/bin/env bash
# upstream-check.sh -- "is it worth updating?" in one command, without touching
# the serving cluster.
#
#   ./scripts/upstream-check.sh                 # launcher drift + image + watchlist
#   ./scripts/upstream-check.sh --image <tag>   # also dry-run our vLLM patches
#                                               # against <tag>'s files on sparkmain
#
# Reports:
#   1. eugr/spark-vllm-docker commits upstream of our fork branch
#      (colonel-otto/spark-vllm-docker:colonel/recipes), flagging the ones that
#      touch Qwen3.8 / Flash-Next / b12x / MTP.
#   2. Whether Docker Hub has a newer eugr/spark-vllm-b12x:latest than the one
#      tagged vllm-node-b12x on sparkmain.
#   3. State of every upstream PR/issue in patches/WATCHLIST.tsv.
#   4. (--image) For each diff in patches/diffs-vs-*/, whether it still applies
#      to that image's stock file, is already present upstream, or conflicts.
#
# Needs: gh (authenticated), curl, python3 locally; ssh access to sparkmain.
# Read-only everywhere.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK="colonel-otto/spark-vllm-docker"
FORK_BRANCH="colonel/recipes"
UPSTREAM="eugr/spark-vllm-docker"
HUB_IMAGE="eugr/spark-vllm-b12x"
HEAD_NODE="${HEAD_NODE:-sparkmain}"
IMAGE=""
[ "${1:-}" = "--image" ] && IMAGE="${2:?--image needs a tag}"
PY="$(command -v python3 || command -v python)"
FOCUS='qwen3\.8|flash.?next|qwen4|b12x|mtp|specul'

echo "=== 1. Launcher: $UPSTREAM main vs $FORK:$FORK_BRANCH ==="
cmp=$(gh api "repos/$FORK/compare/${FORK_BRANCH//\//%2F}...${UPSTREAM%%/*}:main" 2>/dev/null) \
  || { echo "  compare failed (gh auth? branch missing?)"; cmp=""; }
if [ -n "$cmp" ]; then
  summarize=$(cat <<'PYEOF'
import json, re, sys
d = json.load(sys.stdin)
focus = re.compile(sys.argv[1], re.I)
print("  upstream is %d commit(s) ahead of our branch (** = touches our area)" % d.get("ahead_by", 0))
for c in d.get("commits", []):
    msg = c["commit"]["message"].splitlines()[0]
    mark = "  **" if focus.search(msg) else "    "
    print("%s %s %s %s" % (mark, c["sha"][:8], c["commit"]["committer"]["date"][:10], msg[:100]))
hot = [f["filename"] for f in d.get("files", []) if focus.search(f["filename"])]
if hot:
    print("  files touching our area: " + ", ".join(hot[:20]))
PYEOF
)
  echo "$cmp" | "$PY" -c "$summarize" "$FOCUS"
fi

echo
echo "=== 2. Image: $HUB_IMAGE:latest on Docker Hub vs vllm-node-b12x on $HEAD_NODE ==="
hub=$(curl -fsS "https://hub.docker.com/v2/repositories/$HUB_IMAGE/tags/latest" 2>/dev/null \
      | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("last_updated","?"))' 2>/dev/null || echo "?")
local_created=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" \
  "docker image inspect vllm-node-b12x --format '{{.Created}}' 2>/dev/null || echo 'not present'")
echo "  Docker Hub latest pushed: $hub"
echo "  $HEAD_NODE vllm-node-b12x built: $local_created"

echo
echo "=== 3. Watchlist (patches/WATCHLIST.tsv) ==="
grep -vE '^\s*(#|$)' "$HERE/patches/WATCHLIST.tsv" | while IFS=$'\t' read -r repo num item why; do
  # '|' not TAB: TAB is IFS whitespace, so an empty merged_at field would
  # collapse and shift every later column.
  st=$(gh api "repos/$repo/issues/$num" --jq '[(if .pull_request then "PR" else "issue" end), .state, (.pull_request.merged_at // "-"), .updated_at[:10], (.title[:70] | gsub("\\|"; "/"))] | join("|")' 2>/dev/null) \
    || { printf '  %-24s #%-6s ?  (lookup failed)\n' "$repo" "$num"; continue; }
  IFS='|' read -r kind state merged updated title <<< "$st"
  if [ "$merged" != "-" ]; then state="MERGED ${merged:0:10}"; else state="$kind $state"; fi
  printf '  %-24s #%-6s %-17s upd %s  [%s] %s\n' "$repo" "$num" "$state" "$updated" "$item" "$title"
done

if [ -n "$IMAGE" ]; then
  echo
  echo "=== 4. Patch applicability against $IMAGE (on $HEAD_NODE) ==="
  for d in "$HERE"/patches/diffs-vs-*/*.diff; do
    [ -e "$d" ] || continue
    # Diff headers carry the exact package path: "--- a/vllm/<rel path>".
    rel=$(sed -n '1s#^--- a/\([^[:space:]]*\).*#\1#p' "$d")
    [ -n "$rel" ] || { printf '  %-32s %s\n' "$(basename "$d")" "bad diff header"; continue; }
    res=$(ssh -o BatchMode=yes "$HEAD_NODE" "
      tmp=\$(mktemp -d); cat > \$tmp/p.diff; mkdir -p \$tmp/\$(dirname $rel)
      p=\$(docker run --rm --entrypoint sh '$IMAGE' -c 'python3 -c \"import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))\"' 2>/dev/null)
      [ -n \"\$p\" ] || { echo 'no vllm in image'; rm -rf \$tmp; exit; }
      if ! docker run --rm --entrypoint cat '$IMAGE' \"\$p/$rel\" > \$tmp/$rel 2>/dev/null; then
        echo 'file not in image (renamed/removed upstream)'
      elif patch --dry-run -s -p1 -d \$tmp -i \$tmp/p.diff >/dev/null 2>&1; then echo 'applies cleanly'
      elif patch --dry-run -s -R -p1 -d \$tmp -i \$tmp/p.diff >/dev/null 2>&1; then echo 'already present upstream'
      else echo 'CONFLICTS -- upstream changed this file; re-derive or drop'; fi
      rm -rf \$tmp" < "$d" 2>/dev/null)
    printf '  %-52s %s\n' "$rel" "${res:-check failed}"
  done
fi
