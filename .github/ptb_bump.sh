#!/usr/bin/env bash
# Proposes PTB's third_party/TOGSim submodule at this repository's newest main commit whose tests passed:
# one branch (BUMP_BRANCH) force-pushed from the tip of PTB_BASE, one open pull request on it.
# DRY_RUN=1 makes only read calls and prints the writes it would make.
set -euo pipefail

: "${SRC:=PSAL-POSTECH/TOGSim}"
: "${PTB:=PSAL-POSTECH/PyTorchSim-Triton-Backend}"
: "${PTB_BASE:=main}"
: "${BUMP_BRANCH:=bot/togsim-bump}"
: "${SUBMODULE:=third_party/TOGSim}"
: "${DRY_RUN:=0}"
: "${SRC_TOKEN:?read token for $SRC (contents)}"
: "${PTB_TOKEN:?write token for $PTB (contents, pull requests)}"
: "${NEW:?the main commit this run tested}"
LABEL=TOGSim
MARKER='<!-- togsim-bump -->'
NAME='Wonhyuk Yang'
EMAIL='wonhyuk@postech.ac.kr'

api() {  # token method path [json]
  local tok=$1 method=$2 path=$3 data=${4:-}
  if [ "$method" != GET ] && [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: $method $path ${data:0:300}" >&2; echo '{}'; return
  fi
  curl -fsSL -X "$method" -H "Authorization: Bearer $tok" -H "Accept: application/vnd.github+json" \
    ${data:+-d "$data"} "https://api.github.com/$path"
}
src() { api "$SRC_TOKEN" GET "repos/$SRC/$1"; }
ptb() { api "$PTB_TOKEN" "$@"; }

relation() { src "compare/$1...$2" | jq -r .status; }  # how $2 stands to $1: ahead behind identical diverged
pin_at() { ptb GET "repos/$PTB/contents/$SUBMODULE?ref=$1" | jq -r 'select(.type == "submodule") | .sha'; }
subject() {  # a merge commit's subject is its pull request's title
  src "commits/$1" | jq -r '.commit.message | split("\n") as $l
    | if ($l[0] | startswith("Merge pull request")) then ([$l[1:][] | select(length > 0)][0] // $l[0]) else $l[0] end'
}

# A newer main commit runs its own tests and this job after them, cancelling this run; this run proposes
# only its own commit, and only while main still contains it.
TIP=$(src "commits/main" | jq -r .sha)
TARGET=$NEW
case $(relation "$TARGET" "$TIP") in
  ahead|identical) ;;
  *) echo "main ($TIP) no longer contains ${TARGET:0:7}: nothing to propose"; exit 0 ;;
esac
echo "target: $TARGET (main: $TIP)"

OPEN=$(ptb GET "repos/$PTB/pulls?state=open&base=$PTB_BASE&per_page=100")
OURS=$(jq -r --arg b "$BUMP_BRANCH" --arg o "${PTB%%/*}" \
  '[.[] | select(.head.ref == $b and .head.repo.owner.login == $o)][0].number // empty' <<<"$OPEN")

# Any other open pull request this automation opened (the marker says so); a hand-made one stays.
jq -r --arg m "$MARKER" --arg n "$OURS" \
  '.[] | select(((.body // "") | contains($m)) and ((.number|tostring) != $n)) | .number' \
  <<<"$OPEN" | while read -r n; do
  echo "superseded: #$n"
  ptb POST "repos/$PTB/issues/$n/comments" "$(jq -nc --arg b "Superseded by the bump on \`$BUMP_BRANCH\` ($LABEL main ${TARGET:0:7})." '{body:$b}')" >/dev/null
  ptb PATCH "repos/$PTB/pulls/$n" '{"state":"closed"}' >/dev/null
done

OLD=$(pin_at "$PTB_BASE")
[ -n "$OLD" ] || { echo "::error::$PTB $PTB_BASE has no submodule at $SUBMODULE"; exit 1; }
echo "$PTB_BASE pins $OLD"
case $(relation "$OLD" "$TARGET") in
  ahead) ;;
  diverged)
    echo "::warning::$PTB_BASE pins ${OLD:0:7}, which is not on $LABEL main: moving it to ${TARGET:0:7} is left to a person"
    exit 0 ;;
  *)
    echo "the pin is already at or past ${TARGET:0:7}: nothing to propose"
    if [ -n "$OURS" ]; then
      ptb POST "repos/$PTB/issues/$OURS/comments" "$(jq -nc --arg b "$PTB_BASE already pins ${OLD:0:7}, at or past $LABEL main ${TARGET:0:7}." '{body:$b}')" >/dev/null
      ptb PATCH "repos/$PTB/pulls/$OURS" '{"state":"closed"}' >/dev/null
    fi
    exit 0 ;;
esac

BASE_SHA=$(ptb GET "repos/$PTB/branches/$PTB_BASE" | jq -r .commit.sha)
if BRANCH=$(ptb GET "repos/$PTB/branches/$BUMP_BRANCH" 2>/dev/null); then
  PREV=$(pin_at "$BUMP_BRANCH")
  if [ "$PREV" = "$TARGET" ] && [ "$(jq -r '.commit.parents[0].sha' <<<"$BRANCH")" = "$BASE_SHA" ] && [ -n "$OURS" ]; then
    echo "$BUMP_BRANCH already proposes ${TARGET:0:7} on $PTB_BASE's tip"; exit 0
  fi
  if [ -n "$PREV" ] && [ "$PREV" != "$TARGET" ] && [ "$(relation "$PREV" "$TARGET")" != ahead ]; then
    echo "$BUMP_BRANCH proposes ${PREV:0:7}, not behind ${TARGET:0:7}: a newer run got there"; exit 0
  fi
fi

S=${TARGET:0:7}
COMPARE="https://github.com/$SRC/compare/$OLD...$TARGET"
LOG=$(src "compare/$OLD...$TARGET" | jq -r '.commits[] | "- \(.sha[0:7]) \(.commit.message | split("\n")[0])"')
N=$(wc -l <<<"$LOG")
[ "$N" -le 40 ] || LOG="$(head -40 <<<"$LOG")"$'\n'"- ... $((N - 40)) more in the compare view"
TITLE="$LABEL moves to $S: $(subject "$TARGET")"

# Only the gitlink changes: the index takes the new commit at $SUBMODULE and nothing is checked out.
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 --filter=blob:none --no-checkout --branch "$PTB_BASE" \
  "https://x-access-token:$PTB_TOKEN@github.com/$PTB.git" "$WORK/ptb"
cd "$WORK/ptb"
[ "$(git rev-parse HEAD)" = "$BASE_SHA" ] || { echo "::error::$PTB_BASE moved while this ran"; exit 1; }
git read-tree HEAD
git update-index --cacheinfo "160000,$TARGET,$SUBMODULE"
[ "$(git diff --cached --name-only)" = "$SUBMODULE" ] || { git diff --cached --stat; echo "::error::the bump touched more than $SUBMODULE"; exit 1; }
GIT_AUTHOR_NAME=$NAME GIT_AUTHOR_EMAIL=$EMAIL GIT_COMMITTER_NAME=$NAME GIT_COMMITTER_EMAIL=$EMAIL \
  git commit -q -F - <<EOF
$TITLE

$SUBMODULE ${OLD:0:7} -> $S, $LABEL main:
$LOG

$COMPARE
EOF
git show --stat --format='%an <%ae>%n%B' HEAD
[ "$(git ls-tree HEAD "$SUBMODULE" | awk '{print $1, $2, $3}')" = "160000 commit $TARGET" ] || { echo "::error::$SUBMODULE did not take $TARGET"; exit 1; }
if [ "$DRY_RUN" = 1 ]; then echo "DRY_RUN: git push --force origin HEAD:refs/heads/$BUMP_BRANCH"
else git push -q --force origin "HEAD:refs/heads/$BUMP_BRANCH"; fi

BODY="Moves \`$SUBMODULE\` from ${OLD:0:7} to $LABEL main $S ([compare]($COMPARE)).

$LABEL commits in the range, oldest first:
$LOG

PTB-side changes the new $LABEL needs, if any, are not included. PTB's toolchain image tag hashes the submodule's tree, so this pull request's CI builds that image anew.

Opened by $LABEL's test workflow. A newer $LABEL main commit force-pushes \`$BUMP_BRANCH\` and rewrites this pull request, so only the newest bump is ever open.
$MARKER"
if [ -n "$OURS" ]; then
  echo "updating #$OURS"
  ptb PATCH "repos/$PTB/pulls/$OURS" "$(jq -nc --arg t "$TITLE" --arg b "$BODY" '{title:$t, body:$b}')" >/dev/null
else
  echo "opening a pull request"
  ptb POST "repos/$PTB/pulls" "$(jq -nc --arg t "$TITLE" --arg b "$BODY" --arg h "$BUMP_BRANCH" --arg base "$PTB_BASE" \
    '{title:$t, body:$b, head:$h, base:$base}')" | jq -r '.html_url // empty'
fi
[ "$DRY_RUN" != 1 ] || printf '%s\n---\n%s\n' "$TITLE" "$BODY"
