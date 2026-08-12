#!/usr/bin/env bash
#
# pre-push-scan.sh — pre-push secret gate for owl-fleet/labyrinthbench
#
# WHAT THIS IS
#   The public repo owl-fleet/labyrinthbench is authored directly (no
#   fresh-history extraction step), and GitHub push protection is not
#   available on the org's Team plan for private repos — so once the repo
#   goes public there is no server-side secret scan backstop. This script
#   is the airlock: it greps the working tree for homelab-identifying
#   patterns (operator identity, LAN topology, internal service names,
#   credential shapes, private lab-notebook paths) plus a gitleaks pass,
#   and refuses to let a push happen if anything trips.
#
# INSTALL
#   Installed as .git/hooks/pre-push via a one-line shim (this script is
#   version-controlled at scripts/pre-push-scan.sh; hooks are not, so the
#   shim is what actually lives at .git/hooks/pre-push):
#
#     #!/usr/bin/env bash
#     exec "$(git rev-parse --show-toplevel)/scripts/pre-push-scan.sh"
#
#   This script can also be run standalone (`bash scripts/pre-push-scan.sh`)
#   from anywhere inside the tree for a manual check.
#
# EXIT STATUS
#   0  — clean: no category had hits after allowlist filtering, gitleaks
#        (if run) found nothing.
#   1+ — dirty: at least one category had hits, or gitleaks failed, or
#        gitleaks was requested but errored out (not merely "unavailable").
#
# DESIGN NOTES
#   - Never prints a matched secret VALUE, anywhere, for any category —
#     only file:line + category label. This applies uniformly (pattern
#     hits and gitleaks alike) so the hook's own output is safe to paste
#     into chat/CI logs.
#   - The literal trigger strings this script hunts for (operator username,
#     internal service names, host mount paths, etc.) are written with a
#     harmless single-char bracket around one character each — e.g. the
#     operator-identity pattern brackets its middle vowel, the internal
#     service names bracket their hyphen/underscore separators — so the
#     regex still matches the real string in target files, but the
#     PATTERN DEFINITION text sitting here in this script does not
#     literal-match its own regex. That means this script is swept like
#     any other tracked file instead of being carved out as an exception.

set -uo pipefail

# ---------------------------------------------------------------------------
# Locate repo root. Prefer git (tracked-file semantics); fall back to the
# script's parent-parent dir so this also works standing alone in a plain
# directory (no .git yet — e.g. during authoring/staging).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"

IS_GIT=0
if ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    IS_GIT=1
else
    ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd -P)"
fi

cd -- "$ROOT" || { echo "pre-push-scan: cannot cd to repo root ($ROOT)" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Build the file list to scan.
#   - git repo:    `git ls-files` (tracked-file semantics — respects
#                  .gitignore, so build junk under gitignored dirs never
#                  needs to be swept in the first place).
#   - plain dir:   `find`, pruning the same directories by hand.
# Either way, apply the same hard excludes on top: .git, node_modules,
# _site, package-lock.json (belt-and-suspenders even under git, since a
# lockfile is usually tracked, not gitignored).
# ---------------------------------------------------------------------------
RAW_FILES="$(mktemp)"
FILES="$(mktemp)"
trap 'rm -f "$RAW_FILES" "$FILES" "${HITS_RAW:-}" "${HITS_FILTERED:-}"' EXIT

if [[ "$IS_GIT" -eq 1 ]]; then
    git ls-files -z > "$RAW_FILES"
else
    find . \
        \( -name .git -o -name node_modules -o -name _site \) -prune -o \
        -type f -print0 > "$RAW_FILES"
fi

# Strip the hard excludes (path components node_modules/_site anywhere,
# any package-lock.json, anything under .git/) regardless of source.
tr '\0' '\n' < "$RAW_FILES" \
    | sed 's#^\./##' \
    | grep -Ev '(^|/)\.git(/|$)' \
    | grep -Ev '(^|/)node_modules(/|$)' \
    | grep -Ev '(^|/)_site(/|$)' \
    | grep -Ev '(^|/)package-lock\.json$' \
    > "$FILES"

# ---------------------------------------------------------------------------
# Pattern battery. One ERE per category; multiple literal forms are joined
# with `|`. Literal (non-regex-metachar) trigger strings are bracket-tricked
# — see DESIGN NOTES above — purely so this file doesn't flag itself; the
# functional match against real target files is unaffected.
# ---------------------------------------------------------------------------
CATEGORY_ORDER=(
    operator-identity
    email-addresses
    lan-ips
    host-paths
    pem-blocks
    dsn-shapes
    api-key-shapes
    internal-service-names
    private-notebook-paths
)

declare -A CATEGORY_PATTERN=(
    [operator-identity]='jwdeav[e]r'
    [email-addresses]='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    [lan-ips]='192\.168\.[0-9]+\.[0-9]+|10\.[0-9]+\.[0-9]+\.[0-9]+'
    [host-paths]='[/]mnt[/]|[/]boot[/]config[/]'
    [pem-blocks]='-----BEGIN[^-]*PRIVATE KEY-----'
    [dsn-shapes]='postgresql://[^/[:space:]]+:[^/[:space:]]+@'
    [api-key-shapes]='sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16}|xox[a-z]-[A-Za-z0-9-]+'
    [internal-service-names]='oath[-]proxy|oath[d]|OATHD[_]|X[-]Oath[-]Token|wali[-]eval|dev[-]timescaledb|dev[-]ingestion[-]worker|labyrinth[-]bench[-]sandbox'
    [private-notebook-paths]='knowledge[/]projects|lb[-]hud[-]orchestration|plans[/]lb[-]'
)

# ---------------------------------------------------------------------------
# Allowlist — narrowly scoped, file+literal, never pattern-wide. Structured
# as an array so future exceptions are one line each. Each entry is an ERE
# applied to a full `file:line:content` hit line; a hit is dropped if it
# matches ANY allowlist entry.
# ---------------------------------------------------------------------------
ALLOWLIST=(
    # QUICKSTART.md's documented generic example for the
    # host.docker.internal workaround.
    '^QUICKSTART\.md:[0-9]+:.*192\.168\.1\.50'
)

is_allowlisted() {
    local hit_line="$1"
    local rule
    for rule in "${ALLOWLIST[@]}"; do
        if [[ "$hit_line" =~ $rule ]]; then
            return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
# Run the battery. For each category: grep -n over the file list (content
# kept only long enough to apply the allowlist), then strip the matched
# content down to file:line before anything is printed or counted — never
# echo a matched value.
# ---------------------------------------------------------------------------
OVERALL_FAIL=0
declare -A CATEGORY_RESULT

echo "== pre-push-scan: pattern battery =="

for cat in "${CATEGORY_ORDER[@]}"; do
    pattern="${CATEGORY_PATTERN[$cat]}"
    HITS_RAW="$(mktemp)"
    HITS_FILTERED="$(mktemp)"

    : > "$HITS_RAW"
    while IFS= read -r f; do
        [[ -f "$f" ]] || continue
        grep -EnI -- "$pattern" "$f" 2>/dev/null | while IFS= read -r line; do
            printf '%s:%s\n' "$f" "$line"
        done >> "$HITS_RAW"
    done < "$FILES"

    : > "$HITS_FILTERED"
    while IFS= read -r hit; do
        [[ -z "$hit" ]] && continue
        if ! is_allowlisted "$hit"; then
            printf '%s\n' "$hit" >> "$HITS_FILTERED"
        fi
    done < "$HITS_RAW"

    count="$(wc -l < "$HITS_FILTERED" | tr -d ' ')"
    if [[ "$count" -gt 0 ]]; then
        CATEGORY_RESULT[$cat]="FAIL ($count hit(s))"
        OVERALL_FAIL=1
        echo "[FAIL] $cat — $count hit(s):"
        # file:line only — never the matched content.
        awk -F: '{print "    " $1 ":" $2}' "$HITS_FILTERED"
    else
        CATEGORY_RESULT[$cat]="PASS"
    fi

    rm -f "$HITS_RAW" "$HITS_FILTERED"
done

echo
echo "== category summary =="
for cat in "${CATEGORY_ORDER[@]}"; do
    printf '  %-24s %s\n' "$cat" "${CATEGORY_RESULT[$cat]}"
done

# ---------------------------------------------------------------------------
# Gitleaks pass. Optional dependency: run it if docker is available, warn
# loudly (but don't fail) if it isn't.
# ---------------------------------------------------------------------------
echo
echo "== gitleaks =="
GITLEAKS_STATUS="skipped"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    if docker run --rm -v "$PWD:/scan" ghcr.io/gitleaks/gitleaks:latest dir /scan --no-banner --exit-code 1; then
        GITLEAKS_STATUS="PASS"
        echo "[PASS] gitleaks — no leaks found"
    else
        GITLEAKS_STATUS="FAIL"
        OVERALL_FAIL=1
        echo "[FAIL] gitleaks — leaks detected (see output above)"
    fi
else
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!! WARNING: docker not available — gitleaks scan SKIPPED.            !!"
    echo "!! The pattern battery above is NOT a substitute for gitleaks;       !!"
    echo "!! this push is going out without a gitleaks pass.                  !!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
fi

# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------
echo
echo "== gitleaks: $GITLEAKS_STATUS =="
if [[ "$OVERALL_FAIL" -ne 0 ]]; then
    echo "== VERDICT: FAIL — do not push. Review hits above. =="
    exit 1
fi
echo "== VERDICT: PASS — clean. =="
exit 0
