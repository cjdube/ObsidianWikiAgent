#!/bin/bash
# Install this repo's launchd jobs for one vault.
#
#   ./launchd/install.sh ~/Vaults/llm-wiki-learnings
#   ./launchd/install.sh ~/Vaults/llm-wiki-learnings ingest lint
#   ./launchd/install.sh --name learnings ~/Vaults/llm-wiki-learnings
#   ./launchd/install.sh --dry-run ~/Vaults/llm-wiki-learnings
#
# Unlike LocalLLMAgent, this repo does not commit its plists: a real one names a
# personal vault path, so launchd/*.plist is gitignored and only the templates
# are tracked. That made adding a vault a copy-and-hand-edit job across four
# placeholders in up to three files, where a typo in WIKI_LAUNCHD_LOG is
# silently wrong and a typo in WREN_RUN_LOG shows up as a job with no run
# history. This fills them in from the vault path instead.
#
# It writes the real .plist beside the templates (still gitignored, still local)
# so you can read what was installed, then bootstraps it.
#
# Re-running is safe: an already-loaded job is booted out first. Nothing here
# runs a job — every template ships RunAtLoad false, so the first run is on
# schedule. To fire one now: launchctl kickstart -k gui/$(id -u)/<label>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
ALL_JOBS=(ingest lint snapshot)

NAME=""
DRY_RUN=0
VAULT=""
jobs=()

while [ $# -gt 0 ]; do
    case "$1" in
        --name) NAME="${2:?--name needs a value}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "unknown option: $1" >&2; exit 2 ;;
        *) if [ -z "$VAULT" ]; then VAULT="$1"; else jobs+=("$1"); fi; shift ;;
    esac
done

if [ -z "$VAULT" ]; then
    echo "usage: $0 [--name LABEL] [--dry-run] <vault-path> [job...]" >&2
    echo "       jobs: ${ALL_JOBS[*]} (default: ingest snapshot)" >&2
    exit 2
fi

# Resolve before anything else: launchd expands neither ~ nor $HOME, so a
# relative or tilde path that works in this shell becomes a job that fails every
# night with a path nobody typed.
if [ ! -d "$VAULT" ]; then
    echo "error: no such vault directory: $VAULT" >&2
    exit 1
fi
VAULT="$(cd "$VAULT" && pwd)"
VAULT_BASENAME="$(basename "$VAULT")"
: "${NAME:=$VAULT_BASENAME}"

if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "error: $ROOT/.venv/bin/python is missing — create the venv first" >&2
    echo "       (README, Setup). A job pointed at a nonexistent interpreter" >&2
    echo "       loads fine and then fails silently on every run." >&2
    exit 1
fi

# lint is opt-in: it is the one job that can send vault contents to Gemini.
[ ${#jobs[@]} -gt 0 ] || jobs=(ingest snapshot)

mkdir -p "$AGENTS" "$ROOT/logs"

# A vault path is user text going into an XML document through sed, so it has to
# survive two syntaxes, in this order:
#
#   XML   & < > are markup inside <string>. A vault at ~/Vaults/R&D produced a
#         plist that plutil rejected with "unknown ampersand-escape sequence".
#   sed   & in a replacement means "the whole match", so the same path put the
#         placeholder back instead of itself and the run died on the
#         placeholder check, blaming the template. \ and the | delimiter are
#         syntax for the same reason.
#
# XML first: it emits &amp;, whose & then needs the sed escape.
plist_value() {
    printf '%s' "$1" \
        | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' \
        | sed -e 's/[\\&|]/\\&/g'
}

for job in "${jobs[@]}"; do
    case "$job" in
        ingest) template="$ROOT/launchd/template.plist.txt" ;;
        lint) template="$ROOT/launchd/template-lint.plist.txt" ;;
        snapshot) template="$ROOT/launchd/template-snapshot.plist.txt" ;;
        *) echo "unknown job: $job (expected: ${ALL_JOBS[*]})" >&2; exit 2 ;;
    esac

    label="local.wikiagent.$NAME-$job"
    local_copy="$ROOT/launchd/$label.plist"

    # The templates are .txt because they are not valid plists until this runs:
    # <ABSOLUTE_PATH_TO_REPO> reads as an XML tag. Substituting the longest
    # placeholder first is not required here, since none is a prefix of another.
    sed -e "s|<ABSOLUTE_PATH_TO_REPO>|$(plist_value "$ROOT")|g" \
        -e "s|<ABSOLUTE_PATH_TO_VAULT>|$(plist_value "$VAULT")|g" \
        -e "s|<VAULT_BASENAME>|$(plist_value "$VAULT_BASENAME")|g" \
        -e "s|<VAULT_NAME>|$(plist_value "$NAME")|g" \
        "$template" > "$local_copy.tmp"

    if grep -q '<ABSOLUTE_PATH_TO_\|<VAULT_NAME>\|<VAULT_BASENAME>' "$local_copy.tmp"; then
        echo "error: $template still has unreplaced placeholders after sed" >&2
        rm -f "$local_copy.tmp"
        exit 1
    fi

    # plutil before launchctl: a malformed plist is rejected by bootstrap with a
    # bare "Input/output error", which says nothing about what is wrong.
    # Left on disk deliberately, because the next thing to do is read it — but
    # it is a .plist.tmp, which launchd/*.plist in .gitignore does not cover, so
    # .gitignore names it too. One escaped from a failed run and sat untracked.
    if ! plutil -lint "$local_copy.tmp" >/dev/null; then
        echo "error: generated plist is not valid: $local_copy.tmp" >&2
        exit 1
    fi

    # Dry run must not touch $local_copy: re-running with an existing vault's
    # name would overwrite a plist that is currently installed and working,
    # which is the opposite of what "dry" promises.
    if [ "$DRY_RUN" = 1 ]; then
        echo "would install $label, from $(basename "$template")"
        sed 's/^/    /' "$local_copy.tmp"
        rm -f "$local_copy.tmp"
        continue
    fi

    mv "$local_copy.tmp" "$local_copy"
    cp "$local_copy" "$AGENTS/$label.plist"

    # bootout, not unload: bootout is the modern verb and is what pairs with
    # bootstrap. Ignore its failure — a job that isn't loaded yet is fine.
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist"
    echo "installed $label"
done

if [ "$DRY_RUN" = 0 ]; then
    echo
    echo "check:  launchctl list | grep wikiagent"
    echo "logs:   $ROOT/logs/"
fi
