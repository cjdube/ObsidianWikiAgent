#!/bin/bash
# Reload this repo's launchd jobs after a Homebrew Python upgrade.
#
# `brew upgrade python@3.12` deletes the old Cellar directory, and launchd has
# cached the old interpreter's code signature. It then refuses to exec the
# replacement: jobs die at launch with OS_REASON_CODESIGNING (exit -9) *before*
# running any of our code. No log line is written, and notify_failure never gets
# the chance to push — the job simply appears not to have run.
#
# This repo is doubly exposed to the same upgrade. The TCC grant that lets these
# jobs read a vault is keyed to the interpreter's exact path, so an upgrade can
# also revoke folder access; that is why vault_snapshot.py is Python rather than
# a shell script, and it cost one run seven hours on 2026-08-13. This script does
# not fix TCC — only re-granting does — but a reload is what clears the signature
# half, and knowing which half broke is most of the diagnosis.
#
#   ./launchd/reload-after-upgrade.sh           # heal whatever is stale
#   ./launchd/reload-after-upgrade.sh --check   # report only; exit 1 if stale
#   ./launchd/reload-after-upgrade.sh --quiet   # say nothing unless it acts
#
# Safe to run any time: it reloads only what is stale, and never interrupts a
# job that is mid-run. That second rule matters more here than in LocalLLMAgent
# — an ingest with several new sources routinely runs one to three hours, and
# bouncing it would throw that away and leave a half-ingested vault behind.
#
# Staleness is judged two ways, and the first is why the second is not enough:
#
#   proactive — the interpreter's identity (realpath + inode + mtime) recorded
#     in config/.interpreter_id. When it changes, every job that execs it is
#     reloaded, because every one is now carrying a signature launchd rejects.
#     This is the check that matters.
#   reactive  — "needs LWCR update" on an individual job. launchd sets that flag
#     only AFTER a job has tried to exec and failed, so on its own it can never
#     prevent a missed run, only notice one. A backstop for breakage that is not
#     an interpreter swap.
#
# Reactive alone loses a run per job per upgrade: a job that has not fired since
# the upgrade carries no flag, so it reads as healthy and stays broken until its
# next scheduled fire — and that fire is the one that dies. The cost scales with
# the schedule, so here the weekly lint would lose a week.
#
# Run under /bin/bash, never .venv/bin/python: the interpreter is the thing that
# gets invalidated, so a Python healer would be killed by the exact failure it
# exists to repair. /bin/bash is Apple-signed and survives.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="gui/$(id -u)"
AGENTS="$HOME/Library/LaunchAgents"
INTERPRETER="$ROOT/.venv/bin/python"
INTERPRETER_ID="$ROOT/config/.interpreter_id"
LABEL_PREFIX="local.wikiagent."

CHECK_ONLY=0
QUIET=0
case "${1:-}" in
    --check) CHECK_ONLY=1 ;;
    --quiet) QUIET=1 ;;
    "")      ;;
    *)       echo "usage: $(basename "$0") [--check|--quiet]" >&2; exit 2 ;;
esac

say() { [ "$QUIET" = 1 ] || echo "$@"; }

# A repair is a real event — an upgrade silently broke the schedule and this put
# it back. Push it, because the alternative is finding out from whichever job
# didn't run. Best-effort: if the interpreter is the thing that broke, this is
# the call that fails, and it must never fail the repair with it.
push() {
    [ -x "$INTERPRETER" ] || return 0
    (cd "$ROOT" && "$INTERPRETER" -c '
import sys
from agent.notify import notify
notify(sys.argv[1], title="WikiAgent: jobs reloaded after an upgrade")
' "$1") >/dev/null 2>&1 || true
}

# launchd reports the cached-signature mismatch as "needs LWCR update" (launch
# weak code requirement). That is the authoritative signal — exit status -9 alone
# cannot be told apart from any other kill.
needs_reload() {
    launchctl print "$DOMAIN/$1" 2>/dev/null | grep -q "needs LWCR update"
}

is_running() {
    launchctl print "$DOMAIN/$1" 2>/dev/null | grep -qE '^[[:space:]]*state = running'
}

# Read ProgramArguments rather than grepping the file: these plists name the
# interpreter in their comments too, and a plain grep takes that as a match.
uses_interpreter() {
    [ "$(/usr/bin/plutil -extract ProgramArguments.0 raw -o - "$AGENTS/$1.plist" 2>/dev/null)" \
      = "$INTERPRETER" ]
}

# Resolved to its real Cellar path and stamped with inode and mtime, so an
# upgrade that lands a new binary on the SAME path is still detected — the
# symlink chain out of .venv/bin/python does not change on every upgrade, but the
# file it ends at always does. Empty output (missing interpreter) means "unknown"
# and must never overwrite a good recorded value.
fingerprint() {
    local real
    real=$(/usr/bin/readlink -f "$INTERPRETER" 2>/dev/null) || return 0
    [ -n "$real" ] && /usr/bin/stat -f '%N %i %m' "$real" 2>/dev/null
}

labels=()
for plist in "$AGENTS/$LABEL_PREFIX"*.plist; do
    [ -e "$plist" ] || continue
    name="$(basename "$plist")"
    labels+=("${name%.plist}")
done

if [ ${#labels[@]} -eq 0 ]; then
    say "no $LABEL_PREFIX jobs installed — nothing to reload"
    exit 0
fi

current="$(fingerprint || true)"
recorded=""
[ -f "$INTERPRETER_ID" ] && recorded="$(cat "$INTERPRETER_ID")"

interpreter_changed=0
if [ -n "$current" ] && [ -n "$recorded" ] && [ "$current" != "$recorded" ]; then
    interpreter_changed=1
fi

stale=()
for label in "${labels[@]}"; do
    if [ "$interpreter_changed" = 1 ] && uses_interpreter "$label"; then
        stale+=("$label")
    elif needs_reload "$label"; then
        stale+=("$label")
    fi
done

if [ ${#stale[@]} -eq 0 ]; then
    # Record only on a clean pass. Stamping a new fingerprint while jobs are
    # still stale would erase the very difference that identifies them.
    [ -n "$current" ] && printf '%s\n' "$current" > "$INTERPRETER_ID"
    say "all ${#labels[@]} jobs healthy"
    exit 0
fi

if [ "$CHECK_ONLY" = 1 ]; then
    echo "stale: ${stale[*]}"
    exit 1
fi

reloaded=()
skipped=()
for label in "${stale[@]}"; do
    if is_running "$label"; then
        # An ingest mid-run has hours of model work in it and a vault in a
        # half-written state. It already exec'd successfully, so it is not the
        # broken one; leave it and heal it on the next pass.
        skipped+=("$label")
        continue
    fi
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$AGENTS/$label.plist"
    reloaded+=("$label")
done

if [ ${#reloaded[@]} -gt 0 ]; then
    echo "reloaded: ${reloaded[*]}"
    push "reloaded after a Python upgrade: ${reloaded[*]}"
fi
if [ ${#skipped[@]} -gt 0 ]; then
    echo "still running, left alone: ${skipped[*]}"
fi

# Same rule as the clean-pass branch: only stamp when nothing was left behind.
if [ ${#skipped[@]} -eq 0 ] && [ -n "$current" ]; then
    printf '%s\n' "$current" > "$INTERPRETER_ID"
fi
