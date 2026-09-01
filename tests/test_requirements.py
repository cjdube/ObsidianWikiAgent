"""Guards the split between requirements.txt (intent) and requirements.lock
(what actually gets installed).

Two files describing the same dependencies can disagree, and the disagreement is
silent in the direction that matters: bump a pin in requirements.txt, forget to
regenerate the lock, and every install keeps using the old version while the
file you edited says otherwise. Nothing at runtime reads requirements.txt, so
nothing would ever contradict you.

Same guard LocalLLMAgent runs, for the same two files.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_PIN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s;]+)")


def _pins(filename: str) -> dict[str, str]:
    pins = {}
    for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN.match(line)
        if m:
            # Normalized the way pip compares names: case-insensitive, and
            # "-"/"_"/"." interchangeable (charset_normalizer == charset-normalizer).
            pins[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return pins


def test_the_lock_holds_more_than_the_direct_pins():
    """If it doesn't, the lock is not a closure — it's a copy."""
    assert len(_pins("requirements.lock")) > len(_pins("requirements.txt"))


def test_every_direct_pin_appears_in_the_lock_at_the_same_version():
    direct = _pins("requirements.txt")
    lock = _pins("requirements.lock")
    assert direct, "requirements.txt has no pins — parser broken?"

    missing = sorted(name for name in direct if name not in lock)
    assert not missing, (
        "in requirements.txt but absent from requirements.lock "
        f"(regenerate the lock): {missing}"
    )

    mismatched = {
        name: (want, lock[name]) for name, want in direct.items() if lock[name] != want
    }
    assert not mismatched, (
        f"requirements.txt and requirements.lock disagree (txt, lock): {mismatched}"
    )


def test_requirements_txt_pins_exactly_and_never_floats():
    """A `>=` here is what let three transitives move unnoticed; the lock only
    helps if the file it is generated from names one version, not a range."""
    floating = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
        and not line.startswith("#")
        and not _PIN.match(line.strip())
    ]
    assert not floating, f"requirements.txt must pin with == : {floating}"


def test_dev_dependencies_stay_out_of_the_lock():
    """The lock is runtime-only on purpose — see its header. pytest riding in
    would pin the test tool to the ingest's release cadence."""
    assert "pytest" not in _pins("requirements.lock")
