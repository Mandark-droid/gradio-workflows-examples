"""Run every test suite: the repo-level tests plus each example's own.

Examples are tested from their own directory because each defines a
top-level `nodes` package; collecting them together would make `import
nodes` ambiguous.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    suites = [ROOT]
    examples = ROOT / "examples"
    if examples.exists():
        suites += [p for p in sorted(examples.iterdir()) if (p / "tests").is_dir()]

    failed = []
    for suite in suites:
        print(f"\n=== pytest in {suite.relative_to(ROOT) or '.'} ===")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"], cwd=suite
        )
        if result.returncode != 0:
            failed.append(str(suite.relative_to(ROOT) or "."))

    if failed:
        print(f"\nFAILED suites: {', '.join(failed)}")
        return 1
    print("\nAll suites passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
