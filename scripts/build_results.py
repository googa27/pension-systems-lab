"""Build the maintained results narrative from the notebook summary artifact."""

from __future__ import annotations

from pathlib import Path

from chile_demographic_pde.reporting import write_results

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports" / "model_summary.json"
OUTPUT = ROOT / "reports" / "RESULTS.md"


def main() -> None:
    write_results(SUMMARY, OUTPUT)
    print(f"Built {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
