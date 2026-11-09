"""Generate a grouped Markdown table from a baseline JSON result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wastecap.eval.error_analysis import group_failures, markdown_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    text = f"# Baseline error analysis\n\n{markdown_summary(group_failures(result))}"
    if args.output:
        args.output.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
