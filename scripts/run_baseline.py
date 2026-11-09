"""Run the real Qwen baseline and write JSON plus an error-analysis table."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from wastecap.eval.baseline import run_baseline
from wastecap.eval.error_analysis import failure_categories, group_failures, markdown_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen_vl")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--db-path", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output = args.output_dir / f"baseline_{args.model}_{args.split}.json"
    result = run_baseline(
        args.model,
        args.split,
        limit=args.limit,
        db_path=args.db_path,
        output_path=output,
    )
    groups = group_failures(result)
    table_path = args.output_dir / f"baseline_{args.model}_{args.split}_error_analysis.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        f"# Baseline error analysis\n\nSource: `{output}`\n\n"
        f"## Failure categories\n\n{failure_categories(result)}\n\n"
        f"## Taxonomy groups\n\n{markdown_summary(groups)}"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_record"},
            indent=2,
        )
    )
    print(f"Error analysis: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
