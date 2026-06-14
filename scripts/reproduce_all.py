from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from _runner import write_summary


EXPERIMENTS = [
    ("toy_color", "scripts.run_synthetic"),
    ("iris_cancer", "scripts.run_diabetes"),
    ("newsgroups", "scripts.run_newsgroups"),
    ("decoy_mnist", "scripts.run_decoy_mnist"),
    ("find_another_explanation", "scripts.run_unsupervised"),
    ("explanation_visualization", "scripts.visualize_explanations"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    rows = []
    failures = []
    for dataset, module_name in EXPERIMENTS:
        sys.argv = [module_name]
        if args.quick:
            sys.argv.append("--quick")
        sys.argv.extend(["--device", args.device])
        try:
            result = importlib.import_module(module_name).main()
            checks = result.get("reproduction_checks", {})
            status = "reproduced" if checks and all(bool(value) for value in checks.values()) else "partial"
            notes = "; ".join(f"{key}={value}" for key, value in checks.items())
            if "history" in result:
                rows.append(
                    {
                        "experiment_id": result["experiment_id"],
                        "dataset": dataset,
                        "method": "iterative_rrr",
                        "metric": "accuracy",
                        "reproduced_result": result["history"][-1]["accuracy"],
                        "status": status,
                        "notes": notes or "final round",
                    }
                )
            else:
                for method in ["baseline", "rrr"]:
                    rows.append(
                        {
                            "experiment_id": result["experiment_id"],
                            "dataset": dataset,
                            "method": method,
                            "metric": "accuracy",
                            "reproduced_result": result[method]["accuracy"],
                            "status": status,
                            "notes": notes,
                        }
                    )
                if dataset == "decoy_mnist" and "extra_eval" in result:
                    rows.extend(
                        [
                            {
                                "experiment_id": result["experiment_id"],
                                "dataset": dataset,
                                "method": "baseline",
                                "metric": "random_swatch_accuracy",
                                "reproduced_result": result["extra_eval"]["baseline_random_swatch"]["accuracy"],
                                "status": status,
                                "notes": notes,
                            },
                            {
                                "experiment_id": result["experiment_id"],
                                "dataset": dataset,
                                "method": "rrr",
                                "metric": "random_swatch_accuracy",
                                "reproduced_result": result["extra_eval"]["rrr_random_swatch"]["accuracy"],
                                "status": status,
                                "notes": notes,
                            },
                        ]
                    )
        except Exception as exc:
            failures.append((dataset, repr(exc)))
            rows.append(
                {
                    "experiment_id": dataset,
                    "dataset": dataset,
                    "method": "all",
                    "metric": "accuracy",
                    "reproduced_result": "",
                    "status": "failed",
                    "notes": repr(exc),
                }
            )
    write_summary(rows)
    if failures:
        for dataset, error in failures:
            print(f"{dataset}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
