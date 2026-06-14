from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {"exp001_baseline", "exp002_gradcam"}


def should_include(path: Path) -> bool:
    return not any(part in EXCLUDED_PARTS for part in path.parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results.zip")
    args = parser.parse_args()
    output = ROOT / args.output
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for base in [ROOT / "results", ROOT / "experiments"]:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and should_include(path.relative_to(ROOT)):
                    zf.write(path, path.relative_to(ROOT).as_posix())
    print(output)


if __name__ == "__main__":
    main()
