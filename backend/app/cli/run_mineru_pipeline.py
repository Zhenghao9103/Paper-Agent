"""Internal subprocess entrypoint for one MinerU pipeline parse."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(args.config.resolve())
    os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    formula_enable = os.environ.get("MINERU_FORMULA_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    from mineru.cli.common import do_parse

    do_parse(
        str(args.output),
        [args.pdf.stem],
        [args.pdf.read_bytes()],
        ["en"],
        backend="pipeline",
        parse_method="txt",
        formula_enable=formula_enable,
        table_enable=True,
        image_analysis=False,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
