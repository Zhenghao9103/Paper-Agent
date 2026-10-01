from __future__ import annotations

import argparse
import json
from pathlib import Path

from .rag_gold import RAGGoldDataset


def validate_dataset(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset = RAGGoldDataset.model_validate(payload)
    return dataset.summary()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the 20-paper PDF-reviewed RAG gold dataset."
    )
    parser.add_argument("dataset", type=Path, help="Path to the gold dataset JSON.")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
