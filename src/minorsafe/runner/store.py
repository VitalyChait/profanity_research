"""Crash-safe atomic Parquet store for experiment run shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class RunStore:
    """Manages atomic writes and crash-resume deduplication for a single run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.responses_path = self.run_dir / "responses.parquet"

    def completed_keys(self) -> set[tuple[str, int]]:
        if not self.responses_path.exists():
            return set()
        df = pd.read_parquet(self.responses_path, columns=["example_id", "seed"])
        return {(str(r.example_id), int(r.seed)) for r in df.itertuples(index=False)}

    def load_responses(self) -> pd.DataFrame:
        if not self.responses_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.responses_path)

    def append_responses(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        new_df = pd.DataFrame(rows)
        if self.responses_path.exists():
            existing_df = pd.read_parquet(self.responses_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        if "example_id" in combined.columns and "seed" in combined.columns:
            combined = combined.drop_duplicates(subset=["example_id", "seed"], keep="first")

        tmp_path = self.responses_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp_path, index=False)
        tmp_path.replace(self.responses_path)

