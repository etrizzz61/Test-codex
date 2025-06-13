"""Database utilities for card information."""
from pathlib import Path
import json
from typing import Dict, Any


class CardDatabase:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.data = self._load_db()

    def _load_db(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            return {}
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def find_card(self, name: str) -> Dict[str, Any] | None:
        return self.data.get(name)
