from __future__ import annotations

from pathlib import Path

from gmd_active_learning.active_learning.deduplicate import deduplicate_candidates, load_risk_score
from gmd_active_learning.utils.io import read_json


class CandidateSelector:
    def __init__(self, candidate_dir: str | Path) -> None:
        self.candidate_dir = Path(candidate_dir)

    def select(self, max_candidates: int, dedup_threshold: float = 0.1) -> list[Path]:
        candidates = [path for path in self.candidate_dir.iterdir() if path.is_dir()]
        ranked = sorted(candidates, key=load_risk_score, reverse=True)
        deduped = deduplicate_candidates(ranked, threshold=dedup_threshold)
        return deduped[:max_candidates]

    def group_by_anomaly(self) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for candidate in self.candidate_dir.iterdir():
            if not candidate.is_dir():
                continue
            data = read_json(candidate / "monitor_metrics.json")
            reasons = data.get("reasons", [])
            key = reasons[0] if reasons else "unknown"
            groups.setdefault(key, []).append(candidate)
        return groups
