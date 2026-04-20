from pathlib import Path

from gmd_active_learning.active_learning.candidate_selector import CandidateSelector


def main() -> None:
    selector = CandidateSelector(Path("active_learning_candidates"))
    print(selector.group_by_anomaly())


if __name__ == "__main__":
    main()
