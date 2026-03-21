from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VmPrediction:
    vm: str
    label: str
    confidence: float
    historical_accuracy: float


def weighted_vote(predictions: List[VmPrediction]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for p in predictions:
        weight = p.confidence * p.historical_accuracy
        scores[p.label] = scores.get(p.label, 0.0) + weight
    return scores


def main() -> None:
    sample = [
        VmPrediction("VM1", "Pneumonia", 0.82, 0.85),
        VmPrediction("VM2", "Normal", 0.74, 0.80),
        VmPrediction("VM3", "Pneumonia", 0.88, 0.90),
    ]
    scores = weighted_vote(sample)
    print("Aggregated scores:", scores)


if __name__ == "__main__":
    main()
