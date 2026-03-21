import json
import time
from statistics import mean, pstdev


def benchmark_stub(runs: int = 10) -> dict:
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        time.sleep(0.01)
        timings.append((time.perf_counter() - t0) * 1000)
    return {
        "inference_mean_ms": round(mean(timings), 3),
        "inference_std_ms": round(pstdev(timings), 3),
        "status": "pending_real_model",
    }


def main() -> None:
    result = benchmark_stub(10)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
