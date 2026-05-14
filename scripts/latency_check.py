import statistics
import time
from typing import Any

import requests

API_URL = "http://localhost:8000/api/v1/detect"


def measure_latency(text: str, repeat: int = 10) -> None:
    latencies_ms: list[float] = []

    for _ in range(repeat):
        start = time.perf_counter()
        response = requests.post(
            API_URL,
            json={"text": text},
            timeout=30,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.raise_for_status()
        latencies_ms.append(elapsed_ms)

    print("=" * 60)
    print(f"Text length: {len(text)} characters")
    print(f"Repeat: {repeat}")
    print(f"Average latency: {statistics.mean(latencies_ms):.2f} ms")
    print(f"Min latency: {min(latencies_ms):.2f} ms")
    print(f"Max latency: {max(latencies_ms):.2f} ms")
    print(f"Last response: {response.json()}")


def main() -> None:
    short_text = "신고된 댓글 예시입니다."
    long_text = "이 문장은 신고된 댓글의 길이가 긴 경우를 가정한 테스트입니다. " * 20

    measure_latency(short_text, repeat=10)
    measure_latency(long_text, repeat=10)


if __name__ == "__main__":
    main()



'''
============================================================
Text length: 13 characters
Repeat: 10
Average latency: 2094.73 ms
Min latency: 2038.08 ms
Max latency: 2387.72 ms
Last response: {'is_hate_speech': False, 'confidence': 0.9287270307540894, 'category': 'clean', 'action': 'allow', 'message': 'Message allowed.'}
============================================================
Text length: 720 characters
Repeat: 10
Average latency: 2180.80 ms
Min latency: 2148.35 ms
Max latency: 2217.71 ms
Last response: {'is_hate_speech': False, 'confidence': 0.899380087852478, 'category': 'clean', 'action': 'allow', 'message': 'Message allowed.'}
'''