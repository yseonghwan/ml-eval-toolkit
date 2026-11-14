# ml-eval-toolkit

모델 학습 정체와 거리 계산 왜곡을 진단하기 위해 만든 **CNN 특성 맵 디버깅 및 K-Means 정량 평가 CLI·라이브러리**입니다.

주피터 노트북(`.ipynb`) 환경에서 하이퍼파라미터를 수동으로 바꾸며 재학습하던 반복 작업을 줄이고, **명확한 입출력 Spec(`dataclass` 및 JSON 출력)**을 기반으로 사람과 자동화 파이프라인(또는 LLM 에이전트)이 모두 호출할 수 있는 구조로 설계했습니다.

---

## 1. 해결하려는 문제 (Problem Definition)

1. **CNN 입력 정규화 누락 및 계층별 죽은 채널(Dead Channel) 추적**
   - 합성곱 신경망(`Conv2D` 32채널 → `MaxPooling2D` → `Conv2D` 64채널) 학습 시 입력 스케일링(`/ 255.0`)이 누락되거나 특정 계층에서 활성화 값이 소실되어도 단순 손실(Loss) 값만으로는 원인을 특정하기 어렵습니다.
   - 또한 TensorFlow/Keras에서 연산 결정성(`enable_op_determinism`) 활성화 시 `MaxPooling2D`의 XLA 구현 한계로 런타임 에러가 발생하므로, `jit_compile=False` 설정과 계층별 활성화 통계 추출을 모듈화했습니다.
2. **K-Means 유클리드 거리 왜곡 제어 및 최적 K값 자동 검증**
   - 변수 간 스케일 차이가 큰 데이터를 그대로 군집화하면 값의 단위가 큰 변수 하나로 거리 계산이 쏠립니다.
   - 전처리 인터페이스(`Min-Max`, `Z-score`)를 강제하고, 엘보우(Inertia)와 실루엣 계수(Silhouette Score)를 한 번에 산출해 최적 K값을 반환하는 파이프라인을 밑바닥부터 NumPy로 구현했습니다.

---

## 2. 디렉토리 구조 (Directory Structure)

```text
ml-eval-toolkit/
├── .github/
│   └── workflows/
│       └── ci.yml                  # Push/PR 시 Ruff 린트 및 Pytest 단위 테스트 자동 실행
├── src/
│   └── ml_eval_toolkit/
│       ├── __init__.py             # 패키지 공개 API 정의
│       ├── spec.py                 # 입출력 데이터 구조(ClusterEvalResult, CNNDebugReport) 정의
│       ├── kmeans_eval.py          # NumPy 기반 정규화, K-Means++, Inertia, 실루엣 계수 산출
│       ├── cnn_debugger.py         # 입력 스케일 검증, 4D 특성 맵 사멸 채널 분석, 결정적 CNN 빌더
│       └── cli.py                  # 터미널 실행용 CLI 엔트리포인트 (ml-eval)
├── tests/
│   ├── test_kmeans_eval.py         # 정규화 통계치 및 군집 품질 지표 범위(-1.0 ~ 1.0) 검증 테스트
│   └── test_cnn_debugger.py        # 스케일 이상 탐지, 죽은 채널 비율 계산, CLI JSON 출력 테스트
├── pyproject.toml                  # 빌드 의존성, CLI 스크립트 등록, Ruff/Mypy/Pytest 설정
└── requirements.txt                # 핵심 실행 의존성 목록
```

---

## 3. 설치 및 빠른 시작 (Installation & Quick Start)

```bash
# 저장소 클론 및 개발 모드 설치
git clone https://github.com/<yseonghwan>/ml-eval-toolkit.git
cd ml-eval-toolkit
pip install -e ".[dev]"
```

---

## 4. CLI 레퍼런스 (CLI Usage)

설치 후 `ml-eval` 명령어 또는 `python -m ml_eval_toolkit.cli`로 실행합니다. `--json` 플래그를 전달하면 다른 시스템이나 LLM 도구가 파싱할 수 있는 정형 JSON Spec을 출력합니다.

### 4.1 K-Means 정량 평가 파이프라인 실행

```bash
python -m ml_eval_toolkit.cli kmeans --samples 180 --k-min 2 --k-max 5 --scale zscore --json
```

**출력 예시 (JSON Spec)**
```json
{
  "task": "kmeans_evaluation",
  "scale_method": "zscore",
  "recommended_k": 3,
  "best_silhouette_score": 0.6542,
  "evaluations": [
    {
      "k": 2,
      "scale_method": "zscore",
      "inertia": 172.4102,
      "silhouette_score": 0.5418,
      "iterations": 4
    },
    {
      "k": 3,
      "scale_method": "zscore",
      "inertia": 58.1924,
      "silhouette_score": 0.6542,
      "iterations": 5
    }
  ]
}
```

### 4.2 CNN 입력 스케일 및 특성 맵 진단 실행

```bash
# 정규화 누락([0, 255]) 및 죽은 채널 경고 시뮬레이션
python -m ml_eval_toolkit.cli cnn-check --unnormalized-demo --json
```

**출력 예시 (JSON Spec)**
```json
{
  "is_input_normalized": false,
  "input_min": 0.0,
  "input_max": 255.0,
  "input_mean": 127.412,
  "layer_stats": [
    {
      "layer_name": "conv2d",
      "shape": [8, 28, 28, 32],
      "mean_activation": 0.17821,
      "std_activation": 0.21452,
      "max_activation": 1.41205,
      "dead_channel_ratio": 0.0,
      "zero_activation_ratio": 0.3085
    },
    {
      "layer_name": "conv2d_1",
      "shape": [8, 14, 14, 64],
      "mean_activation": 0.04812,
      "std_activation": 0.11204,
      "max_activation": 1.18921,
      "dead_channel_ratio": 0.625,
      "zero_activation_ratio": 0.7421
    }
  ],
  "warning_messages": [
    "Input scale anomaly detected: range [0.0, 255.0]. Divide raw pixel values by 255.0 or apply channel standardization.",
    "Layer 'conv2d_1' has 62.5% dead channels."
  ]
}
```

---

## 5. 라이브러리 API 레퍼런스 (Library Usage)

```python
import numpy as np
from ml_eval_toolkit import evaluate_k_range, run_cnn_diagnostic_report

# 1. K-Means 정량 평가
raw_data = np.random.randn(150, 2) * np.array([1.0, 1000.0])
eval_results = evaluate_k_range(raw_data, k_min=2, k_max=5, scale_method="zscore")
for res in eval_results:
    print(res.k, res.silhouette_score)

# 2. CNN 입력 스케일 및 4D Feature Map 진단
sample_images = np.random.rand(4, 28, 28, 1)  # [0.0, 1.0] 정규화 완료 텐서
conv1_output = np.random.rand(4, 28, 28, 32)
report = run_cnn_diagnostic_report(sample_images, {"conv2d": conv1_output})
print(report.is_input_normalized, report.warning_messages)
```

---

## 6. 테스트 및 CI 검증 (Testing & Linting)

```bash
# 전체 단위 테스트 실행
pytest -v
```
