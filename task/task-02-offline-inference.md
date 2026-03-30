# Task 02: Offline Inference 예제 실행

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

- [ ] Step 1: `basic.py` 실행 및 출력 확인
- [ ] Step 2: `generate.py` CLI 인자 실험
- [ ] Step 3: `chat.py` 실행 및 chat 포맷 이해
- [ ] Step 4: `SamplingParams` 파라미터 직접 실험
- [ ] `vllm/entrypoints/llm.py` 소스 직접 읽기

---

## 목표

vLLM의 `LLM` 클래스를 사용해 오프라인 추론(배치 추론)이 어떻게 동작하는지 이해한다.
서버 없이 Python 코드에서 직접 모델을 로드하고 텍스트를 생성하는 흐름을 익힌다.

---

## 핵심 개념

**Offline Inference란?**
- HTTP 서버 없이 Python 프로세스 안에서 직접 모델을 로드하고 추론하는 방식
- `LLM` 클래스가 모델 로딩, 스케줄링, 추론을 모두 담당
- 배치 처리에 적합 (여러 프롬프트를 한 번에 처리)

**Online Serving과의 차이:**
- Offline: `LLM(model=...).generate(prompts)` — 스크립트 실행
- Online: `vllm serve` — HTTP 서버를 띄워 API 요청 처리

---

## 단계별 실습

### Step 1: 가장 단순한 예제 실행

`facebook/opt-125m`은 125M 파라미터짜리 작은 모델로, HuggingFace에서 자동 다운로드된다.
GPU 메모리 부담이 적어 첫 실험에 적합하다.

```bash
python examples/offline_inference/basic/basic.py
```

예상 출력:
```
Generated Outputs:
------------------------------------------------------------
Prompt:    'Hello, my name is'
Output:    ' John. I am a student at the University of ...'
------------------------------------------------------------
```

코드 위치: `examples/offline_inference/basic/basic.py`

살펴볼 것:
- `LLM(model=...)` — 모델 로딩
- `SamplingParams(temperature=0.8, top_p=0.95)` — 샘플링 파라미터
- `llm.generate(prompts, sampling_params)` — 배치 추론
- `output.outputs[0].text` — 결과 접근 방식

---

### Step 2: generate.py — CLI 인자로 실험

`basic.py`와 동일한 추론이지만 CLI 인자를 지원한다.
`EngineArgs`가 어떻게 CLI와 연결되는지 확인할 수 있다.

```bash
# 기본 실행
python examples/offline_inference/basic/generate.py

# 인자 목록 확인
python examples/offline_inference/basic/generate.py --help

# 파라미터 직접 지정
python examples/offline_inference/basic/generate.py \
    --model facebook/opt-125m \
    --temperature 0 \
    --max-tokens 20
```

코드 위치: `examples/offline_inference/basic/generate.py`

살펴볼 것:
- `EngineArgs.add_cli_args(parser)` — 엔진 설정 전체가 CLI 인자로 노출되는 구조
- `LLM(**args)` — dict를 언패킹해서 LLM에 전달하는 패턴

---

### Step 3: Chat 형식 추론

`basic.py`는 raw text completion이다.
실제 LLM 사용에서 더 일반적인 chat 형식(system/user/assistant 턴)을 실험한다.

```bash
python examples/offline_inference/basic/chat.py \
    --model meta-llama/Llama-3.2-1B-Instruct
```

> Llama 모델은 HuggingFace 접근 권한이 필요할 수 있다.
> 권한 문제가 생기면 `huggingface-cli login`으로 토큰을 설정하거나,
> chat template이 있는 다른 모델로 대체한다 (예: `--model Qwen/Qwen2.5-0.5B-Instruct`).
> `facebook/opt-125m`은 chat template이 없어서 `llm.chat()`에서 에러가 난다.

코드 위치: `examples/offline_inference/basic/chat.py`

살펴볼 것:
- `llm.chat(conversation, sampling_params)` — chat 인터페이스
- conversation 포맷 (role: system / user / assistant)
- 배치 chat 처리 (`conversations = [conversation for _ in range(10)]`)

---

### Step 4: SamplingParams 실험

같은 프롬프트에 파라미터를 바꿔가며 출력이 어떻게 달라지는지 확인한다.

```python
from vllm import LLM, SamplingParams

# LLM은 한 번만 초기화한다 — 매번 새로 만들면 모델을 반복 로드하게 됨
llm = LLM(model="facebook/opt-125m")
prompt = ["The future of AI is"]

# 결정적 출력 (temperature=0)
print(llm.generate(prompt, SamplingParams(temperature=0))[0].outputs[0].text)

# 다양한 출력 (temperature=1.0)
print(llm.generate(prompt, SamplingParams(temperature=1.0))[0].outputs[0].text)

# 출력 길이 제한
print(llm.generate(prompt, SamplingParams(temperature=0, max_tokens=10))[0].outputs[0].text)
```

주요 파라미터:
| 파라미터 | 역할 |
|----------|------|
| `temperature` | 0이면 greedy(결정적), 높을수록 다양한 출력 |
| `top_p` | nucleus sampling 확률 임계값 |
| `max_tokens` | 최대 생성 토큰 수 |
| `n` | 프롬프트당 생성할 출력 개수 |

---

## 코드 흐름 이해

실습 후 아래 흐름을 소스코드에서 직접 추적해볼 것:

```
LLM(model=...) 생성
    └─ vllm/entrypoints/llm.py  →  LLM.__init__
           └─ LLMEngine 또는 AsyncLLMEngine 초기화

llm.generate(prompts, sampling_params)
    └─ vllm/entrypoints/llm.py  →  LLM.generate
           └─ 내부적으로 스케줄러가 배치 구성 후 추론 실행

output.outputs[0].text
    └─ vllm/outputs.py  →  RequestOutput, CompletionOutput
```
