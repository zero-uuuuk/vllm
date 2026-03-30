# Task 01: vLLM 편집 모드 설치

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 설치 과정에서 버그나 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.
> 공식 문서, 소스코드, GitHub Issues를 직접 확인하는 습관을 들일 것.

## TODO

- [ ] EC2 인스턴스 생성 및 초기 설정
- [ ] `uv` 설치 및 가상환경 구성
- [ ] vLLM 편집 모드 설치
- [ ] `import vllm` 으로 설치 확인

---
vLLM 소스코드를 직접 수정하고 즉시 반영되는 환경을 구성한다.
Python 코드 수정에 집중하며, C++/CUDA 빌드는 포함하지 않는다.

---

## 환경: EC2 Instance

### 권장 인스턴스 스펙

| 항목 | 권장 사양 |
|------|-----------|
| 인스턴스 타입 | `g4dn.xlarge` (T4 GPU 1장) 이상 |
| AMI | Deep Learning AMI (Ubuntu 22.04) — PyTorch 포함 |
| 스토리지 | 50GB 이상 (모델 가중치 포함 시 더 필요) |
| Python | 3.12 권장 (3.10~3.13 지원) |

> 공부 목적이라면 `g4dn.xlarge`로 충분하다.
> GPU 없이 CPU만으로 실험하려면 `c5.2xlarge` 등도 가능하나 추론 속도가 매우 느리다.

### EC2 초기 설정

```bash
# 1. 시스템 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# 2. uv 설치 (vLLM 공식 권장 패키지 매니저)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # 또는 새 터미널 열기

# 3. Python 3.12 가상환경 생성
uv venv vllm-env --python 3.12
source vllm-env/bin/activate
```

---

## 편집 모드 설치 (Python only)

```bash
# 1. 레포 클론
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 2. 편집 모드로 설치
#    VLLM_USE_PRECOMPILED=1 → 미리 빌드된 CUDA 바이너리를 다운로드해서 사용
#    -e → editable mode, 소스 수정이 즉시 반영됨
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

### 설치 확인

```bash
python -c "import vllm; print(vllm.__version__)"
```

### 편집 모드란?

`-e` 옵션은 패키지를 복사하지 않고 소스 디렉토리를 직접 참조한다.
`vllm/` 아래 Python 파일을 수정하면 재설치 없이 즉시 반영된다.

---

## Linting 설정 (선택)

> 공부 목적이라면 지금 당장 필요하지 않다.
> PR을 올릴 계획이 생겼을 때 설정해도 충분하다.

설정하면 `git commit` 시 자동으로 코드 스타일 검사가 실행된다.
(ruff 포매터, mypy 타입 체크, clang-format 등)

```bash
uv pip install pre-commit
pre-commit install
```
