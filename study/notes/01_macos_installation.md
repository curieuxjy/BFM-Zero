# macOS 설치 및 문제 해결 가이드

## 환경 정보
- macOS Darwin 25.2.0
- Apple Silicon (arm64)
- Python 3.10.18 (uv가 자동 설치)

## 설치 단계

### 1. Git LFS 설치
```bash
brew install git-lfs
git lfs install
```

### 2. LFS 데이터 다운로드 (예산 초과 시)
GitHub LFS 예산 초과 시 HuggingFace에서 직접 다운로드:

```bash
# 모션 데이터
curl -L -o humanoidverse/data/lafan_29dof.pkl \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/data/lafan_29dof.pkl"

curl -L -o humanoidverse/data/lafan_29dof_10s-clipped.pkl \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/data/lafan_29dof_10s-clipped.pkl"

# 모델 체크포인트
mkdir -p model/checkpoint/model

curl -L -o model/checkpoint/config.json \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/model/checkpoint/config.json"

curl -L -o model/checkpoint/init_kwargs.json \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/model/checkpoint/init_kwargs.json"

curl -L -o model/checkpoint/train_status.json \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/model/checkpoint/train_status.json"

curl -L -o model/checkpoint/model/config.json \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/model/checkpoint/model/config.json"

curl -L -o model/checkpoint/model/init_kwargs.json \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/model/checkpoint/model/init_kwargs.json"

curl -L -o model/checkpoint/model/model.safetensors \
  "https://huggingface.co/LeCAR-Lab/BFM-Zero/resolve/main/model/checkpoint/model/model.safetensors"
```

### 3. 의존성 설치
```bash
uv sync
```

### 4. 누락된 패키지 추가
```bash
uv add imageio
```

---

## macOS 전용 문제 해결

### 문제 1: `model/config.json` 누락

**증상:**
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**원인:** HuggingFace에 `model/config.json`(학습 설정 파일)이 없음

**해결:** `model/config.json` 직접 생성

```json
{
    "agent": {
        "name": "FBcprAuxAgent"
    },
    "env": {
        "name": "humanoidverse_isaac",
        "device": "cpu",
        "lafan_tail_path": "humanoidverse/data/lafan_29dof.pkl",
        "enable_cameras": false,
        "disable_obs_noise": false,
        "disable_domain_randomization": false,
        "relative_config_path": "exp/bfm_zero/bfm_zero",
        "include_last_action": true,
        "hydra_overrides": [
            "num_envs=1",
            "+headless=True",
            "robot=g1/g1_29dof"
        ],
        "context_length": null,
        "include_dr_info": false,
        "included_dr_obs_names": null,
        "include_history_actor": true,
        "include_history_noaction": false,
        "root_height_obs": true
    },
    "work_dir": "model",
    "seed": 0
}
```

---

### 문제 2: MuJoCo Hydra config 누락

**증상:**
```
MissingConfigException: Could not find 'simulator/mujoco'
```

**원인:** `humanoidverse/config/simulator/mujoco.yaml` 파일이 없음

**해결:** `humanoidverse/config/simulator/mujoco.yaml` 생성

```yaml
# @package _global_

simulator:
  _target_: humanoidverse.simulator.mujoco.mujoco.MuJoCo
  _recursive_: False
  config:
    name: "mujoco"
    terrain: ${terrain}
    plane:
        static_friction: 1.0
        dynamic_friction: 1.0
        restitution: 0.0
    sim:
      fps: 200
      control_decimation: 4
      substeps: 1
      render_mode: "human"
      render_interval: 4

    scene:
      num_envs: ${num_envs}
      env_spacing: ${env.config.env_spacing}
```

---

### 문제 3: 로봇 config 이름 불일치

**증상:**
```
MissingConfigException: Could not find 'robot/g1/g1_29dof_new'
```

**원인:** `g1_29dof_new`가 없고 `g1_29dof`만 존재

**해결:** `model/config.json`의 hydra_overrides에 `robot=g1/g1_29dof` 추가

---

### 문제 4: MUJOCO_GL 환경 변수 오류

**증상:**
```
RuntimeError: invalid value for environment variable MUJOCO_GL: egl
```

**원인:**
- `tracking_inference.py`가 `os.environ["MUJOCO_GL"] = "egl"` 하드코딩
- macOS에서 EGL 미지원 (Linux 전용)

**해결:** macOS용 래퍼 스크립트 생성 (`study/scripts/tracking_inference_macos.py`)

```python
import os
import sys

# macOS에서는 glfw 사용 (egl은 Linux 전용)
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ["MUJOCO_GL"] = "egl"
```

---

### 문제 5: quat_rotate 함수 인자 누락

**증상:**
```
RuntimeError: quat_rotate() is missing value for argument 'w_last'
```

**원인:** `humanoidverse/simulator/mujoco/mujoco.py`에서 `quat_rotate` 호출 시 `w_last` 인자 누락

**해결:** `humanoidverse/simulator/mujoco/mujoco.py` 수정

```python
# 변경 전
quat_rotate(base_quat, qvel_tensor[:, 3:6]),

# 변경 후
quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),
```

두 군데 수정 필요:
- `robot_root_states` 프로퍼티 (line ~441)
- `all_root_states` 프로퍼티 (line ~462)

---

## 최종 실행 방법

```bash
# macOS용 스크립트 사용
uv run python study/scripts/tracking_inference_macos.py --model_folder model/

# 옵션
# --headless        : GUI 없이 실행 (기본값: True)
# --no-headless     : GUI 표시
# --save_mp4        : 비디오 저장
# --device cpu      : CPU 사용 (macOS 필수)
# --simulator mujoco: MuJoCo 시뮬레이터 (macOS 필수)
```

---

## 요약: macOS 제약사항

| 항목 | Linux | macOS |
|------|-------|-------|
| Isaac Sim | ✓ | ✗ |
| MuJoCo | ✓ | ✓ |
| CUDA | ✓ | ✗ |
| MPS (Apple GPU) | ✗ | 코드에서 미지원 |
| MUJOCO_GL | egl | glfw |
| device | cuda | cpu |

---

## 파일 목록

### 메인 코드 외부 (study/ 폴더)
1. **`study/configs/simulator/mujoco.yaml`** - MuJoCo Hydra config
2. **`study/patches/mujoco_patch.py`** - quat_rotate 버그 런타임 패치
3. **`study/patches/config_patch.py`** - config 파일 복사 유틸리티
4. **`study/scripts/tracking_inference_macos.py`** - macOS용 추론 스크립트

### 모델 폴더 (gitignore 권장)
1. **`model/config.json`** - 학습 설정 파일 (직접 생성)

### 패치 방식
메인 소스코드(`humanoidverse/`)를 직접 수정하지 않습니다.
- **런타임 패치**: `study/patches/mujoco_patch.py`가 MuJoCo 클래스 메서드를 교체
- **Config 복사**: `study/patches/config_patch.py`가 필요한 yaml 파일을 복사

이 방식으로 upstream 업데이트 시 충돌을 방지합니다.
