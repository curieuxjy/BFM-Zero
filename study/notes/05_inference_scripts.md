# BFM-Zero 추론 스크립트 분석

## 개요

BFM-Zero는 세 가지 추론 모드를 제공합니다:

| 스크립트 | 용도 | 입력 |
|---------|------|------|
| `tracking_inference.py` | 모션 트래킹 | 전문가 궤적 |
| `goal_inference.py` | 목표 도달 | 목표 프레임 |
| `reward_inference.py` | 보상 최적화 | 보상 함수 |

---

## 1. Tracking Inference (모션 트래킹)

### 원리

```python
def tracking_inference(obs) -> torch.Tensor:
    # 전문가 궤적을 Backward Map으로 인코딩
    z = model.backward_map(obs)  # [T, z_dim]

    # 미래 프레임 평균 (smoothing)
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)

    return model.project_z(z)
```

### 실행 방법

```bash
# Linux (Isaac Sim)
python humanoidverse/tracking_inference.py --model_folder model/

# macOS (MuJoCo) - 참조 모션과 정책 출력을 나란히 표시
uv run python study/scripts/tracking_inference_macos.py --model_folder model/

# 다른 모션 지정 (정수 인덱스 0~39)
uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --motion_list 0

# 옵션
--headless          # GUI 없이 실행
--motion_list 25    # 평가할 모션 ID (기본: 25)
--num_steps 500     # 시뮬레이션 스텝 수 (headless)
--disable_dr        # Domain Randomization 비활성화
--disable_obs_noise # Observation noise 비활성화
```

### GUI 모드 화면 구성 (macOS)

macOS 스크립트는 좌우 분할 화면으로 참조 모션과 정책을 비교한다:

```
┌─────────────────────┬─────────────────────┐
│   Reference Motion  │    Policy Output    │
│                     │                     │
│  (모션 라이브러리의  │  (학습된 모델이      │
│   원본 동작)         │   생성하는 동작)     │
│                     │                     │
│  Frame 42/13065     │  Step 42            │
└─────────────────────┴─────────────────────┘
```

**구현 방식**: 동일한 `MjModel`에서 두 개의 `MjData` 인스턴스를 생성하고 각각 별도의 `mujoco.Renderer`로 렌더링한다. 참조 모션의 qpos는 `get_backward_observation()`이 반환하는 `obs_dict`에서 추출하며, 쿼터니언은 내부 포맷(xyzw)에서 MuJoCo 포맷(wxyz)으로 변환된다.

### 출력

```
model/
├── exported/
│   └── FBcprAuxModel.onnx    # ONNX 내보내기
└── tracking_inference/
    └── zs_25.pkl             # z 벡터 저장
```

---

## 2. Goal Inference (목표 도달)

### 원리

```python
def goal_inference(goal_observation):
    # 목표 프레임을 Backward Map으로 인코딩
    z = model.backward_map(goal_observation)
    return model.project_z(z)
```

### 실행 방법

```bash
python humanoidverse/goal_inference.py --model_folder model/
```

### 설정 파일

`humanoidverse/data/robots/g1/goal_frames_lafan29dof.json`:

```json
[
  {
    "motion_id": 25,
    "motion_name": "walk1_subject1",
    "frames": [100, 200, 300]  # 목표로 사용할 프레임
  }
]
```

### 출력

```
model/
└── goal_inference/
    ├── goal_reaching.pkl     # z 벡터 딕셔너리
    └── videos/
        └── goal.mp4          # 목표 도달 비디오
```

---

## 3. Reward Inference (보상 최적화)

### 원리

```python
def reward_inference(task):
    # 태스크 정의 (예: "move-ego-0-0.7")
    # 리플레이 버퍼에서 샘플링하여 보상 가중 평균
    z = weighted_average(B(states), rewards)
    return model.project_z(z)
```

### 지원 태스크

```python
tasks = [
    # 정지
    "move-ego-0-0",           # 제자리 서기
    "move-ego-low0.5-0-0",    # 낮은 자세

    # 이동
    "move-ego-0-0.7",         # 전진 (0.7 m/s)
    "move-ego-90-0.7",        # 왼쪽
    "move-ego-180-0.7",       # 후진
    "move-ego--90-0.7",       # 오른쪽

    # 회전
    "rotate-z-5-0.5",         # 시계 방향 회전
    "rotate-z--5-0.5",        # 반시계 방향

    # 팔 올리기
    "raisearms-l-l",          # 양팔 낮게
    "raisearms-m-m",          # 양팔 중간

    # 이동 + 팔
    "move-arms-0-0.7-m-m",    # 전진하며 팔 올리기

    # 앉기
    "crouch-0",               # 웅크리기
    "sitonground",            # 바닥에 앉기
]
```

### 실행 방법

```bash
python humanoidverse/reward_inference.py --model_folder model/
```

### 출력

```
model/
└── reward_inference/
    ├── reward_locomotion.pkl  # z 벡터 딕셔너리
    └── videos/
        ├── move-ego-0-0.7.mp4
        ├── rotate-z-5-0.5.mp4
        └── ...
```

---

## 4. macOS 호환성

### 필요한 패치

macOS에서 MuJoCo 시뮬레이터를 사용하려면 `quat_rotate` 함수 패치가 필요합니다:

```python
# 원본 (버그)
quat_rotate(base_quat, qvel_tensor[:, 3:6])

# 패치 (수정)
quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True)
```

### 패치 적용 방법

`study/scripts/tracking_inference_macos.py`에서 자동 적용:

```python
# 스크립트 시작 부분에서 패치
from humanoidverse.simulator.mujoco import mujoco as mujoco_module
from humanoidverse.utils.torch_utils import quat_rotate
import torch

MuJoCoClass = mujoco_module.MuJoCo

def _patched_robot_root_states(self):
    ...
    quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True)
    ...

MuJoCoClass.robot_root_states = property(_patched_robot_root_states)
MuJoCoClass.all_root_states = property(_patched_all_root_states)
```

### 환경 변수

```bash
# macOS
export MUJOCO_GL=glfw

# Linux (headless)
export MUJOCO_GL=egl
```

---

## 5. 모델 로딩

### HuggingFace에서 다운로드

```bash
# 모델 다운로드 (huggingface-hub 사용)
huggingface-cli download unitreerobotics/bfm-zero --local-dir model/
```

### 모델 구조

```
model/
├── checkpoint/
│   ├── config.json           # 에이전트 설정
│   ├── init_kwargs.json      # 초기화 파라미터
│   ├── train_status.json     # 학습 상태
│   └── model/
│       ├── config.json
│       ├── init_kwargs.json
│       └── model.safetensors  # 모델 가중치 (~3.4GB)
└── config.json               # 환경 설정
```

### 로딩 코드

```python
from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

model = load_model_from_checkpoint_dir(
    "model/checkpoint",
    device="cpu"  # 또는 "cuda", "mps"
)
model.eval()
```

---

## 6. ONNX 내보내기

모든 추론 스크립트는 ONNX 모델을 내보냅니다:

```python
from humanoidverse.utils.helpers import export_meta_policy_as_onnx

export_meta_policy_as_onnx(
    model,
    output_dir="model/exported",
    filename="FBcprAuxModel.onnx",
    inputs={"actor_obs": torch.randn(1, obs_dim + z_dim)},
    z_dim=model.cfg.archi.z_dim,
    history=True,
    use_29dof=True,
)
```

ONNX 모델은 로봇에 직접 배포할 수 있습니다.

---

## 7. 실행 흐름

```
┌─────────────────────────────────────────────────────────────┐
│  1. 모델 로드                                                │
│     model = load_model_from_checkpoint_dir(...)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 환경 빌드                                                │
│     env_cfg = HumanoidVerseIsaacConfig(**config["env"])    │
│     wrapped_env, _ = env_cfg.build(num_envs)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Z 벡터 계산                                              │
│     - Tracking: z = B(expert_traj)                         │
│     - Goal: z = B(goal_frame)                              │
│     - Reward: z = Σ r_i * B(s_i)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 추론 루프                                                │
│     for step in range(num_steps):                          │
│         action = model.act(obs, z)                         │
│         obs, reward, ... = env.step(action)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 결과 저장                                                │
│     - z 벡터: .pkl                                         │
│     - 비디오: .mp4                                         │
│     - ONNX: .onnx                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 테스트 결과 (macOS)

```
✅ 모델 로딩: 성공 (FBcprAuxModel)
✅ 환경 빌드: 성공 (MuJoCo)
✅ Tracking Inference: 성공
   - Motion ID 25 (walk1_subject1)
   - z shape: [13065, 256]
   - 50 스텝 롤아웃 완료
```

---

## 9. 코드 참조

| 기능 | 파일 |
|------|------|
| Tracking Inference | `humanoidverse/tracking_inference.py` |
| Goal Inference | `humanoidverse/goal_inference.py` |
| Reward Inference | `humanoidverse/reward_inference.py` |
| macOS용 Tracking (참조모션 비교) | `study/scripts/tracking_inference_macos.py` |
| macOS용 Goal Inference | `study/scripts/goal_inference_macos.py` |
| 모델 로딩 | `humanoidverse/agents/load_utils.py` |
| ONNX 내보내기 | `humanoidverse/utils/helpers.py:export_meta_policy_as_onnx` |
| 환경 래퍼 | `humanoidverse/agents/envs/humanoidverse_isaac.py` |

