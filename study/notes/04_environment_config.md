# BFM-Zero 환경 설정 시스템 분석

## 개요

BFM-Zero는 **Hydra** 기반의 계층적 설정 시스템을 사용합니다. 복잡한 로봇 시뮬레이션의 모든 구성 요소를 YAML 파일로 관리합니다.

---

## 1. 설정 파일 구조

```
humanoidverse/config/
├── base.yaml                    # 글로벌 설정
├── base_eval.yaml               # 평가 모드 오버라이드
├── base/
│   ├── hydra.yaml              # Hydra 프레임워크 설정
│   ├── structure.yaml          # 필수 구성 요소 선언
│   └── fabric.yaml             # PyTorch Lightning 설정
├── env/                         # 환경 설정
│   ├── base_task.yaml
│   ├── legged_base.yaml
│   └── legged_motions.yaml
├── simulator/                   # 시뮬레이터 설정
│   ├── isaacsim.yaml
│   └── mujoco.yaml
├── robot/                       # 로봇 설정
│   ├── robot_base.yaml
│   └── g1/
│       ├── g1_29dof.yaml
│       └── g1_29dof_new_effort_limit.yaml
├── rewards/                     # 보상 설정
│   └── reward_bfm_zero.yaml
├── obs/                         # 관찰 설정
│   └── bfm_zero_obs.yaml
├── terrain/                     # 지형 설정
│   └── terrain_locomotion_plane.yaml
├── domain_rand/                 # 도메인 무작위화
│   └── domain_rand.yaml
├── callbacks/                   # 콜백 설정
└── exp/                         # 실험 설정
    └── bfm_zero/
        └── bfm_zero.yaml
```

---

## 2. Hydra 설정 체인

### 실험 설정 예시 (bfm_zero.yaml)

```yaml
defaults:
  - /env: legged_motions
  - /simulator: isaacsim        # 또는 mujoco
  - /domain_rand: domain_rand
  - /rewards: reward_bfm_zero
  - /robot: g1/g1_29dof_new
  - /terrain: terrain_locomotion_plane
  - /obs: bfm_zero_obs
  - /callbacks: im_eval

num_envs: 50
project_name: BFMZero
```

### 실행 시 오버라이드

```bash
# MuJoCo로 전환
python train.py +exp=bfm_zero simulator=mujoco

# 환경 수 변경
python train.py +exp=bfm_zero num_envs=1024
```

---

## 3. 환경 클래스 계층

```
gym.Env
  └─ BaseTask
       └─ LeggedRobotBase
            └─ LeggedRobotMotions
```

### 3.1 BaseTask

**파일**: `humanoidverse/envs/base_task/base_task.py`

```python
class BaseTask(gym.Env):
    def __init__(self, config, device):
        # 시뮬레이터 인스턴스화
        self.simulator = SimulatorClass(config, device)

        # 관찰/액션 공간 정의
        self.observation_space = ...
        self.action_space = ...

    def step(self, actions):
        # 시뮬레이션 스텝 실행
        pass

    def reset(self):
        # 환경 초기화
        pass
```

### 3.2 LeggedRobotBase

**파일**: `humanoidverse/envs/legged_base_task/legged_robot_base.py`

**주요 설정** (legged_base.yaml):

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| env_spacing | 5.0 | 환경 간 거리 (m) |
| max_episode_length_s | 20 | 에피소드 최대 길이 (초) |
| clip_actions | 100.0 | 액션 클리핑 |
| clip_observations | 100.0 | 관찰 클리핑 |

### 3.3 LeggedRobotMotions

**파일**: `humanoidverse/envs/legged_robot_motions/legged_robot_motions.py`

**특징**:
- 모션 라이브러리 (MotionLibRobot) 통합
- 전문가 궤적 추적
- 상반신/하반신 분리 제어

---

## 4. 시뮬레이터 설정

### 시뮬레이터 계층

```
BaseSimulator (추상 클래스)
  ├─ IsaacSim    # NVIDIA GPU 병렬 시뮬레이션
  ├─ IsaacGym   # 레거시 GPU 시뮬레이션
  ├─ MuJoCo     # CPU 기반, 디버깅 용이
  └─ Genesis    # 새로운 시뮬레이터
```

### 4.1 Isaac Sim

**파일**: `humanoidverse/simulator/isaacsim/isaacsim.py`

```yaml
# isaacsim.yaml
sim:
  fps: 200                    # 200Hz (dt = 0.005s)
  control_decimation: 4       # 제어 주파수 = 200/4 = 50Hz
  substeps: 1
  physx:
    num_threads: 10
    solver_type: 1            # TGS (더 안정적)
    num_position_iterations: 4
    num_velocity_iterations: 0
    contact_offset: 0.01
    bounce_threshold_velocity: 0.5
plane:
  static_friction: 1.0
  dynamic_friction: 1.0
  restitution: 0.0
render_mode: "human"          # 또는 None (headless)
render_interval: 4
```

### 4.2 MuJoCo

**파일**: `humanoidverse/simulator/mujoco/mujoco.py`

```yaml
# mujoco.yaml
sim:
  fps: 200
  control_decimation: 4
  substeps: 1
  render_mode: "human"
  render_interval: 4
```

**MuJoCo 장점**:
- CPU 기반으로 가벼움
- 디버깅 용이
- macOS 지원

---

## 5. 로봇 설정 (G1)

### 5.1 관절 구조 (29 DOF)

```
다리 (12 DOF):
  ├─ 좌측 다리 (6)
  │   └─ hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
  └─ 우측 다리 (6)
      └─ hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll

허리 (3 DOF):
  └─ waist_yaw, waist_roll, waist_pitch

팔 (14 DOF):
  ├─ 좌측 팔 (7)
  │   └─ shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
  └─ 우측 팔 (7)
      └─ shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
```

### 5.2 제어 파라미터

```yaml
# g1_29dof.yaml
control_type: P               # 비례 제어

stiffness:                    # P 게인
  hip_yaw: 40.17924
  hip_roll: 99.09843
  hip_pitch: 99.09843
  knee: 99.09843
  ankle: 28.50125
  shoulder: 14.25062
  elbow: 14.25062
  wrist: 14.25-16.78

damping:                      # D 게인
  # 대략 stiffness의 1/16

action_scale: 0.25
action_clip_value: 5.0
normalize_action: True
```

### 5.3 자산 파일

```yaml
# URDF (범용)
urdf_file: "g1/g1_29dof_fakehand.urdf"

# USD (Isaac Sim용)
usd_file: "g1/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"

# MJCF (MuJoCo용)
scene_29dof_freebase_mujoco.xml
```

---

## 6. 관찰 설정

### bfm_zero_obs.yaml

```yaml
use_obs_filter: True
root_height_obs: True

# Actor 관찰 (정책 입력)
actor_obs:
  - base_ang_vel          # 기체 각속도
  - projected_gravity     # 로컬 좌표계 중력
  - dof_pos               # 관절 위치
  - dof_vel               # 관절 속도
  - actions               # 이전 액션
  - history_actor         # 과거 4 스텝 이력
  - max_local_self        # 몸체 국소 정보

# 관찰 스케일
obs_scales:
  base_ang_vel: 0.25
  dof_pos: 1.0
  dof_vel: 1.0
  actions: 1.0
  projected_gravity: 1.0

# 노이즈 스케일 (시뮬-투-리얼)
noise_scales:
  base_ang_vel: 0.2
  projected_gravity: 0.05
  dof_pos: 0.01
  dof_vel: 0.5
```

---

## 7. 보상 설정

### reward_bfm_zero.yaml

```yaml
# 페널티 (음수 보상)
reward_scales:
  penalty_torques: -0.000001      # 토크 사용 최소화
  penalty_undesired_contact: -1   # 원치 않는 접촉
  penalty_action_rate: -0.5       # 액션 변화율
  penalty_ankle_roll: -0.5        # 발목 롤 제한
  penalty_feet_ori: -0.1          # 발 방향
  feet_heading_alignment: -0.1    # 발 방향 일관성
  penalty_slippage: -1.0          # 미끄러짐 방지

  # 한계 페널티
  limits_dof_pos: -10.0
  limits_dof_vel: -5.0
  limits_torque: -5.0

# 접촉 페널티 부위
penalize_contacts_on:
  - pelvis
  - shoulder
  - hip
```

### 커리큘럼 학습

```yaml
reward_penalty_curriculum: False  # 활성화 가능

# 활성화 시
reward_initial_penalty_scale: 0.10
reward_penalty_degree: 0.000003

# 에피소드 길이 기반 조정
level_up_threshold: 42   # 길이 > 42: 난이도 증가
level_down_threshold: 40 # 길이 < 40: 난이도 감소
```

---

## 8. 도메인 무작위화

### domain_rand.yaml

```yaml
# 로봇 푸시
push_robots: True
push_interval_s: [1, 3]        # 1~3초마다 푸시
max_push_vel_xy: 0.5
max_push_ang_vel: 0.5

# 질량 무작위화
randomize_link_mass: True
link_mass_range: [0.95, 1.05]  # ±5%

# 무게중심 무작위화
randomize_base_com: True
base_com_range:
  x: [-0.02, 0.02]
  y: [-0.02, 0.02]
  z: [-0.02, 0.02]

# 마찰 무작위화
randomize_friction: True
friction_range: [0.5, 1.25]

# 기타
randomize_pd_gain: False
randomize_default_dof_pos: True
default_dof_pos_noise_range: [-0.02, 0.02]
randomize_ctrl_delay: False
```

---

## 9. 종료 조건

### 기본 환경

```yaml
termination:
  terminate_when_close_to_dof_pos_limit: False
  terminate_when_close_to_dof_vel_limit: False
  terminate_when_close_to_torque_limit: False
```

### 모션 추적 환경

```yaml
termination:
  terminate_by_contact: False
  terminate_by_gravity: False
  terminate_by_low_height: False
  terminate_when_motion_end: False
  terminate_when_motion_far: False

termination_scales:
  termination_min_base_height: 0.2
  termination_gravity_x: 0.8
  termination_gravity_y: 0.8
  termination_motion_far_threshold: 1.5
```

---

## 10. 설정 흐름 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│  실행 명령                                                   │
│  python train.py +exp=bfm_zero simulator=mujoco             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Hydra 설정 로드                                            │
│  1. base.yaml (글로벌)                                      │
│  2. exp/bfm_zero.yaml (실험)                                │
│  3. 각 defaults 체인 로드                                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  설정 병합                                                   │
│  simulator: mujoco.yaml                                     │
│  env: legged_motions.yaml                                   │
│  robot: g1_29dof.yaml                                       │
│  rewards: reward_bfm_zero.yaml                              │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Workspace 초기화                                           │
│  1. Fabric 설정 (GPU/CPU)                                   │
│  2. 시뮬레이터 인스턴스화 (MuJoCo)                           │
│  3. 환경 생성 (LeggedRobotMotions)                          │
│  4. 에이전트 생성 (FBcprAuxAgent)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  학습 루프                                                   │
│  while steps < num_env_steps:                               │
│      action = agent.act(obs, z)                             │
│      obs, reward = env.step(action)                         │
│      agent.update(batch)                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. 코드 참조

| 기능 | 파일 |
|------|------|
| 기본 설정 | `config/base.yaml` |
| 환경 기본 클래스 | `envs/base_task/base_task.py` |
| 다리 로봇 기본 | `envs/legged_base_task/legged_robot_base.py` |
| 모션 추적 환경 | `envs/legged_robot_motions/legged_robot_motions.py` |
| 시뮬레이터 기본 | `simulator/base_simulator/base_simulator.py` |
| Isaac Sim | `simulator/isaacsim/isaacsim.py` |
| MuJoCo | `simulator/mujoco/mujoco.py` |
| G1 로봇 설정 | `config/robot/g1/g1_29dof.yaml` |
| 보상 설정 | `config/rewards/reward_bfm_zero.yaml` |

---

## 12. 주요 인사이트

1. **모듈식 설계**: 각 구성 요소가 독립적으로 설정 가능
2. **시뮬레이터 교체**: 코드 수정 없이 YAML 한 줄로 전환
3. **커리큘럼 학습**: 보상과 한계를 점진적으로 조정
4. **도메인 무작위화**: Sim-to-Real 전이를 위한 다양한 변형
5. **재현성**: seed, timestamp로 실험 추적

