# Phase 8: 실습 가이드 (Practical Exercises)

BFM-Zero 코드베이스를 깊이 이해하기 위한 실전 실습 가이드. 각 실습은 목적, 코드 수정 위치, 예상 결과, macOS 실행 명령을 포함한다.

**사전 준비:**
```bash
git lfs install && git lfs pull   # 모션 데이터 및 모델 다운로드
uv sync                           # 의존성 설치
```

---

## 1. 텐서 Shape 추적 실습

### 1.1 목적

핵심 텐서(obs, z, action, F(s,a), B(g), M(s,g))가 학습 파이프라인에서 어떤 shape으로 흘러가는지 직접 확인한다. Forward-Backward 표현 학습의 데이터 흐름을 체감하는 것이 목표이다.

### 1.2 실습: 학습 루프에서 텐서 shape 출력

`humanoidverse/agents/fb_cpr/agent.py`의 `update()` 메서드(line 170) 시작부에 print문을 추가한다.

```python
# agent.py update() 메서드, train_batch 샘플링 직후 (line 172 부근)
def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
    expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
    train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

    # === 실습용 shape 출력 ===
    if step % 100000 == 0:
        print(f"\n=== Tensor Shape Debug (step={step}) ===")
        for k, v in train_batch["observation"].items():
            print(f"  train_obs[{k}]: {v.shape}")
        print(f"  train_action: {train_batch['action'].shape}")

    # z 샘플링 직후 (line 212 부근), 네트워크 출력도 확인:
    if step % 100000 == 0:
        with torch.no_grad():
            Fs = self._model._forward_map(train_obs, z, train_action)
            B = self._model._backward_map(train_next_obs)
            print(f"  z: {z.shape}, F(s,a): {Fs.shape}, B(g): {B.shape}")
            print(f"  M(s,g)=F*B^T: {torch.matmul(Fs, B.T).shape}")
            print(f"  Q_critic: {self._model._critic(train_obs, z, train_action).shape}")
            print(f"  D_logits: {self._model._discriminator.compute_logits(train_obs, z).shape}")
```

### 1.3 예상 결과

기본 설정(batch_size=1024, z_dim=256, num_parallel=2)에서:

| 텐서 | Shape | 설명 |
|------|-------|------|
| `train_obs["state"]` | `(1024, 64)` | dof_pos(29)+dof_vel(29)+gravity(3)+ang_vel(3) |
| `train_obs["privileged_state"]` | `(1024, 357)` | 전체 강체 관측 |
| `z` | `(1024, 256)` | 행동 의도 벡터 |
| `F(s,a)` | `(2, 1024, 256)` | 앙상블 2개의 Forward 출력 |
| `B(g)` | `(1024, 256)` | Backward 출력 |
| `M(s,g)` | `(2, 1024, 1024)` | Successor Measure 행렬 |
| `Q_critic` | `(2, 1024, 1)` | 앙상블 Q-value |
| `D_logits` | `(1024, 1)` | Discriminator 로짓 |

### 1.4 macOS 실행: CPU 더미 데이터로 shape 확인

```python
# study/scripts/debug_tensor_shapes.py
import torch, gymnasium
from humanoidverse.agents.fb_cpr.model import FBcprModel, FBcprModelConfig

obs_space = gymnasium.spaces.Dict({
    "state": gymnasium.spaces.Box(low=-10, high=10, shape=(64,)),
    "privileged_state": gymnasium.spaces.Box(low=-10, high=10, shape=(357,)),
    "last_action": gymnasium.spaces.Box(low=-1, high=1, shape=(29,)),
})
cfg = FBcprModelConfig(device="cpu")
model = cfg.build(obs_space, action_dim=29)

batch = 16
obs = {k: torch.randn(batch, v.shape[0]) for k, v in obs_space.spaces.items()}
z = model.sample_z(batch)
action = torch.randn(batch, 29)

print(f"z: {z.shape}, ||z||: {z.norm(dim=-1).mean():.2f}")
print(f"B(obs): {model.backward_map(obs).shape}")
print(f"F(obs,z,a): {model.forward_map(obs, z, action).shape}")
```

```bash
uv run python study/scripts/debug_tensor_shapes.py
```

---

## 2. 학습 과정 시각화 실습

### 2.1 목적

학습 메트릭을 분석하여 모델이 정상적으로 학습되고 있는지 판단하는 능력을 기른다.

### 2.2 주요 메트릭 해석

| 메트릭 | 의미 | 정상 범위 |
|--------|------|-----------|
| `fb_loss` | FB 표현 학습 손실 | 감소 후 안정화 |
| `ortho_loss` | B 벡터 직교성 손실 | 감소 후 안정화 |
| `disc_loss` | Discriminator 손실 | 0.5~2.0 부근 안정 |
| `critic_loss` | Critic TD 손실 | 감소 후 안정화 |
| `actor_loss` | Actor 손실 | 점진적 감소 |
| `mean_disc_reward` | Discriminator 보상 평균 | 음수에서 양수로 전환 |
| `eval/emd` | Earth Mover's Distance | < 0.75 (목표) |

### 2.3 실습: CSV 로그 시각화

```python
# study/scripts/visualize_training.py
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import tyro

def main(log_dir: str, save_png: bool = True):
    log_dir = Path(log_dir)
    df = pd.read_csv(log_dir / "train_log.txt")
    print(f"Loaded {len(df)} entries. Columns: {list(df.columns)}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("BFM-Zero Training Metrics", fontsize=14)
    metrics = [("fb_loss","FB Loss"), ("ortho_loss","Ortho Loss"), ("disc_loss","Disc Loss"),
               ("critic_loss","Critic Loss"), ("actor_loss","Actor Loss"), ("mean_disc_reward","Disc Reward")]

    for ax, (col, title) in zip(axes.flat, metrics):
        if col in df.columns:
            ax.plot(df.get("timestep", range(len(df))), df[col], linewidth=0.8)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_png:
        plt.savefig(log_dir / "training_curves.png", dpi=150)
        print(f"Saved to {log_dir / 'training_curves.png'}")

if __name__ == "__main__":
    tyro.cli(main)
```

```bash
uv run python study/scripts/visualize_training.py --log_dir results/bfmzero-isaac
```

### 2.4 학습 상태 판단 기준

**정상 학습**: fb_loss 감소 후 안정 / ortho_loss 꾸준히 감소 / mean_disc_reward가 양수로 전환 / eval/emd가 0.75 이하로 감소

**학습 실패**: disc_loss가 0 수렴(Discriminator 과강) 또는 발산 / fb_loss NaN / eval/emd 1.5 이상 정체

---

## 3. 모션 트래킹 커스터마이징

### 3.1 목적

특정 모션만 선택적으로 트래킹하고, z 벡터 특성을 분석하는 방법을 익힌다.

### 3.2 사용 가능한 모션 데이터

`lafan_29dof.pkl`에는 총 40개의 모션이 정수 인덱스(0~39)로 접근 가능하다.

| 카테고리 | 인덱스 범위 (대략) | 모션 수 | 설명 |
|----------|-------------------|---------|------|
| **dance** | 0~7 | 8 | dance1~2, subject1~5 |
| **fallAndGetUp** | 8~13 | 6 | 넘어진 후 일어나기 |
| **fight** | 14~18 | 5 | 격투 및 스포츠 동작 |
| **jumps** | 19~21 | 3 | 점프 동작 |
| **run** | 22~25 | 4 | 달리기 |
| **sprint** | 26~27 | 2 | 전력 질주 |
| **walk** | 28~39 | 12 | walk1~4, 다양한 보행 패턴 |

```bash
# 전체 모션 ID 및 이름 확인
uv run python -c "
import joblib
data = joblib.load('humanoidverse/data/lafan_29dof.pkl')
for i, k in enumerate(data.keys()): print(f'  [{i:2d}] {k}')
"
```

### 3.3 실습 A: 특정 모션 ID 트래킹

```bash
# 모션 ID 25번 트래킹 (기본값)
uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --motion_list 25

# 다른 모션으로 변경
uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --motion_list 0   # 댄스
uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --motion_list 22  # 달리기
uv run python study/scripts/tracking_inference_macos.py --model_folder model/ --motion_list 19  # 점프

# 여러 모션 순차 트래킹 (headless 모드)
uv run python study/scripts/tracking_inference_macos.py \
    --model_folder model/ --motion_list 0 5 10 25 --headless
```

**GUI 모드의 화면 구성**: 실행 시 좌우 분할 화면이 표시된다.
- **왼쪽 패널**: 참조 모션 (Reference Motion) — 모션 라이브러리의 원본 동작
- **오른쪽 패널**: 정책 출력 (Policy Output) — 학습된 모델이 생성하는 실제 로봇 동작
- 각 패널에 프레임/스텝 카운터가 오버레이됨
- `q` 또는 `ESC`로 종료

### 3.4 실습 B: z 벡터 분석

트래킹 추론 후 생성된 z 벡터의 코사인 유사도를 비교한다. `model/tracking_inference/` 디렉토리의 `zs_*.pkl` 파일을 로드하여, 같은 동작 유형(걷기 vs 걷기)은 코사인 유사도 0.3~0.7, 다른 유형(걷기 vs 점프)은 -0.2~0.2 범위인지 확인한다.

```python
# study/scripts/analyze_z_vectors.py - 핵심 로직
import joblib, numpy as np
from pathlib import Path

z_files = list(Path("model/tracking_inference").glob("zs_*.pkl"))
z_dict = {f.stem.split("_")[1]: joblib.load(f) for f in z_files}

for mid, z in z_dict.items():
    print(f"Motion {mid}: shape={z.shape}, ||z||={np.linalg.norm(z, axis=-1).mean():.2f}")

# 코사인 유사도: 같은 동작 유형은 높고, 다른 유형은 낮을 것
keys = list(z_dict.keys())
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        cos = np.dot(z_dict[keys[i]][0], z_dict[keys[j]][0]) / (
            np.linalg.norm(z_dict[keys[i]][0]) * np.linalg.norm(z_dict[keys[j]][0]))
        print(f"  Motion {keys[i]} vs {keys[j]}: cos_sim = {cos:.4f}")
```

### 3.5 실습 C: 모션 우선순위 커리큘럼 변경

`train.py`의 `train_bfm_zero()` 함수(line 702-706)에서 prioritization 파라미터를 조정한다.

```python
# 기본값:
prioritization_scale=2.0, prioritization_mode='exp'  # 2^(EMD*scale) 가중치

# 더 공격적인 커리큘럼 (어려운 모션에 더 집중):
prioritization_scale=3.0, prioritization_max_val=3.0

# 균등 분배 모드:
prioritization_mode='bin'  # 같은 EMD 범위의 모션에 동일 확률
```

### 3.6 예상 결과

- z 벡터의 노름: `sqrt(256) = 16.0` (norm_z=True일 때 정확히 이 값)
- 시간에 따른 z 변화: 연속 프레임에서 z가 부드럽게 변화함을 확인

---

## 4. Reward 함수 커스터마이징 실습

### 4.1 목적

커스텀 reward 함수를 작성하여 z 벡터가 보상에 따라 어떻게 달라지는지 체험한다.

### 4.2 실습 A: 커스텀 보상 함수 작성

`humanoidverse/envs/g1_env_helper/rewards.py` 파일 끝에 추가한다.

```python
@dataclasses.dataclass
class StandStillReward(RewardFunction):
    """제자리에 서 있기: 골반 높이 + 직립 자세 + 수평 이동 억제"""
    target_height: float = 0.75

    def compute(self, model, data) -> float:
        pelvis_height = get_xpos(model, data, "pelvis")[-1]
        height_reward = rewards.tolerance(pelvis_height,
            bounds=(self.target_height, float("inf")), margin=self.target_height/2, sigmoid="linear")
        upvector = get_sensor_data(model, data, "upvector_torso")
        upright = rewards.tolerance(np.dot(upvector, np.array([0.073, 0, 1.0])),
            bounds=(0.9, float("inf")), margin=0.5, sigmoid="linear")
        com_vel = get_center_of_mass_linvel(model, data)
        dont_move = rewards.tolerance(np.linalg.norm(com_vel[:2]), bounds=(0, 0.1), margin=0.5)
        return float(height_reward * upright * dont_move)

    @staticmethod
    def reward_from_name(name):
        match = re.search(r"^standstill-?(\d*\.?\d*)$", name)
        if match:
            height = float(match.group(1)) if match.group(1) else 0.75
            return StandStillReward(target_height=height)
        return None
```

### 4.3 실습 B: 커스텀 보상으로 z 추론 및 비교

```bash
uv run python -m humanoidverse.reward_inference \
    --model_folder model/ --simulator mujoco --device cpu --headless \
    --tasks "standstill-0.75" "move-ego-0-0" "move-ego-0-0.7"
```

추론 후 `model/reward_inference/reward_locomotion.pkl`에서 z 벡터를 로드하여 비교한다.

```python
# study/scripts/compare_reward_z.py - 핵심 로직
import joblib, numpy as np
z_dict = joblib.load("model/reward_inference/reward_locomotion.pkl")
# 각 task의 평균 z 벡터 간 코사인 유사도를 계산
# "standstill"과 "move-ego-0-0" (둘 다 제자리)은 높은 유사도 예상
# "standstill"과 "move-ego-0-0.7" (서기 vs 전진)은 낮은 유사도 예상
```

---

## 5. Goal-reaching 실습

### 5.1 목적

goal_inference의 동작 원리를 이해하고, 목표 위치에 따른 z 벡터 변화를 실험한다.

### 5.2 실습 A: Goal Frame 분석

```bash
uv run python -c "
import json
with open('humanoidverse/data/robots/g1/goal_frames_lafan29dof.json') as f:
    goals = json.load(f)
for g in goals[:5]:
    print(f'Motion: {g[\"motion_name\"]}, ID: {g[\"motion_id\"]}, Frames: {g[\"frames\"]}')
"
```

### 5.3 실습 B: macOS에서 goal inference 실행

`study/scripts/tracking_inference_macos.py`의 MuJoCo 패치 패턴을 활용하여 goal_inference를 macOS에서 실행하는 스크립트를 작성한다. 핵심 로직은 다음과 같다.

```python
# study/scripts/goal_inference_macos.py 핵심 구조:
# 1. MuJoCo quat_rotate w_last=True 패치 적용 (tracking_inference_macos.py와 동일)
# 2. 모델 로드 및 환경 구성 (simulator=mujoco, disable_domain_randomization=True)
# 3. goal_frames_lafan29dof.json에서 목표 프레임 로드
# 4. 각 목표에 대해 B(goal_obs)로 z 벡터 계산
# 5. z 벡터를 사용하여 롤아웃 실행

# z 계산의 핵심:
with torch.no_grad():
    goal_obs = {k: v[frame_idx][None, ...] for k, v in gobs.items()}
    z = model.goal_inference(goal_obs)  # B(goal_frame)
    # z의 노름은 항상 sqrt(z_dim) = 16.0
```

```bash
uv run python study/scripts/goal_inference_macos.py --model_folder model/
```

### 5.4 실습 C: Goal z 벡터 관계 분석

같은 모션의 다른 프레임에서 온 z들의 코사인 유사도(0.2~0.8)와, 다른 모션 간 z들의 유사도(-0.3~0.3)를 비교한다. 프레임 간격이 클수록 유사도가 낮아지는 경향을 확인한다.

---

## 6. 환경 설정 실험

### 6.1 목적

Domain Randomization, 로봇 물리 파라미터, 지형 설정을 변경하면서 에이전트 동작에 미치는 영향을 관찰한다.

### 6.2 실습 A: Domain Randomization 파라미터 변경

`humanoidverse/config/domain_rand/domain_rand.yaml` 파일의 주요 파라미터:

| 파라미터 | 기본값 | 역할 |
|----------|--------|------|
| `push_robots` | True | 주기적 외란 적용 |
| `randomize_link_mass` | True | 링크 질량 95~105% 랜덤화 |
| `randomize_friction` | True | 바닥 마찰계수 0.5~1.25 랜덤화 |
| `randomize_base_com` | True | 몸통 무게중심 +/-2cm 오프셋 |
| `randomize_default_dof_pos` | True | 기본 관절 자세에 +/-0.02rad 노이즈 |

```bash
# DR 비활성화 vs 활성화 비교
uv run python study/scripts/tracking_inference_macos.py \
    --model_folder model/ --disable_dr --disable_obs_noise --motion_list 25

uv run python study/scripts/tracking_inference_macos.py \
    --model_folder model/ --motion_list 25
```

### 6.3 실습 B: 로봇 물리 파라미터 변경

`train_bfm_zero()`의 `hydra_overrides` 리스트를 수정한다.

```python
# 기본: 'robot=g1/g1_29dof_hard_waist'
# 변형 1: 표준 허리 강성
'robot=g1/g1_29dof'
# 변형 2: 액션 스케일 확대 (기본 0.25 -> 0.5)
'robot.control.action_scale=0.5'
# 변형 3: 누운 상태에서 시작 (30% 확률)
'env.config.lie_down_init=True', 'env.config.lie_down_init_prob=0.3'
```

### 6.4 실습 C: 마찰 계수 실험

MuJoCo의 `model.geom_friction` 값을 직접 변경하면서 트래킹 성능을 비교한다. `env.simulator.model.geom_friction`에 스케일 팩터(0.3~2.0)를 곱한 후 롤아웃을 실행한다.

**예상 결과**: 마찰 0.3(미끄러움)과 2.0(과도한 마찰) 모두 성능 저하, 0.8~1.2에서 가장 안정적.

---

## 7. 디버깅 기법

### 7.1 목적

학습 중 발생하는 일반적인 문제를 진단하고 해결하는 방법을 익힌다.

### 7.2 학습 발산 체크리스트

**증상**: `fb_loss`가 NaN이 되거나 급격히 증가

1. **Learning Rate 확인**: lr_f=3e-4, lr_b=1e-5(B가 작은 이유: 안정성이 전체에 영향), lr_actor=3e-4
2. **Gradient Norm 추가**: `update_fb()` 끝에 `torch.nn.utils.clip_grad_norm_`으로 모니터링
3. **ortho_coef 확인**: 기본값 100.0, 너무 크면 B 학습 불안정

### 7.3 NaN 디버깅

| 원인 | 증상 | 해결법 |
|------|------|--------|
| 학습률 과대 | loss 폭발 후 NaN | lr을 1/10로 줄이기 |
| 관측값 정규화 문제 | 초기 스텝 NaN | `num_seed_steps` 늘리기 |
| Discriminator 불안정 | disc_loss NaN | `grad_penalty_discriminator` 늘리기 |
| z 벡터 스케일 | project_z 후 NaN | norm_z=True 확인 |

```python
# agent.py의 update()에서 NaN 검사 추가:
def check_nan(tensor_dict, prefix=""):
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor) and torch.isnan(v).any():
            print(f"NaN in {prefix}{k}! shape={v.shape}")
            return True
    return False
```

### 7.4 일반적인 에러

| 에러 | 원인 | 해결법 |
|------|------|--------|
| `CUDA out of memory` | 배치/버퍼 과대 | `batch_size=512`, `buffer_device='cpu'` |
| macOS `quat_rotate` 에러 | w_last 미지정 | `quat_rotate(..., w_last=True)` 패치 적용 |
| `Motion data not found` | LFS 미실행 | `git lfs install && git lfs pull` |
| `terminate` AssertionError | 종료 조건 활성 | BFM-Zero는 모든 종료조건 비활성 필요 |

### 7.5 체크포인트 복구

```bash
# 체크포인트 상태 확인
uv run python -c "
import json; from pathlib import Path
ckpt = Path('results/bfmzero-isaac/checkpoint')
if ckpt.exists():
    status = json.load(open(ckpt / 'train_status.json'))
    print(f'Last step: {status[\"time\"]:,}')
"
# train.py의 create_agent_or_load_checkpoint()가 checkpoint 디렉토리 존재 시 자동 로드
```

---

## 8. 고급 실습: 모델 수정

### 8.1 목적

네트워크 구조를 직접 변경하면서 모델 아키텍처가 학습 성능에 미치는 영향을 이해한다.

### 8.2 실습 A: 네트워크 크기 변경

`train_bfm_zero()` 함수(`train.py` line 587)의 아키텍처 설정을 변경한다.

```python
# 기본 (Large):  hidden_dim=2048, hidden_layers=6  -> ~50M params
# Medium:        hidden_dim=1024, hidden_layers=2  -> ~15M params
# Small:         hidden_dim=512,  hidden_layers=2  -> ~5M params
f=ForwardArchiConfig(hidden_dim=1024, hidden_layers=2, ...),
actor=ActorArchiConfig(hidden_dim=1024, hidden_layers=2, ...),
```

### 8.3 실습 B: z_dim 변경

```python
# train_bfm_zero()의 archi 설정에서:
z_dim=256,   # 기본값. norm = sqrt(256) = 16.0
# z_dim=128  # 표현력 감소, 학습 빠름. norm = sqrt(128) = 11.3
# z_dim=512  # 표현력 증가, M 행렬 메모리 사용 증가. norm = sqrt(512) = 22.6
```

변경 시 BackwardMap, ForwardMap, Actor, Critic, Discriminator 모두 입출력 차원이 자동 조정된다.

### 8.4 실습 C: simple vs residual 모델

```python
# simple: Linear -> LayerNorm -> Tanh -> [Linear -> ReLU]* -> Linear -> ReLU
# residual: Block -> [ResidualBlock(x + LayerNorm -> Linear -> Mish)]* -> Block
f=ForwardArchiConfig(model='residual', hidden_dim=2048, hidden_layers=6, ...),  # 기본
f=ForwardArchiConfig(model='simple',   hidden_dim=1024, hidden_layers=2, ...),  # 비교 실험
```

residual 모델은 깊은 네트워크에서 그래디언트 전파가 안정적이며, BFM-Zero 기본 설정이다.

### 8.5 실습 D: Auxiliary Loss 추가 (고급)

기존 aux_rewards (`train.py` line 664):
```python
aux_rewards=['penalty_action_rate', 'limits_dof_pos', 'penalty_undesired_contact',
             'penalty_feet_ori', 'penalty_ankle_roll', 'penalty_slippage', ...],
aux_rewards_scaling={'penalty_ankle_roll': -4.0, 'limits_dof_pos': -10.0, ...},
```

새 페널티 추가 절차:
1. 환경의 보상 파일에 `_reward_<name>()` 메서드 추가
2. `aux_rewards` 리스트에 이름 추가
3. `aux_rewards_scaling`에 스케일 지정

```python
# 예: 관절 속도 제한 위반 페널티
def _reward_limits_dof_vel(self):
    vel_ratio = self.simulator.dof_vel.abs() / self.dof_vel_limits
    return torch.clamp(vel_ratio - 0.95, min=0.0).sum(dim=-1)
# 설정: aux_rewards=[..., 'limits_dof_vel'], aux_rewards_scaling={'limits_dof_vel': -5.0}
```

### 8.6 macOS에서 아키텍처 비교

CPU에서 더미 데이터로 모델 크기별 파라미터 수와 추론 속도를 비교할 수 있다.

```bash
# study/scripts/model_architecture_compare.py에서
# FBcprModelConfig를 다양한 hidden_dim/hidden_layers로 생성하고
# sum(p.numel() for p in model.parameters())로 파라미터 수 확인
# model.act(obs, z, mean=True)로 추론 시간 벤치마크
uv run python study/scripts/model_architecture_compare.py
```

---

## 부록: 실습 파일 목록

| 파일 | 설명 | 모델 필요 |
|------|------|-----------|
| `study/scripts/tracking_inference_macos.py` | macOS 트래킹 추론 (참조모션 비교 뷰어) | O |
| `study/scripts/goal_inference_macos.py` | macOS goal inference (목표 도달 뷰어) | O |
| `study/scripts/friction_experiment.py` | 마찰 계수에 따른 트래킹 성능 비교 | O |
| `study/scripts/analyze_z_vectors.py` | 모션 ID별 z 벡터 코사인 유사도 분석 | O |
| `study/scripts/compare_reward_z.py` | reward_inference 결과의 z 벡터 비교 | X (pkl 필요) |
| `study/scripts/debug_tensor_shapes.py` | FBcprModel 텐서 shape 확인 (CPU 단독) | X |
| `study/scripts/visualize_training.py` | 학습 로그(train_log.txt) 시각화 | X (로그 필요) |
| `study/scripts/model_architecture_compare.py` | 아키텍처별 파라미터 수/추론 속도 비교 (CPU 단독) | X |
| `study/scripts/example_load_model.py` | 모델 로드 기본 예제 | O |
| `study/scripts/mujoco_viewer_test.py` | MuJoCo 뷰어 기능 테스트 | X |
| `study/scripts/fbcpr_algorithm_annotated.py` | FBcpr 알고리즘 주석 코드 (교육용, 실행 불가) | - |
| `study/scripts/train_annotated.py` | 학습 루프 주석 코드 (교육용, 실행 불가) | - |

모든 실행 가능 스크립트는 `uv run python study/scripts/<filename>.py` 명령으로 실행한다.

## 참고 자료

- **학습 진입점**: `humanoidverse/train.py`
- **FBcpr 에이전트**: `humanoidverse/agents/fb_cpr/agent.py`
- **FBcpr 모델**: `humanoidverse/agents/fb_cpr/model.py`
- **신경망 모델**: `humanoidverse/agents/nn_models.py`
- **보상 함수**: `humanoidverse/envs/g1_env_helper/rewards.py`
- **Domain Rand 설정**: `humanoidverse/config/domain_rand/domain_rand.yaml`
- **macOS 트래킹**: `study/scripts/tracking_inference_macos.py`

관련 스터디 노트:
- 신경망 구조: `study/notes/06_neural_network_architecture.md`
- 리플레이 버퍼: `study/notes/08_replay_buffers.md`
- 평가 시스템: `study/notes/09_evaluation_system.md`
- 환경/시뮬레이터: `study/notes/10_env_simulators.md`
- 모션 라이브러리: `study/notes/11_motion_library.md`
- 추론 심화: `study/notes/12_inference_advanced.md`
