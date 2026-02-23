# train.py 상세 분석

## 개요

`humanoidverse/train.py`는 BFM-Zero의 메인 학습 진입점입니다. FBcpr(Forward-Backward with Contrastive Predictive Representation) 알고리즘을 사용하여 휴머노이드 로봇 제어 정책을 학습합니다.

## 핵심 클래스 구조

```
TrainConfig (Pydantic)
    ├── agent: FBcprAuxAgentConfig
    │       ├── model: FBcprAuxModelConfig
    │       └── train: FBcprAuxAgentTrainConfig
    ├── env: HumanoidVerseIsaacConfig
    └── evaluations: [HumanoidVerseIsaacTrackingEvaluationConfig]
            │
            ▼
        Workspace
            ├── train_env (병렬 환경)
            ├── agent (FBcprAuxAgent)
            ├── replay_buffer (DictBuffer / TrajectoryDictBuffer)
            └── evaluations (평가 모듈)
```

---

## 1. TrainConfig 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `seed` | 0 | 랜덤 시드 |
| `online_parallel_envs` | 50 | 병렬 환경 수 (실제: 1024) |
| `num_env_steps` | 30M | 총 학습 스텝 (실제: 384M) |
| `num_seed_steps` | 50K | 랜덤 액션으로 초기 데이터 수집 |
| `update_agent_every` | 500 | 에이전트 업데이트 주기 |
| `num_agent_updates` | 50 | 업데이트 당 gradient step 수 |
| `buffer_size` | 5M | 리플레이 버퍼 크기 |
| `checkpoint_every_steps` | 5M | 체크포인트 저장 주기 |
| `eval_every_steps` | 1M | 평가 주기 |

### train_bfm_zero() 실제 설정값 (line 587-721)

```python
online_parallel_envs = 1024      # 1024개 병렬 환경
num_env_steps = 384_000_000      # 3.84억 스텝
num_seed_steps = 10_240          # 10K 시드 스텝
update_agent_every = 1024        # 1024 스텝마다 업데이트
num_agent_updates = 16           # 16번 gradient update
buffer_size = 5_120_000          # 512만 버퍼
checkpoint_every_steps = 9_600_000  # 960만 스텝마다 체크포인트
eval_every_steps = 9_600_000     # 960만 스텝마다 평가
```

---

## 2. Workspace 클래스 (line 188-585)

### 2.1 초기화 (`__init__`)

```python
def __init__(self, cfg: TrainConfig):
    # 1. 환경 생성
    self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)

    # 2. 관측/행동 공간 설정
    self.obs_space = self.train_env.single_observation_space
    self.action_space = self.train_env.single_action_space
    del self.obs_space.spaces["time"]  # 시간 정보 제거

    # 3. 에이전트 생성 또는 체크포인트 로드
    self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(...)

    # 4. 평가 모듈 빌드
    self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in cfg.evaluations}
```

### 2.2 학습 루프 (`train_online`, line 261-531)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 데이터 로드                                                   │
│     ├── 전문가 버퍼: load_expert_trajectories_from_motion_lib()  │
│     └── 학습 버퍼: DictBuffer 또는 TrajectoryDictBufferMultiDim  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 메인 루프 (t = 0 → num_env_steps)                           │
│                                                                  │
│  for t in range(0, num_env_steps, online_parallel_envs):        │
│      │                                                           │
│      ├── 체크포인트 저장 (checkpoint_every_steps마다)             │
│      │                                                           │
│      ├── 평가 실행 (eval_every_steps마다)                        │
│      │   └── prioritization 업데이트 (EMD 기반 샘플링 가중치)     │
│      │                                                           │
│      ├── 액션 선택                                               │
│      │   ├── t < num_seed_steps: 랜덤 액션                      │
│      │   └── t >= num_seed_steps: agent.act(obs, z)             │
│      │                                                           │
│      ├── 환경 스텝: env.step(action)                            │
│      │                                                           │
│      ├── 버퍼에 데이터 추가: replay_buffer["train"].extend(data) │
│      │                                                           │
│      └── 에이전트 업데이트 (update_agent_every마다)              │
│          └── num_agent_updates회 반복: agent.update(replay_buffer)│
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Z 컨텍스트 업데이트 (line 411)

```python
context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
```

- `update_z_every_step` (기본 100) 스텝마다 새로운 z 샘플링
- `use_mix_rollout=True`: z_buffer에서 학습된 z 재사용
- `rollout_expert_trajectories=True`: 일부 환경에서 전문가 궤적 z 사용

---

## 3. FBcpr 에이전트 아키텍처

### 3.1 클래스 계층 구조

```
FBAgent (fb/agent.py)
    │
    └── FBcprAgent (fb_cpr/agent.py)
            │
            └── FBcprAuxAgent (fb_cpr_aux/agent.py)
```

### 3.2 모델 구성요소

```
FBcprAuxModel
    ├── _backward_map    : obs → z (256차원 잠재 벡터)
    ├── _forward_map     : (obs, z, action) → next_obs 예측
    ├── _actor           : (obs, z) → action (정책)
    ├── _critic          : (obs, z, action) → Q-value (판별자 보상)
    ├── _discriminator   : (obs, z) → expert vs policy 판별
    ├── _aux_critic      : (obs, z, action) → Q-value (보조 보상)
    └── _obs_normalizer  : 관측값 정규화
```

### 3.3 Target Networks

```python
_target_forward_map   # soft update: τ = 0.01
_target_backward_map  # soft update: τ = 0.01
_target_critic        # soft update: τ = 0.005
_target_aux_critic    # soft update: τ = 0.005
```

---

## 4. 학습 업데이트 흐름 (FBcprAuxAgent.update)

```python
def update(self, replay_buffer, step: int):
    # 1. 배치 샘플링
    expert_batch = replay_buffer["expert_slicer"].sample(batch_size)  # 전문가 데이터
    train_batch = replay_buffer["train"].sample(batch_size)           # 온라인 데이터

    # 2. 관측값 정규화
    train_obs = self._model._obs_normalizer(train_obs)
    expert_obs = self._model._obs_normalizer(expert_obs)

    # 3. 전문가 z 인코딩
    expert_z = self.encode_expert(next_obs=expert_next_obs)

    # 4. 판별자 업데이트 (GAN 스타일)
    metrics = self.update_discriminator(expert_obs, expert_z, train_obs, train_z)

    # 5. z 샘플링 (혼합 분포)
    z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z)

    # 6. z 리라벨링 (relabel_ratio=0.8)
    mask = torch.rand(...) <= relabel_ratio
    train_z = torch.where(mask, z, train_z)

    # 7. Forward-Backward 업데이트
    metrics.update(self.update_fb(obs, action, discount, next_obs, goal, z))

    # 8. Critic 업데이트 (판별자 보상)
    metrics.update(self.update_critic(obs, action, discount, next_obs, z))

    # 9. Auxiliary Critic 업데이트 (보조 보상)
    metrics.update(self.update_aux_critic(obs, action, discount, aux_reward, next_obs, z))

    # 10. Actor 업데이트
    metrics.update(self.update_actor(obs, action, z))

    # 11. Target 네트워크 soft update
    _soft_update_params(forward_map, target_forward_map, τ=0.01)
    _soft_update_params(backward_map, target_backward_map, τ=0.01)
    _soft_update_params(critic, target_critic, τ=0.005)
    _soft_update_params(aux_critic, target_aux_critic, τ=0.005)
```

---

## 5. Z 분포 샘플링 (sample_mixed_z)

세 가지 소스에서 z를 혼합하여 샘플링:

```python
p_goal = train_goal_ratio           # 0.2: 목표 인코딩
p_expert_asm = expert_asm_ratio     # 0.6: 전문가 궤적 인코딩
p_uniform = 1 - p_goal - p_expert   # 0.2: 균등 분포

# 확률적 선택
mix_idxs = torch.multinomial([p_goal, p_expert_asm, p_uniform], batch_size)

z = torch.where(mix_idxs == 0, goal_z, z)       # 목표 인코딩
z = torch.where(mix_idxs == 1, expert_z, z)     # 전문가 인코딩
# mix_idxs == 2: 균등 분포 (기본값)
```

---

## 6. 손실 함수들

### 6.1 Forward-Backward Loss (update_fb)

```python
# M = F · B^T (Successor Feature Matrix)
Fs = forward_map(obs, z, action)      # num_parallel x batch x z_dim
B = backward_map(goal)                 # batch x z_dim
Ms = torch.matmul(Fs, B.T)            # num_parallel x batch x batch

# FB Loss = Off-diagonal + Diagonal
fb_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag_sum
fb_diag = -torch.diagonal(diff).mean()
fb_loss = fb_offdiag + fb_diag

# Orthonormality Loss (B 정규화)
Cov = torch.matmul(B, B.T)
orth_loss = orth_loss_offdiag + orth_loss_diag
fb_loss += ortho_coef * orth_loss  # ortho_coef = 100
```

### 6.2 Discriminator Loss (update_discriminator)

```python
# Binary Cross Entropy (GAN)
expert_logits = discriminator.compute_logits(expert_obs, expert_z)
unlabeled_logits = discriminator.compute_logits(train_obs, train_z)

expert_loss = -F.logsigmoid(expert_logits)      # 전문가: 1에 가깝게
unlabeled_loss = F.softplus(unlabeled_logits)   # 정책: 0에 가깝게

loss = mean(expert_loss + unlabeled_loss)

# WGAN Gradient Penalty
gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
loss += grad_penalty * wgan_gp  # grad_penalty = 10
```

### 6.3 Critic Loss (update_critic)

```python
# TD Learning with Discriminator Reward
reward = discriminator.compute_reward(obs, z)
next_Qs = target_critic(next_obs, z, next_action)
target_Q = reward + discount * next_V

Qs = critic(obs, z, action)
critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, target_Q)
```

### 6.4 Actor Loss (update_actor in FBcprAuxAgent)

```python
# 세 가지 Q-value의 가중 합
Q_fb = (forward_map(obs, z, action) * z).sum(-1)          # FB 보상
Q_discriminator = critic(obs, z, action)                   # 판별자 보상
Q_aux = aux_critic(obs, z, action)                         # 보조 보상

actor_loss = (
    -Q_discriminator.mean() * reg_coeff * weight      # reg_coeff = 0.05
    - Q_aux.mean() * reg_coeff_aux * weight           # reg_coeff_aux = 0.02
    - Q_fb.mean()                                      # 기본 FB 보상
)
```

---

## 7. 보조 보상 (Auxiliary Rewards)

`FBcprAuxAgent`는 추가적인 보조 보상을 사용하여 안전하고 자연스러운 동작을 유도합니다:

```python
aux_rewards = [
    'penalty_torques',           # 토크 페널티 (0.0)
    'penalty_action_rate',       # 액션 변화율 페널티 (-0.1)
    'limits_dof_pos',            # 관절 한계 페널티 (-10.0)
    'limits_torque',             # 토크 한계 페널티 (0.0)
    'penalty_undesired_contact', # 원치않는 접촉 페널티 (-1.0)
    'penalty_feet_ori',          # 발 방향 페널티 (-0.4)
    'penalty_ankle_roll',        # 발목 롤 페널티 (-4.0)
    'penalty_slippage',          # 미끄러짐 페널티 (-2.0)
]
```

---

## 8. Prioritized Sampling

EMD(Earth Mover's Distance) 기반 우선순위 샘플링:

```python
# 평가 후 각 모션의 EMD 계산
priorities = torch.clamp(emd_values, min=0.5, max=2.0) * 2.0

# 우선순위 모드 (prioritization_mode='exp')
priorities = 2 ** priorities

# 모션 라이브러리 가중치 업데이트
motion_lib.update_sampling_weight_by_id(priorities, motion_ids)

# 전문가 버퍼 우선순위 업데이트
expert_buffer.update_priorities(priorities, idxs)
```

---

## 9. 코드 참조

| 기능 | 파일:라인 |
|------|-----------|
| TrainConfig 정의 | `train.py:71-127` |
| train_bfm_zero() 설정 | `train.py:587-721` |
| Workspace.__init__ | `train.py:189-256` |
| train_online() 메인 루프 | `train.py:261-531` |
| FBAgent.update_fb | `fb/agent.py:216-297` |
| FBcprAgent.update_discriminator | `fb_cpr/agent.py:333-365` |
| FBcprAgent.update_critic | `fb_cpr/agent.py:367-405` |
| FBcprAuxAgent.update_actor | `fb_cpr_aux/agent.py:253-298` |
| sample_mixed_z | `fb_cpr/agent.py:127-150` |
| encode_expert | `fb_cpr/agent.py:152-168` |

---

## 10. 학습 실행 명령

```bash
# 기본 학습 (GPU 필요)
uv run python -m humanoidverse.train

# 커스텀 설정
uv run python -m humanoidverse.train \
    --seed 42 \
    --num_env_steps 100000000 \
    --online_parallel_envs 512 \
    --use_wandb
```

---

## 요약

BFM-Zero의 학습은 다음 핵심 요소로 구성됩니다:

1. **Forward-Backward Representation**: 상태-목표 관계를 잠재 공간에서 학습
2. **Contrastive Discriminator**: 전문가 vs 정책 구분으로 암묵적 보상 생성
3. **Mixed Z Distribution**: 목표/전문가/균등 분포 혼합으로 다양한 행동 학습
4. **Auxiliary Rewards**: 안전하고 자연스러운 동작을 위한 추가 페널티
5. **Prioritized Sampling**: EMD 기반 어려운 모션 우선 학습
