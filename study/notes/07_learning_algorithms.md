# 07. FB-CPR 학습 알고리즘 심층 분석

> **관련 노트**: 이 문서는 수학적 원리, 텐서 shape 흐름, 하이퍼파라미터 정리 중심입니다.
> 코드 행 번호 기반 상세 분석은 [`03_fbcpr_algorithm.md`](03_fbcpr_algorithm.md)를 참고하세요.

## 목차

1. [개요: 에이전트 계층 구조](#1-개요-에이전트-계층-구조)
2. [FB 표현 학습의 수학적 원리](#2-fb-표현-학습의-수학적-원리)
3. [Z 샘플링 전략: `sample_z()`와 `sample_mixed_z()`](#3-z-샘플링-전략)
4. [`act()` 메서드: 관측에서 행동까지](#4-act-메서드-관측에서-행동까지)
5. [`update()` 메서드 분석: 전체 학습 루프](#5-update-메서드-분석-전체-학습-루프)
6. [FB Loss 계산 과정](#6-fb-loss-계산-과정)
7. [Orthonormality Loss](#7-orthonormality-loss)
8. [Discriminator 학습과 CPR Loss](#8-discriminator-학습과-cpr-loss)
9. [Critic 학습](#9-critic-학습)
10. [Actor 학습: 다중 목적 최적화](#10-actor-학습-다중-목적-최적화)
11. [Auxiliary Loss (보조 손실)](#11-auxiliary-loss-보조-손실)
12. [커리큘럼 학습: EMD 기반 모션 가중치](#12-커리큘럼-학습-emd-기반-모션-가중치)
13. [텐서 Shape 흐름 추적](#13-텐서-shape-흐름-추적)
14. [학습 하이퍼파라미터와 옵티마이저 설정](#14-학습-하이퍼파라미터와-옵티마이저-설정)
15. [Target Network와 Soft Update](#15-target-network와-soft-update)

---

## 1. 개요: 에이전트 계층 구조

BFM-Zero의 학습 알고리즘은 세 단계의 상속 구조로 설계되어 있다.

```
FBAgent (기본 FB 표현 학습)
  └── FBcprAgent (+ Discriminator, Critic, CPR 정규화)
        └── FBcprAuxAgent (+ Auxiliary Critic, 보조 보상)
```

| 계층 | 소스 파일 | 추가 구성 요소 |
|------|-----------|---------------|
| `FBAgent` | `humanoidverse/agents/fb/agent.py` | Forward map (F), Backward map (B), Actor |
| `FBcprAgent` | `humanoidverse/agents/fb_cpr/agent.py` | Discriminator, Critic |
| `FBcprAuxAgent` | `humanoidverse/agents/fb_cpr_aux/agent.py` | Auxiliary Critic, Auxiliary Reward |

각 계층은 이전 계층의 `update()` 메서드를 확장하며, 새로운 loss 항을 추가하는 방식으로 구성된다.

---

## 2. FB 표현 학습의 수학적 원리

### 2.1 Forward-Backward 표현이란?

FB 표현 학습의 핵심 아이디어는 **상태 점유 측도(state occupancy measure)**를 신경망으로 근사하는 것이다. 정책 pi가 주어졌을 때, 상태 s에서 시작하여 행동 a를 취하고 이후 정책 pi를 따를 때 미래 상태 s'를 방문할 확률 밀도를 **후계 측도(successor measure)** M(s,a,s')라 한다.

FB 표현은 이 M을 두 함수의 내적으로 분해한다:

```
M(s, a, s') ≈ F(s, z, a)^T * B(s')
```

여기서:
- **F (Forward map)**: 상태 s, 잠재 변수 z, 행동 a를 입력으로 받아 z_dim 차원의 벡터를 출력
- **B (Backward map)**: 목표 상태 s'를 입력으로 받아 z_dim 차원의 임베딩을 출력
- **z**: task를 정의하는 잠재 벡터 (z_dim 차원, 기본값 100)

### 2.2 벨만 방정식과 FB Loss

M은 벨만 방정식을 만족한다:

```
M(s, a, s') = delta(s, s') + gamma * E[M(s_next, pi(s_next), s')]
```

여기서 delta는 디랙 델타 함수이다. 이를 FB 분해로 표현하면:

```
F(s, z, a)^T * B(s') = I(s=s') + gamma * F(s_next, z, pi(s_next))^T * B(s')
```

코드에서 이 관계를 행렬 형태로 구현한다 (`fb/agent.py`, 238-245행):

```python
# M = F * B^T, 현재 시점
Fs = self._model._forward_map(obs, z, action)       # [num_parallel, batch, z_dim]
B = self._model._backward_map(goal)                  # [batch, z_dim]
Ms = torch.matmul(Fs, B.T)                           # [num_parallel, batch, batch]

# 목표 M = gamma * M_target, 다음 시점
target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # [num_parallel, batch, z_dim]
target_B = self._model._target_backward_map(goal)                      # [batch, z_dim]
target_Ms = torch.matmul(target_Fs, target_B.T)                        # [num_parallel, batch, batch]

# 벨만 잔차: M(s,a) - gamma * M_target(s',pi(s'))
diff = Ms - discount * target_M  # [num_parallel, batch, batch]
```

### 2.3 Loss 분해: 대각 vs 비대각

diff 행렬에서 대각 원소와 비대각 원소는 다른 의미를 가진다:

- **대각 원소** `diff[i,i]`: 자기 자신에 대한 후계 측도. 벨만 방정식의 `delta(s=s')` 항에 해당하므로 값이 1이 되어야 한다. 따라서 **최대화**해야 한다.
- **비대각 원소** `diff[i,j]` (i != j): 다른 상태에 대한 후계 측도. 벨만 잔차가 0이 되어야 하므로 **최소화**해야 한다.

```python
# 비대각: L2 손실 (0으로 수렴)
fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum

# 대각: 음수 평균 (최대화 → 1로 수렴)
fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]

fb_loss = fb_offdiag + fb_diag
```

`off_diag`는 대각이 0이고 나머지가 1인 마스크로, `setup_training()`에서 미리 계산된다 (122행):

```python
self.off_diag = 1 - torch.eye(batch_size, batch_size, device=self.device)
self.off_diag_sum = self.off_diag.sum()  # = batch_size^2 - batch_size
```

---

## 3. Z 샘플링 전략

Z 벡터는 에이전트가 수행할 "task"를 정의하는 잠재 표현이다. 다양한 소스에서 z를 샘플링하여 범용적인 정책을 학습한다.

### 3.1 기본 z 샘플링: `sample_z()` (`fb/model.py`, 119-121행)

```python
def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
    z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
    return self.project_z(z)
```

가우시안 노이즈에서 z를 샘플링한 후, L2 정규화를 적용한다.

### 3.2 Z 프로젝션: `project_z()` (`fb/model.py`, 123-126행)

```python
def project_z(self, z):
    if self.cfg.archi.norm_z:
        z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
    return z
```

`norm_z=True`(기본값)일 때, z를 단위 구면 위로 정규화하고 `sqrt(z_dim)` 배율을 곱한다. 이렇게 하면 모든 z의 L2 노름이 `sqrt(z_dim)`로 일정해지며, z 공간의 기하학적 구조가 안정화된다.

### 3.3 FBAgent의 `sample_mixed_z()` (`fb/agent.py`, 146-159행)

FBAgent에서는 두 가지 소스에서 z를 혼합한다:

```python
@torch.no_grad()
def sample_mixed_z(self, train_goal=None, *args, **kwargs):
    z = self._model.sample_z(batch_size, device=self.device)  # uniform z

    if train_goal is not None:
        perm = torch.randperm(batch_size, device=self.device)
        train_goal = tree_map(lambda x: x[perm], train_goal)
        goals = self._model._backward_map(train_goal)       # B(next_obs)로 goal z 생성
        goals = self._model.project_z(goals)
        mask = torch.rand((batch_size, 1)) < train_goal_ratio  # 기본 0.5
        z = torch.where(mask, goals, z)                       # 50%는 goal z, 50%는 uniform z
    return z
```

| 비율 | 소스 | 목적 |
|------|------|------|
| `train_goal_ratio` (50%) | `B(next_obs)` | 실제 관측된 상태를 향한 task |
| 나머지 (50%) | 가우시안 + 정규화 | 탐색을 위한 랜덤 task |

### 3.4 FBcprAgent의 `sample_mixed_z()` (`fb_cpr/agent.py`, 126-150행)

FBcprAgent에서는 세 가지 소스를 사용한다:

```python
@torch.no_grad()
def sample_mixed_z(self, train_goal, expert_encodings, *args, **kwargs):
    z = self._model.sample_z(batch_size, device=self.device)

    p_goal = self.cfg.train.train_goal_ratio           # 기본 0.5
    p_expert_asm = self.cfg.train.expert_asm_ratio     # 기본 0
    prob = torch.tensor([p_goal, p_expert_asm, 1 - p_goal - p_expert_asm])

    mix_idxs = torch.multinomial(prob, num_samples=batch_size, replacement=True).reshape(-1, 1)

    # 소스 0: goal z (B로 train obs 인코딩)
    goals = self._model._backward_map(train_goal)
    goals = self._model.project_z(goals)
    z = torch.where(mix_idxs == 0, goals, z)

    # 소스 1: expert z (전문가 궤적 인코딩)
    perm = torch.randperm(batch_size, device=self.device)
    z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

    return z
```

| 비율 | 소스 | 설명 |
|------|------|------|
| `train_goal_ratio` (50%) | `B(train_next_obs)` | 학습 버퍼의 관측을 goal로 인코딩 |
| `expert_asm_ratio` (0%) | `encode_expert()` 결과 | 전문가 궤적 시퀀스의 평균 인코딩 |
| 나머지 (50%) | uniform | 랜덤 탐색 |

### 3.5 전문가 궤적 인코딩: `encode_expert()` (`fb_cpr/agent.py`, 152-168행)

```python
@torch.no_grad()
def encode_expert(self, next_obs):
    B_expert = self._model._backward_map(next_obs).detach()       # [batch, z_dim]
    B_expert = B_expert.view(
        batch_size // seq_length, seq_length, B_expert.shape[-1]  # [N, L, z_dim]
    )
    z_expert = B_expert.mean(dim=1)                                # [N, z_dim] (시간 평균)
    z_expert = self._model.project_z(z_expert)                     # L2 정규화
    z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)  # [batch, z_dim]
    return z_expert
```

배치를 시퀀스 단위(seq_length)로 나누고, 각 시퀀스 내에서 B 임베딩의 시간 평균을 구하여 해당 궤적의 대표 z를 생성한다. 이후 다시 원래 배치 크기로 복원한다.

---

## 4. `act()` 메서드: 관측에서 행동까지

### 4.1 호출 체인

```
FBAgent.act(obs, z, mean=True)
  → FBModel.act(obs, z, mean)
    → FBModel.actor(obs, z, std)          # 정규화된 obs로 분포 생성
      → _obs_normalizer(obs)              # 관측값 정규화
      → _actor(norm_obs, z, actor_std)    # Actor 네트워크 호출
    → dist.mean (mean=True) 또는 dist.sample()
```

참조: `fb/model.py` 128-132행, `fb/agent.py` 143-144행

### 4.2 Actor 네트워크 구조 (`nn_models.py`, 303-333행)

Actor는 obs와 z를 각각 별도의 임베딩으로 변환한 후 결합한다:

```python
class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg):
        obs_dim = filtered_space.shape[0]
        self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
        self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)
        # hidden_layers 만큼의 Linear + ReLU, 마지막에 action_dim 출력
        self.policy = nn.Sequential(...)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # [batch, hidden_dim // 2]
        s_embedding = self.embed_s(obs)                           # [batch, hidden_dim // 2]
        embedding = torch.cat([s_embedding, z_embedding], dim=-1) # [batch, hidden_dim]
        mu = torch.tanh(self.policy(embedding))                   # [batch, action_dim], [-1, 1] 범위
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist
```

핵심 포인트:
- obs와 z를 **별도로** 임베딩하여 각각의 특성을 독립적으로 학습
- `simple_embedding`은 `Linear → LayerNorm → Tanh → (Linear → ReLU)* → Linear → ReLU` 구조
- 출력 크기가 `hidden_dim // 2`이므로, 두 임베딩을 concat하면 `hidden_dim`이 된다
- `tanh`로 행동을 [-1, 1]로 클램프
- `TruncatedNormal` 분포를 반환하여 탐색 시 노이즈 추가 가능

### 4.3 `simple_embedding` 함수 (`nn_models.py`, 228-234행)

```python
def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    # [input_dim] → Linear → LayerNorm → Tanh → (Linear → ReLU)* → Linear → ReLU → [hidden_dim // 2]
    seq = [linear(input_dim, hidden_dim), layernorm(hidden_dim), nn.Tanh()]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
    seq += [linear(hidden_dim, hidden_dim // 2), nn.ReLU()]
    return nn.Sequential(*seq)
```

### 4.4 TruncatedNormal 분포 (`nn_models.py`, 662-683행)

```python
class TruncatedNormal(pyd.Normal):
    def sample(self, clip=None, sample_shape=torch.Size()):
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale                 # scale = std
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)  # 기본 clip=0.3
        x = self.loc + eps               # mu + noise
        return self._clamp(x)            # [-1+eps, 1-eps] 범위로 클램프
```

학습 시에는 `clip=stddev_clip` (기본 0.3)으로 탐색 노이즈를 제한하고, 추론 시에는 `mean`을 직접 사용한다.

---

## 5. `update()` 메서드 분석: 전체 학습 루프

### 5.1 FBAgent.update() (`fb/agent.py`, 161-208행)

가장 기본적인 학습 루프:

```python
def update(self, replay_buffer, step):
    # 1. 미니배치 샘플링
    batch = replay_buffer["train"].sample(batch_size)
    obs, action, next_obs, terminated = ...
    discount = 0.99 * ~terminated

    # 2. 관측값 정규화 (running mean/std 갱신 → 정규화 적용)
    self._model._obs_normalizer(obs)          # 통계량 갱신
    self._model._obs_normalizer(next_obs)
    obs = self._model._obs_normalizer(obs)    # 정규화된 값 적용
    next_obs = self._model._obs_normalizer(next_obs)

    # 3. z 샘플링 (goal z + uniform z 혼합)
    z = self.sample_mixed_z(train_goal=next_obs).clone()
    self.z_buffer.add(z)                      # z 버퍼에 저장 (향후 rollout용)

    # 4. FB loss 업데이트 (Forward, Backward 네트워크)
    metrics = self.update_fb(obs, action, discount, next_obs, goal=next_obs, z=z, ...)

    # 5. Actor loss 업데이트
    metrics.update(self.update_actor(obs, action, z, ...))

    # 6. Target network soft update
    _soft_update_params(forward_params, target_forward_params, tau=0.01)
    _soft_update_params(backward_params, target_backward_params, tau=0.01)

    return metrics
```

### 5.2 FBcprAgent.update() (`fb_cpr/agent.py`, 170-269행)

FB 위에 discriminator와 critic을 추가:

```python
def update(self, replay_buffer, step):
    # 1. 미니배치 샘플링 (학습 버퍼 + 전문가 버퍼)
    expert_batch = replay_buffer["expert_slicer"].sample(batch_size)
    train_batch = replay_buffer["train"].sample(batch_size)

    # 2. 관측값 정규화 (학습 데이터와 전문가 데이터 모두)
    train_obs, train_next_obs = normalize(...)
    expert_obs, expert_next_obs = normalize(...)

    # 3. 전문가 궤적 인코딩 → expert z
    expert_z = self.encode_expert(next_obs=expert_next_obs)
    train_z = train_batch["z"]  # rollout 시 사용된 z

    # 4. ★ Discriminator 학습 (전문가 vs 학습 데이터 분류)
    metrics = self.update_discriminator(expert_obs, expert_z, train_obs, train_z, ...)

    # 5. z 혼합 샘플링 (goal + expert + uniform)
    z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z)

    # 6. z 재라벨링 (relabel_ratio=1.0 → 100% 재라벨)
    if relabel_ratio is not None:
        mask = torch.rand(...) <= relabel_ratio
        train_z = torch.where(mask, z, train_z)

    # 7. FB loss 업데이트
    metrics.update(self.update_fb(...))

    # 8. ★ Critic 학습 (discriminator reward로 Q-function 학습)
    metrics.update(self.update_critic(...))

    # 9. ★ Actor 학습 (Q_fb + Q_discriminator 결합)
    metrics.update(self.update_actor(...))

    # 10. Target network soft update (F, B, Critic 모두)
    _soft_update_params(forward, target_forward, tau=0.01)
    _soft_update_params(backward, target_backward, tau=0.01)
    _soft_update_params(critic, target_critic, tau=0.005)

    return metrics
```

### 5.3 FBcprAuxAgent.update() (`fb_cpr_aux/agent.py`, 83-210행)

FBcprAgent 위에 auxiliary reward와 critic을 추가:

```python
def update(self, replay_buffer, step):
    # ... (FBcprAgent와 동일한 1-7 단계) ...

    # 8. Critic 학습
    metrics.update(self.update_critic(...))

    # 9. ★ Auxiliary reward 계산 (tracking 등 보조 보상의 가중합)
    aux_reward = torch.zeros((batch_size, 1), ...)
    for aux_reward_name in self.cfg.aux_rewards:
        aux_reward += scaling[name] * train_batch["aux_rewards"][name]
    aux_reward = self._model._aux_reward_normalizer(aux_reward)

    # 10. ★ Auxiliary Critic 학습
    metrics.update(self.update_aux_critic(obs, action, discount, aux_reward, next_obs, z))

    # 11. ★ Actor 학습 (Q_fb + Q_discriminator + Q_aux 결합)
    metrics.update(self.update_actor(...))

    # 12. Target network soft update (F, B, Critic, Aux Critic)
    ...
```

---

## 6. FB Loss 계산 과정

`update_fb()` 메서드 (`fb/agent.py`, 216-297행) 의 상세 분석.

### 6.1 Target M 계산 (no_grad)

```python
with torch.no_grad():
    next_action = self.sample_action_from_norm_obs(next_obs, z)
    target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # [2, 1024, 100]
    target_B = self._model._target_backward_map(goal)                      # [1024, 100]
    target_Ms = torch.matmul(target_Fs, target_B.T)                        # [2, 1024, 1024]
    _, _, target_M = self.get_targets_uncertainty(target_Ms, pessimism)     # [1024, 1024]
```

`get_targets_uncertainty`는 앙상블 평균에서 불확실성 페널티를 빼는 함수이다 (`fb/agent.py`, 328-343행):

```python
def get_targets_uncertainty(self, preds, pessimism_penalty):
    preds_mean = preds.mean(dim=0)                    # 앙상블 평균
    # 앙상블 간 쌍별 차이의 평균으로 불확실성 추정
    preds_unc = abs(preds[i] - preds[j]).mean() for all i,j
    return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc
```

FB target의 `fb_pessimism_penalty`는 기본 0.0이므로, 실제로는 앙상블 평균만 사용된다.

### 6.2 현재 M 계산 및 Loss

```python
Fs = self._model._forward_map(obs, z, action)   # [2, 1024, 100]
B = self._model._backward_map(goal)              # [1024, 100]
Ms = torch.matmul(Fs, B.T)                       # [2, 1024, 1024]

diff = Ms - discount * target_M                  # [2, 1024, 1024]

# 비대각: 벨만 잔차의 L2 → 0으로 수렴해야 함
fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum

# 대각: 자기 자신에 대한 측도 → 1로 수렴해야 함 (부호 반전으로 최대화)
fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]

fb_loss = fb_offdiag + fb_diag
```

### 6.3 선택적 Q Loss (q_loss_coef > 0일 때)

implicit reward를 계산하여 TD 오류를 추가로 부과한다 (254-269행):

```python
# implicit reward: r(s) = B(s)^T * Cov(B)^{-1} * z
cov = B^T * B / batch_size            # [z_dim, z_dim]
B_inv_conv = solve(cov, B)            # B * cov^{-1}
implicit_reward = (B_inv_conv * z).sum(dim=-1)

target_Q = implicit_reward + discount * next_Q
Qs = (Fs * z).sum(dim=-1)             # Q(s,a,z) = F(s,z,a)^T * z
q_loss = 0.5 * MSE(Qs, target_Q)
```

---

## 7. Orthonormality Loss

Backward map B의 출력이 직교 정규 기저에 가깝도록 정규화한다 (`fb/agent.py`, 248-252행):

```python
Cov = torch.matmul(B, B.T)           # [1024, 1024], B 임베딩의 공분산
orth_loss_diag = -Cov.diag().mean()   # 대각 → 1로 (정규화)
orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum  # 비대각 → 0으로 (직교)
orth_loss = orth_loss_offdiag + orth_loss_diag

fb_loss += ortho_coef * orth_loss     # ortho_coef = 1.0 (기본)
```

이 loss의 의미:
- **대각 원소가 1**: 각 B(s_i)의 노름이 1에 가까워야 함 (정규성)
- **비대각 원소가 0**: 서로 다른 상태의 B 임베딩이 직교해야 함 (다양성)

이를 통해 B 임베딩 공간의 표현력이 보장되며, z 공간에서의 task 구분이 명확해진다.

---

## 8. Discriminator 학습과 CPR Loss

### 8.1 Discriminator 구조 (`nn_models.py`, 336-369행)

```python
class Discriminator(nn.Module):
    def __init__(self, obs_space, z_dim, cfg):
        # (obs_dim + z_dim) → Linear → LayerNorm → Tanh → (Linear → ReLU)* → Linear → 1
        self.trunk = nn.Sequential(...)

    def compute_logits(self, obs, z):
        x = torch.cat([z, obs], dim=1)    # z와 obs를 concat
        return self.trunk(x)               # logit 출력

    def compute_reward(self, obs, z, eps=1e-7):
        s = torch.sigmoid(self.compute_logits(obs, z))
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()  # log-ratio 보상
        return reward
```

Discriminator는 (obs, z) 쌍이 전문가 데이터에서 온 것인지 학습 데이터에서 온 것인지를 분류한다. 보상은 GAN의 log-ratio 형태로 계산된다.

### 8.2 Discriminator 학습 (`fb_cpr/agent.py`, 333-365행)

```python
def update_discriminator(self, expert_obs, expert_z, train_obs, train_z, grad_penalty):
    expert_logits = self._model._discriminator.compute_logits(expert_obs, expert_z)
    unlabeled_logits = self._model._discriminator.compute_logits(train_obs, train_z)

    # 이진 교차 엔트로피와 동등한 형태
    expert_loss = -torch.nn.functional.logsigmoid(expert_logits)     # 전문가 → 1로
    unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)  # 학습 → 0으로
    loss = torch.mean(expert_loss + unlabeled_loss)

    # WGAN Gradient Penalty (기본 10.0)
    if grad_penalty is not None:
        wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
        loss += grad_penalty * wgan_gp
```

### 8.3 WGAN Gradient Penalty (`fb_cpr/agent.py`, 271-331행)

전문가 데이터와 학습 데이터 사이의 보간점에서 gradient의 L2 노름이 1에 가깝도록 정규화한다:

```python
def gradient_penalty_wgan(self, real_obs, real_z, fake_obs, fake_z):
    alpha = torch.rand(batch_size, 1)
    interpolated_obs = alpha * real_obs + (1 - alpha) * fake_obs
    interpolated_z = alpha * real_z + (1 - alpha) * fake_z

    d_interpolates = discriminator.compute_logits(interpolated_obs, interpolated_z)
    gradients = autograd.grad(outputs=d_interpolates, inputs=[interpolated_obs, interpolated_z], ...)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

이 기법은 학습의 안정성을 크게 높여준다.

---

## 9. Critic 학습

Critic은 discriminator reward에 대한 Q-function을 학습한다 (`fb_cpr/agent.py`, 367-405행).

```python
def update_critic(self, obs, action, discount, next_obs, z):
    with torch.no_grad():
        # 1. Discriminator에서 보상 계산
        reward = self._model._discriminator.compute_reward(obs=obs, z=z)

        # 2. Target Q 계산 (TD target)
        next_action = actor(next_obs, z).sample(clip=0.3)
        next_Qs = target_critic(next_obs, z, next_action)         # [num_parallel, batch, 1]
        Q_mean, Q_unc, next_V = get_targets_uncertainty(next_Qs, critic_pessimism=0.5)
        target_Q = reward + discount * next_V                     # [batch, 1]

    # 3. Critic loss (MSE)
    Qs = critic(obs, z, action)                                   # [num_parallel, batch, 1]
    critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, target_Q.expand(num_parallel, -1, -1))
```

Critic은 Forward map과 동일한 아키텍처(`ForwardArchiConfig`)를 사용하지만, 출력 차원이 `z_dim` 대신 1이다 (`fb_cpr/model.py`, 42행).

---

## 10. Actor 학습: 다중 목적 최적화

Actor는 여러 목적 함수를 동시에 최적화한다. 계층별로 점점 더 많은 목적이 추가된다.

### 10.1 FBAgent의 Actor 업데이트 (`fb/agent.py`, 308-326행)

```python
def update_td3_actor(self, obs, z, clip_grad_norm):
    dist = self._model._actor(obs, z, actor_std)
    action = dist.sample(clip=0.3)

    Fs = self._model._forward_map(obs, z, action)  # [num_parallel, batch, z_dim]
    Qs = (Fs * z).sum(-1)                            # [num_parallel, batch]
    _, _, Q = get_targets_uncertainty(Qs, actor_pessimism=0.5)

    actor_loss = -Q.mean()   # Q 최대화
```

FB에서의 Q-value는 `Q(s,a,z) = F(s,z,a)^T * z`로 정의된다. 이는 z 방향으로의 후계 측도를 최대화하는 것을 의미한다.

### 10.2 FBcprAgent의 Actor 업데이트 (`fb_cpr/agent.py`, 407-443행)

```python
def update_actor(self, obs, action, z, clip_grad_norm):
    dist = self._model._actor(obs, z, actor_std)
    action = dist.sample(clip=0.3)

    # 목적 1: Discriminator Q (전문가 모방)
    Qs_discriminator = self._model._critic(obs, z, action)
    _, _, Q_discriminator = get_targets_uncertainty(Qs_discriminator, pessimism=0.5)

    # 목적 2: FB Q (task 수행)
    Fs = self._model._forward_map(obs, z, action)
    Qs_fb = (Fs * z).sum(-1)
    _, _, Q_fb = get_targets_uncertainty(Qs_fb, pessimism=0.5)

    # 가중 결합
    weight = Q_fb.abs().mean().detach() if scale_reg else 1.0
    actor_loss = -Q_discriminator.mean() * reg_coeff * weight - Q_fb.mean()
```

`scale_reg=True`(기본값)일 때, `weight = |Q_fb|.mean()`으로 discriminator 보상의 스케일을 Q_fb에 맞춘다. 이는 두 목적 함수의 스케일이 달라지는 문제를 방지한다.

### 10.3 FBcprAuxAgent의 Actor 업데이트 (`fb_cpr_aux/agent.py`, 253-298행)

```python
def update_actor(self, obs, action, z, clip_grad_norm):
    # ... Q_discriminator, Q_fb 계산 (FBcprAgent와 동일) ...

    # 목적 3: Auxiliary Q (보조 보상)
    Qs_aux = self._model._aux_critic(obs, z, action)
    _, _, Q_aux = get_targets_uncertainty(Qs_aux, pessimism=0.5)

    weight = Q_fb.abs().mean().detach() if scale_reg else 1.0
    actor_loss = (
        -Q_discriminator.mean() * reg_coeff * weight       # CPR loss
        - Q_aux.mean() * reg_coeff_aux * weight            # Auxiliary loss
        - Q_fb.mean()                                       # FB loss
    )
```

최종 Actor loss 구성:

| 항 | 계수 | 기본값 | 의미 |
|----|------|--------|------|
| `-Q_fb.mean()` | 1 | - | z 방향 task 수행 최대화 |
| `-Q_disc.mean() * reg_coeff * weight` | `reg_coeff` | 1.0 | 전문가 모방 (CPR) |
| `-Q_aux.mean() * reg_coeff_aux * weight` | `reg_coeff_aux` | 1.0 | 보조 보상 최대화 |

---

## 11. Auxiliary Loss (보조 손실)

### 11.1 Auxiliary Reward 수집 (`fb_cpr_aux/agent.py`, 156-167행)

```python
aux_reward = torch.zeros((batch_size, 1), device=self.device)
for aux_reward_name in self.cfg.aux_rewards:
    metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
    aux_reward += scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name]

aux_reward = self._model._aux_reward_normalizer(aux_reward)  # EMA 정규화
```

보조 보상은 환경에서 수집된 `aux_rewards` 딕셔너리에서 가져온다. 예를 들어 tracking 오류, 자세 안정성 등을 포함할 수 있다. 각 보조 보상에 `aux_rewards_scaling`에 정의된 스케일 가중치가 곱해진다.

### 11.2 Auxiliary Critic 학습 (`fb_cpr_aux/agent.py`, 212-251행)

Critic 학습과 동일한 구조이지만, discriminator reward 대신 auxiliary reward를 사용한다:

```python
def update_aux_critic(self, obs, action, discount, aux_reward, next_obs, z):
    with torch.no_grad():
        next_action = actor(next_obs, z).sample(clip=0.3)
        next_Qs = target_aux_critic(next_obs, z, next_action)
        _, _, next_V = get_targets_uncertainty(next_Qs, aux_critic_pessimism=0.5)
        target_Q = aux_reward + discount * next_V

    Qs = aux_critic(obs, z, action)
    aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, target_Q.expand(...))
```

### 11.3 EMA Reward Normalizer (`nn_models.py`, 694-728행)

```python
class EMA(nn.Module):
    """exponential moving average"""
    def __init__(self, tau=0.99, translate=False, scale=False):
        # running mean, running mean_square 관리
        ...

    def forward(self, x):
        m = x.mean()
        sm = x.pow(2).mean()
        self.mean = tau * self.mean + (1 - tau) * m
        self.mean_square = tau * self.mean_square + (1 - tau) * sm
        # bias correction
        var = clamp(ema_mean_square - ema_mean^2, min=eps)
        return (x - translate_mean) / scale_std
```

기본 설정(`translate=False, scale=False`)에서는 실질적으로 아무 변환도 하지 않는다. 필요에 따라 평균 이동(translate)과 분산 스케일링(scale)을 활성화할 수 있다.

---

## 12. 커리큘럼 학습: EMD 기반 모션 가중치

### 12.1 개요

BFM-Zero는 평가 시점에서 각 모션 클립에 대한 **Earth Mover's Distance (EMD)** 를 계산하고, EMD가 큰 (즉, 아직 잘 따라하지 못하는) 모션에 더 높은 샘플링 가중치를 부여한다. 이를 통해 어려운 모션을 더 자주 학습하는 커리큘럼 효과를 달성한다.

### 12.2 EMD 계산 (`agents/evaluations/humanoidverse_isaac.py`, 564-573행)

```python
def emd_numpy(next_obs, tracking_target, prefix=""):
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}
```

EMD는 에이전트가 생성한 궤적과 목표 궤적 사이의 최적 운송 비용을 측정한다. 두 분포를 균등 가중치로 놓고 cost matrix 기반으로 Wasserstein 거리를 계산한다.

### 12.3 우선순위 업데이트 (`train.py`, 350-394행)

```python
if self.cfg.prioritization:
    motions_id, priorities, idxs = [], [], []
    for _, metr in eval_metrics[self.priorization_eval_name].items():
        motions_id.append(metr["motion_id"])
        priorities.append(metr["emd"])                     # EMD 값을 우선순위로
        idxs.append(index_in_buffer[metr["motion_id"]])

    # 클램핑 및 스케일링
    priorities = torch.clamp(
        torch.tensor(priorities),
        min=prioritization_min_val,    # 기본 0.5
        max=prioritization_max_val,    # 기본 5.0
    ) * prioritization_scale           # 기본 2.0

    # 모드에 따른 변환
    if prioritization_mode == "lin":
        pass                            # 선형
    elif prioritization_mode == "exp":
        priorities = 2**priorities      # 지수적
    elif prioritization_mode == "bin":
        bins = torch.floor(priorities)  # 구간별 균등 가중치
        for i in range(int(bins.min()), int(bins.max()) + 1):
            mask = bins == i
            priorities[mask] = 1 / mask.sum()

    # 모션 라이브러리와 버퍼 모두에 우선순위 적용
    train_env._env._motion_lib.update_sampling_weight_by_id(priorities, idxs, ...)
    replay_buffer["expert_slicer"].update_priorities(priorities, idxs)
```

우선순위 모드 비교:

| 모드 | 변환 | 특성 |
|------|------|------|
| `lin` | 변환 없음 | EMD에 비례하여 선형적으로 가중치 부여 |
| `exp` | `2^priority` | EMD가 클수록 기하급수적으로 가중치 증가 |
| `bin` (기본) | 같은 구간 내 균등 배분 | EMD를 구간화하고 각 구간 내에서 균등 가중 |

---

## 13. 텐서 Shape 흐름 추적

기본 설정 기준: `batch_size=1024`, `z_dim=100`, `action_dim=29` (G1 로봇), `hidden_dim=1024`, `num_parallel=2`.

### 13.1 Forward Pass (act)

```
obs: [batch, obs_dim]
 ↓ obs_normalizer
norm_obs: [batch, obs_dim]
 ↓ input_filter
filtered_obs: [batch, filtered_obs_dim]
 ↓ embed_s(filtered_obs)
s_embedding: [batch, 512]  (hidden_dim // 2)
 ↓ embed_z(cat([filtered_obs, z], dim=-1))
z_embedding: [batch, 512]  (입력: [batch, filtered_obs_dim + z_dim])
 ↓ cat([s_embedding, z_embedding])
embedding: [batch, 1024]  (hidden_dim)
 ↓ policy(embedding)
raw_output: [batch, 29]  (action_dim)
 ↓ tanh
mu: [batch, 29]  (범위 [-1, 1])
 ↓ TruncatedNormal(mu, std=0.2)
dist → dist.mean: [batch, 29]
```

### 13.2 Forward Map

```
obs: [batch, obs_dim]
z: [batch, z_dim]
action: [batch, action_dim]
 ↓ expand (num_parallel > 1)
obs: [2, batch, obs_dim]
z: [2, batch, z_dim]
action: [2, batch, action_dim]
 ↓ embed_z(cat([obs, z]))
z_emb: [2, batch, 512]
 ↓ embed_sa(cat([obs, action]))
sa_emb: [2, batch, 512]
 ↓ Fs(cat([sa_emb, z_emb]))
output: [2, batch, 100]  (z_dim)
```

### 13.3 Backward Map

```
goal (next_obs): [batch, obs_dim]
 ↓ input_filter
filtered: [batch, filtered_obs_dim]
 ↓ Linear → LayerNorm → Tanh
hidden: [batch, 256]
 ↓ (Linear → ReLU) * (hidden_layers - 1)
hidden: [batch, 256]
 ↓ Linear
output: [batch, 100]  (z_dim)
 ↓ Norm (sqrt(z_dim) * normalize)
B: [batch, 100]  (L2 노름 = sqrt(100) = 10)
```

### 13.4 FB Loss 텐서 흐름

```
Fs: [2, 1024, 100]       (Forward map 출력)
B:  [1024, 100]           (Backward map 출력)
 ↓ matmul(Fs, B.T)
Ms: [2, 1024, 1024]       (후계 측도 행렬)
 ↓ - discount * target_M
diff: [2, 1024, 1024]     (벨만 잔차)
 ↓ off_diag 마스킹
fb_offdiag: scalar
fb_diag: scalar
fb_loss: scalar
```

### 13.5 Discriminator 흐름

```
obs: [1024, obs_dim]
z: [1024, 100]
 ↓ cat([z, obs], dim=1)
input: [1024, obs_dim + 100]
 ↓ trunk (MLP)
logits: [1024, 1]
 ↓ sigmoid → log ratio
reward: [1024, 1]
```

---

## 14. 학습 하이퍼파라미터와 옵티마이저 설정

### 14.1 FBAgent 기본 하이퍼파라미터 (`fb/agent.py`, 24-44행)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `lr_f` | 1e-4 | Forward map 학습률 |
| `lr_b` | 1e-4 | Backward map 학습률 |
| `lr_actor` | 1e-4 | Actor 학습률 |
| `weight_decay` | 0.0 | 가중치 감쇠 |
| `clip_grad_norm` | 0.0 | Gradient clipping (0=비활성) |
| `fb_target_tau` | 0.01 | FB target network 소프트 업데이트 비율 |
| `ortho_coef` | 1.0 | Orthonormality loss 계수 |
| `train_goal_ratio` | 0.5 | goal z 비율 |
| `fb_pessimism_penalty` | 0.0 | FB target 비관주의 페널티 |
| `actor_pessimism_penalty` | 0.5 | Actor Q 비관주의 페널티 |
| `stddev_clip` | 0.3 | Action 노이즈 클리핑 |
| `batch_size` | 1024 | 미니배치 크기 |
| `discount` | 0.99 | 할인율 |
| `update_z_every_step` | 150 | z 갱신 주기 (rollout 중) |
| `z_buffer_size` | 10000 | z 버퍼 용량 |

### 14.2 FBcprAgent 추가 하이퍼파라미터 (`fb_cpr/agent.py`, 21-38행)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `lr_discriminator` | 1e-4 | Discriminator 학습률 |
| `lr_critic` | 1e-4 | Critic 학습률 |
| `critic_target_tau` | 0.005 | Critic target 소프트 업데이트 비율 |
| `critic_pessimism_penalty` | 0.5 | Critic Q 비관주의 페널티 |
| `reg_coeff` | 1.0 | CPR 정규화 계수 |
| `scale_reg` | True | Q_fb 기반 CPR 스케일링 |
| `expert_asm_ratio` | 0.0 | 전문가 z 비율 |
| `relabel_ratio` | 1.0 | z 재라벨링 비율 |
| `grad_penalty_discriminator` | 10.0 | WGAN gradient penalty 계수 |

### 14.3 FBcprAuxAgent 추가 하이퍼파라미터 (`fb_cpr_aux/agent.py`, 21-24행)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `lr_aux_critic` | 1e-4 | Auxiliary critic 학습률 |
| `reg_coeff_aux` | 1.0 | Auxiliary 정규화 계수 |
| `aux_critic_pessimism_penalty` | 0.5 | Aux critic 비관주의 페널티 |

### 14.4 옵티마이저 설정

모든 네트워크에 Adam 옵티마이저를 사용한다. `setup_training()` 참조:

```python
# fb/agent.py, 96-113행
self.backward_optimizer = torch.optim.Adam(
    self._model._backward_map.parameters(), lr=lr_b, weight_decay=0.0)
self.forward_optimizer = torch.optim.Adam(
    self._model._forward_map.parameters(), lr=lr_f, weight_decay=0.0)
self.actor_optimizer = torch.optim.Adam(
    self._model._actor.parameters(), lr=lr_actor, weight_decay=0.0)

# fb_cpr/agent.py, 98-109행
self.critic_optimizer = torch.optim.Adam(
    self._model._critic.parameters(), lr=lr_critic, weight_decay=0.0)
self.discriminator_optimizer = torch.optim.Adam(
    self._model._discriminator.parameters(), lr=lr_discriminator,
    weight_decay=weight_decay_discriminator)  # discriminator만 별도 weight_decay 가능
```

### 14.5 학습 주기 설정 (`train.py`, 82-93행)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `online_parallel_envs` | 50 | 병렬 환경 수 |
| `num_env_steps` | 30M | 총 환경 스텝 |
| `update_agent_every` | 500 | 에이전트 업데이트 주기 (환경 스텝 기준) |
| `num_seed_steps` | 50K | 초기 시드 스텝 (업데이트 없이 수집만) |
| `num_agent_updates` | 50 | 주기당 업데이트 횟수 |

이를 종합하면, 매 500 환경 스텝마다 50번의 `update()` 호출이 발생한다. 각 `update()`는 1024 크기의 배치를 사용한다.

---

## 15. Target Network와 Soft Update

### 15.1 Soft Update 공식

```python
# nn_models.py, 80-82행
def _soft_update_params(net_params, target_net_params, tau):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)
```

수식으로 표현하면: `theta_target = (1 - tau) * theta_target + tau * theta`

### 15.2 각 네트워크의 tau 값

| Target Network | tau | 소스 네트워크 |
|----------------|-----|--------------|
| `target_forward_map` | 0.01 | `forward_map` |
| `target_backward_map` | 0.01 | `backward_map` |
| `target_critic` | 0.005 | `critic` |
| `target_aux_critic` | 0.005 | `aux_critic` |

FB 네트워크(tau=0.01)가 critic(tau=0.005)보다 더 빠르게 target을 갱신한다.

### 15.3 가중치 초기화 (`nn_models.py`, 61-72행)

```python
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)   # 직교 초기화
        m.bias.data.fill_(0.0)               # bias 0
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)  # 병렬 직교 초기화
        m.bias.data.fill_(0.0)
```

모든 선형 레이어에 직교 초기화를 적용하여 학습 초기의 gradient 흐름을 안정화한다. `DenseParallel`(앙상블용 병렬 레이어)에는 ReLU gain으로 스케일링된 직교 초기화를 사용한다.

---

## 요약: 전체 학습 흐름 다이어그램

```
[환경] ─→ (obs, action, next_obs, terminated) ─→ [Replay Buffer (train)]
[전문가 데이터] ─→ (expert_obs, expert_next_obs) ─→ [Replay Buffer (expert_slicer)]

update() 호출 시:
┌─────────────────────────────────────────────────────────────┐
│ 1. 배치 샘플링: train(1024) + expert(1024)                  │
│ 2. 관측값 정규화 (running mean/std)                          │
│ 3. expert z 인코딩: B(expert_next_obs) → 시퀀스 평균         │
│ 4. Discriminator 업데이트:                                    │
│    L_disc = -logsigmoid(expert) + softplus(train) + GP      │
│ 5. z 혼합 샘플링: goal(50%) + expert(0%) + uniform(50%)      │
│ 6. z 재라벨링 (100%)                                         │
│ 7. FB 업데이트:                                               │
│    L_fb = L_offdiag + L_diag + ortho_coef * L_orth           │
│ 8. Critic 업데이트:                                           │
│    L_critic = MSE(Q, r_disc + gamma * V_target)              │
│ 9. (Aux Critic 업데이트):                                     │
│    L_aux = MSE(Q_aux, r_aux + gamma * V_aux_target)          │
│10. Actor 업데이트:                                            │
│    L_actor = -Q_fb - reg * weight * Q_disc (- reg_aux * Q_aux)│
│11. Soft update: F, B, Critic, (Aux Critic) target networks   │
└─────────────────────────────────────────────────────────────┘
```

---

## 참조 파일 목록

| 파일 | 절대 경로 |
|------|-----------|
| FB Agent | `humanoidverse/agents/fb/agent.py` |
| FB Model | `humanoidverse/agents/fb/model.py` |
| FBcpr Agent | `humanoidverse/agents/fb_cpr/agent.py` |
| FBcpr Model | `humanoidverse/agents/fb_cpr/model.py` |
| FBcprAux Agent | `humanoidverse/agents/fb_cpr_aux/agent.py` |
| FBcprAux Model | `humanoidverse/agents/fb_cpr_aux/model.py` |
| NN Models | `humanoidverse/agents/nn_models.py` |
| Z Buffer | `humanoidverse/agents/misc/zbuffer.py` |
| Train Loop | `humanoidverse/train.py` |
| Evaluation (EMD) | `humanoidverse/agents/evaluations/humanoidverse_isaac.py` |
