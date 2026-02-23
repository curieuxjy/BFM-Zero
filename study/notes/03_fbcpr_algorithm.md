# 03. FB-CPR 알고리즘 분석 (코드 중심)

> **관련 노트**: 이 문서는 에이전트 코드의 행 번호 기반 상세 분석입니다.
> 수학적 원리, 텐서 shape 흐름, 하이퍼파라미터 정리는 [`07_learning_algorithms.md`](07_learning_algorithms.md)를 참고하세요.

## 1. 개요

BFM-Zero의 학습 알고리즘은 세 단계의 상속 구조로 이루어진 에이전트 계층으로 구현된다.

```
FBAgent (fb/agent.py)
    |  Forward-Backward 표현 학습의 기본 구현
    |  - Forward Map F(s, z, a), Backward Map B(s'), Actor pi(s, z)
    |  - FB 손실, 직교성 손실, TD3 스타일 Actor 업데이트
    |
    +-- FBcprAgent (fb_cpr/agent.py)  <-- 주력 에이전트
            |  Contrastive Policy Regularization 추가
            |  - Discriminator D(s, z): 전문가/정책 판별
            |  - Critic Q(s, z, a): 판별자 보상 기반 TD 학습
            |  - Mixed Z 분포: 전문가/목표/균등 혼합
            |  - Z 리라벨링 (relabel_ratio)
            |
            +-- FBcprAuxAgent (fb_cpr_aux/agent.py)
                    보조 보상(Auxiliary Reward) 추가
                    - Aux Critic: 환경 페널티 기반 추가 Q-value
                    - Actor 손실에 3종 Q-value 결합
```

**핵심 아이디어**: Forward-Backward (FB) 표현 학습은 상태 공간의 successor feature를 학습하는 비지도 강화학습 프레임워크이다. 잠재 벡터 $z$를 조건으로 하여, 어떤 목표(goal)든 추론 시 즉시 추적할 수 있는 범용 정책을 학습한다. CPR은 여기에 전문가 시연 데이터를 활용하는 GAN 기반 정규화를 추가하여, 전문가처럼 행동하면서도 다양한 $z$에 반응할 수 있도록 한다.

### 모델 구성 요소 전체 맵

| 구성 요소 | 입력 | 출력 | 역할 |
|-----------|------|------|------|
| `_backward_map` B | obs | $\mathbb{R}^{z\_dim}$ | 관측을 잠재 공간으로 인코딩 |
| `_forward_map` F | obs, z, action | $\mathbb{R}^{z\_dim}$ | Successor Feature 예측 |
| `_actor` $\pi$ | obs, z | action | 정책 (z 조건부) |
| `_critic` Q | obs, z, action | $\mathbb{R}^1$ | 판별자 보상 기반 가치 함수 |
| `_discriminator` D | obs, z | $\mathbb{R}^1$ | 전문가/정책 분류기 |
| `_aux_critic` $Q_{aux}$ | obs, z, action | $\mathbb{R}^1$ | 보조 보상 가치 함수 |

각 구성 요소에 대해 `_target_*` 네트워크가 존재하며, 소프트 업데이트로 안정적 학습 타겟을 제공한다.

---

## 2. fb/agent.py - Forward-Backward 기본 에이전트

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb/agent.py`

### 2.1 클래스 구조와 초기화

`FBAgent`는 FB 표현 학습의 기본 구현을 담당한다.

**설정 클래스 `FBAgentTrainConfig`** (주요 하이퍼파라미터):

```python
class FBAgentTrainConfig(BaseConfig):
    lr_f: float = 1e-4              # Forward Map 학습률
    lr_b: float = 1e-4              # Backward Map 학습률
    lr_actor: float = 1e-4          # Actor 학습률
    fb_target_tau: float = 0.01     # 타겟 네트워크 소프트 업데이트 계수
    ortho_coef: float = 1.0         # 직교성 손실 계수
    train_goal_ratio: float = 0.5   # z 샘플링 시 goal 인코딩 비율
    fb_pessimism_penalty: float = 0.0   # FB 타겟의 비관적 페널티
    actor_pessimism_penalty: float = 0.5 # Actor Q의 비관적 페널티
    stddev_clip: float = 0.3        # 액션 분포 클리핑
    batch_size: int = 1024          # 배치 크기
    discount: float = 0.99          # 할인 계수 gamma
    update_z_every_step: int = 150  # 롤아웃 z 교체 주기
    z_buffer_size: int = 10000      # z 버퍼 크기
```

**초기화 과정** (`__init__`, 65-76행):

```python
def __init__(self, obs_space, action_dim, cfg: FBAgentConfig):
    self._model: FBModel = self.cfg.model.build(obs_space, action_dim)
    self.setup_training()   # 옵티마이저 설정 + weight_init
    self.setup_compile()    # torch.compile / CudaGraph 설정
```

`setup_training()` (90-125행)에서의 초기화 단계:
1. 모델의 모든 가중치에 직교 초기화(`weight_init`) 적용 -- `nn.Linear`에는 `nn.init.orthogonal_`, `DenseParallel`에는 `parallel_orthogonal_`
2. `_prepare_for_train()`으로 타겟 네트워크(`_target_forward_map`, `_target_backward_map`)를 deep copy로 생성
3. 세 개의 Adam 옵티마이저 생성 (Forward, Backward, Actor)
4. 비대각 마스크 `off_diag` 사전 계산: `1 - I_{batch_size}` (batch_size x batch_size 단위행렬의 보수)
5. `ZBuffer` 초기화 (학습에 사용된 z를 저장하여 롤아웃에 재활용)

### 2.2 update_fb() - FB 맵 업데이트

이 메서드(216-297행)가 BFM-Zero 학습의 핵심이다. Forward-Backward 표현의 successor measure를 학습한다.

**수학적 배경**: FB 표현 학습에서 Forward Map $F(s, z, a)$와 Backward Map $B(s')$는 다음 관계를 만족하도록 학습된다:

$$M(s, a, s') = F(s, z, a) \cdot B(s')^T \approx \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t \mathbf{1}[s_t' = s'] \mid s_0 = s, a_0 = a\right]$$

이는 상태 $s$에서 행동 $a$를 취한 후 미래에 상태 $s'$를 방문할 할인된 확률(successor measure)을 근사한다.

**입력/출력**:
- 입력: `obs`, `action`, `discount`, `next_obs`, `goal`, `z`, `q_loss_coef`, `clip_grad_norm`
- 출력: 메트릭 딕셔너리 (`fb_loss`, `fb_diag`, `fb_offdiag`, `orth_loss` 등)

**핵심 로직 (단계별)**:

**Step 1: 타겟 M 행렬 계산** (그래디언트 없음, 228-235행)

```python
with torch.no_grad():
    next_action = self.sample_action_from_norm_obs(next_obs, z)
    target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
    target_B = self._model._target_backward_map(goal)                       # batch x z_dim
    target_Ms = torch.matmul(target_Fs, target_B.T)                        # num_parallel x batch x batch
    _, _, target_M = self.get_targets_uncertainty(target_Ms, fb_pessimism_penalty)
```

`target_Ms[i, j]`는 "상태 $i$에서 출발하여 상태 $j$를 미래에 방문할 확률"의 근사값이다. `get_targets_uncertainty`는 여러 병렬 네트워크(`num_parallel`)의 예측을 결합하여 불확실성을 고려한 비관적 추정치를 반환한다.

**Step 2: 현재 M 행렬 계산** (그래디언트 있음, 238-240행)

```python
Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
B = self._model._backward_map(goal)             # batch x z_dim
Ms = torch.matmul(Fs, B.T)                      # num_parallel x batch x batch
```

**Step 3: FB 손실 계산** (242-245행)

FB 손실은 Bellman 일관성 조건에서 유도된다:

$$\text{diff} = M(s, a, s') - \gamma \cdot M_{\text{target}}(s', \pi(s'), s')$$

이 diff 행렬에서 **대각 요소**와 **비대각 요소**를 별도로 처리한다:

```python
diff = Ms - discount * target_M                           # num_parallel x batch x batch
fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum  # 비대각: MSE
fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]       # 대각: 최대화
fb_loss = fb_offdiag + fb_diag
```

$$\mathcal{L}_{FB} = \underbrace{\frac{1}{2|\text{off-diag}|}\sum_{i \neq j} (\text{diff}_{ij})^2}_{\text{비대각: 0으로}} + \underbrace{(-1) \cdot N_p \cdot \text{mean}(\text{diag}(\text{diff}))}_{\text{대각: 최대화}}$$

- **비대각 요소** ($i \neq j$): 서로 다른 상태 쌍은 0이 되어야 (방문하지 않은 상태)
- **대각 요소** ($i = j$): 같은 상태 쌍은 커져야 (자기 자신은 방문함)

이 비대칭적 손실이 FB 표현의 핵심으로, successor measure를 올바르게 학습하게 한다.

**Step 4: 직교성(Orthonormality) 손실** (248-252행)

Backward Map의 출력이 직교 정규 기저를 형성하도록 정규화한다:

```python
Cov = torch.matmul(B, B.T)                                    # batch x batch
orth_loss_diag = -Cov.diag().mean()                            # 대각: 노름 -> 1
orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum  # 비대각: -> 0
orth_loss = orth_loss_offdiag + orth_loss_diag
fb_loss += self.cfg.train.ortho_coef * orth_loss
```

$$\mathcal{L}_{orth} = \frac{1}{2|\text{off-diag}|}\sum_{i \neq j} (B_i \cdot B_j)^2 - \text{mean}(\|B_i\|^2)$$

이 손실은 B의 출력 벡터들이 서로 수직이면서 단위 노름을 갖게 하여, 잠재 공간에서의 표현이 풍부하고 중복 없이 유지되도록 한다. 목표하는 공분산: $B B^T \to I$ (단위행렬).

**Step 5: (선택) Q-Loss** (254-269행)

`q_loss_coef > 0`일 때 추가되는 보조 손실로, implicit reward를 활용한 Q-learning 목표:

```python
# Implicit reward: r = B(s') @ Cov(B)^{-1} @ z
cov = B.T @ B / B.shape[0]                      # z_dim x z_dim
B_inv_conv = torch.linalg.solve(cov, B, left=False)  # batch x z_dim
implicit_reward = (B_inv_conv * z).sum(dim=-1)   # batch
target_Q = implicit_reward + discount * next_Q

Qs = (Fs * z).sum(dim=-1)                        # num_parallel x batch
q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, target_Q)
```

$$r_{\text{implicit}}(s') = B(s') \cdot \text{Cov}(B)^{-1} \cdot z$$

이는 FB 표현에서 암묵적으로 유도되는 보상 신호이다. B의 공분산의 역행렬을 통해 정규화된 보상을 계산한다.

**Step 6: 역전파 및 옵티마이저 스텝** (272-279행)

```python
self.forward_optimizer.zero_grad(set_to_none=True)
self.backward_optimizer.zero_grad(set_to_none=True)
fb_loss.backward()
if clip_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
    torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
self.forward_optimizer.step()
self.backward_optimizer.step()
```

FB 손실 하나로 Forward와 Backward를 **동시에** 역전파하여 양쪽 옵티마이저로 업데이트한다. gradient clipping은 선택적.

### 2.3 update_actor() - 정책 업데이트

기본 FBAgent에서의 Actor 업데이트(308-326행)는 TD3 스타일이다:

```python
def update_td3_actor(self, obs, z, clip_grad_norm):
    dist = self._model._actor(obs, z, self._model.cfg.actor_std)
    action = dist.sample(clip=self.cfg.train.stddev_clip)
    Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
    Qs = (Fs * z).sum(-1)                           # num_parallel x batch
    _, _, Q = self.get_targets_uncertainty(Qs, actor_pessimism_penalty)
    actor_loss = -Q.mean()
```

$$Q_{FB}(s, z, a) = F(s, z, a) \cdot z$$

$$\mathcal{L}_{actor}^{FB} = -\mathbb{E}_{s, z}\left[Q_{FB}(s, z, \pi(s, z))\right]$$

Forward Map의 출력과 현재 task descriptor $z$의 내적이 Q-value가 된다. 이것이 FB 표현의 핵심 통찰: **$z$에 따라 다른 목표에 대한 가치를 즉시 계산**할 수 있다. 새로운 목표가 주어지면 $z$만 바꾸면 된다.

### 2.4 옵티마이저 설정

세 개의 독립된 Adam 옵티마이저(96-113행):

```python
self.backward_optimizer = Adam(self._model._backward_map.parameters(), lr=lr_b)
self.forward_optimizer = Adam(self._model._forward_map.parameters(), lr=lr_f)
self.actor_optimizer = Adam(self._model._actor.parameters(), lr=lr_actor)
```

FB 손실은 Forward와 Backward를 동시에 역전파하여 양쪽 옵티마이저로 업데이트한다. Actor는 별도로 업데이트된다. `capturable` 파라미터는 CudaGraph 사용 시 옵티마이저 상태를 GPU 메모리에 유지한다.

### 2.5 act() - 추론 시 액션 생성

```python
def act(self, obs, z, mean=True):
    return self._model.act(obs, z, mean)
```

`FBModel.act()` (128-132행)는 관측값을 정규화한 후 Actor 네트워크를 통해 액션 분포를 생성한다. `mean=True`이면 분포의 평균(결정적 정책), `False`이면 샘플링(탐색). 추론 시에는 `mean=True`로 결정적 정책을 사용한다.

### 2.6 get_targets_uncertainty() - 비관적 추정

여러 병렬 네트워크의 예측을 결합하여 불확실성을 고려한 추정치를 반환한다(328-343행):

```python
def get_targets_uncertainty(self, preds, pessimism_penalty):
    preds_mean = preds.mean(dim=0)                          # 앙상블 평균
    preds_uns = preds.unsqueeze(dim=0)                      # 1 x n_parallel x ...
    preds_uns2 = preds.unsqueeze(dim=1)                     # n_parallel x 1 x ...
    preds_diffs = torch.abs(preds_uns - preds_uns2)         # 모든 쌍의 차이
    num_parallel_scaling = N**2 - N                          # 쌍의 수
    preds_unc = preds_diffs.sum(dim=(0, 1)) / num_parallel_scaling  # 평균 차이
    return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc
```

$$\hat{Q} = \bar{Q} - \lambda \cdot \text{Unc}(Q)$$

$$\text{Unc}(Q) = \frac{1}{N(N-1)} \sum_{i \neq j} |Q_i - Q_j|$$

이는 **앙상블 기반 비관적 추정**으로, 불확실한 상태-행동 쌍에 대해 보수적으로 행동하게 한다. $\lambda = 0$이면 순수 평균, $\lambda > 0$이면 불확실성이 높을수록 가치를 낮게 추정한다.

### 2.7 maybe_update_rollout_context() - 롤아웃 z 관리

환경 롤아웃 시 각 병렬 환경에 할당할 $z$ 벡터를 관리한다(354-378행):

```python
def maybe_update_rollout_context(self, z, step_count, replay_buffer):
    mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
    if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
        new_z = self.z_buffer.sample(z.shape[0])  # 학습에서 사용된 z 재활용
    else:
        new_z = self._model.sample_z(z.shape[0])   # 랜덤 z
    z = torch.where(mask_reset_z, new_z, z)
```

- `update_z_every_step` 스텝마다 z를 교체
- `use_mix_rollout=True`이면 ZBuffer에서 이전 학습에 사용된 z를 재활용
- `rollout_expert_trajectories=True`이면 일부 환경(비율: `rollout_expert_trajectories_percentage`)에서 전문가 궤적 z를 사용하여 모방 학습 데이터를 적극 수집

전문가 롤아웃(363-370행)의 경우, `_sample_tracking_z()`가 전문가 버퍼에서 시퀀스를 샘플링하고 B로 인코딩한 후 시간 축 sliding window 평균을 적용한다:

```python
def _sample_tracking_z(self, replay_buffer, batch_dim, traj_length):
    batch = replay_buffer["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)
    z = self._model.backward_map(batch["next"]["observation"])  # NT x z_dim
    z = z.view(batch_dim, traj_length, z.shape[-1])             # N x T x z_dim
    for step in range(traj_length):
        end_idx = min(step + self.cfg.model.seq_length, traj_length)
        z[:, step] = z[:, step:end_idx].mean(dim=1)  # 미래 seq_length 프레임 평균
    return self._model.project_z(z)
```

### 2.8 Z 정규화 (project_z)

모든 z 벡터는 정규화하여 일정한 노름을 유지한다(FBModel, 123-126행):

```python
def project_z(self, z):
    if self.cfg.archi.norm_z:
        z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
    return z
```

$$z_{\text{proj}} = \sqrt{d_z} \cdot \frac{z}{\|z\|}$$

노름이 $\sqrt{d_z}$로 고정되어, z 공간에서의 단위 초구면 위에 분포한다. 예를 들어 `z_dim=100`이면 $\|z\| = 10$.

---

## 3. fb_cpr/agent.py - FB-CPR 에이전트 (핵심)

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb_cpr/agent.py`

FB-CPR은 FBAgent를 상속하여 **Contrastive Policy Regularization**을 추가한다. 전문가 시연 데이터를 활용하여 정책의 품질을 높이는 것이 핵심이다.

**추가된 설정** (`FBcprAgentTrainConfig`, 21-38행):

```python
class FBcprAgentTrainConfig(FBAgentTrainConfig):
    lr_discriminator: float = 1e-4       # 판별자 학습률
    lr_critic: float = 1e-4              # Critic 학습률
    critic_target_tau: float = 0.005     # Critic 타겟 소프트 업데이트 계수
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1                 # 판별자 보상 가중치
    scale_reg: bool = True               # Q_fb로 스케일링 여부
    expert_asm_ratio: float = 0          # z 샘플링 시 전문가 인코딩 비율
    relabel_ratio: float | None = 1      # z 리라벨링 비율 (1 = 전부)
    grad_penalty_discriminator: float = 10.0  # WGAN 그래디언트 페널티 계수
    weight_decay_discriminator: float = 0.0
```

**추가된 구성 요소**: `_critic`(Critic 네트워크), `_discriminator`(판별자), 그리고 각각의 옵티마이저 및 타겟 네트워크.

### 3.1 update() - 전체 학습 루프

FB-CPR의 `update()`(170-269행)는 한 번의 학습 스텝에서 수행하는 모든 과정을 조율한다.

```python
def update(self, replay_buffer, step):
    # (1) 데이터 샘플링
    expert_batch = replay_buffer["expert_slicer"].sample(batch_size)  # 전문가 시연
    train_batch = replay_buffer["train"].sample(batch_size)            # 온라인 경험

    # (2) 관측값 정규화 (running stats 업데이트 후 정규화)
    self._model._obs_normalizer(train_obs)       # running mean/var 업데이트 (train 모드)
    self._model._obs_normalizer(train_next_obs)
    with torch.no_grad(), eval_mode(self._model._obs_normalizer):
        train_obs = self._model._obs_normalizer(train_obs)        # 정규화 (eval 모드)
        train_next_obs = self._model._obs_normalizer(train_next_obs)
        expert_obs = self._model._obs_normalizer(expert_obs)
        expert_next_obs = self._model._obs_normalizer(expert_next_obs)

    # (3) 전문가 z 인코딩
    expert_z = self.encode_expert(next_obs=expert_next_obs)

    # (4) 판별자 업데이트
    metrics = self.update_discriminator(expert_obs, expert_z, train_obs, train_z, grad_penalty)

    # (5) Mixed z 샘플링 및 리라벨링
    z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z)
    if relabel_ratio is not None:
        mask = torch.rand(...) <= relabel_ratio
        train_z = torch.where(mask, z, train_z)

    # (6) FB 업데이트 (상속)
    metrics.update(self.update_fb(train_obs, train_action, discount, train_next_obs, train_next_obs, train_z))

    # (7) Critic 업데이트 (새로 추가)
    metrics.update(self.update_critic(train_obs, train_action, discount, train_next_obs, train_z))

    # (8) Actor 업데이트 (오버라이드)
    metrics.update(self.update_actor(train_obs, train_action, train_z))

    # (9) 타겟 네트워크 소프트 업데이트
    _soft_update_params(forward_map, target_forward_map, tau=0.01)
    _soft_update_params(backward_map, target_backward_map, tau=0.01)
    _soft_update_params(critic, target_critic, tau=0.005)
```

**핵심 관찰점**:
- `update_fb()`에 전달되는 `goal`이 `train_next_obs`라는 것. 즉, 현재 배치의 다음 관측값을 목표로 사용하여 FB 행렬의 대각/비대각 구조를 학습한다.
- `train_z`는 원래 롤아웃에서 사용된 z이지만, `relabel_ratio`에 따라 새로 샘플링된 mixed z로 교체된다.
- 전문가 데이터는 `expert_slicer`라는 특수 샘플러에서 가져오며, 이는 시퀀스 단위로 연속 샘플링을 지원한다.

### 3.2 sample_mixed_z() - z 믹싱 전략

FB-CPR에서 z 분포는 세 가지 소스에서 혼합된다(127-150행):

```python
@torch.no_grad()
def sample_mixed_z(self, train_goal, expert_encodings):
    z = self._model.sample_z(batch_size)  # 기본: 가우시안 -> project_z

    # 확률적 혼합
    prob = [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm]
    mix_idxs = torch.multinomial(prob, num_samples=batch_size, replacement=True).reshape(-1, 1)

    # (1) Goal 인코딩: 온라인 경험의 next_obs를 B로 인코딩
    perm = torch.randperm(batch_size)
    train_goal = train_goal[perm]
    goals = self._model._backward_map(train_goal)
    goals = self._model.project_z(goals)
    z = torch.where(mix_idxs == 0, goals, z)

    # (2) Expert 인코딩: 전문가 궤적의 B 인코딩 (셔플)
    perm = torch.randperm(batch_size)
    z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

    # (3) Uniform: 나머지는 랜덤 z 유지
    return z
```

**FBAgent의 sample_mixed_z와 차이점**: 기본 FBAgent는 goal과 uniform 두 가지만 혼합했지만, FB-CPR은 expert 소스를 추가하여 세 가지를 확률적으로 혼합한다. `torch.multinomial`로 각 배치 요소에 대해 어떤 소스를 사용할지 확률적으로 선택한다.

**세 가지 소스의 의미**:

| 소스 | 비율 키 | 의미 |
|------|---------|------|
| Goal 인코딩 | `train_goal_ratio` | 이미 방문한 상태를 목표로 -- exploitation |
| Expert 인코딩 | `expert_asm_ratio` | 전문가의 행동 패턴 -- imitation |
| Uniform | 나머지 | 탐색을 위한 랜덤 z -- exploration |

### 3.3 encode_expert() - 전문가 궤적 인코딩

전문가 궤적의 연속된 관측값을 Backward Map으로 인코딩하여 단일 z 벡터로 요약한다(152-168행):

```python
@torch.no_grad()
def encode_expert(self, next_obs):
    B_expert = self._model._backward_map(next_obs).detach()  # batch x d
    # 시퀀스 단위로 reshape: batch = N * L
    B_expert = B_expert.view(
        batch_size // seq_length,  # N: 궤적 수
        seq_length,                # L: 시퀀스 길이
        B_expert.shape[-1]         # d: z 차원
    )  # N x L x d
    z_expert = B_expert.mean(dim=1)  # N x d (시퀀스 내 평균)
    z_expert = self._model.project_z(z_expert)
    z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)  # batch x d
    return z_expert
```

**핵심 아이디어**: 전문가 궤적의 여러 타임스텝을 B로 인코딩한 후 **시퀀스 내 평균**을 내면, 그 궤적의 "의도"를 나타내는 z 벡터가 된다. `repeat_interleave`로 같은 궤적의 모든 타임스텝에 동일한 z를 할당한다.

이것이 가능한 이유: B가 학습을 통해 상태를 task-relevant한 잠재 공간으로 인코딩하게 되므로, 궤적의 B 인코딩 평균은 그 궤적이 "무엇을 하려는지"를 요약하게 된다.

### 3.4 update_discriminator() - GAN 스타일 판별기

전문가 (obs, z) 쌍과 온라인 정책의 (obs, z) 쌍을 구분하는 이진 분류기를 학습한다(333-365행):

```python
def update_discriminator(self, expert_obs, expert_z, train_obs, train_z, grad_penalty):
    # 로짓 계산
    expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
    unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)

    # Binary Cross Entropy (동치 형태)
    expert_loss = -torch.nn.functional.logsigmoid(expert_logits)      # 전문가 -> 1
    unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)   # 정책 -> 0
    loss = torch.mean(expert_loss + unlabeled_loss)

    # WGAN Gradient Penalty
    if grad_penalty is not None:
        wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
        loss += grad_penalty * wgan_gp

    self.discriminator_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self.discriminator_optimizer.step()
```

**손실 함수**:

$$\mathcal{L}_{disc} = -\mathbb{E}_{expert}[\log \sigma(D(s, z))] + \mathbb{E}_{policy}[\log(1 + e^{D(s, z)})]$$

여기서 $-\log \sigma(x) = \log(1 + e^{-x})$이고 $\text{softplus}(x) = \log(1 + e^x)$이므로, 이것은 표준 이진 교차 엔트로피와 동치이다. 코드의 주석에서도 "these are equivalent to binary cross entropy"라고 명시하고 있다.

**Discriminator 네트워크의 보상 변환** (`nn_models.py`, 365-369행):

```python
class Discriminator(nn.Module):
    def compute_reward(self, obs, z, eps=1e-7):
        s = self.forward(obs, z)        # sigmoid 출력: D(s,z)
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward
```

$$r_{disc}(s, z) = \log D(s, z) - \log(1 - D(s, z))$$

이것은 **AIRL(Adversarial Inverse RL) 스타일의 보상 변환**으로, 판별자의 sigmoid 출력을 로그 비율(log-odds)로 변환한다. 전문가에 가까울수록 양수, 정책에 가까울수록 음수.

### 3.5 gradient_penalty_wgan() - WGAN 그래디언트 페널티

학습 안정성을 위한 Wasserstein GAN 스타일의 그래디언트 페널티(272-331행):

```python
@torch.compiler.disable  # autograd.grad는 torch.compile과 호환 불가
def gradient_penalty_wgan(self, real_obs, real_z, fake_obs, fake_z):
    alpha = torch.rand(batch_size, 1)

    # 전문가와 정책 데이터의 보간 (dict obs도 지원)
    interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
    interpolated_z = (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)

    d_interpolates = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)

    # 보간점에서의 그래디언트 노름 계산
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolated_obs_list + [interpolated_z],
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True,
        allow_unused=True,  # 판별자가 일부 obs를 사용하지 않을 수 있음
    )
    gradients = [g for g in gradients if g is not None]  # None 필터링
    cat_gradients = torch.cat(gradients, dim=1)
    gradient_penalty = ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
```

$$\mathcal{L}_{GP} = \lambda_{GP} \cdot \mathbb{E}_{\hat{x}}\left[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2\right]$$

여기서 $\hat{x} = \alpha x_{expert} + (1 - \alpha) x_{policy}$, $\alpha \sim U(0, 1)$

그래디언트의 L2 노름이 1에 가깝게 되도록 정규화하여, **판별자가 1-Lipschitz 조건**을 만족하도록 한다. 이는 학습의 안정성과 수렴을 보장한다. `@torch.compiler.disable` 데코레이터로 `autograd.grad` 사용 시 compile 비활성화.

### 3.6 update_critic() - TD 학습

판별자 보상을 기반으로 표준 TD(Temporal Difference) 학습을 수행한다(367-405행):

```python
def update_critic(self, obs, action, discount, next_obs, z):
    num_parallel = self.cfg.model.archi.critic.num_parallel
    with torch.no_grad():
        # (1) 판별자로부터 보상 계산
        reward = self._model._discriminator.compute_reward(obs=obs, z=z)

        # (2) 다음 상태에서의 행동 샘플링
        dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
        next_action = dist.sample(clip=self.cfg.train.stddev_clip)

        # (3) 타겟 Q 계산 (비관적 추정)
        next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
        Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, critic_pessimism_penalty)
        target_Q = reward + discount * next_V
        expanded_targets = target_Q.expand(num_parallel, -1, -1)

    # (4) 현재 Q-value 계산 및 MSE 손실
    Qs = self._model._critic(obs, z, action)  # num_parallel x batch x 1
    critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

    self.critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    self.critic_optimizer.step()
```

$$\mathcal{L}_{critic} = \frac{N_p}{2} \cdot \mathbb{E}\left[\sum_{k=1}^{N_p} (Q_k(s, z, a) - y)^2\right]$$

$$y = r_{disc}(s, z) + \gamma \cdot (\bar{Q}(s', z, a') - \lambda_{pes} \cdot \text{Unc}(Q(s', z, a')))$$

- $r_{disc}$: 판별자 보상 (전문가와 유사할수록 높음)
- $\gamma$: 할인 계수
- 비관적 추정으로 over-estimation 방지
- `num_parallel`개의 앙상블 Q-network를 동시에 학습

### 3.7 update_actor() - 정책 그래디언트 (FB-CPR 버전)

FB-CPR의 Actor 업데이트(407-443행)는 **FB Q-value**와 **Discriminator Q-value** 두 가지를 결합한다:

```python
def update_actor(self, obs, action, z, clip_grad_norm):
    dist = self._model._actor(obs, z, self._model.cfg.actor_std)
    action = dist.sample(clip=self.cfg.train.stddev_clip)

    # (1) Discriminator 보상 기반 Q-value
    Qs_discriminator = self._model._critic(obs, z, action)
    _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, actor_pessimism_penalty)

    # (2) FB 보상 기반 Q-value
    Fs = self._model._forward_map(obs, z, action)
    Qs_fb = (Fs * z).sum(-1)
    _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, actor_pessimism_penalty)

    # (3) 가중 결합
    weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
    actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()

    self.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
    self.actor_optimizer.step()
```

$$\mathcal{L}_{actor}^{CPR} = -\underbrace{\alpha \cdot |Q_{FB}|_{\text{mean}} \cdot Q_{disc}(s, z, \pi(s, z))}_{\text{CPR 정규화 (전문가 모방)}} - \underbrace{Q_{FB}(s, z, \pi(s, z))}_{\text{FB 보상 (목표 추적)}}$$

여기서:
- $\alpha$는 `reg_coeff` (기본값 1)
- $|Q_{FB}|_{\text{mean}}$은 스케일링 계수 (`.detach()`됨, `scale_reg=True`일 때)
- `scale_reg`가 True이면 $Q_{disc}$의 스케일을 $Q_{FB}$에 맞춰줌 -- 두 보상의 크기가 다를 수 있으므로

**두 보상의 역할**:
- $Q_{FB}$: z 조건에 맞는 행동을 하도록 -- "목표 달성"
- $Q_{disc}$: 전문가와 유사한 행동을 하도록 -- "자연스러운 동작"

### 3.8 soft_update_params() - 타겟 네트워크 업데이트

```python
# nn_models.py, 80-82행
def _soft_update_params(net_params, target_net_params, tau):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)
```

$$\theta_{target} \leftarrow (1 - \tau) \cdot \theta_{target} + \tau \cdot \theta_{online}$$

`torch._foreach_*` 연산을 사용하여 파라미터 리스트를 한 번에 업데이트한다 (개별 for 루프보다 효율적). FB-CPR에서의 $\tau$ 값:
- Forward/Backward Map 타겟: $\tau = 0.01$ (느린 업데이트)
- Critic 타겟: $\tau = 0.005$ (더 느린 업데이트)

---

## 4. fb_cpr_aux/agent.py - 보조 손실 에이전트

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb_cpr_aux/agent.py`

### 4.1 FB-CPR과의 차이점

`FBcprAuxAgent`는 `FBcprAgent`를 상속하여 **보조 보상(Auxiliary Reward)** 기반의 추가 Critic을 도입한다. 이를 통해 환경에서 정의한 물리적 페널티(토크 한계, 미끄러짐, 관절 한계 등)를 학습에 반영한다.

**추가된 설정** (21-35행):

```python
class FBcprAuxAgentTrainConfig(FBcprAgentTrainConfig):
    lr_aux_critic: float = 1e-4           # 보조 Critic 학습률
    reg_coeff_aux: float = 1.0            # 보조 보상 가중치
    aux_critic_pessimism_penalty: float = 0.5

class FBcprAuxAgentConfig(BaseConfig):
    aux_rewards: list[str] = []                   # 사용할 보조 보상 이름 리스트
    aux_rewards_scaling: dict[str, float] = {}    # 보상별 스케일링 계수
```

**구조적 추가 사항**:
- `_aux_critic` 네트워크 + `_target_aux_critic` (deep copy)
- `_aux_reward_normalizer`: 보조 보상의 스케일을 정규화
- `aux_critic_optimizer`: Adam 옵티마이저

### 4.2 보조 보상 Critic

`update_aux_critic()`(212-251행)은 `update_critic()`과 동일한 TD 학습 구조이지만, 판별자 보상 대신 환경의 보조 보상을 사용한다:

```python
def update_aux_critic(self, obs, action, discount, aux_reward, next_obs, z):
    with torch.no_grad():
        dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
        next_action = dist.sample(clip=self.cfg.train.stddev_clip)
        next_Qs = self._model._target_aux_critic(next_obs, z, next_action)
        Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, aux_critic_pessimism_penalty)
        target_Q = aux_reward + discount * next_V       # <-- 판별자 대신 보조 보상 사용
        expanded_targets = target_Q.expand(num_parallel, -1, -1)

    Qs = self._model._aux_critic(obs, z, action)
    aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

    self.aux_critic_optimizer.zero_grad(set_to_none=True)
    aux_critic_loss.backward()
    self.aux_critic_optimizer.step()
```

**보조 보상 계산** (update() 내부, 157-167행):

```python
aux_reward = torch.zeros((batch_size, 1), device=self.device)
for aux_reward_name in self.cfg.aux_rewards:
    metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
    aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name]
aux_reward = self._model._aux_reward_normalizer(aux_reward)  # 보상 정규화
```

여러 보조 보상의 가중합을 계산하고, 보상 정규화기(`RewardNormalizer`)를 통해 스케일을 조정한다. 각 보상은 `aux_rewards_scaling` 딕셔너리에서 개별 가중치를 가진다.

### 4.3 추가된 손실 함수 - Actor 업데이트

FB-CPR-Aux의 Actor 손실(253-298행)은 **세 가지 Q-value**를 결합한다:

```python
def update_actor(self, obs, action, z, clip_grad_norm):
    dist = self._model._actor(obs, z, self._model.cfg.actor_std)
    action = dist.sample(clip=self.cfg.train.stddev_clip)

    # (1) Discriminator Q
    Qs_discriminator = self._model._critic(obs, z, action)
    _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, actor_pessimism_penalty)

    # (2) Auxiliary Q (새로 추가)
    Qs_aux = self._model._aux_critic(obs, z, action)
    _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, actor_pessimism_penalty)

    # (3) FB Q
    Fs = self._model._forward_map(obs, z, action)
    Qs_fb = (Fs * z).sum(-1)
    _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, actor_pessimism_penalty)

    # 가중 결합
    weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
    actor_loss = (
        -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
        - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
        - Q_fb.mean()
    )
```

$$\mathcal{L}_{actor}^{Aux} = -\underbrace{\alpha_{disc} \cdot w \cdot Q_{disc}}_{\text{전문가 모방}} - \underbrace{\alpha_{aux} \cdot w \cdot Q_{aux}}_{\text{물리적 제약}} - \underbrace{Q_{FB}}_{\text{목표 추적}}$$

여기서 $w = |\bar{Q}_{FB}|$ (스케일 정규화), $\alpha_{disc}$ = `reg_coeff`, $\alpha_{aux}$ = `reg_coeff_aux`

**세 Q-value의 역할 분담**:
- $Q_{FB}$: "z가 지시하는 목표를 달성하라" (비지도 학습)
- $Q_{disc}$: "전문가처럼 자연스럽게 움직여라" (모방 학습)
- $Q_{aux}$: "물리적으로 안전하게 움직여라" (환경 제약)

**FB-CPR과의 구조적 차이 요약**:

| 항목 | FB-CPR | FB-CPR-Aux |
|------|--------|------------|
| Actor 손실 항 | $Q_{FB} + Q_{disc}$ | $Q_{FB} + Q_{disc} + Q_{aux}$ |
| Critic 수 | 1 (discriminator) | 2 (discriminator + auxiliary) |
| 타겟 네트워크 | F, B, Critic 각각의 target | + Aux_Critic_target 추가 |
| 보상 소스 | 판별자만 | 판별자 + 환경 보조 보상 |
| 보상 정규화 | 없음 | `_aux_reward_normalizer` |
| 옵티마이저 | 5개 | 6개 (+`aux_critic_optimizer`) |

---

## 5. 학습 흐름 전체 그림

### 5.1 한 스텝의 학습 과정 (순서도)

```
+---------------------------------------------------------------------+
|                    FBcprAuxAgent.update() 한 스텝                      |
+---------------------------------------------------------------------+
|                                                                       |
|  [1] 데이터 샘플링                                                     |
|      +-- expert_batch <- replay_buffer["expert_slicer"].sample()     |
|      +-- train_batch  <- replay_buffer["train"].sample()             |
|                                                                       |
|  [2] 관측값 정규화                                                     |
|      +-- _obs_normalizer(train_obs)      <- running stats 업데이트    |
|      +-- 모든 obs를 정규화 (eval 모드, no_grad)                       |
|                                                                       |
|  [3] 전문가 z 인코딩                                                   |
|      expert_z = encode_expert(expert_next_obs)                        |
|      +-- B(expert_obs) -> reshape(N,L,d) -> 시퀀스 평균 -> project_z |
|                                                                       |
|  [4] 판별자 업데이트 <- update_discriminator()                          |
|      +-- expert (obs, z) -> 1 에 가깝게                                |
|      +-- policy (obs, z) -> 0 에 가깝게                                |
|      +-- + WGAN Gradient Penalty                                      |
|                                                                       |
|  [5] Mixed z 샘플링 <- sample_mixed_z()                                |
|      +-- p_goal: 온라인 경험의 goal 인코딩                             |
|      +-- p_expert: 전문가 궤적 인코딩                                  |
|      +-- p_uniform: 랜덤 z                                            |
|      +-- z_buffer.add(z) 저장                                         |
|                                                                       |
|  [6] z 리라벨링                                                        |
|      train_z = where(rand < relabel_ratio, mixed_z, original_z)       |
|                                                                       |
|  [7] FB 업데이트 <- update_fb()                                        |
|      +-- M = F(s,z,a) @ B(s')^T                                      |
|      +-- FB 대각 손실 + 비대각 손실                                    |
|      +-- + 직교성 손실 (B의 Cov -> I)                                  |
|      +-- 역전파: forward_optimizer + backward_optimizer               |
|                                                                       |
|  [8] Critic 업데이트 <- update_critic()                                 |
|      +-- reward = discriminator.compute_reward(obs, z)                |
|      +-- TD 학습: Q(s,z,a) -> r + gamma*V(s')                        |
|      +-- 역전파: critic_optimizer                                     |
|                                                                       |
|  [9] 보조 보상 Critic 업데이트 <- update_aux_critic()                   |
|      +-- aux_reward = sum(scale_i * reward_i)                         |
|      +-- aux_reward = _aux_reward_normalizer(aux_reward)              |
|      +-- TD 학습: Q_aux(s,z,a) -> r_aux + gamma*V_aux(s')            |
|      +-- 역전파: aux_critic_optimizer                                 |
|                                                                       |
|  [10] Actor 업데이트 <- update_actor()                                  |
|       loss = -alpha_disc*w*Q_disc - alpha_aux*w*Q_aux - Q_fb          |
|       +-- 역전파: actor_optimizer                                     |
|                                                                       |
|  [11] 타겟 네트워크 소프트 업데이트                                      |
|       +-- F_target   <- (1-tau_fb)*F_target + tau_fb*F                |
|       +-- B_target   <- (1-tau_fb)*B_target + tau_fb*B                |
|       +-- Q_target   <- (1-tau_cr)*Q_target + tau_cr*Q                |
|       +-- Q_aux_tgt  <- (1-tau_cr)*Q_aux_tgt + tau_cr*Q_aux          |
|                                                                       |
+---------------------------------------------------------------------+
```

### 5.2 손실 함수 총정리

| # | 손실 | 수식 | 최적화 대상 | 역할 |
|---|------|------|------------|------|
| 1 | FB Loss (대각) | $-N_p \cdot \text{mean}(\text{diag}(M - \gamma M_{target}))$ | F, B | Successor measure 대각 최대화 |
| 2 | FB Loss (비대각) | $\frac{1}{2\|\text{off-diag}\|}\sum_{i \neq j} (\text{diff}_{ij})^2$ | F, B | Successor measure 비대각 최소화 |
| 3 | 직교성 Loss | $\frac{1}{2\|\text{off-diag}\|}\sum_{i \neq j}(B_i \cdot B_j)^2 - \text{mean}(\|B_i\|^2)$ | B | B 출력 직교 정규화 |
| 4 | Discriminator Loss | $-\mathbb{E}[\log \sigma(D_{exp})] + \mathbb{E}[\log(1+e^{D_{pol}})]$ | D | 전문가/정책 판별 |
| 5 | GP Loss | $\lambda_{GP}(\|\nabla D(\hat{x})\|_2 - 1)^2$ | D | Lipschitz 제약 |
| 6 | Critic Loss | $\frac{N_p}{2}\|Q - (r_{disc} + \gamma V')\|^2$ | Q | 판별자 보상 가치 추정 |
| 7 | Aux Critic Loss | $\frac{N_p}{2}\|Q_{aux} - (r_{aux} + \gamma V'_{aux})\|^2$ | $Q_{aux}$ | 보조 보상 가치 추정 |
| 8 | Actor Loss | $-\alpha_{disc} w Q_{disc} - \alpha_{aux} w Q_{aux} - Q_{FB}$ | $\pi$ | 정책 최적화 |

### 5.3 하이퍼파라미터 정리

**옵티마이저 관련**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `lr_f` | 1e-4 | Forward Map 학습률 |
| `lr_b` | 1e-4 | Backward Map 학습률 |
| `lr_actor` | 1e-4 | Actor 학습률 |
| `lr_critic` | 1e-4 | Critic 학습률 |
| `lr_discriminator` | 1e-4 | Discriminator 학습률 |
| `lr_aux_critic` | 1e-4 | Aux Critic 학습률 |
| `weight_decay` | 0.0 | L2 정규화 (기본 비활성) |
| `weight_decay_discriminator` | 0.0 | 판별자 전용 weight decay |

**학습 구조**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `batch_size` | 1024 | 미니배치 크기 |
| `discount` | 0.99 | 할인 계수 $\gamma$ |
| `fb_target_tau` | 0.01 | F/B 타겟 소프트 업데이트 $\tau$ |
| `critic_target_tau` | 0.005 | Critic 타겟 소프트 업데이트 $\tau$ |
| `stddev_clip` | 0.3 | 액션 분포 클리핑 범위 |
| `actor_std` | 0.2 | Actor 출력 표준편차 |
| `clip_grad_norm` | 0.0 | 그래디언트 클리핑 (0=비활성) |

**CPR 관련**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `reg_coeff` | 1 | 판별자 보상 가중치 $\alpha_{disc}$ |
| `reg_coeff_aux` | 1.0 | 보조 보상 가중치 $\alpha_{aux}$ |
| `scale_reg` | True | $\|Q_{FB}\|$로 정규화 항 스케일링 |
| `train_goal_ratio` | 0.5 | z 혼합 시 goal 비율 |
| `expert_asm_ratio` | 0 | z 혼합 시 expert 비율 |
| `relabel_ratio` | 1 | z 리라벨링 비율 (1 = 전부 리라벨) |
| `grad_penalty_discriminator` | 10.0 | WGAN GP 계수 $\lambda_{GP}$ |

**비관적 추정**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `fb_pessimism_penalty` | 0.0 | FB 타겟의 비관적 페널티 |
| `actor_pessimism_penalty` | 0.5 | Actor Q의 비관적 페널티 |
| `critic_pessimism_penalty` | 0.5 | Critic 타겟의 비관적 페널티 |
| `aux_critic_pessimism_penalty` | 0.5 | Aux Critic의 비관적 페널티 |

**잠재 공간**:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `z_dim` | 100 | 잠재 벡터 차원 |
| `norm_z` | True | z 정규화 활성화 ($\|z\| = \sqrt{d_z}$) |
| `seq_length` | 1 | 전문가 인코딩 시퀀스 길이 |
| `z_buffer_size` | 10000 | z 버퍼 크기 |
| `update_z_every_step` | 150 | 롤아웃 z 교체 주기 |
| `use_mix_rollout` | False | z_buffer에서 z 재활용 여부 |

---

## 6. 핵심 수학적 개념

### 6.1 Forward-Backward 표현 학습

**Successor Measure**: 상태 $s$에서 행동 $a$를 취한 후, 미래에 상태 $s'$를 방문할 할인된 기대 빈도를 나타내는 측도:

$$M^\pi(s, a, s') = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t \delta(s_t = s') \mid s_0 = s, a_0 = a\right]$$

FB 표현은 이 측도를 저차원으로 분해한다:

$$M^\pi(s, a, s') \approx F(s, z, a) \cdot B(s')^T$$

여기서 $F: \mathcal{S} \times \mathcal{Z} \times \mathcal{A} \to \mathbb{R}^d$, $B: \mathcal{S} \to \mathbb{R}^d$

**핵심 성질**: 임의의 보상 함수 $r(s')$에 대해, Q-value를 다음과 같이 표현할 수 있다:

$$Q(s, z, a) = F(s, z, a) \cdot z$$

여기서 $z = \int B(s') r(s') ds'$ (보상의 B-인코딩). 따라서 **단일 모델로 무한히 많은 보상 함수에 대한 정책을 즉시 생성**할 수 있다. 이것이 "Foundation Model" 접근의 핵심이다.

**Bellman 일관성**: FB 손실의 대각/비대각 구조는 다음 Bellman 방정식에서 유래한다:

$$M(s, a, s') = \delta(s = s') + \gamma \mathbb{E}_{a' \sim \pi}[M(s', a', s')]$$

배치에서 $s_i, s_j'$에 대해:
- $i = j$ (대각): $M_{ii} = 1 + \gamma M_{target,ii}$ → diff의 대각이 양수여야 하므로 **최대화**
- $i \neq j$ (비대각): $M_{ij} = 0 + \gamma M_{target,ij}$ → diff가 0이 되어야 하므로 **MSE로 최소화**

이 비대칭적 처리가 FB 표현을 올바르게 학습시키는 핵심이다. 만약 둘 다 MSE로 처리하면 trivial solution(모든 출력이 0)으로 수렴할 수 있다.

### 6.2 CPR (Contrastive Policy Regularization)

CPR은 FB 정책에 전문가 시연을 활용한 정규화를 추가하는 기법이다.

**동기**: 순수 FB 학습만으로는 현실적인 동작을 보장하기 어렵다. $z$가 지시하는 목표를 달성하더라도 비현실적인 자세로 할 수 있다. 전문가 데이터를 활용하되, 특정 task에 종속되지 않는 방식이 필요하다.

**CPR의 접근**: 판별자 $D(s, z)$가 관측 $s$와 task descriptor $z$의 **쌍**이 전문가에서 온 것인지 정책에서 온 것인지를 구분한다. 핵심은 **z도 함께 조건**으로 사용한다는 점이다.

전문가 데이터의 z는 `encode_expert()`로 생성한다:

$$z_{expert} = \text{project\_z}\left(\frac{1}{L}\sum_{t=1}^{L} B(s_t^{expert})\right)$$

이렇게 하면 판별자는 단순히 "전문가처럼 보이는가"가 아니라 **"이 z로 조건화했을 때 전문가처럼 행동하는가"**를 판단한다. 이를 통해:

1. **z에 대한 일관된 행동을 유도**: z가 같으면 비슷한 동작을 해야 함
2. **전문가 궤적의 다양성을 z 공간으로 전달**: 다양한 전문가 동작이 다양한 z에 매핑
3. **task-agnostic한 정규화 효과**: 특정 task가 아닌 "전문가다운 행동" 전반을 학습

### 6.3 GAN 기반 Discriminator의 역할

판별자는 GAIL(Generative Adversarial Imitation Learning)과 유사하지만 두 가지 핵심 차이점이 있다:

**차이점 1 -- 입력에 z 포함**: $D(s, z)$로 task 조건부 판별. 이를 통해 같은 상태에서도 다른 z에 대해 다른 판단을 내릴 수 있다.

**차이점 2 -- 보상 변환**: sigmoid 출력을 AIRL 스타일 log-odds로 변환:

$$r_{disc}(s, z) = \log \frac{D(s, z)}{1 - D(s, z)}$$

이 보상은 **직접 Actor를 업데이트하지 않고** Critic 네트워크의 TD 학습 타겟으로 사용된다. Actor는 Critic의 Q-value를 최대화한다. 이 중간 단계(Critic)가 중요한 이유:

1. 판별자 보상은 노이즈가 많을 수 있는데, Critic이 이를 시간적으로 평활화
2. Critic의 Q-value는 미래의 누적 보상을 추정하므로 더 안정적인 학습 신호 제공
3. 앙상블 기반 비관적 추정으로 추가적인 안정성 확보

### 6.4 Mixed Z 분포의 의미

z 분포의 세 가지 소스는 각각 다른 학습 목표를 제공한다:

**1) Uniform z (탐색)**:

$$z \sim \mathcal{N}(0, I) \to \text{project\_z}(z) = \sqrt{d_z} \cdot \frac{z}{\|z\|}$$

- 잠재 공간 전체를 균일하게 탐색
- 아직 보지 못한 행동 패턴 발견
- 표현의 일반성(generality) 확보
- 과적합 방지

**2) Goal z (활용)**:

$$z_{goal} = \text{project\_z}(B(s_{next}))$$

- 이미 경험한 상태를 목표로 설정
- 현재 정책으로 도달 가능한 영역 강화 (exploitation)
- Hindsight relabeling 효과: "도달한 곳을 목표로 삼기"
- on-policy에 가까운 학습 효과

**3) Expert z (모방)**:

$$z_{expert} = \text{project\_z}\left(\frac{1}{L}\sum_{t=1}^{L} B(s_t^{expert})\right)$$

- 전문가 궤적의 "의도"를 인코딩
- 전문가 행동을 재현하는 방향으로 z 공간을 조직화
- 현실적 동작 패턴 유도
- 판별자와의 시너지: expert z + expert obs가 positive sample로 사용됨

**z 리라벨링의 의미**: `relabel_ratio=1`이면 배치의 모든 전환(transition)에 대해 z를 새로 샘플링된 mixed z로 교체한다. 이것은 **Hindsight Experience Replay (HER)**와 유사한 효과로, 같은 경험 데이터를 다양한 task context에서 재활용할 수 있게 한다. 원래 롤아웃 시 사용된 z가 아닌 새 z로 학습함으로써:

- 동일한 전환(s, a, s')을 다양한 목표에 대한 경험으로 재해석
- 데이터 효율성을 극대화
- z 공간 전체에 걸친 일반화 능력 향상

---

## 7. 지원 인프라

### 7.1 BaseConfig (base.py)

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/base.py`

모든 설정 클래스의 기반으로, Pydantic의 `BaseModel`을 상속한다:

```python
class BaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="forbid",          # 오타 방지: 정의되지 않은 필드 거부
        strict=True,             # 엄격한 타입 체크 (e.g., "True" -> True 불가)
        use_enum_values=True,    # enum 직렬화 호환
        frozen=True              # 불변 객체 (생성 후 변경 불가)
    )
```

`__init_subclass__`에서 자동으로 `name` 필드의 `Literal` 타입을 설정하여, **다형적 설정 로딩**을 지원한다. 예를 들어 `FBcprAgentConfig`의 `name`은 자동으로 `Literal["FBcprAgentConfig", "FBcprAgent"]`가 된다.

`__getitem__`으로 딕셔너리 스타일 접근(`config["key"]`)도 지원한다.

### 7.2 BaseModel (base_model.py)

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/base_model.py`

모든 신경망 모델의 기반 클래스:

```python
class BaseModel(nn.Module):
    def __init__(self, obs_space, action_dim, config):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = config
```

**주요 기능**:
- `save()`: safetensors 형식으로 모델 가중치 직렬화 + config.json + init_kwargs.json 저장
- `load()`: config에서 모델 재구성 후 가중치 로드
- `_prepare_for_train()`: 타겟 네트워크 deep copy 생성 (서브클래스에서 오버라이드)
- `to()`: 디바이스 이동 시 `self.device` 속성도 함께 업데이트

`load_model()` 함수는 타겟 네트워크 유무를 자동 감지하여 strict/non-strict 로딩을 처리한다:

```python
if strict and any(["target" in key for key in state_dict.keys()]):
    loaded_model._prepare_for_train()  # 타겟 네트워크가 있으면 생성 후 로드
```

### 7.3 ZBuffer (misc/zbuffer.py)

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/misc/zbuffer.py`

학습에 사용된 z 벡터를 순환 버퍼에 저장하여 롤아웃 시 재활용한다:

```python
class ZBuffer:
    def __init__(self, capacity, dim, device):
        self._storage = torch.zeros((capacity, dim), device=device)
        self._idx = 0
        self._is_full = False
```

- `add(data)`: 새 z를 저장. 용량 초과 시 순환적으로 덮어씀 (ring buffer)
- `sample(num)`: `np.random.randint`로 인덱스를 뽑아 랜덤 추출 (`.clone()`으로 복사)
- `empty()`: 아직 데이터가 없는지 확인

이를 통해 학습에서 생성된 "유의미한" z 분포를 롤아웃에 재활용하여, 더 관련성 높은 경험 데이터를 수집할 수 있다 (`use_mix_rollout=True` 시 활용).

### 7.4 Discriminator 네트워크 (nn_models.py)

파일 위치: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/nn_models.py`

```python
class Discriminator(nn.Module):
    def __init__(self, obs_space, z_dim, cfg):
        self.input_filter = cfg.input_filter.build(obs_space)  # obs 일부만 선택 가능
        # MLP: [z_dim + filtered_obs_dim] -> hidden_dim -> ... -> 1
        seq = [nn.Linear(z_dim + filtered_obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        seq += [nn.Linear(hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)
```

세 가지 출력 모드:
- `forward(obs, z)`: sigmoid 확률 $\sigma(D(s, z))$ -- 0~1 범위
- `compute_logits(obs, z)`: 원시 로짓 $D(s, z)$ -- 실수 전체
- `compute_reward(obs, z)`: AIRL 보상 $\log D - \log(1-D)$ -- 실수 전체, clamp로 수치 안정성 확보

`input_filter`는 관측 공간에서 판별에 필요한 부분만 선택할 수 있게 한다 (예: 관절 위치만 사용, 속도 제외 등).

---

## 8. 코드 참조 맵

| 기능 | 파일:행 |
|------|---------|
| FBAgentTrainConfig | `fb/agent.py:24-44` |
| FBAgent.__init__ | `fb/agent.py:65-76` |
| FBAgent.setup_training | `fb/agent.py:90-125` |
| FBAgent.update | `fb/agent.py:161-208` |
| FBAgent.update_fb | `fb/agent.py:216-297` |
| FBAgent.update_td3_actor | `fb/agent.py:308-326` |
| FBAgent.get_targets_uncertainty | `fb/agent.py:328-343` |
| FBAgent.maybe_update_rollout_context | `fb/agent.py:354-378` |
| FBcprAgentTrainConfig | `fb_cpr/agent.py:21-38` |
| FBcprAgent.update | `fb_cpr/agent.py:170-269` |
| FBcprAgent.sample_mixed_z | `fb_cpr/agent.py:127-150` |
| FBcprAgent.encode_expert | `fb_cpr/agent.py:152-168` |
| FBcprAgent.update_discriminator | `fb_cpr/agent.py:333-365` |
| FBcprAgent.gradient_penalty_wgan | `fb_cpr/agent.py:272-331` |
| FBcprAgent.update_critic | `fb_cpr/agent.py:367-405` |
| FBcprAgent.update_actor | `fb_cpr/agent.py:407-443` |
| FBcprAuxAgent.update | `fb_cpr_aux/agent.py:83-210` |
| FBcprAuxAgent.update_aux_critic | `fb_cpr_aux/agent.py:212-251` |
| FBcprAuxAgent.update_actor | `fb_cpr_aux/agent.py:253-298` |
| FBModel (+ project_z, sample_z) | `fb/model.py:72-161` |
| FBcprModel (+ discriminator, critic) | `fb_cpr/model.py:34-61` |
| FBcprAuxModel (+ aux_critic) | `fb_cpr_aux/model.py:30-53` |
| _soft_update_params | `nn_models.py:80-82` |
| weight_init | `nn_models.py:61-72` |
| Discriminator | `nn_models.py:336-369` |
| ZBuffer | `misc/zbuffer.py:12-38` |
| BaseConfig | `base.py:6-36` |
| BaseModel | `base_model.py:74-94` |
