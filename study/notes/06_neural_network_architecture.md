# Phase 1: 신경망 구조 (Neural Network Architecture)

## 1. 개요

BFM-Zero의 신경망은 **Forward-Backward (FB) Representation Learning** 프레임워크 위에 구축되어 있다. 핵심 아이디어는 Successor Feature 분해를 통해 상태-목표 간의 관계를 학습하는 것이다.

전체 구조는 다음과 같은 주요 컴포넌트로 구성된다:

| 컴포넌트 | 역할 | 입력 | 출력 |
|----------|------|------|------|
| **Forward Map (F)** | 상태-행동 쌍의 미래 표현 학습 | obs, z, action | z_dim 벡터 |
| **Backward Map (B)** | 상태(목표)를 latent 공간에 매핑 | obs | z_dim 벡터 |
| **Actor** | 정책 네트워크 (행동 결정) | obs, z | action 분포 |
| **Critic** | Q-value 추정 (FB-CPR 전용) | obs, z, action | 스칼라 값 |
| **Discriminator** | 전문가/비전문가 구분 (FB-CPR 전용) | obs, z | 확률 값 |
| **Obs Normalizer** | 관측값 정규화 | obs | 정규화된 obs |

핵심 수학적 관계: **M(s, g) = F(s, a) * B(g)^T** (Successor Measure 분해)

파일 구조:
- `nn_models.py`: 모든 신경망 빌딩 블록, 가중치 초기화, 유틸리티
- `nn_filters.py`: 딕셔너리 관측값을 텐서로 변환하는 입력 필터
- `nn_filter_models.py`: 관측값의 일부 차원만 사용하는 필터 모델 래퍼
- `fb/model.py`: FB 기본 모델 (F, B, Actor 조합)
- `fb_cpr/model.py`: FB-CPR 모델 (Critic, Discriminator 추가)
- `normalizers.py`: BatchNorm 기반 관측값 정규화


## 2. nn_models.py - 신경망 유틸리티

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/nn_models.py`

### 2.1 가중치 초기화

#### `parallel_orthogonal_(tensor, gain=1)`

병렬 앙상블 레이어(`DenseParallel`)를 위한 직교 초기화 함수이다. 3차원 이상의 텐서에서 각 병렬 슬라이스에 대해 독립적으로 QR 분해를 수행한다.

```python
def parallel_orthogonal_(tensor, gain=1):
    # 2D 텐서는 표준 orthogonal_init으로 처리
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor

    # 3D 이상: 각 병렬 슬라이스에 대해 QR 분해 수행
    n_parallel = tensor.size(0)
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)
    for flat_tensor in torch.unbind(flattened, dim=0):
        q, r = torch.linalg.qr(flat_tensor)
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph  # Haar 분포에 따른 균일화
```

**핵심 원리**: QR 분해 후 `d.sign()`을 곱하는 것은 [Mezzadri 2007](https://arxiv.org/pdf/math-ph/0609050.pdf) 논문의 방법으로, 직교 행렬이 Haar 측도(균일 분포)에 따라 분포하도록 보장한다.

#### `weight_init(m)`

모듈 타입별로 적절한 초기화를 적용하는 콜백 함수이다. `model.apply(weight_init)` 형태로 사용된다.

```python
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)  # 직교 초기화
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)           # 바이어스 0 초기화
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")  # ReLU용 gain 계산
        parallel_orthogonal_(m.weight.data, gain)
```

- `nn.Linear`: 표준 직교 초기화 + 바이어스 0
- `DenseParallel`: ReLU gain 적용한 병렬 직교 초기화
- 기타: `reset_parameters()` 호출

### 2.2 Soft Target Update

타겟 네트워크를 지수이동평균(EMA)으로 업데이트하는 유틸리티이다.

```python
def _soft_update_params(net_params, target_net_params, tau):
    # target = (1 - tau) * target + tau * source
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)
```

`torch._foreach_*` 연산은 여러 텐서에 대한 일괄 연산으로, 개별 루프보다 효율적이다. 기본 `tau` 값은 FB 네트워크 0.01, Critic 네트워크 0.005이다.

### 2.3 DenseParallel - 병렬 선형 레이어

앙상블 학습을 위해 여러 개의 선형 레이어를 하나의 텐서 연산으로 동시에 실행하는 모듈이다. Forward Map과 Critic에서 `num_parallel=2`로 사용되어 불확실성 추정(pessimism)을 가능하게 한다.

```python
class DenseParallel(nn.Module):
    def __init__(self, in_features, out_features, n_parallel, ...):
        # 가중치 shape: (n_parallel, in_features, out_features)
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features)))
        # 바이어스 shape: (n_parallel, 1, out_features)
        self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features)))

    def forward(self, input):
        # batch matrix multiply: (n_parallel, batch, in) @ (n_parallel, in, out) + (n_parallel, 1, out)
        return torch.baddbmm(self.bias, input, self.weight)
```

**핵심**: `torch.baddbmm`은 배치 행렬 곱 + 덧셈을 한 번의 커널 호출로 수행한다. 2개의 독립적인 `nn.Linear`를 실행하는 것보다 GPU 활용이 효율적이다.

### 2.4 ParallelLayerNorm

`DenseParallel`과 함께 사용되는 병렬 LayerNorm이다.

```python
class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, ...):
        # weight shape: (n_parallel, 1, *normalized_shape)
        self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape]))
        self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape]))

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        return (norm_input * self.weight) + self.bias
```

표준 `F.layer_norm`으로 정규화한 후, 병렬 슬라이스별 독립적인 affine 변환을 적용한다.

### 2.5 Embedding 함수

#### `simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1)`

모든 모델 컴포넌트의 입력단에서 사용되는 임베딩 네트워크를 생성한다.

```python
def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2
    # 첫 번째 레이어: Linear -> LayerNorm -> Tanh (안정적 초기 활성화)
    seq = [linear(input_dim, hidden_dim, num_parallel),
           layernorm(hidden_dim, num_parallel), nn.Tanh()]
    # 중간 레이어: Linear -> ReLU
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
    # 마지막 레이어: hidden_dim -> hidden_dim // 2 (두 임베딩을 concat하므로)
    seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
    return nn.Sequential(*seq)
```

**설계 포인트**: 출력이 `hidden_dim // 2`인 이유는 Forward Map과 Actor에서 두 개의 임베딩을 concat하여 `hidden_dim` 크기로 만들기 때문이다. 예를 들어 ForwardMap에서:
- `embed_z`: (obs+z) -> hidden_dim//2 = 512
- `embed_sa`: (obs+action) -> hidden_dim//2 = 512
- concat 결과: 1024 -> Fs 네트워크 입력

#### `residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1)`

잔차 연결(Residual Connection)을 사용하는 대안 임베딩이다. `ResidualBlock`은 `x + MLP(x)` 구조이다.

```python
def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]  # x + LayerNorm -> Linear -> Mish(x)
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
```

Residual 모델은 `model="residual"` 설정으로 선택할 수 있으며, Mish 활성화 함수를 사용한다.

### 2.6 ForwardMap - 전방 매핑

Successor Feature를 학습하는 핵심 네트워크이다. 상태 s에서 행동 a를 취했을 때, 잠재 방향 z에 대한 미래의 누적 특징을 예측한다.

```python
class ForwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg, output_dim=None):
        # 두 개의 임베딩 네트워크 (병렬 앙상블 지원)
        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_sa = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)

        # 출력 네트워크
        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim, cfg.num_parallel), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, output_dim if output_dim else z_dim, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs, z, action):
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            # 앙상블을 위해 입력을 복제: (batch, dim) -> (num_parallel, batch, dim)
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))     # (num_parallel, batch, h_dim//2)
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # (num_parallel, batch, h_dim//2)
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))  # (num_parallel, batch, z_dim)
```

**데이터 흐름 (기본 설정 hidden_dim=1024, z_dim=256, num_parallel=2)**:
```
입력:
  obs: (batch, obs_dim)
  z: (batch, 256)
  action: (batch, action_dim)

expand (num_parallel=2):
  obs: (2, batch, obs_dim)
  z: (2, batch, 256)
  action: (2, batch, action_dim)

embed_z: cat([obs, z]) -> (2, batch, obs_dim+256) -> (2, batch, 512)
embed_sa: cat([obs, action]) -> (2, batch, obs_dim+action_dim) -> (2, batch, 512)

Fs: cat([sa_emb, z_emb]) -> (2, batch, 1024) -> ... -> (2, batch, 256)
```

### 2.7 BackwardMap - 후방 매핑

목표 상태(또는 관측값)를 latent 공간 z에 매핑하는 네트워크이다.

```python
class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg):
        # 단일 MLP (앙상블 없음)
        seq = [nn.Linear(filtered_space.shape[0], cfg.hidden_dim),   # obs_dim -> 256
               nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):                      # hidden_layers=1이면 이 루프 실행 안 됨
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]                   # 256 -> z_dim(256)
        if cfg.norm:
            seq += [Norm()]                                          # L2 정규화 + sqrt(z_dim) 스케일링
        self.net = nn.Sequential(*seq)
```

**Norm 클래스**: 출력 벡터를 단위 구(unit sphere)에 정사영한 후, `sqrt(z_dim)`으로 스케일링한다.

```python
class Norm(nn.Module):
    def forward(self, x):
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)
        # ||output|| = sqrt(z_dim) = sqrt(256) = 16
```

**설계 이유**: B(g)의 노름을 일정하게 유지함으로써, M = F * B^T에서 F의 학습이 B의 스케일 변화에 영향받지 않도록 한다. 기본 설정에서 `hidden_dim=256, hidden_layers=1`로 Forward Map보다 훨씬 작은 네트워크이다.

### 2.8 Actor - 정책 네트워크

관측값과 latent z가 주어졌을 때 행동 분포를 출력하는 네트워크이다.

```python
class Actor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg):
        # ForwardMap과 유사하지만, 앙상블 없음 (num_parallel 미사용)
        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, z, std):
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # (batch, h_dim//2)
        s_embedding = self.embed_s(obs)                          # (batch, h_dim//2)
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)  # (batch, h_dim)
        mu = torch.tanh(self.policy(embedding))                  # [-1, 1] 범위로 클램핑
        std = torch.ones_like(mu) * std                          # 고정 표준편차 (기본값 0.2)
        dist = TruncatedNormal(mu, std)
        return dist
```

**TruncatedNormal 분포**: 평균 mu, 표준편차 std의 정규분포에서 샘플링하되, [-1, 1] 범위로 잘라낸다. `tanh`로 mu를 이미 [-1, 1]로 만들었으므로, 실제로는 mu 근방의 작은 노이즈만 추가되는 효과이다.

```python
class TruncatedNormal(pyd.Normal):
    def sample(self, clip=None, sample_shape=torch.Size()):
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)  # 노이즈 범위 제한 (기본 clip=0.3)
        x = self.loc + eps
        return self._clamp(x)  # [-1+eps, 1-eps]로 최종 클램핑
```

### 2.9 Discriminator - 판별기

전문가 데이터 (obs, z) 쌍과 학습 데이터 쌍을 구분하는 이진 분류기이다.

```python
class Discriminator(nn.Module):
    def __init__(self, obs_space, z_dim, cfg):
        # 단순 MLP: (obs_dim + z_dim) -> hidden_dim -> ... -> 1
        seq = [nn.Linear(obs_dim + z_dim, cfg.hidden_dim),
               nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, 1)]  # 로짓 출력
        self.trunk = nn.Sequential(*seq)

    def forward(self, obs, z):
        s = self.compute_logits(obs, z)
        return torch.sigmoid(s)          # 확률로 변환

    def compute_reward(self, obs, z, eps=1e-7):
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()  # log(D/(1-D)) = logit
        return reward
```

**보상 계산**: `log(D(s,z)) - log(1-D(s,z))`는 GAIL 스타일의 보상이다. 판별기가 전문가 데이터라고 확신할수록 높은 보상을 준다.

기본 설정: `hidden_dim=1024, hidden_layers=3` (다른 네트워크보다 깊음)

### 2.10 VForwardMap - Value Forward Map

Action 없이 상태와 z만으로 value를 추정하는 변형 Forward Map이다. ForwardMap에서 `embed_sa`가 `embed_s`로 대체된다.

```python
class VForwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, output_dim, cfg):
        self.embed_z = simple_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
        self.embed_s = simple_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers, cfg.num_parallel)
```

### 2.11 EMA (Exponential Moving Average)

보상 정규화에 사용되는 지수이동평균 모듈이다.

```python
class EMA(nn.Module):
    def forward(self, x):
        m = x.mean()
        sm = x.pow(2).mean()
        # 바이어스 보정된 EMA 업데이트
        self.mean.data = self.tau * self.mean + (1 - self.tau) * m
        self.mean_square.data = self.tau * self.mean_square + (1 - self.tau) * sm
        self.counter += 1
        norm = 1 - self.tau ** self.counter  # 바이어스 보정 항
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        var = torch.clamp(ema_mean_square - ema_mean ** 2, min=self.epsilon)
        return (x - translate_mean) / scale_std
```


## 3. nn_filters.py - 입력 필터링 시스템

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/nn_filters.py`

BFM-Zero의 관측값은 `gymnasium.spaces.Dict` 형태로 제공된다 (예: `{"proprio": tensor}`). 신경망은 단일 텐서 입력을 기대하므로, 입력 필터가 이 변환을 담당한다.

### 3.1 IdentityInputFilterConfig

아무 변환도 하지 않는 기본 필터이다. 관측 공간이 이미 단일 텐서일 때 사용한다.

```python
class IdentityInputFilterConfig(BaseConfig):
    def build(self, space):
        nn_module = nn.Identity()
        nn_module.output_space = space  # 출력 공간 메타데이터 저장
        return nn_module
```

### 3.2 DictInputFilter

딕셔너리 관측값에서 특정 키를 추출하는 필터이다.

```python
class DictInputFilter(nn.Module):
    def forward(self, _input: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(_input, dict):
            _input = _input[self.cfg.key]  # 예: _input["proprio"]
        # 이미 텐서이면 그대로 반환
        return _input
```

### 3.3 DictInputConcatFilter

딕셔너리의 여러 키를 추출하여 연결(concatenate)하는 필터이다.

```python
class DictInputConcatFilter(nn.Module):
    def forward(self, _input: torch.Tensor | dict[str, torch.Tensor]):
        if isinstance(_input, dict):
            _input = torch.cat([_input[key] for key in self.cfg.key], dim=-1)
        return _input
```

### 3.4 입력 데이터 흐름

```
환경 관측값 (Dict)                입력 필터              신경망
{"proprio": [69dim],    -->  DictInputFilter("proprio")  -->  BackwardMap
 "task": [10dim]}       -->  DictInputConcatFilter(       -->  ForwardMap
                              ["proprio", "task"])
```

`DictInputFilterConfig`의 `key` 파라미터에 따라:
- `key="proprio"` (문자열): `DictInputFilter` 생성, 해당 키만 추출
- `key=["proprio", "task"]` (리스트): `DictInputConcatFilter` 생성, 여러 키 연결

### 3.5 NNFilter 타입 유니온

Pydantic discriminated union으로 설정 파일에서 필터 타입을 선택할 수 있다.

```python
NNFilter = tp.Annotated[
    tp.Union[
        IdentityInputFilterConfig,
        DictInputFilterConfig,
    ],
    pydantic.Field(discriminator="name"),
]
```


## 4. nn_filter_models.py - 필터 모델 래퍼

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/nn_filter_models.py`

관측값의 **특정 인덱스만** 사용하도록 기본 모델을 래핑하는 모듈이다. humenv 환경에서 proprioceptive 관측값의 일부 차원만 사용하고 싶을 때 유용하다.

### 4.1 filter_space 유틸리티

```python
def filter_space(obs_space: gymnasium.spaces.Dict, filter: list[int]) -> gymnasium.spaces.Dict:
    # "proprio" 키에서 지정된 인덱스만 추출한 새 공간 생성
    obs_space = obs_space.spaces["proprio"]
    filtered_space = gymnasium.spaces.Box(
        low=obs_space.low[filter], high=obs_space.high[filter], shape=(len(filter),)
    )
    return gymnasium.spaces.Dict({"proprio": filtered_space})
```

### 4.2 필터 래퍼 패턴

모든 필터 래퍼는 동일한 패턴을 따른다: 입력 관측값에서 `filter` 인덱스에 해당하는 차원만 추출한 후 기본 모델에 전달한다.

```python
class FilterBackwardMap(nn.Module):
    def forward(self, obs):
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base(filtered_obs)

class FilterForwardMap(nn.Module):
    def forward(self, obs, z, action):
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base(filtered_obs, z, action)

class FilterActor(nn.Module):
    def forward(self, obs, z, std):
        if self._filter_z:
            z = z[:, self._filter]  # z도 필터링 가능
        filtered_obs = tree_map(lambda x: x[:, self._filter], obs)
        return self._nn_base(filtered_obs, z, std)
```

`tree_map`을 사용하여 딕셔너리 구조 내부의 텐서에 일관되게 필터링을 적용한다.

### 4.3 설정 상속 구조

```
BackwardArchiConfig ──> BackwardFilterArchiConfig  (filter: list[int] 추가)
ForwardArchiConfig  ──> ForwardFilterArchiConfig
ActorArchiConfig    ──> ActorFilterArchiConfig      (filter_z: bool 추가)
DiscriminatorArchiConfig ──> DiscriminatorFilterArchiConfig
```


## 5. fb/model.py - Forward-Backward 기본 모델

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb/model.py`

### 5.1 모델 구조

`FBModel`은 `BaseModel(nn.Module)`을 상속하며, 세 개의 핵심 네트워크를 조합한다.

```python
class FBModel(BaseModel):
    def __init__(self, obs_space, action_dim, cfg):
        # 세 개의 핵심 네트워크 생성
        self._backward_map = arch.b.build(obs_space, arch.z_dim)    # B: obs -> z
        self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)  # F: (obs,z,a) -> z
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)    # pi: (obs,z) -> action

        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        # 추론 전용 모드로 설정
        self.train(False)
        self.requires_grad_(False)
```

### 5.2 Forward Map (F)

추론 시 `@torch.no_grad()` 데코레이터와 AMP(Automatic Mixed Precision) 지원이 적용된다.

```python
@torch.no_grad()
def forward_map(self, obs, z, action):
    with autocast(device_type=self.device, dtype=self.amp_dtype, enabled=self.cfg.amp):
        return self._forward_map(self._normalize(obs), z, action)
```

**입출력 shape**:
- 입력: obs (batch, obs_dim), z (batch, 256), action (batch, action_dim)
- 출력: (num_parallel, batch, 256) - 앙상블 2개의 예측

### 5.3 Backward Map (B)

```python
@torch.no_grad()
def backward_map(self, obs):
    with autocast(...):
        return self._backward_map(self._normalize(obs))
```

**입출력 shape**:
- 입력: obs (batch, obs_dim)
- 출력: (batch, 256) - 정규화된 z 벡터

### 5.4 Actor

```python
@torch.no_grad()
def actor(self, obs, z, std):
    with autocast(...):
        return self._actor(self._normalize(obs), z, std)
```

**입출력 shape**:
- 입력: obs (batch, obs_dim), z (batch, 256), std (스칼라, 기본 0.2)
- 출력: TruncatedNormal 분포 (mean shape: (batch, action_dim))

### 5.5 Target Network

학습 시 `_prepare_for_train()`으로 타겟 네트워크를 deep copy하여 생성한다.

```python
def _prepare_for_train(self):
    self._target_backward_map = copy.deepcopy(self._backward_map)
    self._target_forward_map = copy.deepcopy(self._forward_map)
```

타겟 네트워크는 학습 루프에서 `soft_update_params`로 느리게(tau=0.01) 업데이트된다. 이는 TD 학습의 안정성을 위한 표준 기법이다.

### 5.6 z 벡터 샘플링과 투영

```python
def sample_z(self, size, device="cpu"):
    z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
    return self.project_z(z)

def project_z(self, z):
    if self.cfg.archi.norm_z:
        z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        # ||z|| = sqrt(256) = 16
    return z
```

**핵심**: z 벡터는 반지름 `sqrt(z_dim)=16`인 초구(hypersphere) 위에 놓인다. B(g)도 Norm 모듈로 같은 스케일링이 적용되므로, `F*B^T`와 `F*z`의 스케일이 일관된다.

### 5.7 추론 메서드들

#### `reward_inference` - 보상 기반 z 추론
```python
def reward_inference(self, next_obs, reward, weight=None):
    wr = reward if weight is None else reward * weight
    for i in range(num_batches):
        B = self.backward_map(next_obs_slice)
        z += torch.matmul(wr.T, B)  # z = sum(w_i * r_i * B(s_i))
    return self.project_z(z)
```

보상이 높은 상태의 B 벡터를 보상 가중 합산하여 z를 구한다.

#### `goal_inference` - 목표 도달 z 추론
```python
def goal_inference(self, next_obs):
    z = self.backward_map(next_obs)  # 목표 상태를 직접 B로 인코딩
    return self.project_z(z)
```

#### `tracking_inference` - 모션 추적 z 추론
```python
def tracking_inference(self, next_obs):
    z = self.backward_map(next_obs)
    for step in range(z.shape[0]):
        end_idx = min(step + self.cfg.seq_length, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)  # 슬라이딩 윈도우 평균
    return self.project_z(z)
```

연속 프레임의 B 벡터를 `seq_length=8` 윈도우로 평균하여 시간적으로 부드러운 z 시퀀스를 생성한다.


## 6. fb_cpr/model.py - FB-CPR 모델

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb_cpr/model.py`

### 6.1 아키텍처 확장

`FBcprModel`은 `FBModel`을 상속하고 **Critic**과 **Discriminator** 두 컴포넌트를 추가한다.

```python
class FBcprModelArchiConfig(FBModelArchiConfig):
    # 기존 FB 컴포넌트 (f, b, actor, z_dim, norm_z)에 추가
    critic: ForwardArchiConfig | ForwardFilterArchiConfig = ForwardArchiConfig()
    discriminator: DiscriminatorArchiConfig | DiscriminatorFilterArchiConfig = DiscriminatorArchiConfig()
```

### 6.2 Critic

Critic은 **ForwardMap과 동일한 아키텍처**를 사용하되, `output_dim=1`로 설정하여 스칼라 Q-value를 출력한다.

```python
class FBcprModel(FBModel):
    def __init__(self, obs_space, action_dim, cfg):
        super().__init__(obs_space, action_dim, cfg)
        self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
```

**Critic의 데이터 흐름**:
```
입력: obs, z, action
  -> embed_z: cat([obs, z]) -> (2, batch, 512)
  -> embed_sa: cat([obs, action]) -> (2, batch, 512)
  -> Fs: cat -> (2, batch, 1024) -> ... -> (2, batch, 1)  ← z_dim 대신 1
출력: Q-value (2, batch, 1) - 2개 앙상블
```

Critic의 학습은 Discriminator 보상을 사용한 TD 학습이다:
```python
# agent.py에서
reward = self._model._discriminator.compute_reward(obs=obs, z=z)
target_Q = reward + discount * next_V
critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)
```

### 6.3 Discriminator

전문가 데이터와 학습 데이터를 구분하는 이진 분류기이다. `nn_models.py`의 `Discriminator` 클래스를 사용한다.

- 입력: (obs, z) 쌍
- 출력: 전문가일 확률 (sigmoid)
- 학습: Binary Cross-Entropy + WGAN Gradient Penalty

```python
# agent.py의 update_discriminator에서
expert_loss = -torch.nn.functional.logsigmoid(expert_logits)    # 전문가 → 1
unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)  # 비전문가 → 0
loss = torch.mean(expert_loss + unlabeled_loss)
# + WGAN gradient penalty (Lipschitz 제약)
```

### 6.4 Target Network 확장

```python
def _prepare_for_train(self):
    super()._prepare_for_train()  # target_backward_map, target_forward_map 생성
    self._target_critic = copy.deepcopy(self._critic)  # target_critic 추가
```

### 6.5 FB 모델과의 차이점

| 항목 | FBModel | FBcprModel |
|------|---------|------------|
| 네트워크 수 | 3 (F, B, Actor) | 5 (+ Critic, Discriminator) |
| 타겟 네트워크 | 2 (target_F, target_B) | 3 (+ target_Critic) |
| Actor 손실 | `-Q_fb.mean()` (FB로만 계산) | `-Q_disc * reg_coeff * weight - Q_fb` (이중 Q) |
| 보상 소스 | 없음 (비지도) | Discriminator 보상 |
| z 샘플링 | 랜덤 + 목표 인코딩 | 랜덤 + 목표 인코딩 + **전문가 인코딩** |

**CPR의 핵심**: Actor 손실에서 FB 기반 Q-value(`Q_fb`)와 Discriminator 기반 Q-value(`Q_discriminator`)를 결합한다.

```python
# agent.py update_actor에서
Qs_fb = (Fs * z).sum(-1)           # FB 기반 Q: F(s,a)*z
Q_discriminator = critic(obs, z, action)  # Discriminator 기반 Q
actor_loss = -Q_discriminator.mean() * reg_coeff * weight - Q_fb.mean()
```


## 7. normalizers.py - 정규화

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/normalizers.py`

### 7.1 BatchNormNormalizer

`nn.BatchNorm1d`를 래핑한 관측값 정규화기이다. `affine=False`로 설정하여 학습 가능한 스케일/시프트 파라미터 없이 순수 통계 기반 정규화만 수행한다.

```python
class BatchNormNormalizer(nn.Module):
    def __init__(self, obs_space, cfg):
        # affine=False: 감마/베타 학습하지 않음 (순수 정규화만)
        self._normalizer = nn.BatchNorm1d(
            num_features=obs_space.shape[0],
            affine=False,
            momentum=cfg.momentum  # 기본값 0.01
        )
```

**동작 방식**: 학습 시 배치 통계(평균, 분산)의 이동평균을 추적하고, 추론 시 이 통계로 정규화한다. `momentum=0.01`이므로 새 배치의 영향이 천천히 반영된다.

### 7.2 ObsNormalizer

딕셔너리 관측 공간의 각 키에 대해 독립적인 정규화기를 관리하는 컨테이너이다.

```python
class ObsNormalizer(nn.Module):
    def __init__(self, obs_space, cfg):
        if isinstance(cfg.normalizers, dict):
            # 키별로 다른 정규화기 적용 가능
            self._normalizers = nn.ModuleDict({
                key: cfg.normalizers[key].build(obs_space[key])
                for key in cfg.normalizers.keys()
            })
        else:
            # 단일 정규화기를 전체 공간에 적용
            self._normalizers = cfg.normalizers.build(obs_space)

    def forward(self, x):
        if isinstance(self.cfg.normalizers, dict):
            normalized_obs = {}
            for key in self._normalizers.keys():
                normalized_obs[key] = self._normalizers[key](x[key])
            return normalized_obs
        else:
            return self._normalizers(x)
```

**사용 패턴** (fb/agent.py에서):
```python
# 1단계: 학습 모드에서 통계 업데이트 (running mean/var 갱신)
self._model._obs_normalizer(train_obs)

# 2단계: eval 모드에서 정규화 적용 (running 통계 사용, 갱신 안 함)
with torch.no_grad(), eval_mode(self._model._obs_normalizer):
    train_obs = self._model._obs_normalizer(train_obs)
```


## 8. 핵심 개념 정리

### 8.1 Successor Feature 분해: M(s, g) = F(s, a) * B(g)^T

Successor Measure M(s, g)는 상태 s에서 정책을 따랐을 때 목표 상태 g를 방문할 누적 할인 확률을 나타낸다. 이를 F와 B의 내적으로 분해한다.

**FB 손실 함수** (fb/agent.py `update_fb`에서):
```python
# Successor Measure 행렬 계산
Fs = self._model._forward_map(obs, z, action)   # (2, batch, 256)
B = self._model._backward_map(goal)              # (batch, 256)
Ms = torch.matmul(Fs, B.T)                       # (2, batch, batch) - M 행렬

# 타겟: Bellman 업데이트
target_Ms = torch.matmul(target_Fs, target_B.T)  # (2, batch, batch)
diff = Ms - discount * target_M                   # (2, batch, batch)

# 대각선: M(s_i, g_i)는 즉시 전이를 반영 → 최대화 (부호 반전)
fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]

# 비대각선: M(s_i, g_j) i!=j는 0에 가까워야 → 최소화 (제곱 손실)
fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
```

대각선 항은 `s_i`에서 출발해서 `g_i`(자기 자신의 미래 상태)에 도달하는 것이므로 값이 커야 하고, 비대각선 항은 다른 궤적의 상태에 도달하는 것이므로 작아야 한다.

**직교성 손실**:
```python
Cov = torch.matmul(B, B.T)          # (batch, batch) - B 벡터간 내적 행렬
orth_loss_diag = -Cov.diag().mean()  # B 벡터의 노름 최대화
orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum  # 서로 직교
```

B 벡터들이 서로 직교하도록(다른 상태는 다른 방향을 가리키도록) 강제한다. 이는 z 공간의 표현력을 극대화한다.

### 8.2 z 벡터 (256차원 latent)의 의미

z 벡터는 **행동 의도(behavioral intent)**를 인코딩한다. `||z|| = sqrt(256) = 16`인 초구 위의 한 점이다.

| z의 출처 | 의미 | 사용 맥락 |
|----------|------|-----------|
| `sample_z` (랜덤) | 무작위 탐색 방향 | 비지도 탐색 |
| `backward_map(goal)` | 목표 상태의 인코딩 | 목표 도달 |
| `tracking_inference` | 모션 궤적의 시간 평균 인코딩 | 모션 추적 |
| `reward_inference` | 보상 가중 상태 인코딩 | 보상 기반 과제 |
| `encode_expert` | 전문가 궤적의 인코딩 | CPR 학습 |

Actor는 z를 조건으로 받아 행동을 결정하므로, z를 바꾸면 로봇의 행동이 바뀐다. 이것이 "promptable control"의 핵심이다.

**z 혼합 분포** (FB-CPR의 `sample_mixed_z`):
```python
prob = [train_goal_ratio,           # 0.2: 학습 버퍼의 목표 인코딩
        expert_asm_ratio,           # 0.6: 전문가 궤적 인코딩
        1 - goal_ratio - expert]    # 0.2: 무작위 z
mix_idxs = torch.multinomial(prob, batch_size, replacement=True)
```

### 8.3 Soft Target Update 메커니즘

타겟 네트워크를 온라인 네트워크의 지수이동평균으로 업데이트한다. TD(Temporal Difference) 학습에서 부트스트랩 타겟의 안정성을 보장하는 표준 기법이다.

```python
# 매 업데이트 스텝마다 실행
_soft_update_params(forward_params, target_forward_params, tau=0.01)     # FB 네트워크
_soft_update_params(backward_params, target_backward_params, tau=0.01)   # FB 네트워크
_soft_update_params(critic_params, target_critic_params, tau=0.005)      # Critic (더 느리게)
```

수식: `theta_target = (1 - tau) * theta_target + tau * theta_online`

- `tau=0.01`: 100스텝당 약 63%가 새 파라미터로 교체됨
- `tau=0.005`: 200스텝당 약 63%가 새 파라미터로 교체됨 (Critic은 더 보수적)


## 9. 파일 간 관계도

### 9.1 모듈 의존성

```
normalizers.py          nn_filters.py
     |                      |
     v                      v
base_model.py          nn_models.py
     |                /     |      \
     |    BackwardMap  ForwardMap   Actor, Discriminator
     |          |         |              |
     v          v         v              v
  fb/model.py (FBModel)
     |    - _backward_map (BackwardMap)
     |    - _forward_map (ForwardMap)
     |    - _actor (Actor)
     |    - _obs_normalizer (ObsNormalizer)
     |
     v
  fb_cpr/model.py (FBcprModel extends FBModel)
     |    - _critic (ForwardMap, output_dim=1)
     |    - _discriminator (Discriminator)
     |
     v
  fb/agent.py (FBAgent) ──> fb_cpr/agent.py (FBcprAgent extends FBAgent)
     - update_fb()              - update_discriminator()
     - update_actor()           - update_critic()
     - soft_update_params()     - update_actor() (오버라이드)
```

### 9.2 학습 시 데이터 흐름

```
[환경] --> obs (Dict: {"proprio": Tensor})
              |
              v
[ObsNormalizer] --> 정규화된 obs
              |
              +--> [BackwardMap] --> B(goal), z_expert
              |
              +--> [ForwardMap] --> F(s,a) --> M = F * B^T (FB 손실)
              |                          \--> Q_fb = F * z (Actor 손실)
              |
              +--> [Actor(obs, z)] --> action 분포
              |
              +--> [Critic(obs, z, action)] --> Q_disc (Discriminator 보상 기반)
              |
              +--> [Discriminator(obs, z)] --> D(obs,z) --> reward = log(D/(1-D))
```

### 9.3 네트워크 크기 요약 (기본 설정)

| 네트워크 | hidden_dim | hidden_layers | embedding_layers | num_parallel | 특이사항 |
|----------|-----------|---------------|-----------------|-------------|---------|
| Forward Map | 1024 | 2 | 2 | 2 | 앙상블 |
| Backward Map | 256 | 1 | - | 1 | Norm 출력 |
| Actor | 1024 | 2 | 2 | 1 | TruncatedNormal |
| Critic | 1024 | 2 | 2 | 2 | output_dim=1 |
| Discriminator | 1024 | 3 | - | 1 | 가장 깊음 |

### 9.4 nn_filter_models.py의 역할

`nn_filter_models.py`는 위 기본 모델들을 래핑하여 관측값의 일부 차원만 사용하는 변형을 제공한다. 설정에서 `name: "ForwardFilterArchi"` 등으로 선택하면 `FilterForwardMap(ForwardMap, filter=[0,1,2,...])` 형태로 생성된다. 이는 특정 로봇 설정에서 불필요한 센서 입력을 제거할 때 사용한다.
