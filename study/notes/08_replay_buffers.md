# Phase 3: 리플레이 버퍼 (Replay Buffers)

## 1. 개요

### 리플레이 버퍼의 역할

리플레이 버퍼(Replay Buffer)는 오프폴리시(off-policy) 강화학습의 핵심 구성요소로, 에이전트가 환경과 상호작용하면서 수집한 경험(experience)을 저장하고, 학습 시 이를 재사용할 수 있게 해준다. 이를 통해:

1. **데이터 효율성**: 한 번 수집한 경험을 여러 번 재사용하여 sample efficiency를 높임
2. **상관관계 제거**: 연속적으로 수집된 데이터 간의 시간적 상관관계를 랜덤 샘플링으로 제거
3. **안정적 학습**: 미니배치 학습에서의 분산(variance)을 줄여 학습을 안정화

### BFM-Zero에서의 버퍼 시스템 구성

BFM-Zero는 세 종류의 버퍼를 사용한다:

| 버퍼 | 클래스 | 용도 | 저장 단위 |
|------|--------|------|-----------|
| Train 버퍼 | `DictBuffer` 또는 `TrajectoryDictBufferMultiDim` | 정책(policy)이 수집한 온라인 데이터 | 전이(transition) 또는 궤적(trajectory) |
| Expert 버퍼 | `TrajectoryDictBuffer` | 전문가 모션 데이터 (모션 캡처) | 궤적(trajectory) |
| Z 버퍼 | `ZBuffer` | 학습 중 생성된 z 벡터 | z 벡터 |

학습 루프(`train.py`)에서 이들은 `replay_buffer` 딕셔너리로 관리된다:
```python
replay_buffer = {}
replay_buffer["train"] = DictBuffer(...)       # 또는 TrajectoryDictBufferMultiDim(...)
replay_buffer["expert_slicer"] = expert_buffer  # TrajectoryDictBuffer
```

---

## 2. transition.py - DictBuffer (Transition 기반 버퍼)

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/buffers/transition.py`

### 2.1 클래스 구조

`DictBuffer`는 Python의 `@dataclasses.dataclass`로 구현된 **순환 버퍼(Circular Buffer)**이다. 핵심 상태 변수는 다음과 같다:

```python
@dataclasses.dataclass(kw_only=True)
class DictBuffer:
    capacity: int           # 버퍼 최대 크기
    device: str = "cpu"     # 텐서 저장 디바이스
    nested_key_separator: str = "-"  # HDF5 저장 시 중첩 키 구분자

    def __post_init__(self) -> None:
        self.storage = None     # 실제 데이터 저장소 (딕셔너리)
        self._idx = 0           # 현재 쓰기 위치 (커서)
        self._is_full = False   # 버퍼가 한 번이라도 꽉 찬 적이 있는지
```

- `storage`는 `None`으로 초기화되며, 첫 `extend()` 호출 시 데이터 형태에 맞게 자동 생성된다 (lazy initialization).
- `__len__`은 버퍼가 꽉 찼으면 `capacity`를, 아니면 현재 `_idx`를 반환한다.

### 2.2 extend() - 데이터 추가

`extend()` 메서드는 배치 단위로 데이터를 버퍼에 추가한다. 중첩된 딕셔너리 구조를 **재귀적으로** 처리하는 것이 특징이다.

```python
@torch.no_grad
def extend(self, data: Dict) -> None:
    if self.storage is None:
        self.storage = {}
        initialize_storage(data, self.storage, self.capacity, self.device, n_dim=self._ndim())
        # ...
    # 재귀적으로 중첩 딕셔너리 처리
    def add_new_data(data, storage, expected_dim: int):
        for k, v in data.items():
            if isinstance(v, Mapping):
                add_new_data(v, storage=storage[k], expected_dim=expected_dim)
            else:
                end = self._idx + v.shape[0]
                if end >= self.capacity:
                    # 순환(wrap) 처리
                    diff = self.capacity - self._idx
                    storage[k][self._idx :] = _to_torch(v[:diff], device=self.device)
                    storage[k][: v.shape[0] - diff] = _to_torch(v[diff:], device=self.device)
                    self._is_full = True
                else:
                    storage[k][self._idx : end] = _to_torch(v, device=self.device)
    # ...
    self._idx = (self._idx + data_dim) % self.capacity
```

**핵심 동작**:
- 첫 호출 시 `initialize_storage()`가 `capacity` 크기의 0-텐서를 미리 할당 (pre-allocation)
- 데이터가 버퍼 끝을 넘으면 두 부분으로 나눠서 앞부분을 덮어씀 (wrap-around)
- `_to_torch()`는 `functools.singledispatch` 기반으로 numpy, torch, 스칼라 등 다양한 타입을 자동 변환

### 2.3 sample() - 랜덤 샘플링

```python
@torch.no_grad
def sample(self, batch_size) -> Dict[str, torch.Tensor]:
    self.ind = torch.randint(0, len(self), (batch_size,))
    return extract_values(self.storage, self.ind)
```

`extract_values()`는 중첩 딕셔너리를 재귀 순회하며 동일한 인덱스로 모든 값을 추출한다:

```python
def extract_values(d: Dict, idxs: List | torch.Tensor | np.ndarray) -> Dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result[k] = extract_values(v, idxs)
        else:
            result[k] = v[idxs]
    return result
```

### 2.4 FIFO 동작 방식

DictBuffer는 **FIFO(First-In-First-Out) 순환 버퍼**이다:

```
capacity = 8, 데이터 크기 = 3

초기:     [_, _, _, _, _, _, _, _]   _idx=0
extend1:  [A, A, A, _, _, _, _, _]   _idx=3
extend2:  [A, A, A, B, B, B, _, _]   _idx=6
extend3:  [C, C, A, B, B, B, C, C]   _idx=2, _is_full=True  (wrap!)
extend4:  [C, C, D, D, D, B, C, C]   _idx=5                  (가장 오래된 데이터 덮어씀)
```

`_idx`는 항상 `(self._idx + data_dim) % self.capacity`로 갱신되어 순환한다.

### 2.5 저장/로딩 (HDF5)

버퍼는 HDF5 형식으로 디스크에 저장/로딩할 수 있다. 중첩 딕셔너리의 키는 `nested_key_separator`("-")로 flatten된다:

```python
# 저장 예: {"observation": {"state": tensor}} -> HDF5 키: "observation-state"
def save_field(data, prefix: str = "", nested_key: str = "-"):
    for k, v in data.items():
        if isinstance(v, Mapping):
            save_field(v, prefix=f"{prefix}{k}{nested_key}")
        else:
            hf.create_dataset(f"{prefix}{k}", data=v[: len(self)].cpu().detach().numpy())
```

로딩 시 `_idx`, `_is_full` 등 메타데이터도 `config.json`에서 복원하여 이어서 쓰기가 가능하다.

---

## 3. trajectory.py - TrajectoryDictBuffer (궤적 기반 버퍼)

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/buffers/trajectory.py`

### 3.1 TrajectoryDictBuffer vs DictBuffer 차이점

| 특성 | DictBuffer | TrajectoryDictBuffer |
|------|-----------|---------------------|
| 저장 단위 | 개별 전이(transition) | 에피소드(궤적) 목록 |
| 초기화 | lazy (첫 extend 시) | 생성자에서 전체 데이터 로드 |
| 데이터 추가 | `extend()`로 동적 추가 | 생성 시 고정 (읽기 전용) |
| 샘플링 | 완전 랜덤 | **연속 서브시퀀스** 샘플링 |
| 용도 | Policy 온라인 데이터 | Expert 모션 데이터 |

`TrajectoryDictBuffer`는 **전문가 모션 데이터 전용 읽기 전용 버퍼**이다. 모션 캡처 데이터를 에피소드 단위로 받아 하나의 연속 텐서로 합치되, 각 에피소드의 시작/끝 위치를 추적한다.

### 3.2 motion_id 기반 추적

생성자에서 에피소드 목록을 받아 처리한다:

```python
class TrajectoryDictBuffer:
    def __init__(self, episodes: List[dict], device: str = "cpu", seq_length: int = 1,
                 output_key_t: List[str] = ["observation"],
                 output_key_tp1: List[str] = ["observation"],
                 end_key: Tuple[str] | str = "done",
                 motion_id_key: Tuple[str] | str = "motion_id") -> None:
        self.storage = {}
        self.motion_ids = []
        for ep in episodes:
            add_new_data(ep, self.storage)
            # 에피소드 종료 마킹
            if not key_exists(ep, self.end_key):
                episode_start = torch.zeros((ep[_k].shape[0], 1), dtype=torch.bool)
                episode_start[-1] = True
                get_key(self.storage, self.end_key).append(episode_start)
            self.motion_ids.append(get_key(ep, self.motion_id_key)[0].item())
        concat_dict(self.storage)  # 리스트를 하나의 텐서로 concat
```

`motion_ids`는 각 에피소드의 모션 ID를 저장하며, 이후 **우선순위 샘플링**과 **평가(evaluation)**에서 특정 모션을 식별하는 데 사용된다.

생성 후 `find_start_stop_traj()`로 각 궤적의 시작/종료 인덱스와 길이를 계산한다:

```python
self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
    done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=None
)
```

### 3.3 seq_length를 이용한 궤적 슬라이싱

`sample()` 메서드는 **연속 서브시퀀스**를 샘플링한다. 이는 Forward-Backward 표현 학습에서 시간적 연속성을 보존하기 위해 필수적이다.

```python
def sample(self, batch_size: int = 1, seq_length: int | None = None):
    seq_length = seq_length or self.seq_length
    # batch_size는 seq_length의 배수여야 함
    num_slices = batch_size // seq_length

    # 충분히 긴 궤적만 선택
    traj_idx = self.lengths >= (seq_length + offset)

    # 궤적 인덱스 샘플링 (우선순위 기반 또는 균일)
    idxs = self._get_idxs(
        seq_length=seq_length,
        num_slices=num_slices,
        lengths=self.lengths[traj_idx],
        start_idx=self.start_idx[traj_idx],
        storage_length=self.capacity,
        priorities=self.priorities[traj_idx],
    )

    # 현재 시점 데이터
    for k in self.output_key_t:
        output[k] = tree_map(lambda x: x[idxs], self.storage[k])
    # 다음 시점 데이터 (인덱스 +1)
    idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
    for k in self.output_key_tp1:
        output["next"][k] = tree_map(lambda x: x[idxs], self.storage[k])
    return output
```

**샘플링 과정** (함수 `get_idxs()`):
1. `priorities`에 따라 `torch.multinomial()`로 궤적을 선택
2. 선택된 궤적 내에서 랜덤 시작점을 결정 (`torch.rand() * end_point`)
3. 시작점부터 `seq_length`만큼의 연속 인덱스를 생성

```python
def get_idxs(seq_length, num_slices, lengths, start_idx, storage_length, priorities):
    if priorities is not None:
        traj_idx = torch.multinomial(priorities, num_slices, replacement=True)
    else:
        traj_idx = torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    end_point = lengths[traj_idx] - seq_length - 1
    relative_starts = (torch.rand(num_slices, device=lengths.device) * end_point).floor()
    # ...
    idxs = _tensor_slices_from_startend(seq_length, starts, storage_length=storage_length)
    return idxs
```

**output_key_t / output_key_tp1**: 현재 시점(t)과 다음 시점(t+1)에서 어떤 키를 추출할지 지정한다. 기본적으로 둘 다 `["observation"]`이며, 이는 FB 표현 학습에서 s_t, s_{t+1} 쌍이 필요하기 때문이다.

### 3.4 TrajectoryDictBufferMultiDim

`TrajectoryDictBufferMultiDim`은 `DictBuffer`를 상속하면서 궤적 기반 샘플링을 지원하는 **하이브리드 버퍼**이다. 정책이 수집한 온라인 데이터를 저장하면서도 연속 서브시퀀스를 샘플링할 수 있다.

```python
@dataclasses.dataclass(kw_only=True)
class TrajectoryDictBufferMultiDim(DictBuffer):
    n_dim: int = 1        # 2면 다차원 (병렬 환경)
    seq_length: int = 1
    output_key_t: List[str] = ["observation"]
    output_key_tp1: List[str] = ["observation"]
    end_key: Tuple[str] | str = "done"
```

**DictBuffer와의 차이점**:
- `extend()` 호출 시 `_recompute_start_stop = True`로 설정하여 다음 `sample()` 시 궤적 경계를 재계산
- `n_dim=2`일 때 `(시간, 병렬환경, ...)`의 2차원 인덱싱 지원
- `capacity`가 `buffer_size // online_parallel_envs`로 설정됨 (시간 축 기준)

학습 루프에서의 초기화:
```python
# train.py에서
replay_buffer["train"] = TrajectoryDictBufferMultiDim(
    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,
    device=self.cfg.buffer_device,
    n_dim=2,
    end_key="truncated",
    output_key_t=["observation", "action", "z", "terminated", "truncated", "step_count", "reward"],
    output_key_tp1=["observation", "terminated"],
)
```

### 3.5 우선순위 샘플링 (EMD 기반)

`TrajectoryDictBuffer`는 **우선순위 샘플링**을 지원한다. 초기에는 균일 분포이지만, 평가 결과에 따라 가중치가 업데이트된다:

```python
# 초기화: 균일 분포
self.priorities = torch.ones(len(self.lengths), device=self.device, dtype=torch.float32) / len(self.lengths)

# 업데이트: 정규화하여 확률 분포 유지
def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor) -> None:
    self.priorities[idxs] = priorities
    self.priorities = self.priorities / torch.sum(self.priorities)
```

`train.py`에서 평가 후 EMD(Earth Mover's Distance) 값을 기반으로 우선순위를 갱신한다:

```python
# EMD가 높은 모션(추적이 잘 안 되는 모션)에 더 높은 우선순위 부여
priorities = torch.clamp(
    torch.tensor(priorities, dtype=torch.float32),
    min=self.cfg.prioritization_min_val,    # 0.5
    max=self.cfg.prioritization_max_val,    # 2.0
) * self.cfg.prioritization_scale           # 2.0

# 모드별 변환
if self.cfg.prioritization_mode == "exp":
    priorities = 2**priorities  # 지수적 스케일링
elif self.cfg.prioritization_mode == "bin":
    # 빈(bin) 단위로 균일하게
    bins = torch.floor(priorities)
    for i in range(int(bins.min()), int(bins.max()) + 1):
        mask = bins == i
        priorities[mask] = 1 / mask.sum()
```

---

## 4. load_data.py - 데이터 로딩

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/buffers/load_data.py`

### 4.1 load_expert_trajectories()

디스크에 저장된 HDF5 모션 데이터를 `TrajectoryDictBuffer`로 로드한다:

```python
def load_expert_trajectories(
    motions: str | Path,        # 모션 파일 목록 (.txt)
    motions_root: str | Path,   # 모션 데이터 루트 경로
    seq_length: int,            # 서브시퀀스 길이
    device: str,                # 저장 디바이스
    obs_dict_mapper: Callable | None = None,  # 관측값 변환 함수
) -> TrajectoryDictBuffer:
    with open(motions, "r") as txtf:
        h5files = [el.strip().replace(" ", "") for el in txtf.readlines()]
    episodes = []
    for h5 in h5files:
        h5 = canonicalize(h5, base_path=motions_root)
        _ep = load_episode_based_h5(h5, keys=None)
        for el in _ep:
            el["observation"] = tree_map(lambda x: x.astype(np.float32), el["observation"])
            if obs_dict_mapper is not None:
                el["observation"] = obs_dict_mapper(el["observation"])
            del el["file_name"]
        episodes.extend(_ep)
    buffer = TrajectoryDictBuffer(episodes, seq_length=seq_length, device=device)
    return buffer
```

**처리 흐름**:
1. `.txt` 파일에서 HDF5 파일 경로 목록을 읽음
2. 각 HDF5 파일을 `humenv`의 `load_episode_based_h5()`로 에피소드 단위 로드
3. 관측값을 float32로 변환
4. 선택적으로 `obs_dict_mapper`를 적용하여 관측값을 딕셔너리 형태로 변환
5. 모든 에피소드를 `TrajectoryDictBuffer`에 합침

그러나 **BFM-Zero의 기본 학습 설정**(`load_isaac_expert_data=True`)에서는 이 함수 대신 `load_expert_trajectories_from_motion_lib()`을 사용한다. 이 함수는 Isaac Sim의 모션 라이브러리에서 직접 관측값을 계산하여 로드한다.

### 4.2 load_buffer()

체크포인트에서 저장된 버퍼를 복원할 때 사용한다:

```python
def load_buffer(path: str, device: str | None = None) -> DictBuffer:
    path = Path(path)
    with (path / "config.json").open() as f:
        loaded_config = json.load(f)
    target_class = loaded_config["__target__"]

    if target_class.endswith("DictBuffer"):
        return DictBuffer.load(path, device=device)
    elif target_class.endswith("TrajectoryDictBufferMultiDim"):
        return TrajectoryDictBufferMultiDim.load(path, device=device)
```

`config.json`의 `__target__` 필드로 원래 클래스를 식별하여 적절한 로더를 호출한다.

### 4.3 데이터 포맷과 전처리

Expert 데이터(Isaac Sim 기반)는 다음 구조로 저장된다:

```python
ep = {
    "observation": {
        "state": state,                    # [T, dof_pos + dof_vel + gravity + ang_vel]
        "last_action": bogus_actions,      # [T, action_dim] (0으로 채움)
        "privileged_state": max_local_self_obs,  # [T, privileged_dim]
    },
    "terminated": torch.zeros(T, dtype=bool),
    "truncated": truncated,               # 마지막 프레임만 True
    "motion_id": torch.ones(T, dtype=torch.long) * i,
}
```

`state`에는 `dof_pos - default_dof_pos` (기본 자세 대비 관절 위치), `dof_vel`, `projected_gravity`, `ang_vel`이 연결(concat)된다.

---

## 5. zbuffer.py - Z 벡터 버퍼

**파일 경로**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/misc/zbuffer.py`

### 5.1 ZBuffer 구조

`ZBuffer`는 학습 과정에서 생성된 **z 벡터(태스크 임베딩)**를 저장하는 경량 순환 버퍼이다.

```python
class ZBuffer:
    def __init__(self, capacity: int, dim: int, device, dtype=torch.float32):
        self._storage = torch.zeros((capacity, dim), device=device, dtype=dtype)
        self._idx = 0
        self._is_full = False
        self.capacity = capacity
        self.device = device
```

`DictBuffer`와 달리 **고정된 2D 텐서** 하나만 사용하며, 딕셔너리 구조가 필요 없다. z 벡터의 차원은 `z_dim` (기본 256)이다.

### 5.2 순환 버퍼 동작

```python
def add(self, data: torch.Tensor) -> None:
    if self._idx + data.shape[0] >= self.capacity:
        diff = self.capacity - self._idx
        self._storage[self._idx : self._idx + data.shape[0]] = data[:diff]
        self._storage[: data.shape[0] - diff] = data[diff:]
        self._is_full = True
    else:
        self._storage[self._idx : self._idx + data.shape[0]] = data
    self._idx = (self._idx + data.shape[0]) % self.capacity
```

DictBuffer의 순환 로직과 동일한 패턴이지만, 단일 텐서에 대해서만 동작하므로 훨씬 단순하다.

### 5.3 sample() 메서드

```python
def sample(self, num, device=None) -> torch.Tensor:
    idx = np.random.randint(0, len(self), size=num)
    return self._storage[idx].clone().to(device if device is not None else self.device)
```

`clone()`을 호출하여 원본 저장소를 오염시키지 않도록 복사본을 반환한다. `device` 파라미터를 통해 필요시 다른 디바이스로 전송할 수 있다.

### 5.4 ZBuffer의 용도

`FBAgent.setup_training()`에서 초기화되고:
```python
self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self._model.device)
# z_buffer_size = 8192 (기본)
```

학습 업데이트 시 혼합 z를 생성한 후 저장된다:
```python
# FBcprAgent.update()에서
z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
self.z_buffer.add(z)  # 매 업데이트마다 z를 축적
```

롤아웃 시 `use_mix_rollout=True`이면, 환경에서 행동할 때 사용할 z를 ZBuffer에서 샘플링한다:
```python
# FBAgent.maybe_update_rollout_context()에서
if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
    new_z = self.z_buffer.sample(z.shape[0], device=self._model.device)
else:
    new_z = self._model.sample_z(z.shape[0], device=self._model.device)
```

이를 통해 학습된 z 분포에서 롤아웃 z를 샘플링하여, 학습에 유용한 데이터를 수집할 확률을 높인다.

---

## 6. 버퍼 시스템 전체 흐름

### 6.1 학습 루프에서의 버퍼 사용 패턴

```
[환경 상호작용]                    [학습 업데이트]
     |                                  |
     v                                  v
obs, action, reward             expert_batch = expert_slicer.sample(batch_size)
     |                          train_batch  = train.sample(batch_size)
     v                                  |
replay_buffer["train"].extend(data)     v
                                 agent.update(replay_buffer, step)
                                        |
                                        v
                                 z = sample_mixed_z(...)
                                 z_buffer.add(z)
                                        |
                                        v
                                 update_fb(), update_critic(),
                                 update_actor(), update_discriminator()
```

구체적인 흐름:

1. **데이터 수집**: 환경에서 `step()`을 통해 전이 데이터를 수집
2. **버퍼 저장**: `replay_buffer["train"].extend(data)` (매 스텝)
3. **주기적 업데이트** (`update_agent_every=1024` 스텝마다):
   - Expert 버퍼에서 배치 샘플링: `expert_slicer.sample(batch_size)` -- **연속 서브시퀀스**
   - Train 버퍼에서 배치 샘플링: `train.sample(batch_size)` -- **연속 서브시퀀스 또는 랜덤**
   - `num_agent_updates=16`회 반복

### 6.2 Expert 데이터 vs Policy 데이터

**Expert 데이터** (`expert_slicer`):
- 모션 캡처 데이터로부터 생성된 **고정된** 궤적
- 시뮬레이터에서 역운동학(IK)으로 계산한 관절 위치/속도
- `TrajectoryDictBuffer`에 저장 (읽기 전용, 동적 추가 불가)
- Backward Map(B)을 통해 z로 인코딩되어 **expert z**가 됨

**Policy 데이터** (`train`):
- 현재 정책이 환경과 상호작용하면서 수집하는 **온라인** 데이터
- `DictBuffer` 또는 `TrajectoryDictBufferMultiDim`에 저장 (FIFO 순환)
- 관측값, 행동, 보상, z 벡터, 종료 플래그 등이 함께 저장됨

`FBcprAgent.update()`에서 두 데이터를 함께 사용하는 방식:
```python
def update(self, replay_buffer, step):
    expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
    train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
    # Expert: Backward Map으로 z 인코딩 -> discriminator 학습의 "real" 데이터
    expert_z = self.encode_expert(next_obs=expert_next_obs)
    # Train: 저장된 z 사용 -> discriminator 학습의 "fake" 데이터
    train_z = train_batch["z"].to(self.device)
```

### 6.3 커리큘럼 학습과 우선순위 샘플링의 연결

우선순위 샘플링은 **커리큘럼 학습** 효과를 만들어낸다:

1. **평가**: 주기적으로 (`eval_every_steps`) 에이전트가 각 모션을 얼마나 잘 추적하는지 EMD로 측정
2. **우선순위 계산**: EMD가 높은 모션(잘 못 따라하는 모션)에 높은 우선순위 부여
3. **Expert 버퍼 업데이트**: `expert_slicer.update_priorities(priorities, idxs)`
4. **모션 라이브러리 업데이트**: 시뮬레이터의 모션 샘플링 가중치도 동시에 갱신
5. **효과**: 다음 학습 구간에서 어려운 모션에 더 집중하여 학습

이 과정이 `train.py`의 `eval()` 후 자동으로 수행된다:

```python
# 모션 라이브러리 가중치 업데이트 (시뮬레이터 레벨)
train_env._env._motion_lib.update_sampling_weight_by_id(
    priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
)
# Expert 버퍼 가중치 업데이트 (학습 레벨)
replay_buffer["expert_slicer"].update_priorities(
    priorities=priorities, idxs=torch.tensor(np.array(idxs))
)
```

---

## 7. 핵심 개념 정리

### 딕셔너리 기반 PyTree 구조

BFM-Zero의 버퍼 시스템은 관측값(observation)이 **중첩 딕셔너리**로 구성된 환경에 맞춤 설계되었다. 예를 들어:

```python
observation = {
    "state": tensor([batch, 67]),            # 기본 상태 (관절 위치/속도, 중력, 각속도)
    "last_action": tensor([batch, 29]),       # 이전 행동
    "privileged_state": tensor([batch, 217]), # 특권 관측값 (전신 상태)
    "history_actor": tensor([batch, 580]),    # 이력 정보
}
```

모든 버퍼 연산(extend, sample, save, load)은 이 중첩 구조를 **재귀적으로** 처리한다. PyTorch의 `tree_map()`을 활용하여 동일한 변환을 모든 리프 텐서에 적용한다.

### Nested 데이터 처리 유틸리티

`trajectory.py`에 정의된 헬퍼 함수들:

```python
def key_exists(data, key):   # 중첩 키 존재 확인
def set_key(data, key, value): # 중첩 키에 값 설정
def get_key(data, key):      # 중첩 키에서 값 추출
```

`end_key`가 튜플이면 중첩 딕셔너리를 재귀적으로 탐색한다. 문자열이면 최상위 키로 접근한다.

### 메모리 관리

1. **Pre-allocation**: `initialize_storage()`가 `capacity` 크기의 0-텐서를 미리 할당하여 동적 할당을 방지
2. **In-place 업데이트**: `extend()`가 기존 텐서에 슬라이스 대입으로 데이터를 덮어씀 (`storage[k][idx:end] = ...`)
3. **디바이스 분리**: `buffer_device`를 `cpu`로 설정하면 GPU 메모리를 절약할 수 있으나, 기본 설정(`cuda`)에서는 GPU에 저장하여 학습 시 전송 오버헤드를 제거
4. **`@torch.no_grad`**: `extend()`와 `sample()`에 적용하여 불필요한 그래디언트 추적 방지
5. **`torch.compile`**: `get_idxs` 함수를 `torch.compile(mode="reduce-overhead")`로 컴파일하여 인덱스 연산 최적화

### 기본 학습 설정에서의 버퍼 크기

`train_bfm_zero()` 함수의 기본값 기준:

| 파라미터 | 값 | 설명 |
|---------|------|------|
| `buffer_size` | 5,120,000 | 전체 버퍼 크기 (전이 수) |
| `online_parallel_envs` | 1,024 | 병렬 환경 수 |
| 실제 capacity | 5,000 | `buffer_size // online_parallel_envs` (시간 스텝) |
| `z_buffer_size` | 8,192 | Z 벡터 버퍼 크기 |
| `batch_size` | 1,024 | 미니배치 크기 |
| `seq_length` | 8 | 연속 서브시퀀스 길이 |
| `num_slices` | 128 | `batch_size // seq_length` (샘플링되는 서브시퀀스 수) |
