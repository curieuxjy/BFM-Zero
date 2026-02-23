# Phase 4: 평가 시스템 (Evaluation System)

## 1. 개요

BFM-Zero의 평가 시스템은 학습된 에이전트가 참조 모션을 얼마나 정확하게 재현하는지 측정하는 핵심 컴포넌트이다. 학습 과정에서 주기적으로 실행되어 모델 품질을 모니터링하고, **커리큘럼 학습(prioritized sampling)**에 직접적으로 피드백을 제공한다.

### 핵심 평가 지표: EMD (Earth Mover's Distance)

- 에이전트가 생성한 관절 궤적(trajectory)과 참조 모션 궤적 간의 **분포적 거리**를 측정
- 단순 프레임별 오차가 아닌, 두 궤적의 **전체적인 유사도**를 최적 수송(optimal transport) 관점에서 평가
- **학습 목표: `eval/emd < 0.75`** (50~100M 스텝 학습 후 달성 가능)

### 평가가 학습에 미치는 영향

평가 결과(모션별 EMD)는 expert 데이터 샘플링 확률을 갱신하는 데 사용된다. EMD가 높은(잘 못 따라하는) 모션은 학습 중 더 자주 샘플링되어, 전체적으로 균형 잡힌 트래킹 성능을 달성하도록 유도한다.


## 2. evaluations/base.py - 평가 프레임워크

파일 경로: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/evaluations/base.py`

이 파일은 모든 평가 설정의 기반이 되는 추상 클래스와 유틸리티 함수를 정의한다.

### 2.1 BaseEvalConfig

모든 평가 설정 클래스의 부모 클래스이다. `BaseConfig`(Pydantic 모델)를 상속하며, 공통 설정을 제공한다:

```python
class BaseEvalConfig(BaseConfig):
    """Abstract class for evaluation configurations."""
    generate_videos: bool = False       # 비디오 생성 여부
    videos_dir: str = "videos"          # 비디오 저장 디렉토리
    video_name_prefix: str = "unknown_agent"  # 비디오 파일명 접두사

    @classmethod
    def requires_replay_buffer(self):
        return False  # 대부분의 평가는 replay buffer 불필요
```

- `generate_videos`: 평가 중 에이전트의 동작을 비디오로 녹화할지 결정
- `requires_replay_buffer()`: 보상 기반 평가(G1EnvRewardEvaluation)에서만 `True`를 반환. 보상 추론에 replay buffer의 경험 데이터가 필요하기 때문

### 2.2 extract_model 함수

에이전트 객체에서 실제 모델을 추출하는 유틸리티:

```python
def extract_model(agent_or_model):
    if isinstance(agent_or_model, BaseModel):  # 이미 모델이면 그대로 반환
        return agent_or_model
    return agent_or_model._model  # 에이전트에서 ._model 추출
```

평가 함수들은 `agent_or_model`을 받아, 에이전트든 모델이든 일관되게 처리한다. 이렇게 하면 평가 코드에서 에이전트 전체 객체의 `act()` 메서드를 호출할 수도 있고, 모델만 따로 사용할 수도 있다.


## 3. evaluations/humanoidverse_isaac.py - 트래킹 평가 (핵심)

파일 경로: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/evaluations/humanoidverse_isaac.py`

이 파일이 BFM-Zero 학습의 **핵심 평가 모듈**이다. Isaac Sim/IsaacGym 시뮬레이터에서 에이전트의 모션 트래킹 능력을 직접 측정한다.

### 3.1 Episode 클래스

평가 중 한 에피소드의 데이터를 수집하는 데이터 컨테이너:

```python
@dataclasses.dataclass
class Episode:
    storage: Dict | None = None

    def initialise(self, observation, info):
        # 첫 관측 저장, defaultdict(list) 초기화
        ...

    def add(self, observation, reward, action, terminated, truncated, info):
        # 매 스텝마다 데이터 추가
        ...

    def get(self) -> Dict[str, np.ndarray]:
        # 수집된 데이터를 numpy 배열 딕셔너리로 반환
        ...
```

- `initialise()`: 에피소드 시작 시 첫 관측(observation)과 정보(info)를 저장
- `add()`: 매 시뮬레이션 스텝마다 observation, reward, action, terminated, truncated, info를 기록
- `get()`: 수집된 리스트 데이터를 `np.array`로 변환하여 반환. 이후 지표 계산에 사용
- info 키 중 `_`로 시작하거나 `final_observation`, `final_info`인 것은 무시 (gymnasium 내부 관례)

### 3.2 HumanoidVerseIsaacTrackingEvaluation 클래스

설정과 실행의 분리:

```python
class HumanoidVerseIsaacTrackingEvaluationConfig(BaseEvalConfig):
    name_in_logs: str = "humanoidverse_tracking_eval"
    env: HumanoidVerseIsaacConfig | None = None  # None이면 학습 환경 공유
    num_envs: int = 1024
    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False
    disable_tqdm: bool = True

    def build(self):
        return HumanoidVerseIsaacTrackingEvaluation(self)
```

- `env: None`이면 학습에 사용 중인 환경을 직접 공유한다. Isaac Sim은 Python 프로세스당 하나만 생성 가능하기 때문에 싱글톤 패턴이 강제된다
- `num_envs: 1024`는 병렬 환경 수. 모든 모션을 동시에 평가할 수 있도록 모션 수 이상으로 설정

**`run()` 메서드 - 평가 실행의 진입점:**

```python
def run(self, *, timestep, agent_or_model, logger, env=None, **kwargs):
    # 1. 환경 결정 (cfg.env or 전달받은 env)
    # 2. motion_ids 목록 확보 (모든 유니크 모션)
    self.motion_ids = list(range(env._env._motion_lib._num_unique_motions))

    # 3. 모션을 청크 단위로 나누어 평가
    for motion_id_chunk_start in range(0, len(self.motion_ids), n_envs):
        motion_id_chunk = self.motion_ids[...]
        motion_chunk_results = _async_tracking_worker(...)

    # 4. 결과 집계: 평균과 표준편차
    aggregate = collections.defaultdict(list)
    for _, metr in metrics.items():
        for k, v in metr.items():
            if isinstance(v, numbers.Number):
                aggregate[k].append(v)
    for k, v in aggregate.items():
        wandb_dict[k] = np.mean(v)
        wandb_dict[f"{k}#std"] = np.std(v)

    # 5. 평가 후 정리: 학습 모드 복원
    env._env._motion_lib.load_motions_for_training()
    env._env.set_is_training()
    return metrics, wandb_dict
```

핵심 포인트:
- 평가가 끝나면 반드시 `load_motions_for_training()`과 `set_is_training()`을 호출하여 학습 환경 상태를 복원
- 반환값은 `(metrics, wandb_dict)` 튜플. metrics는 모션별 상세 결과, wandb_dict는 집계된 평균/표준편차

### 3.3 Expert 궤적 수집과 z 계산 (get_backward_observation)

평가의 첫 단계는 각 모션에 대한 참조 관측(backward observation)을 구성하고, 이를 통해 잠재 벡터 z를 계산하는 것이다.

```python
def get_backward_observation(env, motion_id, include_last_action, velocity_multiplier=1.0):
    # 1. 모션 길이에 맞는 시간 인덱스 생성
    motion_times = torch.arange(
        int(np.ceil((env._motion_lib._motion_lengths[motion_id] / env.dt).cpu()))
    ).to(env.device) * env.dt

    # 2. motion library에서 참조 상태 추출
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)
    ref_body_pos = motion_state["rg_pos_t"]      # 강체 위치
    ref_body_rots = motion_state["rg_rot_t"]      # 강체 회전 (쿼터니언)
    ref_body_vels = motion_state["body_vel_t"]    # 강체 선형 속도
    ref_body_angular_vels = motion_state["body_ang_vel_t"]  # 강체 각속도
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]  # 기본 자세 빼기
    ref_dof_vel = motion_state["dof_vel"]

    # 3. compute_humanoid_observations_max로 관측 벡터 구성
    obs_dict = compute_humanoid_observations_max(...)
    max_local_self_obs = torch.cat([v for v in obs_dict.values()], dim=-1)

    # 4. G1Env 호환 관측 구성
    g1env_state = torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1)
    g1env_obs = {"state": g1env_state, "privileged_state": max_local_self_obs}
    return g1env_obs, ref_dict
```

이 함수가 반환하는 `g1env_obs`는 모션의 "미래 상태"를 backward 네트워크에 입력하여 z를 계산하는 데 사용된다. `ref_dict`에는 초기화용 참조 데이터(위치, 회전, 속도 등)가 포함된다.

**z 계산 과정 (_async_tracking_worker 내부):**

```python
# tracking_target: 모션의 모든 프레임에 대한 관측
tracking_target, tracking_target_dict = get_backward_observation(env._env, m_id, ...)

# backward_map: 관측 -> 잠재 벡터 z (미래 상태의 "표현")
# 첫 번째 z는 "다음 상태"를 추적해야 하므로 [1:]부터 사용
z = model.backward_map(tree_map(lambda x: x[1:], tracking_target)).clone()

# z 평활화: 현재 스텝에서 끝까지의 평균으로 대체
# -> 에이전트가 미래 전체를 고려한 행동을 취하게 함
for step in range(z.shape[0]):
    end_idx = min(step + 1, z.shape[0])
    z[step] = z[step:end_idx].mean(dim=0)

# z를 정규화된 컨텍스트로 변환
ctx = model.project_z(z)
```

이 과정의 핵심 insight:
1. `backward_map`은 "이 관측에 도달하려면 어떤 행동 의도(z)가 필요한가"를 계산
2. z를 미래 방향으로 평균하면 **장기적으로 일관된 모션 추적**이 가능해진다
3. `project_z`는 z를 단위 구(unit sphere) 위에 정규화: `sqrt(dim) * normalize(z)`

### 3.4 Body Tracking 대상 (xpos_bodies)

평가 시 추적하는 신체 부위 목록 (24개 링크, 손목 제외):

```python
xpos_bodies = [
    "pelvis",                        # 골반 (루트)
    "left_hip_pitch_link",           # 왼쪽 고관절 피치
    "left_hip_roll_link",            # 왼쪽 고관절 롤
    "left_hip_yaw_link",             # 왼쪽 고관절 요
    "left_knee_link",                # 왼쪽 무릎
    "left_ankle_pitch_link",         # 왼쪽 발목 피치
    "left_ankle_roll_link",          # 왼쪽 발목 롤
    "right_hip_pitch_link", ...      # 오른쪽 다리 (대칭)
    "waist_yaw_link",                # 허리 요
    "waist_roll_link",               # 허리 롤
    "torso_link",                    # 몸통
    "left_shoulder_pitch_link", ...  # 왼팔 (어깨~팔꿈치)
    "right_shoulder_pitch_link", ... # 오른팔 (어깨~팔꿈치)
    # 손목 링크는 주석 처리: wrist_roll, wrist_pitch, wrist_yaw
]
```

손목을 제외한 이유: G1 로봇의 29 DOF 제어에서 손목은 미세 조작용이므로, 전체 자세 트래킹 평가에서는 노이즈만 추가할 수 있어 제외하였다.

### 3.5 모션-환경 할당과 도메인 랜덤화

```python
def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    env_idxs = list(range(num_envs))
    random.shuffle(env_idxs)  # 환경 인덱스를 셔플!
    shuffled_env_idxs = torch.tensor(env_idxs, device=device, dtype=torch.long)
    assigned = motion_ids[shuffled_env_idxs % num_motions]
    ...
```

**핵심 설계 의도**: Isaac Sim은 환경(env)마다 **고정된** 도메인 랜덤화(질량, 마찰계수 등)를 적용한다. 만약 같은 모션을 항상 같은 환경에서 평가하면, 특정 물리 파라미터에서만 평가하는 편향이 생긴다. 셔플을 통해 매 평가마다 모션이 다른 환경(다른 물리 파라미터)에서 실행되어 **공정한 평가**가 이루어진다.

### 3.6 평가 실행 루프 (_async_tracking_worker)

평가의 실제 실행 로직. 핵심 흐름:

```python
def _async_tracking_worker(inputs, env, disable_tqdm, include_results_from_all_envs):
    # 1. 모션 라이브러리에서 모든 모션 로드
    if not isaac_env._motion_lib.all_motions_loaded:
        isaac_env._motion_lib.load_motions(random_sample=False, ...)

    # 2. 모션을 환경에 할당
    assigned_motions, motion_to_envs = group_assign_motions_to_envs_with_map(motion_ids, num_envs)

    # 3. 각 모션에 대해 ctx(컨텍스트)와 초기 상태 사전 계산
    for m_id in motion_ids:
        tracking_target, tracking_target_dict = get_backward_observation(env._env, m_id, ...)
        z = model.backward_map(tree_map(lambda x: x[1:], tracking_target)).clone()
        ctx = model.project_z(z)
        ctx_dict[m_id] = ctx
        # 초기 DOF/루트 상태도 모션의 첫 프레임에서 추출
        ...

    # 4. 환경 리셋 (모션 첫 프레임의 자세로)
    observation, info = env.reset(target_states=target_states, to_numpy=False)

    # 5. 시뮬레이션 루프: 모든 환경에서 동시에 실행
    for step in tqdm(range(max_ctx_len)):
        # 각 환경에 해당 모션의 시간 스텝별 ctx 배치
        ctx_batch = []
        for env_id in range(num_envs):
            m_id = assigned_motions[env_id].item()
            ctx_t = ctx_dict[m_id][step % ctx.shape[0]]
            ctx_batch.append(ctx_t)
        ctx_batch = torch.stack(ctx_batch)

        # 에이전트의 행동 결정 (mean=True로 결정적 정책)
        action = agent.act(observation, ctx_batch, mean=True)
        observation, reward, terminated, truncated, info = env.step(action, to_numpy=False)

        # 강체 위치, 관절 상태 기록
        xpos_log.append(isaac_env.simulator._rigid_body_pos.reshape(num_envs, -1, 3))
        joint_pos.append(isaac_env.simulator.dof_state[..., 0])

    # 6. 모션별 지표 계산
    for m_id, envs_with_current_motion in motion_to_envs.items():
        local_metrics = _calc_metrics({
            "tracking_target": tracking_targets[m_id],
            "motion_id": m_id,
            "motion_file": isaac_env._motion_lib.curr_motion_keys[m_id],
            "observation": ...,
            "joint_pos": ...,
            "target_joint_pos": ...,
        })
```

주목할 점:
- `mean=True`: 평가 시에는 확률적 정책이 아닌 **결정적 정책**(평균 액션)을 사용
- `step % ctx.shape[0]`: 모션 길이가 다른 경우 순환(modulo) 처리. 가장 긴 모션이 끝날 때까지 모든 환경이 동작
- 모든 환경이 **동시에** 시뮬레이션되어 GPU 활용도를 극대화

### 3.7 EMD (Earth Mover's Distance) 계산 방식

```python
def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix=""):
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # 비용 행렬 계산: 유클리드 거리
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    # 균등 분포 가정
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    # POT 라이브러리로 최적 수송 비용 계산
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}
```

**거리 행렬 계산:**
```python
def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))
```

EMD 계산의 의미:
1. 에이전트가 실제로 생성한 궤적의 각 프레임(`agent_obs`)과 참조 모션의 각 프레임(`tracked_obs`)을 점(point)으로 취급
2. 두 점 집합 간의 **비용 행렬**을 유클리드 거리로 계산 (T_agent x T_target 행렬)
3. 각 점 집합에 **균등한 확률 질량**을 부여 (1/N)
4. `ot.emd2()`로 한 분포를 다른 분포로 변환하는 **최소 비용**을 계산
5. 이 비용이 낮을수록 두 궤적이 유사하다는 의미

**왜 프레임별 L2가 아닌 EMD를 사용하는가?**
- 프레임별 L2는 시간적 정렬에 민감: 동일 모션이라도 약간의 시간 지연이 있으면 큰 오차로 잡힘
- EMD는 **순서에 관계없이** 전체 궤적의 분포적 유사도를 측정하므로, 시간적 미세 차이에 강건(robust)

### 3.8 추가 지표: distance, proximity, MPJPE

```python
def distance_proximity(next_obs, tracking_target, bound=2.0, margin=2, prefix=""):
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    # proximity: bound 이내면 1.0, 초과하면 선형 감쇠, bound+margin 초과면 0
    stats[f"{prefix}proximity"] = (
        in_bounds_mask +
        ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)
    ).mean()
    stats[f"{prefix}distance"] = dist.mean()
```

- `proximity`: 0~1 사이의 값으로, 1에 가까울수록 참조에 가깝다는 의미
- `distance`: 프레임별 평균 L2 거리 (시간 정렬된 상태에서의 직접 비교)

```python
def compute_joint_pos_metrics(joint_pos, target_joint_pos):
    # MPJPE (Mean Per Joint Position Error) - mm 단위
    stats["mpjpe_l"] = torch.norm(joint_pos - target_joint_pos, dim=-1).mean(-1) * 1000

    # 속도 오차 (유한 차분으로 계산)
    vel_gt = target_joint_pos[:, 1:] - target_joint_pos[:, :-1]
    vel_pred = joint_pos[:, 1:] - joint_pos[:, :-1]
    stats["vel_dist"] = torch.norm(vel_pred - vel_gt, dim=-1).mean(-1) * 1000

    # 가속도 오차 (2차 유한 차분)
    accel_gt = target_joint_pos[:, :-2] - 2 * target_joint_pos[:, 1:-1] + target_joint_pos[:, 2:]
    accel_pred = joint_pos[:, :-2] - 2 * joint_pos[:, 1:-1] + joint_pos[:, 2:]
    stats["accel_dist"] = torch.norm(accel_pred - accel_gt, dim=-1).mean(-1) * 100
```

지표 해석:
- **`mpjpe_l`**: 관절 위치 정확도 (밀리라디안 단위). 절대적인 자세 정확도
- **`vel_dist`**: 속도 추적 정확도. 모션의 동적 특성을 얼마나 잘 재현하는지
- **`accel_dist`**: 가속도 추적 정확도. 모션의 **부드러움(smoothness)**을 측정

### 3.9 지표 계산 통합 (_calc_metrics)

```python
def _calc_metrics(ep):
    metr = {}
    # 관측 공간에서의 EMD/proximity (처음 23차원: 관절 위치만 사용)
    QPOS_START = 23 + 3   # (코드 상단에 정의)
    QPOS_END = 23 + 3 + 23
    QVEL_IDX = 23

    next_obs = torch.tensor(ep["observation"]["state"][:, :QVEL_IDX], dtype=torch.float32)
    tracking_target = torch.tensor(ep["tracking_target"]["state"][:, :QVEL_IDX], dtype=torch.float32)

    # 1. 관측 공간 기반 지표
    metr.update(distance_proximity(next_obs, tracking_target, prefix="obs_state_"))
    metr.update(emd_numpy(next_obs, tracking_target, prefix="obs_state_"))

    # 2. 관절 위치 기반 지표 (MPJPE, vel, accel)
    phc_metrics = compute_joint_pos_metrics(joint_pos=ep["joint_pos"], target_joint_pos=ep["target_joint_pos"])
    metr.update(phc_metrics)

    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}  # 모션 파일명을 키로 사용
```

중요: `state[:, :QVEL_IDX]`에서 `QVEL_IDX=23`은 관절 위치(dof_pos)의 차원 수. state는 `[dof_pos(23) | dof_vel(23) | projected_gravity(3) | ang_vel(3)]`으로 구성되어 있고, EMD 계산에는 **관절 위치만** 사용한다. 속도와 중력 벡터는 노이즈가 크기 때문.


## 4. evaluations/g1env.py - G1 환경 평가

파일 경로: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/evaluations/g1env.py`

MuJoCo 기반의 G1 환경에서 평가를 수행하는 모듈. 트래킹 평가와 보상 기반 평가 두 가지를 지원한다.

### 4.1 G1EnvTrackingEvaluation - 트래킹 평가

```python
class G1EnvTrackingEvaluationConfig(BaseEvalConfig):
    name: Literal["g1env_tracking_eval"] = "g1env_tracking_eval"
    motions: str               # 모션 데이터 경로
    motions_root: str          # 모션 루트 디렉토리
    tracking_env_cfg: G1EnvConfig | G1EnvRandConfig  # MuJoCo 환경 설정
    num_envs: int = 50         # 병렬 환경 수 (Isaac보다 훨씬 적음)

class G1EnvTrackingEvaluation:
    def run(self, *, timestep, agent_or_model, logger, **kwargs):
        model = extract_model(agent_or_model)
        eval_agent = TrackingWrapper(model=model)   # 모델을 트래킹 래퍼로 감쌈
        tracking_eval = TrackingEvaluation(
            motions=self.cfg.motions,
            motion_base_path=self.cfg.motions_root,
            env_config=self.cfg.tracking_env_cfg,
            num_envs=self.cfg.num_envs,
        )
        tracking_metrics = tracking_eval.run(agent=eval_agent, disable_tqdm=True)
        # 결과 집계 (humanoidverse_isaac.py와 동일한 패턴)
        ...
```

### 4.2 G1EnvRewardEvaluation - 보상 기반 평가

```python
class G1EnvRewardEvaluationConfig(BaseEvalConfig):
    tasks: list[str]                    # 평가할 태스크 목록
    reward_env_cfg: G1EnvConfig | G1EnvRandConfig
    num_episodes: int = 10
    max_workers: int = 12
    process_executor: bool = True
    num_inference_workers: int = 1
    num_inference_samples: int = 50_000

    @classmethod
    def requires_replay_buffer(self):
        return True  # 보상 추론에 replay buffer 필요!

class G1EnvRewardEvaluation:
    def run(self, *, timestep, agent_or_model, replay_buffer, logger, **kwargs):
        model = extract_model(agent_or_model)
        eval_agent = RewardWrapper(
            model=model,
            inference_dataset=replay_buffer["train"],  # 학습 버퍼 사용
            num_samples_per_inference=self.cfg.num_inference_samples,
            inference_function="reward_wr_inference",
            ...
        )
        reward_eval = RewardEvaluation(tasks=self.cfg.tasks, ...)
        reward_metrics = reward_eval.run(agent=eval_agent, disable_tqdm=True)
```

보상 기반 평가의 핵심 차이점: `RewardWrapper`는 replay buffer에서 (qpos, qvel, action) 샘플을 추출하고, 보상 함수로 relabeling한 뒤, `reward_wr_inference`를 통해 **보상을 최대화하는 z**를 추론한다.

### 4.3 humanoidverse_isaac.py와의 차이점

| 특성 | humanoidverse_isaac.py | g1env.py |
|------|----------------------|----------|
| **시뮬레이터** | Isaac Sim/IsaacGym (GPU) | MuJoCo (CPU) |
| **병렬화** | GPU 벡터 환경 (1024개) | ProcessPoolExecutor (50개) |
| **환경 생성** | 싱글톤 (학습 환경 공유) | 매번 새로 생성 가능 |
| **z 계산** | 직접 backward_map + project_z | TrackingWrapper.tracking_inference |
| **도메인 랜덤화** | 환경별 고정, 셔플로 대응 | 설정에 따라 G1EnvRandConfig |
| **보상 평가** | 미지원 | G1EnvRewardEvaluation으로 지원 |
| **xpos 추적** | 코드에 있으나 주석 처리됨 | 활성화 (body xpos 기반 EMD도 계산) |
| **성공 판정** | 미포함 | `success_phc_linf_xpos`, `success_phc_mean_xpos` 포함 |


## 5. agents/envs/ - 환경 통합 레이어

디렉토리 경로: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/envs/`

### 5.1 HumanoidVerseIsaacConfig

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/envs/humanoidverse_isaac.py`

Isaac 환경의 전체 설정 및 빌드를 담당하는 Pydantic 설정 클래스:

```python
class HumanoidVerseIsaacConfig(BaseConfig):
    name: Literal["humanoidverse_isaac"] = "humanoidverse_isaac"
    device: str = "cuda:0"
    lafan_tail_path: str                    # 모션 데이터 경로
    enable_cameras: bool = False            # 카메라 렌더링 활성화
    max_episode_length_s: float | None = None  # 최대 에피소드 길이 (초)
    disable_obs_noise: bool = False         # 관측 노이즈 비활성화
    disable_domain_randomization: bool = False  # 도메인 랜덤화 비활성화
    include_last_action: bool = True        # 이전 행동을 관측에 포함
    hydra_overrides: List[str] = []         # Hydra 설정 오버라이드
    context_length: int | None = None       # 히스토리 컨텍스트 길이
    include_history_actor: bool = False     # 액터 히스토리 포함
    root_height_obs: bool = False           # 루트 높이 관측 포함
```

**`build()` 메서드 - 싱글톤 환경 생성:**

```python
def build(self, num_envs: int = 1):
    global _humanoidverse_env_singleton
    if _humanoidverse_env_singleton is not None:
        # 이미 존재하면 설정 일치 확인 후 반환
        if num_envs != _humanoidverse_env_singleton.num_envs:
            raise ValueError("...")
        return _humanoidverse_env_singleton, {}

    # Hydra로 Isaac 설정 로드 및 커스텀화
    with hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_DIR):
        cfg = hydra.compose(config_name=self.relative_config_path, overrides=hydra_overrides)

    # 안전성 검증: 종료 조건이 모두 꺼져 있는지 확인
    assert cfg.env.config.termination.terminate_when_close_to_dof_pos_limit is False
    assert cfg.env.config.termination.terminate_by_contact is False
    # ... (7개 조건 모두 False 확인)

    # 환경 생성
    isaac_env = LeggedRobotMotions(cfg.env.config, device=self.device)
    env = HumanoidVerseVectorEnv(isaac_env, ...)
    _humanoidverse_env_singleton = env
    return env, {"unresolved_conf": unresolved_conf}
```

중요한 설계 결정:
- **종료 조건 전부 비활성화**: BFM-Zero는 unsupervised RL이므로, 에피소드 조기 종료(넘어짐, 접촉 등)를 사용하지 않는다. 에이전트가 다양한 상태를 자유롭게 탐색하도록 허용
- **싱글톤 패턴**: Isaac Sim은 프로세스당 하나만 존재할 수 있으므로 강제

### 5.2 HumanoidVerseVectorEnv

Gymnasium `VectorEnv`를 상속하여 HumanoidVerse 환경을 표준 인터페이스로 감싼다:

```python
class HumanoidVerseVectorEnv(VectorEnv):
    def __init__(self, env: LeggedRobotMotions, ...):
        self._env = env
        # G1Env 호환 관측 공간 구성
        # 히스토리 핸들러 초기화
        # 기본 자세 리셋 타겟 사전 계산

    def _get_g1env_observation(self, to_numpy=True):
        """Isaac 상태를 G1Env 관측 형식으로 변환"""
        raw_obs = self._env.obs_buf_dict_raw["actor_obs"]
        g1env_state = torch.cat([
            raw_obs["dof_pos"],           # 관절 위치
            raw_obs["dof_vel"],           # 관절 속도
            raw_obs["projected_gravity"], # 투영된 중력
            raw_obs["base_ang_vel"],      # 기저 각속도
        ], dim=-1)
        privileged_state = raw_obs["max_local_self"]  # 전체 강체 관측
        observation = {
            "state": g1env_state,
            "privileged_state": privileged_state,
        }
```

이 래퍼의 핵심 역할:
1. **관측 변환**: Isaac Sim 내부 형식 -> G1Env 호환 딕셔너리 형식
2. **step/reset 표준화**: Gymnasium VectorEnv 인터페이스 준수
3. **히스토리 관리**: 컨텍스트 기반 학습을 위한 과거 관측/행동 추적
4. **qpos/qvel 추출**: MuJoCo 형식의 상태를 함께 제공 (보상 relabeling 등에 사용)

### 5.3 IsaacRendererWithMuJoco

Isaac Sim 상태를 MuJoCo로 시각화하는 렌더러:

```python
class IsaacRendererWithMuJoco:
    """Isaac 상태를 MuJoCo로 렌더링. 29 DOF (36-D qpos: 7 free + 29 joints)만 지원"""

    def __init__(self, render_size=512):
        self.mujoco_env, _ = G1EnvConfig(render_height=render_size, render_width=render_size).build(num_envs=1)

    def render(self, hv_env, env_idxs=None):
        # Isaac의 루트 상태 + 관절 위치를 MuJoCo qpos로 변환
        base_pos = hv_env.simulator.robot_root_states[:, [0,1,2,6,3,4,5]]  # (x,y,z,w,qx,qy,qz) 순서 변환
        joint_pos = hv_env.simulator.dof_pos
        mujoco_qpos = np.concatenate([base_pos, joint_pos], axis=1)  # (n_envs, 36)
        # 각 환경에 대해 MuJoCo 렌더링
        for env_idx in env_idxs:
            self.mujoco_env.reset(options={"qpos": mujoco_qpos[env_idx], "qvel": qvel})
            all_images.append(self.mujoco_env.render())
```

### 5.4 Expert 궤적 로딩 (load_expert_trajectories_from_motion_lib)

학습 시작 시 모션 라이브러리에서 expert 데이터를 추출하는 함수:

```python
def load_expert_trajectories_from_motion_lib(env, agent_cfg, device="cpu", add_history_noaction=False):
    env._motion_lib.load_motions_for_training()  # 모든 모션 로드
    episodes = []
    for i in range(env._motion_lib._num_unique_motions):
        # 각 모션에 대해:
        # 1. 시간 인덱스 생성
        # 2. motion_lib에서 상태 추출
        # 3. compute_humanoid_observations_max로 관측 구성
        # 4. G1Env 형식의 state 구성: [dof_pos, dof_vel, projected_gravity, ang_vel]
        # 5. 에피소드 딕셔너리 생성

        ep = {
            "observation": {
                "state": state,
                "last_action": bogus_actions,  # 참조 데이터에는 실제 action이 없으므로 0
                "privileged_state": max_local_self_obs,
            },
            "terminated": torch.zeros(...),
            "truncated": truncated,  # 모션 마지막 프레임에서만 True
            "motion_id": torch.ones(...) * i,
        }
        episodes.append(ep)

    expert_buffer = TrajectoryDictBuffer(episodes=episodes, seq_length=agent_cfg.model.seq_length, ...)
```

이 버퍼는 backward 네트워크의 학습에 사용된다: 모션 라이브러리의 참조 궤적을 관측 공간으로 변환하여, 에이전트가 "도달해야 할 상태"의 표현(z)을 학습할 수 있게 한다.


## 6. 평가 지표 정리

### 6.1 EMD란 무엇인가

**Earth Mover's Distance (Wasserstein-1 Distance)**는 두 확률 분포 사이의 거리를 "한 분포를 다른 분포로 옮기는 데 필요한 최소 작업량"으로 정의한다.

BFM-Zero에서의 구체적 적용:
1. 에이전트 궤적: `{s_1, s_2, ..., s_T}` (T개 프레임, 각각 23차원 관절 위치)
2. 참조 궤적: `{s*_1, s*_2, ..., s*_T}` (동일 구조)
3. 각 점에 균등 질량 `1/T` 부여
4. 한 점 집합에서 다른 집합으로 질량을 옮기는 최소 비용 = EMD

**수식으로:**
```
EMD = min_{gamma} sum_{i,j} gamma_{i,j} * ||s_i - s*_j||_2
subject to: sum_j gamma_{i,j} = 1/T,  sum_i gamma_{i,j} = 1/T
```

여기서 `gamma`는 수송 계획(transport plan), `||.||_2`는 유클리드 거리.

### 6.2 eval/emd < 0.75 기준의 의미

CLAUDE.md에서 명시한 학습 목표:
> Training target: After 50-100M steps, eval/emd should be lower than 0.75.

이 기준의 해석:
- **모든 평가 모션의 평균 EMD**가 0.75 미만이어야 함
- 0.75는 23차원 관절 위치 공간에서의 Wasserstein 거리. 대략 각 관절이 평균적으로 ~0.03 라디안 이내의 오차를 가진다는 의미
- EMD가 0.75 이상이면 에이전트가 참조 모션의 전반적인 자세 분포를 제대로 재현하지 못하고 있다는 의미
- 학습 초기(~10M 스텝)에는 EMD가 1.5~2.0 이상일 수 있으며, 학습이 진행됨에 따라 점진적으로 감소

### 6.3 모션별 평가 결과 해석

`_calc_metrics`가 반환하는 지표 요약:

| 지표 | 의미 | 좋은 값 |
|------|------|---------|
| `obs_state_emd` | 관측 공간 EMD (관절 위치 기반) | < 0.75 |
| `obs_state_distance` | 프레임별 평균 L2 거리 | 낮을수록 좋음 |
| `obs_state_proximity` | bound 이내 비율 (0~1) | 1에 가까울수록 좋음 |
| `emd` | 관절 위치 직접 비교 EMD | 낮을수록 좋음 |
| `mpjpe_l` | Mean Per Joint Position Error (mm) | 낮을수록 좋음 |
| `vel_dist` | 속도 추적 오차 (mm/step) | 낮을수록 좋음 |
| `accel_dist` | 가속도 추적 오차 | 낮을수록 좋음 |
| `distance` / `proximity` | 관절 위치 직접 거리/근접도 | 낮음 / 높음 |

wandb에 로깅될 때는 각 지표의 **평균**과 **표준편차(#std)**가 함께 기록된다:
```python
wandb_dict[k] = np.mean(v)           # e.g., "eval/humanoidverse_tracking_eval/obs_state_emd"
wandb_dict[f"{k}#std"] = np.std(v)   # e.g., "eval/humanoidverse_tracking_eval/obs_state_emd#std"
```


## 7. 학습-평가 연결

### 7.1 train.py에서 평가가 호출되는 시점

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/train.py`

```python
# Workspace.__init__에서 평가 인스턴스 생성
self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}

# 학습 루프 내에서 주기적 호출
eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
# ...
for t in range(self._checkpoint_time, self.cfg.num_env_steps + ...):
    if (self.evaluate and eval_time_checker.check(t)) or (self.evaluate and t == self._checkpoint_time):
        eval_metrics = self.eval(t, replay_buffer=replay_buffer)
```

기본 설정에서:
- `eval_every_steps = 9_600_000` (약 960만 스텝마다)
- `online_parallel_envs = 1024`이므로 약 9,375 업데이트 반복마다
- 총 `384_000_000` 스텝 학습 중 약 40회 평가 수행

**평가 전후 환경 관리:**
```python
# 평가 전: 에이전트를 평가 모드로 전환
self.agent._model.train(False)

# 평가 실행 (학습 환경을 공유)
evaluation_metrics, wandb_dict = evaluation.run(
    timestep=t, agent_or_model=self.agent, ..., env=self.train_env
)

# 평가 후: 환경 리셋 필수 (Isaac에서는 평가 중 환경 상태가 변경됨)
if uses_humanoidverse_eval:
    td, info = train_env.reset()
    terminated = np.zeros(...)

# 에이전트를 학습 모드로 복원
self.agent._model.train()
```

또한 평가 **직전** 스텝에서는 truncated를 강제로 True로 설정하여 깔끔한 에피소드 경계를 만든다:
```python
next_t = t + self.cfg.online_parallel_envs
if self.evaluate and eval_time_checker.check(next_t):
    if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
        new_truncated = np.ones_like(new_truncated, dtype=bool)
```

### 7.2 평가 결과가 학습에 미치는 영향 (커리큘럼/우선순위 학습)

평가 결과는 두 가지 경로로 학습에 피드백된다:

**경로 1: 모션 라이브러리 샘플링 확률 갱신**

```python
if self.cfg.prioritization:
    # 1. 각 모션의 EMD 추출
    for _, metr in eval_metrics[self.priorization_eval_name].items():
        motions_id.append(metr["motion_id"])
        priorities.append(metr["emd"])

    # 2. EMD를 우선순위로 변환 (클램핑 + 스케일링)
    priorities = torch.clamp(
        torch.tensor(priorities),
        min=self.cfg.prioritization_min_val,   # 0.5
        max=self.cfg.prioritization_max_val,   # 2.0
    ) * self.cfg.prioritization_scale            # 2.0

    # 3. 우선순위 모드에 따른 변환
    if self.cfg.prioritization_mode == "exp":
        priorities = 2 ** priorities  # 지수적 스케일링
    elif self.cfg.prioritization_mode == "bin":
        # 비닝: 같은 EMD 범위의 모션은 동일 확률
        bins = torch.floor(priorities)
        for i in range(int(bins.min()), int(bins.max()) + 1):
            mask = bins == i
            if n > 0:
                priorities[mask] = 1 / n

    # 4. 모션 라이브러리의 샘플링 분포 갱신
    train_env._env._motion_lib.update_sampling_weight_by_id(
        priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer
    )
```

기본 설정에서 `prioritization_mode="exp"`, `prioritization_scale=2.0`이므로:
- EMD=0.5인 모션: 우선순위 = 2^(0.5*2) = 2^1 = 2
- EMD=1.5인 모션: 우선순위 = 2^(1.5*2) = 2^3 = 8 (4배 더 자주 샘플링)
- EMD=2.0인 모션: 우선순위 = 2^(2.0*2) = 2^4 = 16 (8배 더 자주 샘플링)

즉, **잘 못 따라하는 모션일수록 지수적으로 더 많이 연습**하게 된다.

**경로 2: Expert 버퍼 샘플링 확률 갱신**

```python
    # expert 버퍼도 동일한 우선순위로 갱신
    replay_buffer["expert_slicer"].update_priorities(
        priorities=priorities.to(self.cfg.buffer_device),
        idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
    )
```

이를 통해 backward 네트워크가 학습할 때도 잘 못 추적하는 모션의 expert 궤적을 더 자주 학습하게 된다.

### 7.3 전체 데이터 흐름 요약

```
[학습 루프]
    |
    v (매 eval_every_steps마다)
[평가 실행]
    |-- backward_map(참조 모션) -> z 계산
    |-- project_z(z) -> ctx 생성
    |-- 시뮬레이션 루프: agent.act(obs, ctx) -> action -> env.step
    |-- _calc_metrics: EMD, MPJPE, distance, proximity 등 계산
    |
    v
[결과 피드백]
    |-- wandb 로깅 (모니터링)
    |-- motion_lib.update_sampling_weight_by_id (환경에서 더 어려운 모션 자주 등장)
    |-- expert_buffer.update_priorities (expert 데이터에서 더 어려운 모션 자주 학습)
    |
    v
[다음 학습 주기] -> 어려운 모션에 집중 -> EMD 점진적 하락
```

이 피드백 루프가 BFM-Zero의 **커리큘럼 학습**의 핵심이다. 평가 없이는 모든 모션이 균등 확률로 샘플링되어, 쉬운 모션에 학습이 편향될 수 있다.
