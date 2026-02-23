# Phase 7: Goal/Reward Inference 스크립트 심층 분석

## 1. 개요

BFM-Zero는 세 가지 inference 모드를 지원하며, 각각 잠재 벡터 `z`를 획득하는 방법이 다르다. `z`는 Forward-Backward(FB) 표현 학습에서 학습된 행동 임베딩(behavior embedding)으로, 정책(actor)이 `z`를 조건으로 받아 휴머노이드 로봇의 행동을 결정한다.

| Inference 모드 | z 획득 방식 | 핵심 아이디어 |
|---|---|---|
| **Tracking** | `z = B(expert_trajectory)` | 전문가 모션 시퀀스를 Backward map으로 인코딩 |
| **Goal** | `z = B(goal_pose)` | 단일 목표 프레임을 Backward map으로 인코딩 |
| **Reward** | `z = sum(w_i * r_i * B(s_i))` | 보상 함수로 가중 평균된 Backward 특징 벡터 |

세 모드 모두 핵심적으로 **Backward Map `B`**를 사용하지만, 입력의 성격과 집계 방식이 다르다.


## 2. tracking_inference.py 복습

### 2.1 z = B(expert_trajectory) 방식

Tracking inference는 전문가의 모션 시퀀스 전체를 Backward map에 통과시켜 프레임별 `z`를 얻고, 이를 시간 윈도우로 평균하여 사용한다.

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/tracking_inference.py`

```python
# tracking_inference.py 내부 로컬 함수 (line 69-74)
def tracking_inference(obs) -> torch.Tensor:
    z = model.backward_map(obs)
    for step in range(z.shape[0]):
        end_idx = min(step + 1, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return model.project_z(z)
```

모델 내부의 `tracking_inference` 메서드 (`FBModel`, line 156-161):

```python
def tracking_inference(self, next_obs):
    z = self.backward_map(next_obs)
    for step in range(z.shape[0]):
        end_idx = min(step + self.cfg.seq_length, z.shape[0])
        z[step] = z[step:end_idx].mean(dim=0)
    return self.project_z(z)
```

**차이점**: 스크립트에서는 `seq_length=1`을 사용하여 미래 평균을 하지 않지만, 모델 메서드에서는 `self.cfg.seq_length`만큼의 윈도우로 미래 프레임을 평균한다.

### 2.2 전체 흐름 요약

1. 모델 로드 및 ONNX 내보내기
2. 환경 생성 (`HumanoidVerseIsaacConfig.build()`)
3. `env.set_is_evaluating(MOTION_ID)` -- 특정 모션 설정
4. `get_backward_observation(env, motion_id)` -- 모션 라이브러리에서 전체 프레임의 관측 구성
5. `tracking_inference(obs)` -- Backward map으로 프레임별 `z` 계산
6. 롤아웃: `model.act(observation, z[i], mean=True)` -- `z[i]`를 시간 순으로 변경하며 실행
7. 비디오 렌더링 (선택)


## 3. goal_inference.py - Goal-Reaching Inference

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/goal_inference.py`

### 3.1 전체 구조와 실행 흐름

Goal inference는 **단일 목표 자세(goal pose)**를 Backward map에 통과시켜 하나의 `z`를 얻는다. 이 `z`로 정책을 구동하면 로봇이 해당 자세에 도달하려 시도한다.

전체 흐름:

```
1. 모델 로드 + config 로드
2. ONNX 내보내기
3. 환경 생성
4. goal_frames JSON 로드 (목표 프레임 정의)
5. 각 goal에 대해:
   a. env.set_is_evaluating(motion_id)
   b. get_backward_observation() 으로 모션 전체 관측 획득
   c. 특정 프레임 인덱스의 관측만 추출
   d. model.goal_inference(goal_observation) -> z
   e. z_dict에 저장
6. z_dict를 goal_reaching.pkl로 저장
7. (선택) 저장된 z들로 순회하며 비디오 렌더링
```

### 3.2 Goal Frame 정의 (JSON에서 로딩)

목표 프레임은 JSON 파일에서 로드된다.

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/data/robots/g1/goal_frames_lafan29dof.json`

```json
[
    {
        "motion_id": 0,
        "frames": [2193, 2230, 3350, 5015],
        "motion_name": "fallAndGetUp1_subject4"
    },
    {
        "motion_id": 1,
        "frames": [505, 4024, 4700],
        "motion_name": "dance1_subject3"
    },
    ...
]
```

각 항목은:
- `motion_id`: 모션 라이브러리 내의 모션 인덱스
- `frames`: 해당 모션 내에서 목표로 사용할 프레임 인덱스 리스트
- `motion_name`: 모션의 이름 (파일명 기반)

이 JSON은 수동으로 선별된 것으로, 댄스 동작이나 격투 동작, 걷기 등에서 의미 있는 자세를 골라낸 프레임들이다.

### 3.3 model.goal_inference() 메서드

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb/model.py` (line 152-154)

```python
def goal_inference(self, next_obs):
    z = self.backward_map(next_obs)
    return self.project_z(z)
```

**가장 간단한 형태**이다. 단일 관측(goal pose)을 Backward map에 넣고, `project_z`로 정규화만 하면 끝이다.

`project_z` (line 123-126):
```python
def project_z(self, z):
    if self.cfg.archi.norm_z:
        z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
    return z
```

`norm_z=True`일 때, z 벡터의 L2 노름을 `sqrt(z_dim)`으로 정규화한다. 이는 FB 표현 학습에서 z 공간의 구조를 유지하기 위한 것이다.

### 3.4 z_dict 저장 (goal_reaching.pkl)

```python
# goal_inference.py (line 97-117)
z_dict = {}
with torch.no_grad():
    for goal in pbar:
        env.set_is_evaluating(goal["motion_id"])
        gobs, gobs_dict = get_backward_observation(env, 0,
            use_root_height_obs=use_root_height_obs, velocity_multiplier=0)
        num_frames = next(iter(gobs.values())).shape[0]
        for frame_idx in goal["frames"]:
            if frame_idx >= num_frames:
                continue
            goal_name = f"{goal['motion_name']}_{frame_idx}"
            goal_observation = {k: v[frame_idx][None,...] for k,v in gobs.items()}
            goal_observation = tree_map(
                lambda x: torch.tensor(x, device=model.device, dtype=torch.float32),
                goal_observation)
            z_dict[goal_name] = model.goal_inference(goal_observation).cpu().numpy()
```

핵심 포인트:
- **`velocity_multiplier=0`**: 목표 자세의 속도를 0으로 설정한다. Goal-reaching에서는 "어떤 자세에 도달하라"이므로 속도 정보는 필요없다.
- **`v[frame_idx][None,...]`**: 프레임 차원에서 특정 인덱스를 골라내고 배치 차원을 추가한다.
- `z_dict` key 형식: `"{motion_name}_{frame_idx}"` (예: `"dance1_subject3_505"`)
- 저장 경로: `{model_folder}/goal_inference/goal_reaching.pkl`

### 3.5 Goal-Reaching 비디오 렌더링

```python
# goal_inference.py (line 134-149)
goal_idx = -1
goal_names = list(z_dict.keys())

while counter < episode_len:
    if counter % 100 == 0:
        goal_idx = (goal_idx + 1) % len(goal_names)
        z = z_dict[goal_names[goal_idx]].copy()
        z = torch.tensor(z, device=model.device, dtype=torch.float32)

    action = model.act(observation, z.repeat(num_envs, 1), mean=True)
    observation, reward, terminated, truncated, info = wrapped_env.step(action, to_numpy=False)
    if save_mp4:
        frames.append(rgb_renderer.render(wrapped_env._env, 0)[0])
    counter += 1
```

**100스텝마다 다음 목표 자세로 전환**한다. 이를 통해 하나의 비디오에서 여러 goal-reaching 동작을 관찰할 수 있다. Tracking과 달리 **하나의 고정된 z**로 100스텝 동안 행동한다 (시간에 따라 z가 변하지 않는다).

### 3.6 핵심 코드 분석

**Tracking과의 핵심 차이**:

| 구분 | Tracking | Goal |
|---|---|---|
| 입력 | 전체 모션 시퀀스 (T프레임) | 단일 프레임 (1프레임) |
| z 갯수 | 프레임별 z 벡터 T개 | 하나의 z 벡터 |
| 롤아웃 시 z | `z[i % len(z)]` (시간에 따라 변화) | 고정된 z 하나로 계속 사용 |
| 속도 정보 | 원본 속도 유지 | `velocity_multiplier=0` |
| 의미 | "이 동작을 따라해라" | "이 자세에 도달해라" |


## 4. reward_inference.py - Reward-Based Task Inference

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/reward_inference.py`

### 4.1 전체 구조와 실행 흐름

Reward inference는 **사전 정의된 보상 함수**를 사용하여, 학습 중 수집된 리플레이 버퍼의 상태들에 보상을 부여하고, 이 보상으로 가중 평균된 `z`를 계산한다. 이는 "이 보상을 최대화하는 행동 임베딩 z를 찾아라"에 해당한다.

전체 흐름:

```
1. 모델 로드 + config 로드
2. ONNX 내보내기
3. 환경 생성
4. 리플레이 버퍼 로드 (학습 시 저장된 transition 데이터)
5. RewardWrapperHV 생성 (모델 + 버퍼 + MuJoCo 모델)
6. 각 태스크에 대해:
   a. reward_eval_agent.reward_inference(task=task_name)
   b. -> 보상 함수 생성 -> 버퍼 샘플링 -> relabel -> z 계산
   c. z_dict에 저장
7. z_dict를 reward_locomotion.pkl로 저장
8. (선택) 각 태스크의 z로 롤아웃하여 비디오 렌더링
```

### 4.2 RewardWrapperHV 클래스

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/bench/reward_eval_hv.py` (line 64-128)

```python
@dataclasses.dataclass(kw_only=True)
class RewardWrapperHV(BaseHumEnvBenchWrapper):
    inference_dataset: Any                  # 리플레이 버퍼
    num_samples_per_inference: int          # 보상 평가에 사용할 샘플 수
    inference_function: str                 # "reward_wr_inference"
    max_workers: int                        # 병렬 보상 계산 워커 수
    process_executor: bool = False          # ProcessPoolExecutor 사용 여부
    env_model: str = "...scene_29dof_freebase_noadditional_actuators.xml"
```

`reward_inference` 메서드의 핵심 로직:

```python
def reward_inference(self, task: str) -> torch.Tensor:
    # 1. MuJoCo 모델 로드 (최초 1회)
    if isinstance(self.env_model, str):
        self.env_model = mujoco.MjModel.from_xml_path(self.env_model)

    # 2. 리플레이 버퍼에서 샘플링
    if self.num_samples_per_inference >= self.inference_dataset.size():
        data = self.inference_dataset.get_full_buffer()
    else:
        data = self.inference_dataset.sample(self.num_samples_per_inference)

    # 3. qpos, qvel, action 추출
    qpos = get_next("qpos", data)
    qvel = get_next("qvel", data)
    action = data["action"]

    # 4. 보상 함수로 각 transition에 보상 부여 (relabeling)
    rewards = relabel(
        self.env_model,
        qpos, qvel, action,
        make_from_name(task),   # 태스크 이름 -> 보상 함수 객체
        max_workers=self.max_workers,
        process_executor=self.process_executor,
    )

    # 5. 보상과 observation으로 z 계산
    td = {
        "reward": torch.tensor(rewards, dtype=torch.float32, device=self.device),
    }
    if "B" in data:
        td["B_vect"] = data["B"]         # 사전 계산된 B 벡터가 있으면 사용
    else:
        td["next_obs"] = get_next("observation", data)  # 없으면 observation 사용

    inference_fn = getattr(self.model, self.inference_function, None)
    ctxs = inference_fn(**td).reshape(1, -1)
    return ctxs
```

**Relabeling 과정**이 핵심이다. 학습 중 수집된 상태(qpos, qvel, action)에 대해 새로운 보상 함수를 적용하여 보상을 재계산한다. 이것이 FB 표현 학습의 강점으로, 학습 시 사용하지 않았던 보상 함수에 대해서도 사후적으로(zero-shot) z를 계산할 수 있다.

### 4.3 사전 정의된 보상 태스크 목록

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/rewards.py`

태스크 이름은 정규식 패턴으로 파싱되며, `make_from_name(task)` 함수가 이름을 보상 클래스 인스턴스로 변환한다.

#### 보상 함수 클래스 요약

| 클래스 | 이름 패턴 | 설명 |
|---|---|---|
| `LocomotionReward` | `move-ego-{angle}-{speed}` | 자기중심 좌표계에서 특정 방향/속도로 이동 |
| `LocomotionReward` (low) | `move-ego-low{height}-{angle}-{speed}` | 낮은 자세로 이동 |
| `RotationReward` | `rotate-{axis}-{ang_vel}-{height}` | 특정 축 중심 회전 |
| `ArmsReward` | `raisearms-{left}-{right}` | 팔 높이 제어 (l/m/h) |
| `MoveArmsReward` | `move-arms-{angle}-{speed}-{left}-{right}` | 이동 + 팔 높이 동시 제어 |
| `SpinArmsReward` | `spin-arms-{ang_vel}-{left}-{right}` | 회전 + 팔 높이 동시 제어 |
| `SitOnGroundReward` | `sitonground`, `crouch-{height}` | 앉기/쭈그리기 |
| `JumpReward` | `jump-{height}` | 점프 |

#### 각 태스크 설명

**이동(Locomotion) 관련**:
- `move-ego-0-0`: 제자리 서기 (speed=0, 수평 이동 없이 직립 유지)
- `move-ego-low0.5-0-0`: 낮은 자세(높이 0.5m)로 서기
- `move-ego-0-0.7`: 전방 0.7m/s로 걷기
- `move-ego-90-0.3`: 옆(90도)으로 0.3m/s로 이동
- `move-ego-180-0.3`: 후방으로 0.3m/s로 이동
- `move-ego--90-0.3`: 반대쪽(270도)으로 0.3m/s로 이동

**회전(Rotation)**:
- `rotate-z-5-0.5`: z축 기준 시계방향 5rad/s 회전 (골반 높이 0.5m 유지)
- `rotate-z--5-0.5`: z축 기준 반시계방향 5rad/s 회전

**팔 올리기(Arms)**:
- `raisearms-l-l`: 양쪽 팔 낮은 위치 (l = low: 0.6~0.8m)
- `raisearms-l-m`: 왼팔 낮게, 오른팔 중간 (m = medium: 1.0m+)
- `raisearms-m-l`: 왼팔 중간, 오른팔 낮게
- `raisearms-m-m`: 양쪽 팔 중간 높이

**이동 + 팔(Move + Arms)**:
- `move-arms-0-0.7-m-m`: 전방 0.7m/s + 양팔 중간
- `move-arms-90-0.7-l-m`: 옆 0.7m/s + 왼팔 낮게/오른팔 중간
- (총 16개 조합)

**회전 + 팔(Spin + Arms)**:
- `spin-arms-5-l-l`: 시계방향 회전 + 양팔 낮게
- `spin-arms--5-m-l`: 반시계방향 + 왼팔 중간/오른팔 낮게

**앉기(Sit)**:
- `crouch-0`: 골반 높이 0m로 쭈그리기 (무릎이 바닥에 안 닿도록)
- `crouch-0.25`: 골반 높이 0.25m로 쭈그리기
- `sitonground`: 바닥에 앉기 (무릎 바닥 접촉 허용)

#### 보상 함수 구조 예시 (LocomotionReward)

```python
# rewards.py (line 249-383)
class LocomotionReward(RewardFunction):
    move_speed: float = 5
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True

    def compute(self, model, data) -> float:
        # 1. 골반 높이 보상 (서 있는지)
        root_height = get_xpos(model, data, "pelvis")[-1]
        standing = rewards.tolerance(root_height,
            bounds=(self.stand_height, float("inf")), ...)

        # 2. 직립 자세 보상 (torso가 위를 향하는지)
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(...)
        stand_reward = standing * cost_orientation

        # 3. 이동 속도 보상
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_speed <= 0.01:
            # 서기: 수평 이동과 회전 모두 억제
            return stand_reward * dont_move * dont_rotate
        else:
            # 걷기: 목표 속도와 방향에 대한 보상
            move = rewards.tolerance(com_velocity, bounds=(...))
            angle_reward = (dot(target_direction, actual_direction) + 1) / 2
            return stand_reward * move * angle_reward
```

각 보상 함수는 여러 조건의 **곱(product)**으로 구성되어, 모든 조건이 동시에 만족되어야 높은 보상을 받는다.

### 4.4 z 계산 방식 (보상 최적화)

모델의 `reward_wr_inference` 메서드가 호출된다.

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/fb/model.py` (line 134-150)

```python
def reward_inference(self, next_obs, reward, weight=None):
    batch_size = tree_get_batch_size(next_obs)
    num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
    z = 0
    wr = reward if weight is None else reward * weight
    for i in range(num_batches):
        start_idx = i * self.cfg.inference_batch_size
        end_idx = (i + 1) * self.cfg.inference_batch_size
        next_obs_slice = tree_map(lambda x: x[start_idx:end_idx].to(self.device), next_obs)
        B = self.backward_map(next_obs_slice)
        z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
    return self.project_z(z)

def reward_wr_inference(self, next_obs, reward):
    return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))
```

수학적으로 이를 풀어쓰면:

```
z = project_z( sum_i  w_i * r_i * B(s_i) )

여기서:
  w_i = softmax(10 * r_i)   (높은 보상에 가중치 집중)
  r_i = reward_function(qpos_i, qvel_i, action_i)
  B(s_i) = backward_map(observation_i)
```

- `reward * weight` = `r * softmax(10r)`: 보상이 높은 상태에 지수적으로 더 많은 가중치를 부여한다.
- `torch.matmul(wr.T, B)`: 가중 보상 벡터와 B 행렬의 행렬곱으로 z를 한 번에 계산한다.
- 이 방식은 논문의 **"successor measure" 기반 보상 추론**에 해당한다.

`inference_batch_size=500_000`으로 대량의 샘플을 배치 처리하며, 총 `num_samples=150_000`개의 샘플에 대해 계산한다.

### 4.5 reward_locomotion.pkl 저장

```python
# reward_inference.py (line 169-182)
z_dict = {}
for r in range(n_inferences):          # 여러 번 반복 추론 가능
    for task in tasks:
        z = reward_eval_agent.reward_inference(task=task)
        z_dict[task] = z_dict.get(task, []) + [z.cpu()]
        # 매 태스크 후 즉시 저장 (중간 결과 보존)
        with open(os.path.join(path, "reward_locomotion.pkl"), "wb") as f:
            joblib.dump(z_dict, f)
```

- `n_inferences > 1`이면 같은 태스크에 대해 여러 z를 생성한다 (샘플링의 확률적 특성).
- `z_dict[task]`는 **리스트**로, 여러 번의 추론 결과를 저장한다.
- 저장 경로: `{model_folder}/reward_inference/reward_locomotion.pkl`

### 4.6 핵심 코드 분석

**리플레이 버퍼 로딩**:

```python
# reward_inference.py (line 146-158)
buffer_path = model_folder / "checkpoint/buffers/train_reduced"
if buffer_path.is_dir():
    dataset = DictBuffer.load(buffer_path, device="cpu")    # 축소 버퍼
else:
    buffer_path = model_folder / "checkpoint/buffers/train"
    dataset = TrajectoryDictBufferMultiDim.load(buffer_path, device="cpu")  # 원본 버퍼
```

학습 중 저장된 transition 버퍼를 로드한다. `train_reduced`가 있으면 우선 사용 (메모리 절약용).

**RewardWrapperHV 생성**:

```python
# reward_inference.py (line 159-168)
reward_eval_agent = RewardWrapperHV(
    model=model,
    inference_dataset=dataset,
    num_samples_per_inference=num_samples,     # 150,000
    inference_function="reward_wr_inference",  # 가중 보상 추론 사용
    max_workers=24,
    process_executor=True,                     # 멀티프로세스로 보상 계산
    env_model=str(HUMANOIDVERSE_DIR / "data" / "robots" / "g1" /
              "scene_29dof_freebase_noadditional_actuators.xml"),
)
```

**Relabel 함수 (병렬 보상 계산)**:

```python
# reward_eval_hv.py (line 186-217)
def relabel(model, qpos, qvel, action, reward_fn, max_workers=5, ...):
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i:i+chunk_size], qvel[i:i+chunk_size], action[i:i+chunk_size])
            for i in range(0, qpos.shape[0], chunk_size)]

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
        result = exe.map(f, args)
    return np.concatenate([r for r in result])
```

150,000개의 샘플을 24개의 프로세스로 나누어 병렬 보상 계산을 수행한다. 각 워커는:
```python
def _relabel_worker(x, model, reward_fn):
    qpos, qvel, action = x
    rewards = np.zeros((qpos.shape[0], 1))
    for i in range(qpos.shape[0]):
        rewards[i] = reward_fn(model, qpos[i], qvel[i], action[i])
    return rewards
```

각 (qpos, qvel, action) 튜플에 대해 `mujoco.mj_forward`를 호출하고 보상을 계산한다. MuJoCo의 forward kinematics를 사용하므로 시뮬레이션을 실행하지 않고도 주어진 상태에서의 보상을 계산할 수 있다.


## 5. 세 가지 Inference 비교

### 5.1 z 획득 방법 비교 표

| 항목 | Tracking | Goal | Reward |
|---|---|---|---|
| **z 계산 공식** | `B(obs_sequence)` + 시간 윈도우 평균 | `B(goal_frame)` | `sum(w*r*B(s))` |
| **입력 데이터** | 전문가 모션 시퀀스 | 단일 목표 프레임 | 리플레이 버퍼 + 보상 함수 |
| **속도 정보** | 원본 유지 | `velocity_multiplier=0` | 버퍼 내 원본 상태 |
| **z 갯수** | 프레임 수(T) 만큼 | 1개 | 태스크당 1개 |
| **롤아웃 시 z 변화** | 시간에 따라 변경 | 고정 | 고정 |
| **외부 데이터 필요** | 모션 라이브러리 | 모션 라이브러리 + JSON | 리플레이 버퍼 |
| **출력 파일** | `zs_{motion_id}.pkl` | `goal_reaching.pkl` | `reward_locomotion.pkl` |

### 5.2 사용 시나리오

- **Tracking**: "이 댄스 동작을 그대로 따라해라" -- 전문가 시연이 있을 때
- **Goal**: "이 자세(포즈)에 도달해라" -- 특정 목표 자세가 있을 때
- **Reward**: "빠르게 전진하면서 팔을 올려라" -- 자연어로 설명 가능한 태스크를 보상 함수로 정의할 때

### 5.3 입출력 비교

```
Tracking:
  입력: model_folder, motion_list=[25]
  출력: model_folder/tracking_inference/zs_25.pkl
        model_folder/exported/FBModel.onnx

Goal:
  입력: model_folder, goal_frames_lafan29dof.json
  출력: model_folder/goal_inference/goal_reaching.pkl
        model_folder/exported/FBModel.onnx

Reward:
  입력: model_folder, tasks 리스트, 리플레이 버퍼
  출력: model_folder/reward_inference/reward_locomotion.pkl
        model_folder/exported/FBModel.onnx
```


## 6. IsaacRendererWithMuJoco

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/envs/humanoidverse_isaac.py` (line 202-246)

### 6.1 렌더링 파이프라인

Isaac Sim이나 IsaacGym에서 시뮬레이션이 실행되더라도, 비디오 렌더링은 MuJoCo를 통해 수행된다. 이유는 MuJoCo의 렌더러가 더 가볍고 headless 환경에서도 쉽게 사용할 수 있기 때문이다.

```python
class IsaacRendererWithMuJoco:
    """Renders Isaac state via MuJoCo. Only 29 DOF (36-D qpos: 7 free + 29 joints) is supported."""

    def __init__(self, render_size: int = 512):
        from humanoidverse.utils.g1_env_config import G1EnvConfig
        self.mujoco_env, _ = G1EnvConfig(render_height=render_size, render_width=render_size).build(num_envs=1)
```

**변환 과정** (`render` 메서드):

```python
def render(self, hv_env, env_idxs=None):
    # 1. Isaac 시뮬레이터에서 로봇 상태 추출
    base_pos = hv_env.simulator.robot_root_states[:, [0, 1, 2, 6, 3, 4, 5]]  # pos + quat(wxyz->xyzw)
    joint_pos = hv_env.simulator.dof_pos   # 29 DOF

    # 2. MuJoCo qpos 형식으로 변환 (36-D: 7 free + 29 joints)
    mujoco_qpos = np.concatenate([base_pos, joint_pos], axis=1)

    # 3. MuJoCo 환경에 상태 설정 후 렌더링
    for env_idx in env_idxs:
        self.mujoco_env.reset(options={"qpos": mujoco_qpos[env_idx], "qvel": qvel})
        all_images.append(self.mujoco_env.render())
    return all_images
```

**쿼터니언 순서 변환** 주의: Isaac은 `(x,y,z,w)` 또는 `(w,x,y,z)` 순서를 상황에 따라 사용하고, MuJoCo는 `(w,x,y,z)` 순서를 사용한다. `robot_root_states`에서 `[:, [0,1,2,6,3,4,5]]` 인덱싱은 `[px, py, pz, qw, qx, qy, qz]` 순서로 재배치하는 것이다.

### 6.2 MuJoCo 기반 시각화

`from_qpos` 메서드는 qpos 배열에서 직접 프레임을 렌더링한다:

```python
def from_qpos(self, qpos):
    """36-D qpos (7 free + 29 joints) only."""
    frames = []
    for q in qpos:
        q = np.asarray(q).ravel()
        if q.size != 36:
            raise ValueError(...)
        self.mujoco_env.reset(options={"qpos": q})
        frames.append(self.mujoco_env.render())
    return frames
```

Tracking inference에서 전문가 모션의 비디오를 렌더링할 때 사용된다 (`expert_video = rgb_renderer.from_qpos(expert_qpos)`).


## 7. Wrappers

### 7.1 HumanoidEnvBench 래퍼

파일: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/wrappers/humenvbench.py`

세 가지 래퍼 클래스가 있으며, 모두 `BaseHumEnvBenchWrapper`를 상속한다:

**BaseHumEnvBenchWrapper** (line 30-66):
- `model`을 감싸서 입출력 변환 처리
- `act()`: obs와 z를 텐서로 변환 후 모델의 `act` 호출, numpy 출력 지원
- `unwrapped_model`: 래퍼 체인의 최하단 모델 접근

**RewardWrapper** (line 84-146):
- `reward_inference(task)`: 원래의 humenv 기반 보상 추론 (G1Env 환경 사용)
- `make_humenv`로 환경을 생성하여 보상 함수 획득
- humenv의 qpos/qvel 형식 사용

**GoalWrapper** (line 176-180):
```python
class GoalWrapper(BaseHumEnvBenchWrapper):
    def goal_inference(self, goal_pose):
        next_obs = tree_map(lambda x: to_torch(x, device=self.device, dtype=self._dtype), goal_pose)
        ctx = self.unwrapped_model.goal_inference(next_obs=next_obs).reshape(1, -1)
        return ctx
```

**TrackingWrapper** (line 183-188):
```python
class TrackingWrapper(BaseHumEnvBenchWrapper):
    def tracking_inference(self, next_obs):
        next_obs = tree_map(lambda x: to_torch(x, device=self.device, dtype=self._dtype), next_obs)
        ctx = self.unwrapped_model.tracking_inference(next_obs=next_obs)
        return ctx
```

### 7.2 RewardWrapperHV vs RewardWrapper 구조 차이

| 항목 | RewardWrapper (humenvbench.py) | RewardWrapperHV (reward_eval_hv.py) |
|---|---|---|
| **환경** | humenv (MuJoCo 기반) | HumanoidVerse (Isaac/MuJoCo) |
| **보상 함수** | humenv의 RewardFunction | g1_env_helper의 RewardFunction |
| **보상 생성** | `env.unwrapped.task` | `make_from_name(task)` |
| **MuJoCo 모델** | `env.unwrapped.model` | 직접 XML에서 로드 |
| **용도** | 벤치마크 평가용 | HumanoidVerse 통합용 |

`RewardWrapperHV`가 이 프로젝트에서 실제로 사용되는 클래스이며, humenv 환경 없이도 MuJoCo 모델만으로 보상을 계산할 수 있도록 개조되었다.


## 8. 실제 사용 예시

### 8.1 macOS에서 goal_inference 실행하기

Goal inference는 환경 생성이 필요하지만 GPU가 필수는 아니다. macOS에서 실행하려면 기존 tracking_inference_macos.py와 유사한 패턴으로 수정이 필요하다:

```bash
# Linux/CUDA 환경 (기본)
uv run python -m humanoidverse.goal_inference \
    --model_folder model/ \
    --simulator mujoco \
    --headless \
    --save_mp4 \
    --device cuda

# macOS에서는 device=cpu, simulator=mujoco 필수
# 그리고 MUJOCO_GL=glfw 또는 osmesa로 변경 필요
```

주요 고려사항:
- `device="cpu"` 설정 필수 (macOS에는 CUDA 없음)
- `simulator="mujoco"` 설정 필수 (Isaac Sim은 Linux 전용)
- `MUJOCO_GL` 환경변수를 `glfw`(GUI) 또는 `osmesa`(headless)로 설정
- MuJoCo의 `quat_rotate` w_last 패치가 필요할 수 있음

### 8.2 커스텀 보상 태스크 정의하기

새로운 보상 태스크를 추가하려면 `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/rewards.py`에 새 클래스를 추가한다:

1. `RewardFunction` 추상 클래스를 상속
2. `compute(model, data)` 메서드 구현 -- MuJoCo 상태에서 보상 계산
3. `reward_from_name(name)` 정적 메서드 구현 -- 이름 패턴 매칭

```python
@dataclasses.dataclass
class MyCustomReward(RewardFunction):
    target_height: float = 1.0

    def compute(self, model, data) -> float:
        pelvis_height = get_xpos(model, data, "pelvis")[-1]
        return rewards.tolerance(pelvis_height, bounds=(self.target_height, float("inf")), ...)

    @staticmethod
    def reward_from_name(name):
        pattern = r"^mycustom-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            return MyCustomReward(target_height=float(match.group(1)))
        return None
```

그 후 `reward_inference.py`의 `tasks` 리스트에 `"mycustom-1.0"` 같은 이름을 추가하면 자동으로 인식된다. `make_from_name` 함수가 `rewards.py` 모듈의 모든 `RewardFunction` 서브클래스를 순회하며 `reward_from_name`을 호출하기 때문이다:

```python
# robot.py (line 289-299)
def make_from_name(name):
    module_n = str(__name__).replace("robot", "rewards")
    all_rewards = inspect.getmembers(sys.modules[module_n], inspect.isclass)
    for reward_class_name, reward_cls in all_rewards:
        if not inspect.isabstract(reward_cls):
            reward_obj = reward_cls.reward_from_name(name)
            if reward_obj is not None:
                return reward_obj
    raise ValueError(f"Unknown reward name: {name}")
```

---

## 정리: FB 표현 학습의 3가지 promptable 인터페이스

BFM-Zero의 핵심 아이디어는 **하나의 학습된 모델로 세 가지 서로 다른 방식의 "프롬프트"를 받을 수 있다**는 것이다:

1. **Tracking (시연 프롬프트)**: "이렇게 움직여" -> 전문가 시퀀스에서 z 추출
2. **Goal (자세 프롬프트)**: "이 자세로 가" -> 목표 프레임에서 z 추출
3. **Reward (보상 프롬프트)**: "빠르게 걸으면서 팔을 올려" -> 보상 함수에서 z 추출

모든 경로는 궁극적으로 **Backward Map B**를 통해 잠재 벡터 z로 수렴하며, 정책 `pi(a|s,z) = actor(obs, z)`가 z에 따라 행동을 결정한다. 이 구조가 FB 표현 학습의 "zero-shot" 일반화 능력의 핵심이다.
