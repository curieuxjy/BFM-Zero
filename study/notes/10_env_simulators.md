# Phase 6: 환경 & 시뮬레이터 (Environment & Simulators)

## 1. 개요

BFM-Zero의 환경-시뮬레이터 아키텍처는 **4개의 레이어**로 분리되어 있다.

```
+----------------------------------------------------------+
|  Gymnasium VectorEnv Wrapper (gymnasium_wrapper.py)       |  <-- 에이전트(FBcpr)가 보는 인터페이스
|  - reset() / step() 표준 인터페이스                       |
|  - observation/action space 정의                          |
+----------------------------------------------------------+
|  모션 트래킹 환경 (legged_robot_motions.py)               |  <-- 모션 데이터 로딩, 참조 트래킹 계산
|  - LeggedRobotMotions -> LeggedRobotBase -> BaseTask     |
|  - 보상 계산, 에피소드 관리, 종료 조건                    |
+----------------------------------------------------------+
|  G1 환경 헬퍼 (g1_env_helper/)                            |  <-- MuJoCo 기반 독립 환경 (추론/평가용)
|  - G1Base -> G1Env / G1Env29dof                           |
|  - 보상 함수, 충돌 감지, 관찰 구성                       |
+----------------------------------------------------------+
|  시뮬레이터 백엔드 (simulator/)                           |  <-- 물리 엔진 추상화
|  - BaseSimulator -> MuJoCo / IsaacSim                     |
|  - setup(), load_assets(), create_envs()                  |
|  - simulate_at_each_physics_step()                        |
+----------------------------------------------------------+
```

핵심 설계 원칙:
- **환경 로직**과 **물리 시뮬레이션**을 완전히 분리하여 동일한 환경 코드가 MuJoCo, Isaac Sim, IsaacGym, Genesis 등 다양한 백엔드에서 동작
- **Gymnasium 호환** 래퍼를 통해 표준 RL 라이브러리와 통합
- **훈련용 환경**(LeggedRobotMotions)과 **추론/평가용 환경**(g1_env_helper)이 별도로 존재

---

## 2. gymnasium_wrapper.py - Gymnasium 벡터환경 래퍼

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/gymnasium_wrapper.py`

### 2.1 HumanoidVerseVectorEnv 클래스

`gymnasium.vector.VectorEnv`를 상속하여 HumanoidVerse 환경을 표준 Gymnasium 인터페이스로 감싼다. FBcpr 에이전트의 `TrajectoryBuffer`가 직접 상호작용하는 대상이다.

```python
class HumanoidVerseVectorEnv(VectorEnv):
    def __init__(self, env: LeggedRobotMotions, add_time_aware_observation: bool = True):
        super().__init__()
        self._env = env
        self.num_envs = self._env.num_envs
        self.add_time_aware_observation = add_time_aware_observation
```

내부적으로 `LeggedRobotMotions` 인스턴스를 받아서 감싸는 구조이다. 초기화 시 `reset()`을 한 번 호출하여 실제 관찰 공간의 형태를 파악한다.

### 2.2 reset() / step() 인터페이스

**reset()**:
```python
def reset(self, *, seed=None, options=None):
    _, info = self.base_env.reset_all()
    observation = self._get_g1env_observation()
    return observation, info
```

**step()**:
```python
def step(self, actions):
    actions = {"actions": actions}
    _, reward, reset, new_info = self.base_env.step(actions)
    reset = reset.bool()
    time_outs = new_info["time_outs"].bool()
    terminated = torch.logical_and(reset, ~time_outs)  # 실제 실패로 인한 종료
    truncated = time_outs  # 시간 제한에 의한 종료
    observation = self._get_g1env_observation()
    return observation, reward, terminated, truncated, new_info
```

`terminated`와 `truncated`를 구분하는 것이 중요하다. `time_outs`는 에피소드 최대 길이 도달을, `terminated`는 넘어짐 등 실제 실패 조건을 의미한다.

### 2.3 observation/action space 정의

관찰 공간은 Dictionary 형태로 구성된다:

```python
def _get_g1env_observation(self):
    raw_obs = self._env.obs_buf_dict_raw["actor_obs"]
    g1env_state = torch.cat([
        raw_obs["dof_pos"],              # 관절 위치 (29D)
        raw_obs["dof_vel"],              # 관절 속도 (29D)
        raw_obs["projected_gravity"],    # 투영 중력 (3D)
        raw_obs["base_ang_vel"],         # 기저 각속도 (3D)
    ], dim=-1)
    last_action = raw_obs["actions"]                    # 이전 행동
    privileged_state = raw_obs["max_local_self"]        # 특권 상태 (357D)
    observation = {
        "state": g1env_state,              # 64D (29+29+3+3)
        "last_action": last_action,         # 29D
        "privileged_state": privileged_state, # 357D
    }
    if self.add_time_aware_observation:
        observation["time"] = self._env.episode_length_buf.unsqueeze(-1)
    return observation
```

- **state**: dof_pos(29) + dof_vel(29) + projected_gravity(3) + base_ang_vel(3) = 64차원
- **last_action**: 이전 스텝의 행동 (29차원)
- **privileged_state**: max_local_self 관찰 (357차원) -- 시뮬레이터에서만 접근 가능한 전체 강체 상태
- **time**: 현재 에피소드 스텝 수 (시간 인식 관찰)

행동 공간은 `single_action_space`를 `num_envs`만큼 타일링:
```python
action_space_shape = (self.num_envs,) + self.single_action_space.shape
```

---

## 3. G1 환경 (g1_env_helper/)

g1_env_helper는 **MuJoCo 기반의 독립 환경**으로, 주로 **추론(inference)** 시에 사용된다. 훈련용 LeggedRobotMotions와는 별도의 경로이다.

### 3.1 base.py - G1Base 클래스

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/base.py`

`gymnasium.Env`를 직접 상속하는 순수 MuJoCo 환경 기반 클래스:

```python
class G1Base(gymnasium.Env):
    def __init__(self, xml_path: str, config: G1EnvConfigsType, seed: int = 0):
        self._ctrl_dt = config.ctrl_dt        # 제어 타임스텝 (0.02s = 50Hz)
        self._sim_dt = config.sim_dt           # 시뮬레이션 타임스텝 (0.005s = 200Hz)
        self._mj_model = get_mujoco_model(xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_data = mujoco.MjData(self._mj_model)
```

핵심 속성:
- `n_substeps`: 제어 1스텝 당 시뮬레이션 서브스텝 수 = `ctrl_dt / sim_dt` = 0.02/0.005 = **4회**
- `action_size`: `self._mj_model.nu` -- MuJoCo 모델의 액추에이터 수
- `render()`: `mujoco.Renderer`를 사용한 렌더링 (400x400 기본)

### 3.2 rewards.py - 보상 함수 상세

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/rewards.py`

이 파일은 **추론 시 태스크 보상**을 정의한다. `RewardFunction` 추상 클래스를 기반으로 다양한 태스크를 구현한다.

#### 보상 클래스 계층구조

```
RewardFunction (ABC)
  +-- ZeroReward            # 보상 없음 (reward-free 모드)
  +-- LocomotionReward      # 이동 보상 (방향+속도)
  +-- JumpReward            # 점프 보상 (높이)
  +-- RotationReward        # 회전 보상 (축별 각속도)
  +-- ArmsReward            # 팔 자세 보상
  +-- SitOnGroundReward     # 앉기/웅크리기 보상
  +-- ToTheKnee             # 무릎 꿇기 보상
  +-- MoveArmsReward        # 이동+팔 자세 복합 보상
  +-- SpinArmsReward        # 회전+팔 자세 복합 보상
```

#### REWARD_LIMITS 구조

팔 높이를 범주화하여 보상을 계산하는 데 사용:
```python
REWARD_LIMITS = {
    "l": [0.6, 0.8, 0.2],      # 낮음: 높이 0.6~0.8m, margin 0.2
    "m": [1.0, float("inf"), 0.1],  # 중간: 높이 1.0m 이상, margin 0.1
}
```
각 항목은 `[하한, 상한, margin]`으로, `dm_control.utils.rewards.tolerance()` 함수에 전달된다.

#### LocomotionReward 상세

가장 핵심적인 보상 함수 -- 이동 태스크의 보상을 계산:

```python
@dataclasses.dataclass
class LocomotionReward(RewardFunction):
    move_speed: float = 5        # 목표 이동 속도
    stand_height: float = 0.5    # 최소 서기 높이
    move_angle: float = 0        # 목표 이동 방향 (도)
    egocentric_target: bool = True  # 자기중심 좌표계 사용
    stay_low: bool = False       # 낮은 자세 유지
```

보상 = `small_control * stand_reward * move * angle_reward`

각 요소:
1. **standing**: pelvis 높이가 `stand_height` 이상인지 (tolerance 함수)
2. **cost_orientation**: 몸통 upvector가 `[0.073, 0, 1.0]`에 가까운지
3. **move**: 질량중심 속도가 목표 속도 범위 내인지
4. **angle_reward**: 이동 방향이 목표 방향과 일치하는지 (내적)

속도 0일 때는 "움직이지 않기" + "회전하지 않기" 보상으로 전환.

#### 이름 기반 보상 생성 패턴

모든 보상 클래스에 `reward_from_name()` 정적 메서드가 있어 문자열에서 보상 인스턴스를 생성:
```python
# 예시: "move-ego-45-3.0" -> 45도 방향, 3.0 m/s 이동
# 예시: "jump-1.4" -> 1.4m 높이 점프
# 예시: "rotate-z-5.0-0.8" -> z축 5.0 rad/s 회전, 골반 높이 0.8m
# 예시: "raisearms-m-l" -> 왼팔 중간, 오른팔 낮음
```

#### 커리큘럼 학습 보상

reward_bfm_zero.yaml에서 커리큘럼 관련 설정:
```yaml
reward_penalty_curriculum: False
reward_initial_penalty_scale: 0.10
reward_penalty_degree: 0.000003
reward_penalty_level_up_threshold: 42
```

평균 에피소드 길이를 기준으로 페널티 스케일을 조정하는 커리큘럼 메커니즘이 있다 (기본적으로 비활성화).

### 3.3 robot_29dof.py - 29-DOF 로봇 정의

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/robot_29dof.py`

Unitree G1 로봇의 29 자유도 MuJoCo 환경을 구현한다.

#### 기본 설정 (default_config_29dof)

```python
def default_config_29dof() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,        # 제어 주기: 50Hz
        sim_dt=0.005,        # 시뮬레이션 주기: 200Hz
        obs_model="state",   # 관찰 모드
        soft_joint_pos_limit_factor=0.95,  # 소프트 관절 한계
        soft_torque_limit_factor=0.95,
    )
```

#### 관절 구성 (29-DOF)

| 부위 | 관절 | DOF | 토크 한계(N*m) |
|------|------|-----|----------------|
| 왼쪽 다리 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | 6 | 88, 88, 88, 139, 50, 50 |
| 오른쪽 다리 | (동일) | 6 | 88, 88, 88, 139, 50, 50 |
| 허리 | waist_yaw, waist_roll, waist_pitch | 3 | 88, 50, 50 |
| 왼팔 | shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw | 7 | 25, 25, 25, 25, 25, 5, 5 |
| 오른팔 | (동일) | 7 | 25, 25, 25, 25, 25, 5, 5 |
| **합계** | | **29** | |

#### PD 제어 파라미터

```python
stiffness = {
    hip_yaw: 100, hip_roll: 100, hip_pitch: 100,
    knee: 200,
    ankle_pitch: 20, ankle_roll: 20,
    waist_yaw/roll/pitch: 400,
    shoulder_pitch: 90, shoulder_roll: 60, shoulder_yaw: 20,
    elbow: 60, wrist: 4,
}
```

#### 노이즈 설정

```python
noise_config = {
    level: 0.0,  # 0.0=비활성, 1.0=활성
    scales: {
        joint_pos: 0.03,    # 관절 위치 노이즈
        joint_vel: 1.5,     # 관절 속도 노이즈
        gravity: 0.05,      # 중력 벡터 노이즈
        linvel: 0.1,        # 선속도 노이즈
        gyro: 0.2,          # 자이로 노이즈
        torque: 0.1,        # 토크 노이즈
    },
    torque_range: [0.5, 1.5],  # 토크 노이즈 범위
}
```

#### 토크 계산 (PD 제어)

```python
def _compute_torques(self, actions):
    actions_scaled = actions * self._config.ctrl_config.action_scale  # 0.25
    # F(t) = K_p * (target - q(t)) - K_d * dq(t)
    torques = (
        self._kp_scale * self.p_gains * (actions_scaled + self._default_pose_29 - self._mj_data.qpos[7:])
        - self._kd_scale * self.d_gains * self._mj_data.qvel[6:]
    )
```

핵심: 행동은 **기본 자세로부터의 오프셋**으로 해석된다. `target_angle = action_scale * action + default_pose`.

#### step 흐름

```python
def step(self, action):
    action = action * self._config.ctrl_config.action_clip_value  # 5.0
    for _ in range(self.n_substeps):   # 4회 (ctrl_dt/sim_dt)
        motor_targets = self._compute_torques(extendend_action)
        step(self._mj_model, self._mj_data, motor_targets, 1)
```

#### 23-DOF 모드

29-DOF에서 손목 3관절을 제거하여 23-DOF 모드로 전환 가능:
```python
QPOS_IDX_23_IN_29 = [0,1,...,25, 29,30,31,32]  # 손목 관절 인덱스 제외
ACT_IDX_23_IN_29 = [0,1,...,18, 22,23,24,25]
```

### 3.4 robot.py - G1Env 로봇 기반

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/robot.py`

`G1Env`는 `G1Base`를 상속하며, 태스크 보상 기반 추론에 사용되는 키네마틱 환경이다.

주요 특징:
- **키네마틱 모드**: 물리 시뮬레이션 없이 `mj_forward()`만 호출하여 상태 전파
- **태스크 설정**: `set_task(task)` 메서드로 문자열 또는 RewardFunction 객체를 지정
- **보조 보상(aux_rewards)**: `_sim2real_costs()`에서 키네마틱 환경이므로 모두 0 반환

```python
def step(self, action):
    action_scale = 0.25
    self._mj_data.qpos[7:] = self._default_pose + action_scale * np.clip(action, -2.0, 2.0)
    self._mj_data.qvel[6:] = 0.0
    mujoco.mj_forward(self._mj_model, self._mj_data)
    reward = self.task.compute(self._mj_model, self._mj_data)
```

관찰 유형 3가지:
```python
class ObsType(Enum):
    proprioceptive = 0         # 고유감각만
    pixel = 1                  # 이미지만
    proprioceptive_pixel = 2   # 둘 다
```

### 3.5 collision.py - 충돌 감지

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/g1_env_helper/collision.py`

MuJoCo의 접촉 데이터를 분석하는 유틸리티:

```python
def get_collision_info(contact, geom1, geom2):
    """두 geom 간 충돌의 거리와 법선 벡터를 반환"""
    mask = (np.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (np.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = np.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, :3]
    return dist, normal

def geoms_colliding(data, geom1, geom2):
    """두 geom이 충돌 중인지 반환"""
    return get_collision_info(data.contact, geom1, geom2)[0] < 0
```

`dist < 0`이면 두 geom이 겹쳐 있는 것(침투)이므로 충돌로 판정.

---

## 4. legged_robot_motions.py - 모션 트래킹 환경

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/envs/legged_robot_motions/legged_robot_motions.py`

### 4.1 LeggedRobotMotions 클래스

**훈련 시 사용되는 핵심 환경**. 상속 계층:

```
LeggedRobotMotions -> LeggedRobotBase -> BaseTask
```

초기화 시 모션 라이브러리, 확장 바디, 트래킹 설정을 순차적으로 초기화한다.

### 4.2 모션 데이터와의 연동

```python
def _init_motion_lib(self):
    self._motion_lib = MotionLibRobot(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
    self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
```

모션 라이브러리(`MotionLibRobot`)는 `data/lafan_29dof.pkl` (평가용) 또는 `data/lafan_29dof_10s-clipped.pkl` (훈련용)에서 모션 클립을 로드한다.

모션 리샘플링:
```python
def _resample_motion_time_and_ids(self, env_ids):
    self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
    self.motion_len[env_ids] = self._motion_lib.get_motion_length(self.motion_ids[env_ids])
    self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])
```

훈련 중 `resample_time_interval` 주기마다 모션을 리샘플링하여 다양한 모션에 노출:
```python
if self.common_step_counter % self.resample_time_interval == 0:
    self.resample_motion()
```

### 4.3 관찰 공간 구성

`_pre_compute_observations_callback()`에서 매 스텝 계산되는 관찰 요소들:

#### 참조 모션과의 차이 계산

```python
# 위치 차이
self.dif_global_body_pos = self.ref_body_pos_extend - self._rigid_body_pos_extend
# 회전 차이
self.dif_global_body_rot = quat_mul(self.ref_body_rot_extend, quat_conjugate(self._rigid_body_rot_extend))
# 속도 차이
self.dif_global_body_vel = self.ref_body_vel_extend - self._rigid_body_vel_extend
# 각속도 차이
self.dif_global_body_ang_vel = self.ref_body_ang_vel_extend - self._rigid_body_ang_vel_extend
# 관절 위치/속도 차이
self.dif_joint_angles = ref_joint_pos - self.simulator.dof_pos
self.dif_joint_velocities = ref_joint_vel - self.simulator.dof_vel
```

#### 확장 바디 (Extend Bodies)

G1 로봇에 물리적으로 존재하지 않는 가상 바디를 추가. 대표적으로 **head_link**:
```yaml
extend_config:
  - joint_name: "head_link"
    parent_name: "torso_link"
    pos: [0.0, 0.0, 0.35]
    rot: [1.0, 0.0, 0.0, 0.0]
```

이 가상 바디의 위치는 부모 바디(torso_link)의 위치+회전에서 계산된다:
```python
extend_curr_pos = my_quat_rotate(self.extend_body_rot_in_parent_xyzw, rotated_pos_in_parent)
    + self.simulator._rigid_body_pos[:, self.extend_body_parent_ids]
```

#### max_local_self 관찰 (357차원)

`compute_humanoid_observations_max()` (TorchScript 최적화)에서 계산:

```python
obs_dict = {
    'root_height': root_h,                    # 1D
    'local_body_pos': local_body_pos,          # (num_bodies-1)*3 D
    'local_body_rot': local_body_rot_obs,      # num_bodies*6 D (tan-norm 표현)
    'local_body_vel': local_body_vel,          # num_bodies*3 D
    'local_body_ang_vel': local_body_ang_vel,  # num_bodies*3 D
}
```

모든 바디의 위치, 회전, 속도, 각속도를 **로컬 좌표계**(heading 방향 제거)로 변환하여 관찰로 사용한다.

#### VR 3-포인트 트래킹

VR 텔레오퍼레이션을 위한 3점(양손+머리) 추적:
```python
ref_vr_3point_pos = self.ref_body_pos_extend[:, self.motion_tracking_id, :]
vr_2root_pos = ref_vr_3point_pos - self.simulator.robot_root_states[:, 0:3]
self._obs_vr_3point_pos = my_quat_rotate(heading_inv_rot_vr, vr_2root_pos)
```

### 4.4 종료 조건

```python
def _check_termination(self):
    super()._check_termination()
    if self.config.termination.terminate_when_motion_far:
        if self.is_evaluating:
            reset_buf_motion_far = torch.norm(self.dif_global_body_pos, dim=-1).mean(dim=-1) > 0.5
        else:
            reset_buf_motion_far = torch.any(
                torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold,
                dim=-1
            )
```

- 평가 시: 평균 바디 차이가 0.5m 초과 시 종료
- 훈련 시: 임의 바디 차이가 임계값 초과 시 종료 (커리큘럼으로 조정 가능)
- 모션 종료 시 타임아웃 처리

### 4.5 루트 상태 리셋

리셋 시 참조 모션의 상태로 초기화 (Reference State Initialization):
```python
def _reset_root_states(self, env_ids, target_state=None):
    ref_root_pos = motion_res['root_pos'][env_ids]
    ref_root_rot = motion_res['root_rot'][env_ids]
    # 노이즈 추가
    self.target_robot_root_states[env_ids, :3] = ref_root_pos + randn * root_pos_noise
```

**lie_down_init** 옵션: 일정 확률로 로봇을 누운 상태에서 시작시켜 회복 학습을 유도:
```python
if self.config.get("lie_down_init", False):
    mask = (torch.rand(len(env_ids)) < prob)
    ref_root_pos[mask, 2] = 0.5  # z=0.5m
    rot_quat = quat_from_angle_axis(-pi/2, [1,0,0])  # x축 -90도 회전
    ref_root_rot[mask] = quat_mul(rot_quat, ref_root_rot[mask])
```

---

## 5. 시뮬레이터 백엔드

### 5.1 base_simulator.py - 추상 인터페이스

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/simulator/base_simulator/base_simulator.py`

모든 시뮬레이터가 구현해야 하는 추상 인터페이스:

```python
class BaseSimulator:
    def __init__(self, config, device):
        self.config = config
        self.sim_device = device
        # 모든 백엔드가 노출해야 하는 텐서
        self._rigid_body_pos: torch.Tensor    # 강체 위치 [num_envs, num_bodies, 3]
        self._rigid_body_rot: torch.Tensor    # 강체 회전 [num_envs, num_bodies, 4]
        self._rigid_body_vel: torch.Tensor    # 강체 선속도 [num_envs, num_bodies, 3]
        self._rigid_body_ang_vel: torch.Tensor # 강체 각속도 [num_envs, num_bodies, 3]
```

주요 추상 메서드:

| 메서드 | 역할 |
|--------|------|
| `setup()` | 시뮬레이터 초기화, 모델 로딩 |
| `setup_terrain(mesh_type)` | 지형 설정 (plane/heightfield/trimesh) |
| `load_assets(robot_config)` | 로봇 에셋 로딩, DOF/바디 이름 저장 |
| `create_envs(num_envs, ...)` | 병렬 환경 생성 |
| `prepare_sim()` | 시뮬레이션 준비 |
| `refresh_sim_tensors()` | 상태 텐서 갱신 |
| `apply_torques_at_dof(torques)` | 관절 토크 적용 |
| `set_actor_root_state_tensor(env_ids, states)` | 루트 상태 설정 |
| `set_dof_state_tensor(env_ids, states)` | DOF 상태 설정 |
| `simulate_at_each_physics_step()` | 1 물리 스텝 진행 |
| `get_dof_limits_properties()` | DOF 한계값 반환 |

### 5.2 MuJoCo 백엔드

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/simulator/mujoco/mujoco.py`

#### 모델 로딩

```python
class MuJoCo(BaseSimulator):
    def setup(self):
        self.model_path = str(hv_root / "data/robots/g1/scene_29dof_freebase_mujoco.xml")
        self.freebase = True
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.sim_dt = 1 / self.simulator_config.sim.fps  # 1/200 = 0.005s
        self.model.opt.timestep = self.sim_dt
```

- XML (MJCF) 형식의 모델 파일 사용
- freebase 모드: pelvis가 free joint로 6DOF 추가

#### 환경 제한

```python
def create_envs(self, num_envs, env_origins, base_init_state):
    self.num_envs = 1  # MuJoCo는 단일 환경만 지원!
```

MuJoCo는 **GPU 병렬화를 지원하지 않으므로** 항상 num_envs=1로 동작한다. 훈련 시에는 Isaac Sim이 필요하고, MuJoCo는 추론/평가 용도로 사용된다.

#### 상태 텐서 (Properties)

MuJoCo의 numpy 데이터를 PyTorch 텐서로 변환:

```python
@property
def robot_root_states(self):
    base_quat = self.base_quat  # wxyz -> xyzw 변환
    return torch.cat([
        torch.tensor([self.data.qpos[0:3]]),     # 위치
        base_quat,                                 # 회전 (xyzw)
        qvel[:, 0:3],                              # 선속도
        quat_rotate(base_quat, qvel[:, 3:6]),     # 각속도 (글로벌->로컬)
    ], dim=-1)

@property
def base_quat(self):
    # MuJoCo는 wxyz, 내부적으로는 xyzw 사용
    return torch.tensor([self.data.qpos[3:7]])[..., [1, 2, 3, 0]]
```

쿼터니언 순서 변환 주의: MuJoCo는 **wxyz**, HumanoidVerse 내부는 **xyzw** 사용.

#### Domain Randomization

```python
def episodic_domain_randomization(self, env_ids):
    # 기본값 복원 후 랜덤화
    self.model.dof_frictionloss[:] = self.default_dof_frictionloss
    self.model.body_mass[:] = self.default_body_mass
    self.model.geom_friction[:] = self.default_geom_friction

    if self.domain_rand_config.get("randomize_link_mass", False):
        dmass = np.random.uniform(low=range[0], high=range[1], size=(self.model.nbody,))
        self.model.body_mass[:] = self.model.body_mass * dmass

    if self.domain_rand_config.get("randomize_friction", False):
        # 관절 마찰 (dof_frictionloss)과 geom 마찰 모두 랜덤화
        self.model.dof_frictionloss[6:] = defaults * np.random.uniform(...)
        self.model.geom_friction[:] = defaults * np.random.uniform(...)

    if self.domain_rand_config.get("randomize_base_com", False):
        # torso_link 질량에 랜덤 오프셋 추가
        self.model.body_mass[self.torso_id] += np.random.uniform(...)
```

#### 물리 시뮬레이션

```python
def simulate_at_each_physics_step(self):
    mujoco.mj_step(self.model, self.data)
    if self.viewer is not None:
        self.viewer.sync()

def apply_torques_at_dof(self, torques):
    if self.freebase:
        self.data.ctrl[6:] = torques  # 앞 6개는 free joint
    else:
        self.data.ctrl[:] = torques
```

### 5.3 Isaac Sim 백엔드

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/simulator/isaacsim/isaacsim.py`

#### SimulationContext 및 PhysX 설정

```python
class IsaacSim(BaseSimulator):
    def __init__(self, config, device):
        sim_config = SimulationCfg(
            dt=1./self.simulator_config.sim.fps,   # 0.005s
            render_interval=self.simulator_config.sim.render_interval,
            device=self.sim_device,                 # "cuda:0"
            physx=PhysxCfg(
                bounce_threshold_velocity=0.5,
                solver_type=1,                      # TGS solver
                max_position_iteration_count=4,
                max_velocity_iteration_count=0,
            )
        )
        self.sim = SimulationContext(sim_config)
```

#### GPU 가속 병렬 환경

Isaac Sim의 핵심 장점 -- 수천 개 환경을 **GPU에서 병렬 실행**:

```python
def create_envs(self, num_envs, env_origins, base_init_state):
    self.num_envs = num_envs  # 수백~수천 개 환경 병렬 실행!
    return self.scene, self._robot
```

Scene 설정:
```python
scene_config = InteractiveSceneCfg(
    num_envs=self.simulator_config.scene.num_envs,
    env_spacing=self.simulator_config.scene.env_spacing,
    replicate_physics=True  # 물리 복제로 메모리 절약
)
```

#### Articulation 설정

USD 파일을 사용한 로봇 에셋 로딩:
```python
spawn = sim_utils.UsdFileCfg(
    usd_path=asset_abs_path,
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=not bool(self.env_config.robot.asset.self_collisions),
        solver_position_iteration_count=4,
    ),
)
```

#### 액추에이터 설정

```python
actuators["all"] = ImplicitActuatorCfg(
    joint_names_expr=[...],
    effort_limit_sim={joint: limit for ...},    # 토크 한계
    velocity_limit_sim={joint: limit for ...},  # 속도 한계
    stiffness=stiffness_dict,                    # PD 강성 (K_p)
    damping=damping_dict,                        # PD 감쇠 (K_d)
    armature={joint: value for ...},             # 관성 보상
    friction={joint: value for ...},             # 관절 마찰
)
```

#### 지형 설정

Isaac Sim은 다양한 지형 유형을 지원:
```python
if terrain_type == "flat":
    sub_terrains["flat"] = terrain_gen.MeshPlaneTerrainCfg(proportion=proportion)
elif terrain_type == "rough":
    sub_terrains["rough"] = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion, noise_range=(0.02, 0.10), noise_step=0.02
    )
elif terrain_type == "low_obst":
    sub_terrains["low_obst"] = terrain_gen.MeshRandomGridTerrainCfg(
        proportion=proportion, grid_height_range=(0.05, 0.2), platform_width=2.0
    )
```

#### 이벤트 기반 Domain Randomization

Isaac Sim은 `EventManager`를 사용한 구조화된 랜덤화:

```python
# 질량 랜덤화
self.events_cfg.scale_body_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={"mass_distribution_params": (0.95, 1.05), "operation": "scale"},
)

# 마찰 랜덤화
self.events_cfg.random_joint_friction = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={"static_friction_range": (0.5, 1.25), "dynamic_friction_range": (0.5, 1.25)},
)

# 로봇 밀기 (push)
self.events_cfg.push_robots = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="on_push_by_setting_velocity",
    params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.5, 0.5), ...}},
)
```

이벤트 모드:
- `startup`: 시뮬레이션 시작 시 1회 적용
- `on_push_by_setting_velocity`: 주기적 외란 적용

#### 상태 텐서

Isaac Sim은 직접 GPU 텐서를 반환 (CPU 복사 불필요):

```python
@property
def dof_pos(self):
    return self._robot.data.joint_pos[:, self.dof_ids]  # GPU 텐서

@property
def _rigid_body_pos(self):
    return self._robot.data.body_pos_w[:, self.body_ids, :]  # GPU 텐서

@property
def base_quat(self):
    return self._robot.data.root_state_w[:, [4,5,6,3]]  # wxyz->xyzw 변환
```

#### 물리 시뮬레이션

```python
def simulate_at_each_physics_step(self):
    self.scene.write_data_to_sim()     # GPU -> 물리 엔진
    self.sim.step(render=False)        # 물리 시뮬레이션 1스텝
    if step % render_interval == 0 and is_rendering:
        self.sim.render()              # 선택적 렌더링
    self.scene.update(dt=self.sim_dt)  # 물리 엔진 -> GPU
```

### 5.4 MuJoCo vs Isaac Sim 비교

| 항목 | MuJoCo | Isaac Sim |
|------|--------|-----------|
| **물리 엔진** | MuJoCo (C 기반) | PhysX (GPU 가속) |
| **병렬 환경** | 1개 (단일 인스턴스) | 수천 개 (GPU 병렬) |
| **장치** | CPU 전용 | CUDA GPU |
| **모델 포맷** | XML (MJCF) | USD |
| **쿼터니언** | wxyz | wxyz (내부 xyzw 변환) |
| **텐서 변환** | numpy -> torch (매 스텝) | 직접 GPU 텐서 |
| **지형** | 기본 평면 | heightfield, trimesh, 다중 지형 |
| **Domain Rand** | 직접 모델 파라미터 수정 | EventManager 이벤트 기반 |
| **용도** | 추론, macOS 지원, 디버깅 | 대규모 훈련 (Linux GPU 필수) |
| **OS 지원** | Windows/macOS/Linux | Linux 전용 |

---

## 6. 설정 파일 분석 (YAML)

### 6.1 reward_bfm_zero.yaml - 보상 가중치

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/config/rewards/reward_bfm_zero.yaml`

```yaml
rewards:
  reward_scales:
    penalty_torques: -0.000001          # 토크 크기 페널티 (매우 작음)
    penalty_undesired_contact: -1        # 원치 않는 접촉 (pelvis, shoulder, hip)
    penalty_action_rate: -0.5            # 행동 변화율 (스무스한 제어 유도)
    penalty_ankle_roll: -0.5             # 발목 롤 페널티
    penalty_feet_ori: -0.1              # 발 방향 페널티
    feet_heading_alignment: -0.1        # 발 heading 정렬
    penalty_slippage: -1.0              # 미끄러짐 페널티
    limits_dof_pos: -10.0               # 관절 위치 한계 위반
    limits_dof_vel: -5.0                # 관절 속도 한계 위반
    limits_torque: -5.0                 # 토크 한계 위반
```

핵심 관찰:
- 모든 보상이 **음수 (페널티)**임 -- BFM-Zero의 reward-free 학습에서 보상은 페널티만으로 구성
- `penalty_action_rate`이 -0.5로 비교적 크며, 이는 실제 로봇 전이를 위한 스무스한 제어에 중요
- `limits_dof_pos`가 -10.0으로 가장 크며, 관절 한계 위반을 강력히 억제

```yaml
  reward_limit:
    soft_dof_pos_limit: 0.95     # 관절 위치 소프트 한계 (하드 한계의 95%)
    soft_dof_vel_limit: 0.95     # 관절 속도 소프트 한계
    soft_torque_limit: 0.95      # 토크 소프트 한계
```

커리큘럼 보상 관련:
```yaml
  reward_penalty_curriculum: False        # 기본 비활성
  reward_initial_penalty_scale: 0.10      # 초기 페널티 스케일 (10%)
  reward_penalty_degree: 0.000003         # 스케일 변화 속도
  reward_penalty_level_up_threshold: 42   # 에피소드 길이 기준
  reward_penalty_reward_names: [          # 커리큘럼 적용 대상
    "penalty_torques", "penalty_action_rate",
    "limits_dof_pos", "limits_torque",
    "penalty_slippage", "penalty_undesired_contact",
    "penalty_ankle_roll", "penalty_feet_ori"
  ]
```

### 6.2 mujoco.yaml / isaacsim.yaml - 시뮬레이터 파라미터

**MuJoCo 설정** (`/Users/jungyeon/Documents/BFM-Zero/humanoidverse/config/simulator/mujoco.yaml`):
```yaml
simulator:
  _target_: humanoidverse.simulator.mujoco.mujoco.MuJoCo
  config:
    name: "mujoco"
    plane:
      static_friction: 1.0
      dynamic_friction: 1.0
      restitution: 0.0
    sim:
      fps: 200                # 시뮬레이션 200Hz (dt=0.005s)
      control_decimation: 4   # 제어 4스텝마다 = 50Hz
      substeps: 1
      render_interval: 4
```

**Isaac Sim 설정** (`/Users/jungyeon/Documents/BFM-Zero/humanoidverse/config/simulator/isaacsim.yaml`):
```yaml
simulator:
  _target_: humanoidverse.simulator.isaacsim.isaacsim.IsaacSim
  config:
    name: "isaacsim"
    sim:
      fps: 200                        # 동일한 200Hz
      control_decimation: 4            # 동일한 50Hz 제어
      physx:
        solver_type: 1                 # TGS (Temporal Gauss-Seidel) solver
        num_position_iterations: 4     # 위치 반복 횟수
        num_velocity_iterations: 0     # 속도 반복 횟수
        bounce_threshold_velocity: 0.5
        max_depenetration_velocity: 1.0
    scene:
      replicate_physics: True          # 물리 복제 (메모리 절약)
```

두 시뮬레이터 모두 **200Hz 시뮬레이션, 50Hz 제어** (control_decimation=4)로 동일한 시간 구조를 사용한다.

### 6.3 g1_29dof.yaml - 로봇 설정

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/config/robot/g1/g1_29dof.yaml`

#### 바디 구성 (30개 = 29 링크 + pelvis)

```yaml
robot:
  num_bodies: 30
  body_names:
    - pelvis
    # 왼쪽 다리 (6)
    - left_hip_pitch_link, left_hip_roll_link, left_hip_yaw_link
    - left_knee_link, left_ankle_pitch_link, left_ankle_roll_link
    # 오른쪽 다리 (6)
    - right_hip_pitch_link, ...
    # 허리 (3)
    - waist_yaw_link, waist_roll_link, torso_link
    # 왼팔 (7)
    - left_shoulder_pitch_link, ..., left_wrist_yaw_link
    # 오른팔 (7)
    - right_shoulder_pitch_link, ..., right_wrist_yaw_link
```

#### PD 제어 게인

```yaml
  control:
    stiffness:  # K_p [N*m/rad]
      hip_yaw: 40.17924
      hip_roll: 99.09843
      hip_pitch: 99.09843
      knee: 99.09843
      ankle_pitch: 28.50125
      ankle_roll: 28.50125
      waist_yaw: 40.17924
      shoulder_pitch: 14.25062
      elbow: 14.25062
      wrist_roll: 14.25062
    damping:    # K_d [N*m*s/rad]
      hip_yaw: 2.55789
      hip_roll: 6.30880
      knee: 6.30880
      ankle_pitch: 1.81445
```

다리 관절이 가장 높은 강성(~99 N*m/rad), 손목이 가장 낮은 강성(~14 N*m/rad)을 가진다.

#### 관절 한계

```yaml
  dof_pos_lower_limit_list:
    [-2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,  # 왼쪽 다리
     -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,  # 오른쪽 다리
     -2.618, -0.52, -0.52,                                       # 허리
     -3.0892, -1.5882, -2.618, -1.0472,                         # 왼쪽 어깨+팔꿈치
     -1.972, -1.614, -1.614,                                     # 왼쪽 손목
     ...]
```

비대칭 구조 주의: 왼쪽 hip_roll [-0.5236, 2.9671] vs 오른쪽 hip_roll [-2.9671, 0.5236]

#### 초기 상태

```yaml
  init_state:
    pos: [0.0, 0.0, 0.8]           # 높이 0.8m에서 시작
    default_joint_angles:
      left_hip_pitch_joint: -0.1    # 약간 굽힌 자세
      left_knee_joint: 0.3          # 무릎 30도 굽힘
      left_ankle_pitch_joint: -0.2  # 발목 보상
```

#### 모션 라이브러리 설정

```yaml
  motion:
    motion_file: 'data/lafan_29dof.pkl'
    standardize_motion_length: True
    standardize_motion_length_value: 10  # 10초로 표준화
    extend_config:
      - joint_name: "head_link"
        parent_name: "torso_link"
        pos: [0.0, 0.0, 0.35]
        rot: [1.0, 0.0, 0.0, 0.0]
```

#### 변형 설정 비교

3개의 g1_29dof YAML 변형:

| 설정 | g1_29dof.yaml | g1_29dof_hard_waist.yaml | g1_29dof_new_effort_limit.yaml |
|------|---------------|--------------------------|-------------------------------|
| waist stiffness | 40.17, 28.5, 28.5 | **300, 300, 300** | 40.17, 28.5, 28.5 |
| waist damping | 2.56, 1.81, 1.81 | **5, 5, 5** | 2.56, 1.81, 1.81 |
| hip effort limit | 88, 88, 88 | 88, 88, 88 | **139, 139, 88** |

- `hard_waist`: 허리 강성을 7~10배 높여 상체 안정성 강화
- `new_effort_limit`: 엉덩이 pitch/roll 토크 한계를 88 -> 139 N*m으로 상향

---

## 7. Domain Randomization

**파일 위치**: `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/config/domain_rand/domain_rand.yaml`

### 7.1 물리 파라미터 랜덤화

```yaml
domain_rand:
  # 링크 질량 랜덤화
  randomize_link_mass: True
  link_mass_range: [0.95, 1.05]    # 원래 질량의 95~105%

  # 마찰 랜덤화
  randomize_friction: True
  friction_range: [0.5, 1.25]      # 마찰 계수 스케일 0.5~1.25

  # 기저 질량중심(COM) 랜덤화
  randomize_base_com: True
  base_com_range:
    x: [-0.02, 0.02]    # torso_link COM에 +/- 2cm 오프셋
    y: [-0.02, 0.02]
    z: [-0.02, 0.02]

  # PD 게인 랜덤화 (기본 비활성)
  randomize_pd_gain: False
  kp_range: [0.75, 1.25]
  kd_range: [0.75, 1.25]
```

### 7.2 외란 및 제어 랜덤화

```yaml
  # 로봇 밀기 (외부 외란)
  push_robots: True
  push_interval_s: [1, 3]              # 1~3초 간격으로 밀기
  max_push_vel_xy: 0.5                  # 최대 밀기 속도 0.5 m/s
  max_push_ang_vel: 0.5                 # 최대 밀기 각속도 0.5 rad/s
  push_robot_recovery_time: 2.0         # 밀기 후 회복 시간 (이 동안 종료 억제)

  # 기본 관절 자세 랜덤화
  randomize_default_dof_pos: True
  default_dof_pos_noise_range: [-0.02, 0.02]   # +/- 0.02 rad

  # 제어 지연 (기본 비활성)
  randomize_ctrl_delay: False
  ctrl_delay_step_range: [0, 1]        # 0~1스텝 지연 (최대 ~5ms)
```

### 7.3 Observation Noise

G1Env29dof의 관찰 노이즈 설정 (robot_29dof.py):
```python
noise_config = {
    level: 0.0,               # 0.0=비활성, 1.0=활성
    scales: {
        joint_pos: 0.03,      # +/- 0.03 rad
        joint_vel: 1.5,       # +/- 1.5 rad/s
        gravity: 0.05,        # +/- 0.05
        linvel: 0.1,          # +/- 0.1 m/s
        gyro: 0.2,            # +/- 0.2 rad/s
        torque: 0.1,          # 토크 출력 노이즈 (torque_lim_scale과 결합)
    },
}
```

노이즈 적용 예시:
```python
noisy_joint_angles = joint_angles + rng.uniform(-1, 1) * noise_level * noise_scales.joint_pos
noisy_gravity = gravity + rng.uniform(-1, 1) * noise_level * noise_scales.gravity
```

### 7.4 Sim-to-Real 전이 의미

Domain Randomization은 **Sim-to-Real 전이**의 핵심 기법이다:

1. **질량 랜덤화**: 실제 로봇의 무게 오차, 부하 변화 대응
2. **마찰 랜덤화**: 다양한 바닥면 조건 대응
3. **COM 랜덤화**: 무게 분포 오차 대응
4. **외란(Push)**: 외부 충격에 대한 로버스트성 확보
5. **관찰 노이즈**: 센서 노이즈 대응
6. **제어 지연**: 통신 지연, 처리 지연 대응
7. **기본 자세 랜덤화**: 조립 오차, 캘리브레이션 오차 대응

이러한 랜덤화를 통해 시뮬레이션에서 학습한 정책이 실제 환경의 불확실성에도 로버스트하게 동작하도록 한다.

---

## 8. 핵심 개념 정리

### 환경-시뮬레이터 분리의 이점

1. **백엔드 교체 용이**: 동일한 환경 로직으로 MuJoCo(디버깅/추론) <-> Isaac Sim(대규모 훈련) 전환 가능
2. **플랫폼 독립성**: MuJoCo로 macOS에서 추론, Isaac Sim으로 Linux GPU 서버에서 훈련
3. **테스트 용이성**: MuJoCo 단일 환경에서 빠르게 디버깅 후 수천 환경으로 확장

### 이중 환경 구조 (g1_env_helper vs LeggedRobotMotions)

- **LeggedRobotMotions**: 훈련용. 시뮬레이터 백엔드를 추상화하여 GPU 병렬 환경에서 모션 트래킹 학습
- **g1_env_helper (G1Env/G1Env29dof)**: 추론용. 순수 MuJoCo 기반. 태스크 보상 기반 평가에 사용

### 데이터 흐름

```
[Agent (FBcpr)]
    |  actions (29D)
    v
[HumanoidVerseVectorEnv]  -- Gymnasium 래퍼
    |
    v
[LeggedRobotMotions]  -- 모션 트래킹 환경
    |  토크 계산 (PD 제어)
    v
[BaseSimulator (MuJoCo/IsaacSim)]  -- 물리 시뮬레이션
    |  mj_step() / sim.step()
    v
[물리 상태 (qpos, qvel, xpos, xquat, ...)]
    |  텐서 변환
    v
[관찰 (state 64D + privileged_state 357D + last_action 29D + time 1D)]
    |
    v
[Agent (FBcpr)]
```

### 시간 구조

```
시뮬레이션: 200 Hz (dt = 0.005s)
      |-----|-----|-----|-----|
제어:          50 Hz (dt = 0.02s)
      |--------------------->|
      control_decimation = 4
```

1 제어 스텝 = 4 시뮬레이션 스텝. 정책은 50Hz로 행동을 출력하고, 물리는 200Hz로 시뮬레이션된다.
