# Phase 6: 모션 라이브러리 & 유틸리티 (Motion Library & Utilities)

## 1. 개요

BFM-Zero에서 모션 데이터는 휴머노이드 로봇 제어의 핵심 입력이다. 전체 파이프라인은 다음과 같이 구성된다:

```
[.pkl 모션 파일] --> [MotionLibBase: 로딩/필터링/샘플링]
                         |
                    [Humanoid_Batch: MJCF 파싱 + Forward Kinematics]
                         |
                    [모션 상태 텐서: gts, grs, lrs, gvs, gavs, dvs, dof_pos]
                         |
                    [학습 환경에서 참조 모션으로 사용]
                         |
                    [FBcpr 에이전트: Backward Map으로 z 인코딩 --> Forward Policy로 행동 생성]
```

**핵심 데이터 흐름:**
1. `.pkl` 파일에서 `pose_aa` (axis-angle)와 `root_trans_offset` (루트 이동) 로딩
2. `Humanoid_Batch.fk_batch()`로 Forward Kinematics 수행 --> 관절별 글로벌 위치/회전 계산
3. `MotionLibBase`가 모든 모션을 연결(concatenate)하여 하나의 큰 텐서로 저장
4. 학습 시 `get_motion_state(motion_ids, motion_times)`으로 특정 시점의 모션 상태를 보간하여 반환

---

## 2. skeleton.py - 스켈레톤 표현

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/motion_lib/skeleton.py`

이 파일은 로봇의 골격 구조를 트리 형태로 표현하는 핵심 모듈이다. 세 가지 주요 클래스 계층이 있다:
`SkeletonTree` --> `SkeletonState` --> `SkeletonMotion`

### 2.1 SkeletonTree 클래스

`SkeletonTree`는 로봇의 관절 계층 구조를 정의하는 기초 클래스다. 세 가지 핵심 속성을 가진다:

```python
class SkeletonTree(Serializable):
    def __init__(self, node_names, parent_indices, local_translation):
        self._node_names = node_names           # List[str]: 관절 이름 목록
        self._parent_indices = parent_indices.long()  # Tensor: 부모 관절 인덱스 (-1은 루트)
        self._local_translation = local_translation   # Tensor: 부모 대비 로컬 위치 오프셋 (N, 3)
        self._node_indices = {self.node_names[i]: i for i in range(len(self))}  # 이름->인덱스 매핑
```

**예시:** 4족 로봇(ant)의 경우:
- `parent_indices = tensor([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11])`
- 인덱스 `-1`은 루트 노드(torso)를 의미
- 인덱스 `0`은 부모가 torso인 다리 시작점

### 2.2 부모-자식 관절 계층

`parent_indices` 텐서가 트리 구조의 핵심이다. 이를 통해:

```python
def parent_of(self, node_name):
    """이름으로 부모 관절을 찾는다"""
    return self[int(self.parent_indices[self.index(node_name)].item())]
```

노드 제거/유지 메서드도 제공한다. `drop_nodes_by_names()`는 특정 관절을 제거하면서 트리 구조를 재구성한다:

```python
def drop_nodes_by_names(self, node_names, pairwise_translation=None):
    # 제거할 노드를 건너뛰면서 부모 인덱스를 재계산
    # 제거된 노드의 local_translation을 합산하여 거리 보정
    while tb_node_index != -1 and self[tb_node_index] in node_names:
        local_translation += self.local_translation[tb_node_index, :]
        tb_node_index = parent_indices[tb_node_index]
```

이 기능은 리타겟팅 시 소스 스켈레톤과 타겟 스켈레톤의 관절 수가 다를 때 필수적이다.

### 2.3 MJCF에서 스켈레톤 구축

MuJoCo XML (MJCF) 파일에서 스켈레톤을 파싱하는 클래스 메서드:

```python
@classmethod
def from_mjcf(cls, path: str) -> "SkeletonTree":
    tree = ET.parse(path)
    xml_world_body = tree.getroot().find("worldbody")
    xml_body_root = xml_world_body.find("body")

    # 재귀적으로 모든 body 노드를 순회
    def _add_xml_node(xml_node, parent_index, node_index):
        node_name = xml_node.attrib.get("name")
        pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
        node_names.append(node_name)
        parent_indices.append(parent_index)
        local_translation.append(pos)
        # 자식 body를 재귀 탐색
        for next_node in xml_node.findall("body"):
            node_index = _add_xml_node(next_node, curr_index, node_index)
```

**MJCF 구조 예시 (G1 로봇):**
```xml
<worldbody>
  <body name="pelvis" pos="0 0 0.793">
    <body name="left_hip_pitch_link" pos="0 0.0955 -0.1065">
      <body name="left_hip_roll_link" pos="0 0 0">
        ...
      </body>
    </body>
  </body>
</worldbody>
```

### 2.4 JSON 직렬화

`Serializable` 기반 클래스가 JSON과 `.npy` 형식으로의 직렬화/역직렬화를 지원한다:

```python
class NumpyEncoder(json.JSONEncoder):
    """numpy 타입을 JSON으로 변환"""
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return dict(__ndarray__=obj.tolist(), dtype=str(obj.dtype), shape=obj.shape)

def json_numpy_obj_hook(dct):
    """JSON에서 numpy 배열을 복원"""
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = np.asarray(dct["__ndarray__"], dtype=dct["dtype"])
        return data.reshape(dct["shape"])
```

`SkeletonTree.to_dict()`는 `OrderedDict`로 변환하고, `from_dict()`로 복원한다. 파일 형식은 `__name__` 키로 클래스를 식별한다.

---

## 3. motion_lib_base.py - 모션 라이브러리

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/motion_lib/motion_lib_base.py`

### 3.1 MotionLibBase 클래스

모션 데이터의 로딩, 저장, 샘플링을 관리하는 중앙 클래스다:

```python
class MotionLibBase():
    def __init__(self, motion_lib_cfg, num_envs, device):
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)  # 시뮬레이션 FPS (기본 50Hz)
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height=motion_lib_cfg.get("fix_height", FixHeightMode.no_fix))
```

**주요 텐서 저장 구조 (load_motions 후):**

| 변수명 | 의미 | 형태 |
|--------|------|------|
| `self.gts` | 글로벌 관절 위치 (global translations) | `[총_프레임, 관절수, 3]` |
| `self.grs` | 글로벌 관절 회전 (global rotations, xyzw) | `[총_프레임, 관절수, 4]` |
| `self.lrs` | 로컬 관절 회전 (local rotations) | `[총_프레임, 관절수, 4]` |
| `self.gvs` | 글로벌 선속도 (global velocities) | `[총_프레임, 관절수, 3]` |
| `self.gavs` | 글로벌 각속도 (global angular velocities) | `[총_프레임, 관절수, 3]` |
| `self.dvs` | 관절 속도 (dof velocities) | `[총_프레임, 관절수]` |
| `self.dof_pos` | 관절 위치 (dof positions) | `[총_프레임, 관절수]` |

모든 모션이 하나의 텐서로 연결(concatenate)되며, `self.length_starts`가 각 모션의 시작 프레임 인덱스를 추적한다.

### 3.2 .pkl 파일에서 모션 로딩

`load_data()`는 두 가지 모드를 지원한다:

```python
def load_data(self, motion_file, min_length=-1, im_eval=False):
    if osp.isfile(motion_file):
        self.mode = MotionlibMode.file           # 단일 .pkl 파일
        self._motion_data_load = joblib.load(motion_file)  # dict of dicts
    else:
        self.mode = MotionlibMode.directory       # 디렉토리 내 여러 .pkl
        self._motion_data_load = glob.glob(osp.join(motion_file, "*.pkl"))
```

**단일 파일 모드:** `.pkl` 파일은 Python 딕셔너리로, 각 키가 모션 이름이고 값이 해당 모션 데이터:
```python
{
    "motion_name_1": {
        "pose_aa": array,          # axis-angle 포즈 (T, J, 3)
        "root_trans_offset": array, # 루트 이동 (T, 3)
        "fps": 30,                  # 프레임 레이트
        "pose_quat_global": array,  # 글로벌 쿼터니언 (필터링 용)
        ...
    },
    "motion_name_2": { ... },
}
```

**필터링 옵션:**
- `min_length != -1`: 최소 프레임 수 이상인 모션만 로드
- `im_eval=True`: 모션을 길이 내림차순으로 정렬 (평가용)

### 3.3 FixHeightMode (no_fix, full_fix, ankle_fix)

높이 보정 모드를 제어하는 Enum:

```python
class FixHeightMode(Enum):
    no_fix = 0      # 높이 보정 없음
    full_fix = 1    # 메시 기반 전체 높이 보정
    ankle_fix = 2   # 발목 기준 높이 보정
```

`fix_trans_height()` 메서드가 실제 보정을 수행한다:

```python
def fix_trans_height(self, pose_aa, trans, fix_height_mode):
    if fix_height_mode == FixHeightMode.no_fix:
        return trans, 0
    with torch.no_grad():
        mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
        height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()  # 메시 최저점
        trans[..., 2] -= height_diff  # z축 이동으로 바닥에 밀착
```

이 기능은 모션 캡처 데이터에서 로봇 모션으로 리타겟팅할 때 발이 바닥에 정확히 닿도록 보정하는 데 사용된다.

### 3.4 모션 필터링과 셋업

모션 로딩은 학습(training)과 평가(evaluation) 두 가지 경로가 있다:

```python
def load_motions_for_training(self, max_num_seqs=None):
    # max_num_seqs가 None이면 전체 로딩
    # max_num_seqs > 고유 모션 수면 전체 로딩
    # 그 외에는 _sampling_prob에 따라 랜덤 샘플링
    self.load_motions(random_sample=True, num_motions_to_load=max_num_seqs)

def load_motions_for_evaluation(self, start_idx=0):
    # 평가 시에는 start_idx부터 순차 로딩
    self.load_motions(random_sample=False, num_motions_to_load=self.num_envs, start_idx=start_idx)
```

**Auto PMCP (Prioritized Motion Curriculum Planning)** -- 실패한 모션에 더 높은 샘플링 확률 부여:

```python
def update_soft_sampling_weight(self, failed_keys):
    if len(failed_keys) > 0:
        indexes = [all_keys.index(k) for k in failed_keys]
        self._termination_history[indexes] += 1
        # 실패 횟수에 비례하여 샘플링 확률 조정
        self._sampling_prob[:] = termination_history / termination_history.sum()
```

### 3.5 모션 상태 조회 (get_motion_state)

학습 중 특정 시점의 모션 상태를 보간하여 반환하는 핵심 메서드:

```python
def get_motion_state(self, motion_ids, motion_times, offset=None):
    # 1. 프레임 인덱스와 보간 가중치 계산
    frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

    # 2. 연결된 텐서에서 실제 인덱스 계산
    f0l = frame_idx0 + self.length_starts[motion_ids]
    f1l = frame_idx1 + self.length_starts[motion_ids]

    # 3. 선형 보간 (위치, 속도)
    rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1

    # 4. 구면 선형 보간 (회전) -- slerp 사용
    rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
```

`_calc_frame_blend()`는 연속적인 시간을 두 이웃 프레임과 블렌드 비율로 변환한다:

```python
def _calc_frame_blend(self, time, len, num_frames, dt):
    phase = time / len  # 0~1 정규화
    phase = torch.clip(phase, 0.0, 1.0)
    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)
    return frame_idx0, frame_idx1, blend
```

### 3.6 SMPL 모션 데이터 지원

SMPL 인체 모델 기반의 모션 데이터도 선택적으로 로드 가능:

```python
smpl_motion_file = motion_lib_cfg.get("smpl_motion_file", None)
if smpl_motion_file is not None:
    self.smpl_data = joblib.load(smpl_motion_file)
    self.smpl_data = [self.smpl_data[k] for k in self._motion_data_keys]
```

이는 사람의 모션 캡처 데이터를 로봇에 리타겟팅하기 위한 참조 데이터로 사용된다.

---

## 4. torch_humanoid_batch.py - 배치 처리

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/motion_lib/torch_humanoid_batch.py`

### 4.1 Humanoid_Batch 클래스

MJCF 파일에서 로봇의 물리적 구조를 파싱하고, 배치 Forward Kinematics를 수행하는 클래스다:

```python
class Humanoid_Batch:
    def __init__(self, cfg, device=torch.device("cpu")):
        self.mjcf_file = self.asset_root / self.asset_file
        self.mjcf_data = self.from_mjcf(self.mjcf_file)

        self.body_names = copy.deepcopy(mjcf_data['node_names'])
        self._parents = mjcf_data['parent_indices']
        self._offsets = mjcf_data['local_translation'][None, ].to(device)      # 관절 오프셋 (1, J, 3)
        self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)  # 로컬 회전 (1, J, 4) wxyz
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation)   # 회전 행렬로 변환 (1, J, 3, 3)
        self.joints_range = mjcf_data['joints_range'].to(device)               # 관절 범위
```

**`from_mjcf()` vs `SkeletonTree.from_mjcf()`의 차이:**
- `SkeletonTree.from_mjcf()`: 이름, 부모 인덱스, 위치 오프셋만 파싱
- `Humanoid_Batch.from_mjcf()`: 추가로 **로컬 회전(quat)**, **관절 범위(range)**, **body-to-joint 매핑** 파싱

`extend_config`를 통해 추가 관절(예: 손끝 센서)을 동적으로 추가할 수 있다:

```python
for extend_config in cfg.extend_config:
    self.body_names_augment += [extend_config.joint_name]
    self._parents = torch.cat([self._parents, torch.tensor([self.body_names.index(extend_config.parent_name)])])
    self._offsets = torch.cat([self._offsets, torch.tensor([[extend_config.pos]])], dim=1)
    self._local_rotation = torch.cat([self._local_rotation, torch.tensor([[extend_config.rot]])], dim=1)
```

### 4.2 배치 FK (Forward Kinematics)

`fk_batch()`는 axis-angle 포즈와 루트 이동으로부터 모든 관절의 글로벌 위치와 회전을 계산한다:

```python
def fk_batch(self, pose, trans, convert_to_mat=True, return_full=False, dt=1/30):
    # 1. axis-angle --> 쿼터니언 --> 회전 행렬 변환
    pose_quat = axis_angle_to_quaternion(pose.clone())  # wxyz 포맷
    pose_mat = quaternion_to_matrix(pose_quat)            # (B, T, J, 3, 3)

    # 2. Forward Kinematics 수행
    wbody_pos, wbody_mat = self.forward_kinematics_batch(
        pose_mat[:, :, 1:],      # 자식 관절들의 회전 (루트 제외)
        pose_mat[:, :, 0:1],     # 루트 회전
        trans                     # 루트 이동
    )

    # 3. 회전 행렬 --> 쿼터니언 (xyzw 포맷으로 변환)
    wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))
```

`forward_kinematics_batch()`의 핵심 루프:

```python
def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
    for i in range(J):
        if self._parents[i] == -1:
            positions_world.append(root_positions)
            rotations_world.append(root_rotations)
        else:
            # 부모의 글로벌 회전으로 오프셋을 회전시켜 위치 계산
            jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0],
                                 expanded_offsets[:, :, i, :, None]).squeeze(-1)
                    + positions_world[self._parents[i]])
            # 글로벌 회전 = 부모 글로벌 회전 * 로컬 고정 회전 * 입력 회전
            rot_mat = torch.matmul(rotations_world[self._parents[i]],
                                   torch.matmul(self._local_rotation_mat[:, (i):(i+1)],
                                                rotations[:, :, (i-1):i, :]))
            positions_world.append(jpos)
            rotations_world.append(rot_mat)
```

**FK 수식:** 관절 `i`의 글로벌 변환:
```
T_global[i] = T_global[parent[i]] * T_local_fixed[i] * T_input[i]
position[i] = R_global[parent[i]] * offset[i] + position[parent[i]]
```

여기서 `T_local_fixed`는 MJCF에서 정의된 고정 로컬 회전이고, `T_input`은 포즈 데이터에서 오는 가변 회전이다.

`return_full=True`일 때 추가로 계산되는 값들:

```python
if return_full:
    rigidbody_linear_velocity = self._compute_velocity(wbody_pos, dt)       # 수치 미분 + 가우시안 필터
    rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)  # 쿼터니언 차분
    return_dict.local_rotation = wxyz_to_xyzw(pose_quat)
    return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]  # 1-DOF 관절: axis-angle 크기가 곧 관절 위치
    return_dict.dof_vels = (dof_pos[:, 1:] - dof_pos[:, :-1]) / dt
    return_dict.fps = int(1/dt)
```

**중요 설계 결정:** `dof_pos = pose.sum(dim=-1)`는 각 관절이 1-DOF (한 축만 회전)인 Unitree 로봇에서 axis-angle 벡터 `(x, y, z)`를 스칼라 관절 각도로 변환하는 트릭이다. 하나의 축만 비영인 경우 `sum()`이 해당 축의 값을 반환한다.

### 4.3 메시 FK (mesh_fk)

3D 메시를 로봇 포즈에 맞게 변환하여 충돌 검사나 높이 보정에 사용한다:

```python
def mesh_fk(self, pose=None, trans=None):
    fk_res = self.fk_batch(pose, trans)
    for geom in geoms:
        body_trans = g_trans[body_idx].numpy()
        body_rot = g_rot[body_idx].numpy()
        mesh_obj.rotate(global_rot.T, center=(0, 0, 0))
        mesh_obj.translate(body_trans)
        joined_mesh_obj.append(mesh_obj)
    # 모든 메시를 합쳐서 하나의 메시로 반환
    merged_mesh = joined_mesh_obj[0]
    for mesh in joined_mesh_obj[1:]:
        merged_mesh += mesh
```

이 합쳐진 메시의 최저 z 좌표를 사용하여 `fix_trans_height()`에서 높이 보정을 수행한다.

---

## 5. motion_utils/ - 회전 변환

**디렉토리 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/motion_lib/motion_utils/`

### 5.1 rotation_conversions.py

Facebook Research에서 제공한 회전 표현 변환 유틸리티 (PyTorch3D 기반). 모든 쿼터니언은 **wxyz** (real-first) 포맷을 사용한다.

**변환 함수 전체 목록:**

| 함수 | 입력 | 출력 |
|------|------|------|
| `quaternion_to_matrix()` | `(..., 4)` 쿼터니언 | `(..., 3, 3)` 회전행렬 |
| `matrix_to_quaternion()` | `(..., 3, 3)` 회전행렬 | `(..., 4)` 쿼터니언 |
| `euler_angles_to_matrix()` | `(..., 3)` 오일러각 + convention | `(..., 3, 3)` 회전행렬 |
| `matrix_to_euler_angles()` | `(..., 3, 3)` 회전행렬 + convention | `(..., 3)` 오일러각 |
| `axis_angle_to_quaternion()` | `(..., 3)` axis-angle | `(..., 4)` 쿼터니언 |
| `quaternion_to_axis_angle()` | `(..., 4)` 쿼터니언 | `(..., 3)` axis-angle |
| `axis_angle_to_matrix()` | `(..., 3)` axis-angle | `(..., 3, 3)` 회전행렬 |
| `matrix_to_axis_angle()` | `(..., 3, 3)` 회전행렬 | `(..., 3)` axis-angle |
| `rotation_6d_to_matrix()` | `(..., 6)` 6D 표현 | `(..., 3, 3)` 회전행렬 |
| `matrix_to_rotation_6d()` | `(..., 3, 3)` 회전행렬 | `(..., 6)` 6D 표현 |

### 5.2 쿼터니언 <-> 회전행렬 <-> 오일러각 변환

**쿼터니언 --> 회전행렬:**

```python
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)  # wxyz 포맷: r=w, i=x, j=y, k=z
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack((
        1 - two_s * (j*j + k*k),  two_s * (i*j - k*r),  two_s * (i*k + j*r),
        two_s * (i*j + k*r),  1 - two_s * (i*i + k*k),  two_s * (j*k - i*r),
        two_s * (i*k - j*r),  two_s * (j*k + i*r),  1 - two_s * (i*i + j*j),
    ), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))
```

**회전행렬 --> 쿼터니언:** 수치적으로 안정적인 Shepperd 방법을 사용:

```python
def matrix_to_quaternion(matrix):
    # 4가지 경우(w, x, y, z가 최대인 경우)에 대해 각각 계산
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22,  # w가 최대
        1.0 + m00 - m11 - m22,  # x가 최대
        1.0 - m00 + m11 - m22,  # y가 최대
        1.0 - m00 - m11 + m22,  # z가 최대
    ], dim=-1))
    # 가장 큰 절대값을 가진 요소를 기준으로 선택
    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :]
```

**오일러각 --> 회전행렬:** 축별 회전 행렬을 생성하고 곱한다:

```python
def euler_angles_to_matrix(euler_angles, convention):
    # convention 예: "XYZ", "ZYX"
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])
```

### 5.3 axis-angle 표현

Axis-angle은 `(..., 3)` 벡터로, 방향이 회전축이고 크기(norm)가 회전 각도(라디안)이다:

```python
def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # 회전 각도
    half_angles = angles * 0.5
    # 수치 안정성: 작은 각도에서 테일러 전개 사용
    sin_half_angles_over_angles[small_angles] = (0.5 - (angles[small_angles]**2) / 48)
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    # 출력: wxyz 포맷
```

**6D 회전 표현 (Zhou et al., CVPR 2019):** 신경망 학습에서 회전 표현의 연속성 문제를 해결하기 위한 표현:

```python
def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)                           # 첫 번째 정규직교 벡터
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1        # Gram-Schmidt 직교화
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)                       # 외적으로 세 번째 벡터
    return torch.stack((b1, b2, b3), dim=-2)
```

### 5.4 flags.py

전역 플래그 관리 유틸리티:

```python
flags = Flags({
    'test': False,
    'debug': False,
    "real_traj": False,
    "im_eval": False,
})
```

---

## 6. 유틸리티 함수들

### 6.1 helpers.py - export_meta_policy_as_onnx(), get_backward_observation()

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/helpers.py`

**`get_backward_observation()`** -- Backward Map 학습을 위한 참조 관측 생성:

```python
def get_backward_observation(env, motion_id, use_root_height_obs=False, velocity_multiplier=1.0):
    # 모션의 전체 타임스텝에 대한 시간 배열 생성
    motion_times = torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id]/env.dt).cpu()))).to(env.device) * env.dt

    # 모션 상태 조회 (보간 포함)
    motion_state = env._motion_lib.get_motion_state(motion_id, motion_times)

    # 참조 관측값 구성
    ref_body_pos = motion_state["rg_pos_t"]           # 확장 관절 포함 글로벌 위치
    ref_body_rots = motion_state["rg_rot_t"]          # 글로벌 회전
    ref_body_vels = motion_state["body_vel_t"] * velocity_multiplier
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]  # 기본 자세 대비 오프셋
    ref_dof_vel = motion_state["dof_vel"] * velocity_multiplier
```

이 함수는 `use_obs_filter=True`일 때 BFM-Zero 전용 관측 형식을 반환한다:

```python
bfmzero_obs = {
    "state": torch.cat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel], dim=-1),
    "last_action": bogus_actions,
    "privileged_state": max_local_self_obs  # 전체 관절 정보
}
```

**`export_meta_policy_as_onnx()`** -- ONNX 모델 내보내기:

```python
def export_meta_policy_as_onnx(inference_model, path, exported_policy_name, example_obs_dict, z_dim, history=False, use_29dof=True):
    class PPOWrapper(nn.Module):
        def forward(self, actor_obs):
            actor_obs, ctx = actor_obs[:, :-z_dim], actor_obs[:, -z_dim:]  # 관측과 z를 분리
            if use_29dof:
                state_end = 64         # 29-DOF: state 차원 = 64
                action_end = state_end + 29
            else:
                state_end = 52         # 23-DOF: state 차원 = 52
                action_end = state_end + 23
            # ...
            return self.actor.act(actor_dict, ctx)
    # ONNX 변환
    torch.onnx.export(wrapper, example_input_list, path, opset_version=13,
                      input_names=["actor_obs"], output_names=["action"])
```

**`pre_process_config()`** -- 관측 차원을 자동 계산:

```python
def pre_process_config(config):
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
    config.robot.algo_obs_dim_dict = obs_dim_dict
```

### 6.2 math.py - 수학 함수

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/math.py`

NVIDIA IsaacGym에서 가져온 기본 수학 유틸리티:

```python
def quat_apply_yaw(quat, vec):
    """쿼터니언에서 yaw 성분만 추출하여 벡터에 적용"""
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.  # x, y 성분 제거 --> yaw만 남김
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec, w_last=True)

def wrap_to_pi(angles):
    """각도를 [-pi, pi] 범위로 정규화"""
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def torch_rand_sqrt_float(lower, upper, shape, device):
    """제곱근 분포 랜덤 생성 (중심에 더 많은 샘플)"""
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower
```

### 6.3 torch_utils.py - quat_rotate, quat_multiply 등

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/torch_utils.py`

이 파일은 프로젝트의 가장 핵심적인 쿼터니언/변환 유틸리티를 포함한다. 대부분의 함수가 `w_last` 매개변수를 지원하여 `xyzw`와 `wxyz` 두 포맷 모두를 처리한다.

**쿼터니언 회전:**

```python
@torch.jit.script
def quat_rotate(q, v, w_last: bool):
    """쿼터니언으로 벡터를 회전"""
    if w_last:
        q_w = q[:, -1]      # xyzw 포맷
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]       # wxyz 포맷
        q_vec = q[:, 1:]
    # 최적화된 회전 공식 (q*v*q^-1 대신 직접 계산)
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(N, 1, 3), v.view(N, 3, 1)).squeeze(-1) * 2.0
    return a + b + c
```

**쿼터니언 곱셈 (Hamilton product):**

```python
@torch.jit.script
def quat_mul(a, b, w_last: bool):
    """두 쿼터니언의 곱 (회전 합성)"""
    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    # ... 최적화된 Hamilton product 계산 ...
    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1)
    else:
        quat = torch.stack([w, x, y, z], dim=-1)
```

**구면 선형 보간 (SLERP):**

```python
@torch.jit.script
def slerp(q0, q1, t):
    """두 쿼터니언 사이의 구면 선형 보간"""
    cos_half_theta = torch.sum(q0 * q1, dim=-1)
    # 반구 보정: 내적이 음수면 q1을 반전
    neg_mask = cos_half_theta < 0
    q1[neg_mask] = -q1[neg_mask]
    # slerp 공식
    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta
    new_q = ratioA * q0 + ratioB * q1
    # 특수 경우 처리 (거의 같은 회전, 거의 반대 회전)
    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
```

**변환(Transform) 연산:**

```python
@torch.jit.script
def transform_from_rotation_translation(r=None, t=None):
    """회전(4D) + 이동(3D) = 변환(7D)으로 결합"""
    return torch.cat([r, t], dim=-1)

@torch.jit.script
def transform_mul(x, y):
    """두 변환 합성: T1 * T2"""
    z = transform_from_rotation_translation(
        r=quat_mul_norm(transform_rotation(x), transform_rotation(y), w_last=True),
        t=quat_rotate(transform_rotation(x), transform_translation(y), w_last=True)
          + transform_translation(x),
    )
    return z
```

**Heading 관련 함수:**

```python
@torch.jit.script
def calc_heading(q):
    """쿼터니언에서 yaw(heading) 각도 추출"""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1  # x축 기준 방향
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

@torch.jit.script
def calc_heading_quat_inv(q, w_last: bool):
    """heading의 역회전 쿼터니언 -- 로봇의 heading을 제거하여 로컬 프레임으로 변환"""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1  # z축 (수직축)
    heading_q = quat_from_angle_axis(-heading, axis, w_last=w_last)
    return heading_q
```

**포맷 변환 헬퍼:**

```python
def wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]

def xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]
```

### 6.4 pytree_utils.py - tree_concat, tree_numpy_to_tensor 등

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/agents/pytree_utils.py`

PyTree는 중첩된 dict/list/tensor 구조를 재귀적으로 처리하기 위한 유틸리티다. FBcpr 에이전트의 replay buffer에서 관측 데이터를 효율적으로 관리하는 데 사용된다.

```python
def tree_clone(pytree):
    """PyTree 내 모든 텐서를 복제"""
    return tree_map(clone_if_tensor, pytree)

def tree_numpy_to_tensor(pytree):
    """PyTree 내 모든 numpy 배열을 torch 텐서로 변환"""
    def convert(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
    return tree_map(convert, pytree)

def tree_concat(list_of_pytree_of_tensors, dim=0):
    """같은 구조의 PyTree 리스트를 leaf 단위로 concatenate"""
    tds = tuple(map(lambda x: TensorDict.from_pytree(x, auto_batch_size=True), list_of_pytree_of_tensors))
    concatenated = torch.cat(tds, dim=dim)
    if isinstance(concatenated, TensorDict):
        return concatenated.to_pytree()
    return concatenated
```

**사용 예시:** replay buffer에서 여러 trajectory의 관측값을 합칠 때:

```python
# 각 trajectory는 {"state": tensor, "last_action": tensor, ...} 구조
all_obs = tree_concat([traj1_obs, traj2_obs, traj3_obs], dim=0)
# 결과: {"state": cat(state1, state2, state3), "last_action": cat(act1, act2, act3), ...}
```

배치 크기 검증도 제공한다:

```python
def tree_check_batch_size(pytree, batch_size, prefix=""):
    """PyTree 내 모든 텐서의 첫 번째 차원이 batch_size와 일치하는지 검증"""
    if isinstance(pytree, torch.Tensor):
        if pytree.shape[0] != batch_size:
            raise ValueError(f"Batch size mismatch at {prefix}")
```

### 6.5 logging.py - 로깅

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/utils/logging.py`

Hydra의 표준 logging을 loguru로 브리지하는 유틸리티:

```python
class HydraLoggerBridge(logging.Handler):
    def emit(self, record):
        level = logger.level(record.levelname).name
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

@contextmanager
def capture_stdout_to_loguru():
    """stdout을 loguru로 리다이렉트하는 컨텍스트 매니저"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    sys.stdout = LoguruStream()
    # ... yield ...
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
```

---

## 7. 모션 데이터 포맷

### 7.1 lafan_29dof.pkl 구조

**파일 경로:** `/Users/jungyeon/Documents/BFM-Zero/humanoidverse/data/lafan_29dof.pkl`

LAFAN1 데이터셋을 29-DOF G1 로봇에 리타겟팅한 평가용 모션 데이터다. `joblib.load()`로 로딩하면 딕셔너리가 반환된다.

```python
{
    "motion_key_1": {
        "pose_aa": np.array,           # axis-angle 포즈, shape (T, J, 3)
                                        # T=프레임 수, J=관절 수(30), 3=axis-angle
        "root_trans_offset": np.array,  # 루트 이동, shape (T, 3)
        "fps": 30,                      # 프레임 레이트
        "pose_quat_global": np.array,   # 글로벌 쿼터니언, shape (T, J, 4) -- 필터링에 사용
        # 선택적 필드:
        "beta": np.array,              # SMPL 체형 파라미터 (존재 시)
        "action": np.array,            # 행동 레이블 (존재 시)
    },
    "motion_key_2": { ... },
    ...
}
```

### 7.2 lafan_29dof_10s-clipped.pkl (학습용)

10초로 클리핑된 학습용 버전이다. 학습 시 긴 모션은 메모리를 많이 소모하므로, 10초 이하로 잘라서 효율적으로 학습할 수 있다.

`MotionLibBase.load_motion_with_skeleton()`에서 추가 클리핑도 가능하다:

```python
seq_len = curr_file['root_trans_offset'].shape[0]
if max_len == -1 or seq_len < max_len:
    start, end = 0, seq_len
else:
    start = random.randint(0, seq_len - max_len)  # 랜덤 시작점에서 max_len만큼
    end = start + max_len
```

### 7.3 데이터 필드 상세

| 필드명 | 형태 | 설명 |
|--------|------|------|
| `pose_aa` | `(T, J, 3)` | axis-angle 포즈. `J[0]`은 루트 회전, `J[1:]`은 관절 회전 |
| `root_trans_offset` | `(T, 3)` | 루트 위치 (x, y, z). 첫 프레임이 원점 근처 |
| `fps` | `int` | 프레임 레이트. 보통 30 |
| `pose_quat_global` | `(T, J, 4)` | 글로벌 쿼터니언 (xyzw). 길이 필터링에 사용 |

**FK 후 생성되는 텐서들:**

| 텐서 | 형태 | 생성 함수 |
|-------|------|----------|
| `global_translation` | `(T, J, 3)` | `forward_kinematics_batch()` |
| `global_rotation` | `(T, J, 4)` | `wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))` |
| `local_rotation` | `(T, J, 4)` | `wxyz_to_xyzw(pose_quat)` |
| `global_velocity` | `(T, J, 3)` | `_compute_velocity()` 수치 미분 |
| `global_angular_velocity` | `(T, J, 3)` | `_compute_angular_velocity()` 쿼터니언 차분 |
| `dof_pos` | `(T, D)` | `pose.sum(dim=-1)[..., 1:]` (1-DOF 관절) |
| `dof_vels` | `(T, D)` | `(dof_pos[t+1] - dof_pos[t]) / dt` |

---

## 8. 핵심 개념 정리

### 8.1 Forward Kinematics와 로봇 제어

**Forward Kinematics (FK)** 는 관절 각도들로부터 각 관절/링크의 글로벌 위치와 방향을 계산하는 과정이다.

BFM-Zero에서 FK가 사용되는 두 가지 장면:

1. **모션 데이터 전처리**: axis-angle 포즈 --> FK --> 글로벌 관절 위치/회전/속도
2. **시뮬레이션 중 참조 추적**: 참조 모션의 글로벌 위치와 시뮬레이터의 현재 상태를 비교

FK의 재귀 구조:
```
T_global[root] = T_root_input
T_global[child] = T_global[parent] * T_offset * T_local_fixed * T_joint_input
```

여기서:
- `T_root_input`: 루트의 위치와 방향
- `T_offset`: MJCF에서 정의된 부모-자식 간 위치 오프셋
- `T_local_fixed`: MJCF에서 정의된 고정 로컬 회전 (관절 축 정의)
- `T_joint_input`: 포즈 데이터에서 오는 관절 회전

### 8.2 쿼터니언 연산의 중요성 (w_last 이슈 포함)

쿼터니언 `q = (x, y, z, w)` 또는 `q = (w, x, y, z)`는 3D 회전을 표현하는 4차원 벡터다. 오일러각 대비 장점:
- **짐벌 락(Gimbal Lock) 없음**: 모든 방향의 회전을 부드럽게 표현
- **보간 가능**: SLERP으로 두 회전 사이를 구면 상에서 부드럽게 보간
- **곱셈으로 합성**: 두 회전을 순서대로 적용하려면 쿼터니언 곱셈

**w_last 이슈 -- 이 프로젝트에서 가장 주의해야 할 점:**

코드베이스에서 두 가지 쿼터니언 포맷이 혼용된다:

| 포맷 | 순서 | 사용처 |
|------|------|--------|
| `xyzw` (w_last=True) | `[x, y, z, w]` | IsaacGym, `torch_utils.py`의 대부분 함수, `grs` 텐서 |
| `wxyz` (w_last=False) | `[w, x, y, z]` | PyTorch3D, `rotation_conversions.py`, `axis_angle_to_quaternion()` |

**변환이 필요한 지점:**

```python
# Humanoid_Batch.fk_batch()에서:
pose_quat = axis_angle_to_quaternion(pose)       # 출력: wxyz
pose_mat = quaternion_to_matrix(pose_quat)        # 입력: wxyz 기대
wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))  # wxyz --> xyzw 변환
return_dict.local_rotation = wxyz_to_xyzw(pose_quat)       # wxyz --> xyzw 변환
```

`torch_utils.py`의 함수들은 `w_last: bool` 매개변수로 포맷을 명시한다:

```python
def quat_mul(a, b, w_last: bool):
    if w_last:   # xyzw
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    else:        # wxyz
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
```

**주의:** `skeleton.py`의 `SkeletonState`는 내부적으로 `w_last=True` (xyzw)를 사용하고, `SkeletonMotion`도 마찬가지다. 하지만 `rotation_conversions.py`와 `Humanoid_Batch`의 내부 `_local_rotation`은 `wxyz`를 사용한다. FK 결과를 반환할 때 반드시 `wxyz_to_xyzw()` 변환이 필요하다.

### 8.3 모션 리타겟팅이란

**모션 리타겟팅(Motion Retargeting)** 은 한 스켈레톤(예: 사람 모션 캡처)에서 다른 스켈레톤(예: G1 로봇)으로 모션을 옮기는 과정이다.

`SkeletonState.retarget_to()` 메서드가 5단계 리타겟팅을 구현한다:

```
Step 1: 소스에서 매핑되지 않는 관절 제거 (drop_nodes_by_names)
Step 2: 소스를 타겟 방향에 맞게 회전 (rotation_to_target_skeleton)
Step 3: 루트 이동을 타겟 스케일에 맞게 정규화 (scale_to_target_skeleton)
Step 4: 소스의 T-pose 대비 상대 회전을 타겟의 T-pose에 적용
Step 5: 글로벌 회전과 루트 이동을 결합하여 최종 결과 생성
```

핵심은 **Step 4의 상대 회전 전이**:

```python
# 소스의 현재 포즈와 T-pose의 차이 (상대 회전)
global_rotation_diff = quat_mul_norm(
    source_state.global_rotation,
    quat_inverse(source_tpose.global_rotation, w_last=True),
    w_last=True
)
# 이 상대 회전을 타겟의 T-pose에 적용
new_global_rotation = quat_mul_norm(
    global_rotation_diff,
    target_tpose_global_rotation,
    w_last=True
)
```

이렇게 하면 소스와 타겟의 체형(비율, 관절 위치)이 달라도 동일한 "의도"의 동작이 전달된다. BFM-Zero에서는 이미 리타겟팅된 `.pkl` 파일을 사용하므로 학습 시에는 이 코드가 직접 호출되지 않지만, 새로운 모션 데이터를 추가할 때 필요한 파이프라인이다.

---

## 9. 전체 데이터 파이프라인 요약

```
[LAFAN1 모션 캡처] --리타겟팅--> [lafan_29dof.pkl]
                                     |
                                     v
[MotionLibBase.__init__()]
    |-- SkeletonTree.from_mjcf(g1.xml)     # 로봇 구조 파싱
    |-- load_data(motion_file)              # .pkl 로딩 (joblib)
    |-- setup_constants()                   # 샘플링 확률 초기화
    |
[MotionLibBase.load_motions()]
    |-- Humanoid_Batch.fk_batch()           # axis-angle -> FK -> 글로벌 상태
    |   |-- axis_angle_to_quaternion()      # (T, J, 3) -> (T, J, 4) wxyz
    |   |-- quaternion_to_matrix()          # (T, J, 4) -> (T, J, 3, 3)
    |   |-- forward_kinematics_batch()      # 재귀 FK
    |   |-- _compute_velocity()             # np.gradient + gaussian_filter
    |   |-- _compute_angular_velocity()     # quat_diff + angle_axis
    |   |-- wxyz_to_xyzw()                  # wxyz -> xyzw 변환
    |
    |-- 모든 모션을 연결 -> self.gts, self.grs, self.lrs, ...
    |-- self.length_starts 계산             # 각 모션의 시작 프레임 인덱스
    |
[학습 루프]
    |-- sample_motions(n)                   # 샘플링 확률에 따라 모션 선택
    |-- sample_time(motion_ids)             # 랜덤 시점 선택
    |-- get_motion_state(motion_ids, times) # 보간된 모션 상태 반환
    |   |-- _calc_frame_blend()             # 프레임 인덱스 + 블렌드 비율
    |   |-- slerp()                         # 쿼터니언 보간
    |   |-- 선형 보간                        # 위치/속도
    |
    |-- 환경에서 참조 모션과 비교 -> 보상 계산 -> 에이전트 학습
```
