"""
train.py 상세 주석 버전

이 파일은 humanoidverse/train.py의 학습 코드를 상세히 주석 처리한 버전입니다.
실제 실행용이 아닌 학습/분석 용도입니다.

원본: humanoidverse/train.py
"""

# =============================================================================
# 1. 임포트 및 환경 설정
# =============================================================================

import os

# 평가 모듈 임포트
# - HumanoidVerseIsaacTrackingEvaluation: 모션 트래킹 EMD 평가
# - HumanoidVerseIsaacTrackingEvaluationConfig: 평가 설정
from humanoidverse.agents.evaluations.humanoidverse_isaac import (
    HumanoidVerseIsaacTrackingEvaluation,
    HumanoidVerseIsaacTrackingEvaluationConfig,
)

# 환경 및 데이터 로딩
# - load_expert_trajectories_from_motion_lib: 모션 라이브러리에서 전문가 궤적 로드
# - HumanoidVerseIsaacConfig: Isaac Sim/MuJoCo 환경 설정
from humanoidverse.agents.envs.humanoidverse_isaac import (
    load_expert_trajectories_from_motion_lib,
    HumanoidVerseIsaacConfig,
)

# OpenMP 스레드 수 제한 (병렬 환경에서 CPU 과부하 방지)
os.environ["OMP_NUM_THREADS"] = "1"

import torch

# PyTorch 행렬 연산 정밀도 설정
# "high": TensorFloat32 사용, 속도와 정밀도 균형
torch.set_float32_matmul_precision("high")

import json
import time
import typing as tp
import warnings
from pathlib import Path
from typing import Dict, List
from torch.utils._pytree import tree_map  # 딕셔너리/리스트에 함수 일괄 적용

import exca as xk          # 클러스터 실행 인프라
import gymnasium           # 강화학습 환경 인터페이스
import numpy as np
import pydantic            # 설정 검증 및 직렬화
import torch
import tyro                # CLI 파싱
import wandb               # 실험 로깅 (Weights & Biases)
from packaging.version import Version
from torch.utils._pytree import tree_map
from tqdm import tqdm      # 진행률 표시

# 에이전트 관련 임포트
from humanoidverse.agents.base import BaseConfig
from humanoidverse.agents.buffers.load_data import load_expert_trajectories
from humanoidverse.agents.buffers.trajectory import TrajectoryDictBufferMultiDim
from humanoidverse.agents.buffers.transition import DictBuffer, dtype_numpytotorch_lower_precision
from humanoidverse.agents.fb_cpr.agent import FBcprAgentConfig
from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentConfig
from humanoidverse.agents.misc.loggers import CSVLogger
from humanoidverse.agents.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

# =============================================================================
# 2. 상수 정의
# =============================================================================

TRAIN_LOG_FILENAME = "train_log.txt"           # 학습 로그 파일
REWARD_EVAL_LOG_FILENAME = "reward_eval_log.csv"    # 보상 평가 로그
TRACKING_EVAL_LOG_FILENAME = "tracking_eval_log.csv" # 트래킹 평가 로그
CHECKPOINT_DIR_NAME = "checkpoint"             # 체크포인트 디렉토리명

# 환경 설정 → 전문가 데이터 관측값 매퍼
# HumanoidVerseIsaacConfig는 매퍼 불필요 (None)
_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER = {
    HumanoidVerseIsaacConfig: None,
}

# =============================================================================
# 3. 타입 정의
# =============================================================================

# 평가 타입 정의 (discriminator로 구분)
Evaluation = tp.Annotated[
    tp.Union[
        HumanoidVerseIsaacTrackingEvaluationConfig,
    ],
    pydantic.Field(discriminator="name"),
]

# 에이전트 타입: FBcpr 또는 FBcprAux
Agent = FBcprAgentConfig | FBcprAuxAgentConfig


# =============================================================================
# 4. TrainConfig - 학습 설정 클래스
# =============================================================================

class TrainConfig(BaseConfig):
    """
    학습 전체 설정을 담는 Pydantic 모델

    주요 섹션:
    - agent: 에이전트 설정 (FBcpr/FBcprAux)
    - env: 환경 설정 (HumanoidVerseIsaac)
    - 학습 하이퍼파라미터
    - 버퍼 설정
    - 평가 설정
    - WandB 로깅 설정
    """

    # -------------------------------------------------------------------------
    # 에이전트 및 환경 설정
    # -------------------------------------------------------------------------

    # 에이전트 설정 (FBcprAgent 또는 FBcprAuxAgent)
    # discriminator="name"으로 JSON에서 어떤 에이전트인지 구분
    agent: Agent = pydantic.Field(discriminator="name")

    # 모션 데이터 경로 (외부 파일에서 로드 시)
    motions: str | None = None
    motions_root: str | None = None

    # 환경 설정
    env: HumanoidVerseIsaacConfig = pydantic.Field(discriminator="name")

    # 작업 디렉토리 (체크포인트, 로그 저장)
    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("g1mujoco_train"))

    # -------------------------------------------------------------------------
    # 학습 하이퍼파라미터
    # -------------------------------------------------------------------------

    seed: int = 0                      # 랜덤 시드
    online_parallel_envs: int = 50     # 병렬 환경 수 (실제: 1024)

    # 로깅 주기 (환경 스텝 단위)
    log_every_updates: int = 100_000

    # 총 학습 스텝 수
    num_env_steps: int = 30_000_000    # 실제: 384M

    # 에이전트 업데이트 주기 (환경 스텝 단위)
    # 이 값 만큼 환경에서 데이터를 수집한 후 업데이트
    update_agent_every: int = 500      # 실제: 1024

    # 랜덤 액션으로 초기 데이터 수집 스텝
    # 학습 시작 전 버퍼를 채우기 위함
    num_seed_steps: int = 50_000       # 실제: 10240

    # 한 번 업데이트 시 gradient step 횟수
    num_agent_updates: int = 50        # 실제: 16

    # 체크포인트 저장 주기 (환경 스텝 단위)
    checkpoint_every_steps: int = 5_000_000  # 실제: 9.6M
    checkpoint_buffer: bool = True     # 버퍼도 함께 저장할지

    # -------------------------------------------------------------------------
    # 우선순위 샘플링 (Prioritized Sampling)
    # -------------------------------------------------------------------------

    # EMD 기반 우선순위 샘플링 활성화
    prioritization: bool = False       # 실제: True
    prioritization_min_val: float = 0.5   # 최소 우선순위
    prioritization_max_val: float = 5     # 최대 우선순위 (실제: 2.0)
    prioritization_scale: float = 2       # 스케일 팩터
    prioritization_mode: str = "bin"      # "bin", "exp", "lin" (실제: "exp")

    padding_beginning: int = 0
    padding_end: int = 0

    # -------------------------------------------------------------------------
    # 리플레이 버퍼 설정
    # -------------------------------------------------------------------------

    # 궤적 버퍼 사용 여부 (에피소드 단위 저장)
    use_trajectory_buffer: bool = False  # 실제: True
    buffer_size: int = 5_000_000         # 버퍼 용량

    # -------------------------------------------------------------------------
    # WandB 로깅 설정
    # -------------------------------------------------------------------------

    use_wandb: bool = False
    wandb_ename: str | None = None     # entity (사용자/팀)
    wandb_gname: str | None = None     # group
    wandb_pname: str | None = None     # project

    # -------------------------------------------------------------------------
    # 기타 설정
    # -------------------------------------------------------------------------

    # Isaac 전문가 데이터 로드 여부
    load_isaac_expert_data: bool = True

    # 버퍼 저장 장치 (cpu/cuda)
    buffer_device: str = "cpu"         # 실제: "cuda"

    # tqdm 비활성화 (콘솔 스팸 방지)
    disable_tqdm: bool = True

    # -------------------------------------------------------------------------
    # 평가 설정
    # -------------------------------------------------------------------------

    # 평가 모듈 리스트
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])

    # 평가 주기 (환경 스텝 단위)
    eval_every_steps: int = 1_000_000  # 실제: 9.6M

    # 태그 (메타데이터)
    tags: dict = pydantic.Field(default_factory=lambda: {})

    # exca 인프라 설정 (클러스터 실행용)
    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        """
        설정 검증 (Pydantic post-init hook)

        검증 항목:
        1. Isaac 전문가 데이터는 HumanoidVerseIsaacConfig에서만 사용 가능
        2. 우선순위 샘플링은 트래킹 평가 필요
        3. 평가 이름 중복 불가
        """
        # Isaac 전문가 데이터 검증
        if self.load_isaac_expert_data and not isinstance(self.env, HumanoidVerseIsaacConfig):
            raise ValueError("Loading expert isaac data is only supported for HumanoidVerseIsaacConfig")

        # 우선순위 샘플링 검증
        if self.prioritization:
            has_prioritization_eval = False
            for eval_type in self.evaluations:
                if isinstance(eval_type, (HumanoidVerseIsaacTrackingEvaluationConfig)):
                    has_prioritization_eval = True
                    break
            if not has_prioritization_eval:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        # 모션 데이터 검증
        if self.motions is None or self.motions_root is None:
            if self.prioritization:
                raise ValueError("Prioritization requires expert data to be provided (motions and motions_root)")
            elif self.agent == FBcprAgentConfig:
                raise ValueError("FBcprAgent requires expert data to be provided (motions and motions_root)")

        # 평가 이름 중복 검증
        if isinstance(self.evaluations, list):
            log_names = set()
            for eval_cfg in self.evaluations:
                if eval_cfg.name_in_logs in log_names:
                    raise ValueError(
                        f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                    )
                log_names.add(eval_cfg.name_in_logs)

    def build(self):
        """Workspace 인스턴스 생성"""
        return Workspace(self)


# =============================================================================
# 5. 체크포인트 로드 함수
# =============================================================================

def create_agent_or_load_checkpoint(
    work_dir: Path,
    cfg: TrainConfig,
    agent_build_kwargs: dict[str, tp.Any]
):
    """
    에이전트 생성 또는 체크포인트에서 로드

    Args:
        work_dir: 작업 디렉토리
        cfg: 학습 설정
        agent_build_kwargs: 에이전트 생성 인자 (obs_space, action_dim)

    Returns:
        agent: 에이전트 인스턴스
        cfg: 설정 (변경 없음)
        checkpoint_time: 체크포인트 시점 (없으면 0)
    """
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0

    if checkpoint_dir.exists():
        # 체크포인트에서 학습 상태 로드
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        print(f"Loading the agent at time {checkpoint_time}")
        # 에이전트 로드 (모델 가중치 + 옵티마이저 상태)
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        # 새 에이전트 생성
        agent = cfg.agent.build(**agent_build_kwargs)

    return agent, cfg, checkpoint_time


# =============================================================================
# 6. WandB 초기화
# =============================================================================

def init_wandb(cfg: TrainConfig):
    """WandB 실험 로깅 초기화"""
    exp_name = "BFM-Zero"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()  # Pydantic → dict
    wandb.init(
        entity=cfg.wandb_ename,
        project=cfg.wandb_pname,
        group=cfg.wandb_gname,
        name=wandb_name,
        config=wandb_config,
        dir="./_wandb"
    )


# =============================================================================
# 7. Workspace - 메인 학습 클래스
# =============================================================================

class Workspace:
    """
    학습 작업 공간

    역할:
    - 환경 및 에이전트 초기화
    - 학습 루프 실행
    - 평가 및 체크포인트 관리
    """

    def __init__(self, cfg: TrainConfig) -> None:
        """
        Workspace 초기화

        주요 단계:
        1. 환경 생성
        2. 관측/행동 공간 설정
        3. 에이전트 생성/로드
        4. 평가 모듈 초기화
        5. 로거 초기화
        """
        self.cfg = cfg

        # ---------------------------------------------------------------------
        # 환경 생성
        # ---------------------------------------------------------------------

        # Isaac 환경은 재생성이 불가능하므로 여기서 한 번만 생성
        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            from omegaconf import OmegaConf

            # 병렬 환경 생성 (online_parallel_envs개)
            self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
            self.obs_space = self.train_env.single_observation_space
            self.action_space = self.train_env.single_action_space
        else:
            # 다른 환경은 나중에 생성
            sample_env, _ = cfg.env.build(num_envs=1)
            self.obs_space = sample_env.observation_space
            self.action_space = sample_env.action_space

        # 관측 공간 검증
        assert "time" in self.obs_space.keys(), \
            "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, \
            "Only 1D action space is supported (first dim should be vector env)"

        # 에이전트에는 time 정보 전달하지 않음
        del self.obs_space.spaces["time"]

        # 행동 차원 (G1 로봇: 29)
        self.action_dim = self.action_space.shape[0]

        # ---------------------------------------------------------------------
        # 작업 디렉토리 설정
        # ---------------------------------------------------------------------

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        # Hydra 설정 저장
        if isinstance(cfg.env, HumanoidVerseIsaacConfig):
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["unresolved_conf"], file)

        # 학습 로거
        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        # ---------------------------------------------------------------------
        # 랜덤 시드 설정
        # ---------------------------------------------------------------------

        set_seed_everywhere(self.cfg.seed)

        # ---------------------------------------------------------------------
        # 에이전트 생성/로드
        # ---------------------------------------------------------------------

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir,
            self.cfg,
            agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        # 학습 모드 활성화
        self.agent._model.train()

        # ---------------------------------------------------------------------
        # 평가 모듈 초기화
        # ---------------------------------------------------------------------

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        else:
            self.evaluations = {eval_cfg: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0

        # 평가 로거
        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}

        # ---------------------------------------------------------------------
        # WandB 초기화
        # ---------------------------------------------------------------------

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        # 설정 저장
        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

        # ---------------------------------------------------------------------
        # 우선순위 샘플링 설정
        # ---------------------------------------------------------------------

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, HumanoidVerseIsaacTrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

        self.training_with_expert_data = True
        self.manager = None

    # =========================================================================
    # 학습 메인 메서드
    # =========================================================================

    def train(self):
        """학습 시작"""
        self.start_time = time.time()
        self.train_online()

    def train_online(self) -> None:
        """
        온라인 학습 메인 루프

        학습 흐름:
        1. 전문가 데이터 로드
        2. 리플레이 버퍼 초기화
        3. 메인 루프:
           - 환경에서 데이터 수집
           - 버퍼에 저장
           - 에이전트 업데이트
           - 평가 및 체크포인트
        """

        # ---------------------------------------------------------------------
        # 1. 전문가 데이터 로드
        # ---------------------------------------------------------------------

        if self.training_with_expert_data:
            if self.cfg.load_isaac_expert_data:
                # 모션 라이브러리에서 직접 로드 (Isaac 환경)
                expert_buffer = load_expert_trajectories_from_motion_lib(
                    self.train_env._env,
                    self.cfg.agent,
                    device=self.cfg.buffer_device
                )
            else:
                # 외부 파일에서 로드
                print("Loading expert trajectories")
                expert_buffer = load_expert_trajectories(
                    self.cfg.motions,
                    self.cfg.motions_root,
                    seq_length=self.agent.cfg.model.seq_length,
                    device=self.cfg.buffer_device,
                    obs_dict_mapper=_ENC_CONFIG_TO_EXPERT_DATA_OBS_MAPPER[self.cfg.env.__class__],
                )

        print("Creating the training environment")

        # ---------------------------------------------------------------------
        # 2. 환경 설정
        # ---------------------------------------------------------------------

        if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            # 이미 생성된 환경 사용
            train_env = self.train_env
            train_env_info = self.train_env_info
        else:
            # 새 환경 생성
            train_env, train_env_info = self.cfg.env.build(num_envs=self.cfg.online_parallel_envs)

        # ---------------------------------------------------------------------
        # 3. 리플레이 버퍼 초기화
        # ---------------------------------------------------------------------

        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME

        # 체크포인트에서 버퍼 로드
        if (checkpoint_dir / "buffers/train").exists():
            print("Loading checkpointed buffer")
            if self.cfg.use_trajectory_buffer:
                replay_buffer["train"] = TrajectoryDictBufferMultiDim.load(
                    checkpoint_dir / "buffers/train",
                    device=self.cfg.buffer_device
                )
            else:
                replay_buffer["train"] = DictBuffer.load(
                    checkpoint_dir / "buffers/train",
                    device=self.cfg.buffer_device
                )
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            # 새 버퍼 생성
            if self.cfg.use_trajectory_buffer:
                # 궤적 버퍼: 에피소드 단위로 저장
                output_key_t = ["observation", "action", "z", "terminated", "truncated", "step_count", "reward"]

                # FBcprAux는 보조 보상도 저장
                if isinstance(self.cfg.agent, (FBcprAuxAgentConfig)):
                    output_key_t.append("aux_rewards")

                replay_buffer["train"] = TrajectoryDictBufferMultiDim(
                    capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,
                    device=self.cfg.buffer_device,
                    n_dim=2,
                    end_key="truncated",
                    output_key_t=output_key_t,
                    output_key_tp1=["observation", "terminated"],
                )
            else:
                # 전이 버퍼: (s, a, r, s', done) 단위로 저장
                replay_buffer["train"] = DictBuffer(
                    capacity=self.cfg.buffer_size,
                    device=self.cfg.buffer_device
                )

        # 전문가 버퍼 추가
        if self.training_with_expert_data:
            replay_buffer["expert_slicer"] = expert_buffer

        # ---------------------------------------------------------------------
        # 4. 메인 학습 루프
        # ---------------------------------------------------------------------

        print("Starting training")
        progb = tqdm(total=self.cfg.num_env_steps, disable=self.cfg.disable_tqdm)

        # 환경 리셋
        td, info = train_env.reset()

        # 종료 플래그 초기화
        # - terminated: 에피소드 자연 종료 (예: 넘어짐)
        # - truncated: 시간 제한으로 종료
        # - done: terminated | truncated
        terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
        done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)

        total_metrics, context = None, None
        start_time = time.time()
        fps_start_time = time.time()

        # 주기적 작업 체커
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        # HumanoidVerse 평가 사용 여부 확인
        eval_instances = []
        for evaluation_name in self.evaluations.keys():
            evaluation = self.evaluations[evaluation_name]
            eval_instances.append(isinstance(evaluation, HumanoidVerseIsaacTrackingEvaluation))
        uses_humanoidverse_eval = True if any(eval_instances) else False

        # ---------------------------------------------------------------------
        # 메인 루프: t = 0 → num_env_steps
        # online_parallel_envs 단위로 증가 (1024개 환경 동시 실행)
        # ---------------------------------------------------------------------

        for t in range(
            self._checkpoint_time,
            self.cfg.num_env_steps + self.cfg.online_parallel_envs,
            self.cfg.online_parallel_envs
        ):
            # -----------------------------------------------------------------
            # 4.1 체크포인트 저장
            # -----------------------------------------------------------------

            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            # -----------------------------------------------------------------
            # 4.2 평가 실행
            # -----------------------------------------------------------------

            if (self.evaluate and eval_time_checker.check(t)) or (self.evaluate and t == self._checkpoint_time):
                eval_metrics = self.eval(t, replay_buffer=replay_buffer)
                eval_time_checker.update_last_step(t)

                # HumanoidVerse 평가 후 환경 리셋 필요
                if uses_humanoidverse_eval:
                    td, info = train_env.reset()
                    terminated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    truncated = np.zeros(self.cfg.online_parallel_envs, dtype=bool)
                    done = np.zeros(self.cfg.online_parallel_envs, dtype=bool)

                # 우선순위 샘플링 업데이트 (EMD 기반)
                if self.cfg.prioritization:
                    assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert_slicer"].motion_ids)

                    # 각 모션의 EMD를 우선순위로 변환
                    index_in_buffer, name_in_buffer = {}, {}
                    for i, motion_id in enumerate(replay_buffer["expert_slicer"].motion_ids):
                        index_in_buffer[motion_id] = i
                        if hasattr(replay_buffer["expert_slicer"], "file_names"):
                            name_in_buffer[motion_id] = replay_buffer["expert_slicer"].file_names[i]

                    motions_id, priorities, idxs = [], [], []
                    for _, metr in eval_metrics[self.priorization_eval_name].items():
                        motions_id.append(metr["motion_id"])
                        priorities.append(metr["emd"])  # EMD: Earth Mover's Distance
                        idxs.append(index_in_buffer[metr["motion_id"]])

                    # 우선순위 클램핑 및 스케일링
                    priorities = (
                        torch.clamp(
                            torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                            min=self.cfg.prioritization_min_val,
                            max=self.cfg.prioritization_max_val,
                        )
                        * self.cfg.prioritization_scale
                    )

                    # 우선순위 모드 적용
                    if self.cfg.prioritization_mode == "lin":
                        pass  # 선형
                    elif self.cfg.prioritization_mode == "exp":
                        priorities = 2**priorities  # 지수
                    elif self.cfg.prioritization_mode == "bin":
                        # 빈 기반: 같은 빈 내에서 균등 분배
                        bins = torch.floor(priorities)
                        for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                            mask = bins == i
                            n = mask.sum().item()
                            if n > 0:
                                priorities[mask] = 1 / n
                    else:
                        raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")

                    # 모션 라이브러리 가중치 업데이트
                    train_env._env._motion_lib.update_sampling_weight_by_id(
                        priorities=list(priorities),
                        motions_id=idxs,
                        file_name=name_in_buffer
                    )

                    # 전문가 버퍼 우선순위 업데이트
                    replay_buffer["expert_slicer"].update_priorities(
                        priorities=priorities.to(self.cfg.buffer_device),
                        idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
                    )

            # -----------------------------------------------------------------
            # 4.3 액션 선택 및 환경 스텝
            # -----------------------------------------------------------------

            with torch.no_grad():
                # 관측값을 텐서로 변환
                obs = tree_map(
                    lambda x: torch.tensor(x, dtype=dtype_numpytotorch_lower_precision(x.dtype), device=self.agent.device),
                    td
                )
                # 시간 정보 분리 (에이전트에 전달하지 않음)
                step_count = obs.pop("time")

                # 히스토리 컨텍스트 처리 (시퀀스 모델용)
                history_context = None
                if "history" in obs:
                    if len(obs["history"]["action"]) == 0:
                        # 초기 컨텍스트
                        history_context = self.agent._model._context_encoder.get_initial_context(
                            self.cfg.online_parallel_envs
                        )
                    else:
                        # 히스토리 인코딩
                        history_context = self.agent.history_inference(
                            obs=obs["history"]["observation"],
                            action=obs["history"]["action"]
                        )[:, -1].clone()

                # Z 컨텍스트 업데이트
                # - update_z_every_step마다 새 z 샘플링
                # - use_mix_rollout: z_buffer에서 학습된 z 사용
                # - rollout_expert_trajectories: 일부 환경에서 전문가 z 사용
                context = self.agent.maybe_update_rollout_context(
                    z=context,
                    step_count=step_count,
                    replay_buffer=replay_buffer
                )

                # 액션 선택
                if t < self.cfg.num_seed_steps:
                    # 시드 스텝: 랜덤 액션으로 버퍼 채우기
                    action = train_env.action_space.sample().astype(np.float32)
                else:
                    # 정책 액션
                    if history_context is not None:
                        action = self.agent.act(obs=obs, z=context, context=history_context, mean=False)
                    else:
                        action = self.agent.act(obs=obs, z=context, mean=False)

                    # CPU 환경용 변환
                    if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                        action = action.cpu().detach().numpy()

            # 환경 스텝 실행
            new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)

            # 다음 스텝에 평가가 있으면 truncated 설정 (환경 공유 문제)
            next_t = t + self.cfg.online_parallel_envs
            if (self.evaluate and eval_time_checker.check(next_t)) or (self.evaluate and next_t == self._checkpoint_time):
                if isinstance(self.cfg.env, HumanoidVerseIsaacConfig) and uses_humanoidverse_eval:
                    new_truncated = np.ones_like(new_truncated, dtype=bool)
                    truncated = np.ones_like(new_truncated, dtype=bool)

            # -----------------------------------------------------------------
            # 4.4 버퍼에 데이터 저장
            # -----------------------------------------------------------------

            if Version(gymnasium.__version__) >= Version("1.0"):
                if self.cfg.use_trajectory_buffer:
                    # 궤적 버퍼: 시간축 추가 [None, ...]
                    data = {
                        "observation": tree_map(lambda x: x[None, ...], obs),
                        "action": action[None, ...],
                        "terminated": terminated[None, ..., None],
                        "truncated": truncated[None, ..., None],
                        "step_count": step_count[None, ..., None],
                        "reward": new_reward[None, ..., None],
                    }
                    data["observation"].pop("history", None)

                    if context is not None:
                        data["z"] = context[None, ...]
                    if history_context is not None:
                        data["history_context"] = history_context[None, ...]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][None, ...]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][None, ...]
                    if "aux_rewards" in new_info:
                        # 보조 보상 저장 (FBcprAux용)
                        data["aux_rewards"] = {
                            k: v[None, ..., None]
                            for k, v in new_info["aux_rewards"].items()
                            if not k.startswith("_")
                        }
                else:
                    # 전이 버퍼: 리셋되지 않은 환경만 저장
                    indexes = ~done

                    real_next_obs = tree_map(
                        lambda x: x.astype(np.float32 if x.dtype == np.float64 else x.dtype)[indexes],
                        new_td
                    )
                    _ = real_next_obs.pop("time")
                    _ = real_next_obs.pop("history", None)

                    data = {
                        "observation": tree_map(lambda x: x[indexes], obs),
                        "action": action[indexes],
                        "step_count": step_count[indexes],
                        "reward": new_reward[indexes].reshape(-1, 1),
                        "next": {
                            "observation": real_next_obs,
                            "terminated": new_terminated[indexes].reshape(-1, 1),
                            "truncated": new_truncated[indexes].reshape(-1, 1),
                        },
                    }
                    data["observation"].pop("history", None)

                    if context is not None:
                        data["z"] = context[indexes]
                    if history_context is not None:
                        data["history_context"] = history_context[indexes]
                    if "qpos" in info:
                        data["qpos"] = info["qpos"][indexes]
                        data["next"]["qpos"] = new_info["qpos"][indexes]
                    if "qvel" in info:
                        data["qvel"] = info["qvel"][indexes]
                        data["next"]["qvel"] = new_info["qvel"][indexes]
                    if "aux_rewards" in new_info:
                        data["aux_rewards"] = {
                            k: v[indexes].reshape(-1, 1)
                            for k, v in new_info["aux_rewards"].items()
                            if not k.startswith("_")
                        }
            else:
                raise NotImplementedError("still some work to do for gymnasium < 1.0")

            # 버퍼에 추가
            replay_buffer["train"].extend(data)

            # -----------------------------------------------------------------
            # 4.5 에이전트 업데이트
            # -----------------------------------------------------------------

            if len(replay_buffer["train"]) > 0 and t > self.cfg.num_seed_steps and update_agent_time_checker.check(t):
                update_agent_time_checker.update_last_step(t)

                # num_agent_updates회 gradient step
                for _ in range(self.cfg.num_agent_updates):
                    # 에이전트 업데이트 (핵심 학습 로직)
                    # - 전문가/온라인 버퍼에서 샘플링
                    # - 판별자, FB, Critic, Actor 업데이트
                    metrics = self.agent.update(replay_buffer, t)

                    # 메트릭 누적
                    if total_metrics is None:
                        num_metrics_updates = 1
                        total_metrics = {k: metrics[k].float().clone() for k in metrics.keys()}
                    else:
                        num_metrics_updates += 1
                        total_metrics = {k: total_metrics[k] + metrics[k].float() for k in metrics.keys()}

            # -----------------------------------------------------------------
            # 4.6 로깅
            # -----------------------------------------------------------------

            if log_time_checker.check(t) and total_metrics is not None:
                log_time_checker.update_last_step(t)

                # 평균 메트릭 계산
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / num_metrics_updates
                    m_dict[k] = np.round(tmp.mean().item(), 6)

                m_dict["duration [minutes]"] = (time.time() - start_time) / 60
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)

                # WandB 로깅
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )

                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()

                # CSV 로깅
                m_dict["timestep"] = t
                self.train_logger.log(m_dict)

            # -----------------------------------------------------------------
            # 4.7 상태 업데이트
            # -----------------------------------------------------------------

            progb.update(self.cfg.online_parallel_envs)
            td = new_td
            terminated = new_terminated
            truncated = new_truncated
            done = np.logical_or(new_terminated.ravel(), new_truncated.ravel())
            info = new_info

        # 환경 종료
        train_env.close()

    # =========================================================================
    # 평가 메서드
    # =========================================================================

    def eval(self, t, replay_buffer):
        """
        평가 실행

        Args:
            t: 현재 타임스텝
            replay_buffer: 리플레이 버퍼 (우선순위 업데이트용)

        Returns:
            evaluation_results: 평가 결과 딕셔너리
        """
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        for evaluation_name in self.evaluations.keys():
            logger = self.eval_loggers[evaluation_name]
            evaluation = self.evaluations[evaluation_name]

            # 평가 시 모델 평가 모드로 전환
            if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                self.agent._model.to("cpu")
            self.agent._model.train(False)

            # 평가 실행
            if isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                    env=self.train_env  # 환경 공유
                )
            else:
                evaluation_metrics, wandb_dict = evaluation.run(
                    timestep=t,
                    agent_or_model=self.agent,
                    replay_buffer=replay_buffer,
                    logger=logger,
                )

            # WandB 로깅
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # 학습 모드로 복귀
        if not isinstance(self.cfg.env, HumanoidVerseIsaacConfig):
            self.agent._model.to(self.cfg.agent.model.device)
        self.agent._model.train()

        return evaluation_results

    # =========================================================================
    # 체크포인트 저장
    # =========================================================================

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        """
        체크포인트 저장

        저장 내용:
        - 에이전트 (모델 가중치 + 옵티마이저)
        - 리플레이 버퍼 (옵션)
        - 학습 상태 (현재 타임스텝)
        """
        print(f"Checkpointing at time {time}")

        # 에이전트 저장
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))

        # 버퍼 저장
        if self.cfg.checkpoint_buffer:
            replay_buffer["train"].save(self.work_dir / CHECKPOINT_DIR_NAME / "buffers" / "train")

        # 학습 상태 저장
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)


# =============================================================================
# 8. train_bfm_zero() - 실제 학습 설정
# =============================================================================

def train_bfm_zero():
    """
    BFM-Zero 학습 설정 및 실행

    이 함수에서 모든 하이퍼파라미터가 하드코딩되어 있습니다.
    실제 학습에 사용되는 설정입니다.
    """
    # 모델 아키텍처 임포트
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgentTrainConfig
    from humanoidverse.agents.nn_models import (
        ForwardArchiConfig,
        BackwardArchiConfig,
        ActorArchiConfig,
        DiscriminatorArchiConfig,
        RewardNormalizerConfig
    )
    from humanoidverse.agents.normalizers import ObsNormalizerConfig, BatchNormNormalizerConfig
    from humanoidverse.agents.nn_filters import DictInputFilterConfig

    # -------------------------------------------------------------------------
    # 전체 학습 설정
    # -------------------------------------------------------------------------

    cfg = TrainConfig(
        name='TrainConfig',

        # ---------------------------------------------------------------------
        # 에이전트 설정 (FBcprAux)
        # ---------------------------------------------------------------------
        agent=FBcprAuxAgentConfig(
            name='FBcprAuxAgent',

            # 모델 설정
            model=FBcprAuxModelConfig(
                name='FBcprAuxModel',
                device='cuda',                    # GPU 사용

                # 아키텍처 설정
                archi=FBcprAuxModelArchiConfig(
                    name='FBcprAuxModelArchiConfig',
                    z_dim=256,                    # 잠재 벡터 차원
                    norm_z=True,                  # z 정규화

                    # Forward Map: (obs, z, action) → next_obs 예측
                    f=ForwardArchiConfig(
                        name='ForwardArchi',
                        hidden_dim=2048,
                        model='residual',         # Residual MLP
                        hidden_layers=6,
                        embedding_layers=2,
                        num_parallel=2,           # 앙상블 (불확실성 추정)
                        ensemble_mode='batch',
                        input_filter=DictInputFilterConfig(
                            name='DictInputFilterConfig',
                            key=['state', 'privileged_state', 'last_action', 'history_actor']
                        )
                    ),

                    # Backward Map: obs → z (목표/상태 인코딩)
                    b=BackwardArchiConfig(
                        name='BackwardArchi',
                        hidden_dim=256,
                        hidden_layers=1,
                        norm=True,
                        input_filter=DictInputFilterConfig(
                            name='DictInputFilterConfig',
                            key=['state', 'privileged_state']
                        )
                    ),

                    # Actor: (obs, z) → action (정책)
                    actor=ActorArchiConfig(
                        name='actor',
                        model='residual',
                        hidden_dim=2048,
                        hidden_layers=6,
                        embedding_layers=2,
                        input_filter=DictInputFilterConfig(
                            name='DictInputFilterConfig',
                            key=['state', 'last_action', 'history_actor']
                        )
                    ),

                    # Critic: 판별자 보상용 Q-function
                    critic=ForwardArchiConfig(
                        name='ForwardArchi',
                        hidden_dim=2048,
                        model='residual',
                        hidden_layers=6,
                        embedding_layers=2,
                        num_parallel=2,
                        ensemble_mode='batch',
                        input_filter=DictInputFilterConfig(
                            name='DictInputFilterConfig',
                            key=['state', 'privileged_state', 'last_action', 'history_actor']
                        )
                    ),

                    # Discriminator: 전문가 vs 정책 판별
                    discriminator=DiscriminatorArchiConfig(
                        name='DiscriminatorArchi',
                        hidden_dim=1024,
                        hidden_layers=3,
                        input_filter=DictInputFilterConfig(
                            name='DictInputFilterConfig',
                            key=['state', 'privileged_state']
                        )
                    ),

                    # Auxiliary Critic: 보조 보상용 Q-function
                    aux_critic=ForwardArchiConfig(
                        name='ForwardArchi',
                        hidden_dim=2048,
                        model='residual',
                        hidden_layers=6,
                        embedding_layers=2,
                        num_parallel=2,
                        ensemble_mode='batch',
                        input_filter=DictInputFilterConfig(
                            name='DictInputFilterConfig',
                            key=['state', 'privileged_state', 'last_action', 'history_actor']
                        )
                    )
                ),

                # 관측값 정규화 (BatchNorm)
                obs_normalizer=ObsNormalizerConfig(
                    name='ObsNormalizerConfig',
                    normalizers={
                        'state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'privileged_state': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'last_action': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01),
                        'history_actor': BatchNormNormalizerConfig(name='BatchNormNormalizerConfig', momentum=0.01)
                    },
                    allow_mismatching_keys=True
                ),

                inference_batch_size=500000,      # 추론 배치 크기
                seq_length=8,                     # 시퀀스 길이
                actor_std=0.05,                   # 정책 표준편차
                amp=False,                        # 혼합 정밀도 비활성화
                norm_aux_reward=RewardNormalizerConfig(
                    name='RewardNormalizer',
                    translate=False,
                    scale=True
                )
            ),

            # 학습 하이퍼파라미터
            train=FBcprAuxAgentTrainConfig(
                name='FBcprAuxAgentTrainConfig',

                # 학습률
                lr_f=0.0003,              # Forward Map
                lr_b=1e-05,               # Backward Map
                lr_actor=0.0003,          # Actor
                lr_discriminator=1e-05,   # Discriminator
                lr_critic=0.0003,         # Critic
                lr_aux_critic=0.0003,     # Auxiliary Critic

                weight_decay=0.0,
                clip_grad_norm=0.0,       # 그래디언트 클리핑 비활성화

                # Target 네트워크 업데이트 (soft update)
                fb_target_tau=0.01,       # FB target tau
                critic_target_tau=0.005,  # Critic target tau

                # 손실 가중치
                ortho_coef=100.0,         # 직교성 손실 계수
                reg_coeff=0.05,           # 판별자 보상 정규화 계수
                reg_coeff_aux=0.02,       # 보조 보상 정규화 계수
                scale_reg=True,

                # Z 분포 비율
                train_goal_ratio=0.2,     # 20% 목표 인코딩
                expert_asm_ratio=0.6,     # 60% 전문가 궤적 인코딩
                # 나머지 20%: 균등 분포

                relabel_ratio=0.8,        # 80% 리라벨링

                # 불확실성 페널티
                fb_pessimism_penalty=0.0,
                actor_pessimism_penalty=0.5,
                critic_pessimism_penalty=0.5,
                aux_critic_pessimism_penalty=0.5,

                stddev_clip=0.3,          # 액션 샘플링 클리핑
                q_loss_coef=0.0,

                batch_size=1024,          # 배치 크기
                discount=0.98,            # 할인율

                # 롤아웃 설정
                use_mix_rollout=True,     # 혼합 z 분포 사용
                update_z_every_step=100,  # z 업데이트 주기
                z_buffer_size=8192,

                # 전문가 롤아웃
                rollout_expert_trajectories=True,
                rollout_expert_trajectories_length=250,
                rollout_expert_trajectories_percentage=0.5,

                # 판별자 설정
                grad_penalty_discriminator=10.0,  # WGAN-GP
                weight_decay_discriminator=0.0,
            ),

            # 보조 보상 설정
            aux_rewards=[
                'penalty_torques',           # 토크 페널티
                'penalty_action_rate',       # 액션 변화율 페널티
                'limits_dof_pos',            # 관절 한계 페널티
                'limits_torque',             # 토크 한계 페널티
                'penalty_undesired_contact', # 원치않는 접촉 페널티
                'penalty_feet_ori',          # 발 방향 페널티
                'penalty_ankle_roll',        # 발목 롤 페널티
                'penalty_slippage',          # 미끄러짐 페널티
            ],
            aux_rewards_scaling={
                'penalty_action_rate': -0.1,
                'penalty_feet_ori': -0.4,
                'penalty_ankle_roll': -4.0,
                'limits_dof_pos': -10.0,
                'penalty_slippage': -2.0,
                'penalty_undesired_contact': -1.0,
                'penalty_torques': 0.0,      # 비활성화
                'limits_torque': 0.0,        # 비활성화
            },

            cudagraphs=False,
            compile=True,                 # torch.compile 사용
        ),

        motions='',
        motions_root='',

        # ---------------------------------------------------------------------
        # 환경 설정 (HumanoidVerse + Isaac)
        # ---------------------------------------------------------------------
        env=HumanoidVerseIsaacConfig(
            name='humanoidverse_isaac',
            device='cuda:0',

            # 모션 데이터 경로 (10초 클립)
            lafan_tail_path='humanoidverse/data/lafan_29dof_10s-clipped.pkl',

            enable_cameras=False,
            camera_render_save_dir='isaac_videos',
            max_episode_length_s=None,
            disable_obs_noise=False,
            disable_domain_randomization=False,

            # Hydra 설정 경로
            relative_config_path='exp/bfm_zero/bfm_zero',

            include_last_action=True,

            # Hydra 오버라이드
            hydra_overrides=[
                'robot=g1/g1_29dof_hard_waist',   # G1 로봇 설정
                'robot.control.action_scale=0.25',
                'robot.control.action_clip_value=5.0',
                'robot.control.normalize_action_to=5.0',
                'env.config.lie_down_init=True',  # 누운 상태에서 시작
                'env.config.lie_down_init_prob=0.3',  # 30% 확률
            ],

            context_length=None,
            include_dr_info=False,
            included_dr_obs_names=None,
            include_history_actor=True,
            include_history_noaction=False,
            make_config_g1env_compatible=False,
            root_height_obs=True,
        ),

        # ---------------------------------------------------------------------
        # 학습 파라미터
        # ---------------------------------------------------------------------
        work_dir='results/bfmzero-isaac',
        seed=4728,

        online_parallel_envs=1024,        # 1024개 병렬 환경
        num_env_steps=384_000_000,        # 3.84억 스텝

        log_every_updates=384_000,        # 38.4만 스텝마다 로깅
        update_agent_every=1024,          # 1024 스텝마다 업데이트
        num_seed_steps=10_240,            # 1만 시드 스텝
        num_agent_updates=16,             # 16번 gradient step

        checkpoint_every_steps=9_600_000, # 960만 스텝마다 체크포인트
        checkpoint_buffer=True,

        # 우선순위 샘플링
        prioritization=True,
        prioritization_min_val=0.5,
        prioritization_max_val=2.0,
        prioritization_scale=2.0,
        prioritization_mode='exp',        # 지수 스케일링

        # 버퍼 설정
        use_trajectory_buffer=True,
        buffer_size=5_120_000,

        # WandB
        use_wandb=False,
        wandb_ename='yitangl',
        wandb_gname='bfmzero-isaac',
        wandb_pname='bfmzero-isaac',

        load_isaac_expert_data=True,
        buffer_device='cuda',
        disable_tqdm=True,

        # ---------------------------------------------------------------------
        # 평가 설정
        # ---------------------------------------------------------------------
        evaluations=[
            HumanoidVerseIsaacTrackingEvaluationConfig(
                name='HumanoidVerseIsaacTrackingEvaluationConfig',
                generate_videos=False,
                videos_dir='videos',
                video_name_prefix='unknown_agent',
                name_in_logs='humanoidverse_tracking_eval',
                env=None,
                num_envs=1024,
                n_episodes_per_motion=1
            )
        ],
        eval_every_steps=9_600_000,       # 960만 스텝마다 평가

        tags={},
    )

    # 학습 실행
    workspace = cfg.build()
    workspace.train()


# =============================================================================
# 9. 메인 진입점
# =============================================================================

if __name__ == "__main__":
    # 학습 실행
    train_bfm_zero()

# 실행 명령:
# uv run --no-cache -m humanoidverse.meta_online_entry_point
