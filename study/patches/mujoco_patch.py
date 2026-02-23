"""
MuJoCo 시뮬레이터 macOS 패치

원본 코드의 버그를 런타임에 수정합니다:
1. quat_rotate 함수 호출 시 w_last=True 인자 누락 문제

사용법:
    from study.patches.mujoco_patch import apply_mujoco_patches
    apply_mujoco_patches()
"""

import torch


def apply_mujoco_patches():
    """MuJoCo 시뮬레이터 패치 적용"""

    # humanoidverse.simulator.mujoco.mujoco 모듈이 이미 로드된 후에 패치
    try:
        from humanoidverse.simulator.mujoco import mujoco as mujoco_module
        from humanoidverse.utils.torch_utils import quat_rotate

        MuJoCoClass = mujoco_module.MuJoCo

        # 원본 robot_root_states 프로퍼티 저장
        original_robot_root_states = MuJoCoClass.robot_root_states.fget
        original_all_root_states = MuJoCoClass.all_root_states.fget

        # 패치된 robot_root_states
        def patched_robot_root_states(self):
            base_quat = self.base_quat
            qvel_tensor = torch.tensor([self.data.qvel[0:6]], device=self.device, dtype=torch.float32)
            return torch.cat(
                [
                    torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32),
                    base_quat,
                    qvel_tensor[:, 0:3],
                    quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),  # 패치: w_last=True 추가
                ], dim=-1
            )

        # 패치된 all_root_states
        def patched_all_root_states(self):
            base_quat = self.base_quat
            qvel_tensor = torch.tensor([self.data.qvel[0:6]], device=self.device, dtype=torch.float32)
            return torch.cat(
                [
                    torch.tensor([self.data.qpos[0:3]], device=self.device, dtype=torch.float32),
                    base_quat,
                    qvel_tensor[:, 0:3],
                    quat_rotate(base_quat, qvel_tensor[:, 3:6], w_last=True),  # 패치: w_last=True 추가
                ], dim=-1
            )

        # 프로퍼티 교체
        MuJoCoClass.robot_root_states = property(patched_robot_root_states)
        MuJoCoClass.all_root_states = property(patched_all_root_states)

        print("[patch] MuJoCo quat_rotate w_last 패치 적용됨")
        return True

    except ImportError as e:
        print(f"[patch] MuJoCo 패치 실패: {e}")
        return False


def setup_mujoco_config():
    """
    MuJoCo Hydra config 설정

    humanoidverse/config/simulator/mujoco.yaml이 없는 경우
    동적으로 config를 등록합니다.
    """
    import os
    from pathlib import Path

    # study/configs/simulator/mujoco.yaml 경로
    study_config = Path(__file__).parent.parent / "configs" / "simulator" / "mujoco.yaml"

    # humanoidverse config 경로
    hv_config = Path(__file__).parent.parent.parent / "humanoidverse" / "config" / "simulator" / "mujoco.yaml"

    if not hv_config.exists() and study_config.exists():
        # 심볼릭 링크 또는 복사는 하지 않고, Hydra search path에 추가하는 방식 사용
        print(f"[config] MuJoCo config 필요: {hv_config}")
        print(f"[config] study/configs/simulator/mujoco.yaml을 humanoidverse/config/simulator/로 복사하세요")
        return False

    return True
