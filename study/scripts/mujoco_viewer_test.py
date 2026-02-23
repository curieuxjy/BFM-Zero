"""
MuJoCo 뷰어 테스트 스크립트

사용법:
    uv run python study/scripts/mujoco_viewer_test.py
"""
import os
import sys

# macOS에서는 glfw 사용
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import mujoco.viewer
from pathlib import Path
import time

def main():
    # G1 로봇 모델 경로
    hv_root = Path(__file__).resolve().parent.parent.parent / "humanoidverse"
    model_path = hv_root / "data/robots/g1/scene_29dof_freebase_mujoco.xml"

    print(f"Loading model from: {model_path}")

    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    # MuJoCo 모델 로드
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print("Model loaded successfully!")
    print(f"  - nq (generalized coords): {model.nq}")
    print(f"  - nv (generalized velocities): {model.nv}")
    print(f"  - nu (actuators): {model.nu}")

    # 뷰어 실행
    print("\nLaunching MuJoCo viewer...")
    print("Close the window to exit.")

    # macOS에서는 launch_passive 대신 launch 사용 (blocking)
    # launch는 창을 닫을 때까지 블로킹됨
    mujoco.viewer.launch(model, data)

    print("Viewer closed.")


if __name__ == "__main__":
    main()
