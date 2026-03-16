"""
compare_three_inferences.py - 3가지 Inference 모드 심층 비교

Tracking / Goal / Reward Inference의 z 벡터 계산 방식을 비교합니다.
실제 모델을 로드하여 각 모드의 z를 계산하고, 코사인 유사도를 분석합니다.

실행:
  uv run python study/scripts/compare_three_inferences.py --model_folder model/

  # 특정 스텝만
  uv run python study/scripts/compare_three_inferences.py --model_folder model/ --step 1
  uv run python study/scripts/compare_three_inferences.py --model_folder model/ --step 2
  uv run python study/scripts/compare_three_inferences.py --model_folder model/ --step 3
  uv run python study/scripts/compare_three_inferences.py --model_folder model/ --step 4

Step 1: 3가지 모드의 구조 비교 (모델 불필요)
Step 2: Tracking Inference - 모션 시퀀스 → z 시퀀스
Step 3: Goal Inference - 단일 프레임 → z 벡터
Step 4: 3가지 모드 z 벡터 비교 분석
"""

import argparse
import os
import sys

if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"


# ─────────────────────────────────────────────────────────────
# Step 1: 3가지 모드 구조 비교 (모델 불필요)
# ─────────────────────────────────────────────────────────────
def step1_compare_structure():
    """3가지 inference 모드의 핵심 차이를 표로 정리"""

    print("=" * 75)
    print("Step 1: 3가지 Inference 모드 구조 비교")
    print("=" * 75)

    print("""
┌──────────────────────────────────────────────────────────────────────────┐
│                    z 벡터 계산 방식 비교                                 │
├──────────┬───────────────────────────────────────────────────────────────┤
│ Tracking │ z[t] = project_z( window_avg( B(obs[t:t+W]) ) )             │
│          │ → 매 프레임마다 다른 z (시퀀스)                               │
│          │ → 모션을 "따라가는" 행동                                      │
├──────────┼───────────────────────────────────────────────────────────────┤
│ Goal     │ z = project_z( B(goal_frame, velocity=0) )                  │
│          │ → 단일 z 벡터                                                │
│          │ → 목표 자세로 "도달하는" 행동                                  │
├──────────┼───────────────────────────────────────────────────────────────┤
│ Reward   │ z = project_z( Σ softmax(10·r_i) · r_i · B(s_i) )          │
│          │ → 버퍼 데이터의 보상 가중 평균으로 단일 z                      │
│          │ → 보상을 "최대화하는" 행동                                     │
└──────────┴───────────────────────────────────────────────────────────────┘
""")

    headers = [
        "항목",
        "Tracking",
        "Goal",
        "Reward",
    ]
    rows = [
        ["입력 데이터",
         "전문가 모션 시퀀스\n(lafan_29dof.pkl)",
         "목표 프레임 1개\n(goal_frames JSON)",
         "리플레이 버퍼\n(checkpoint/buffers)"],
        ["z 개수",
         "T개 (프레임 수만큼)",
         "1개 (목표당)",
         "1개 (태스크당)"],
        ["rollout 중 z 변화",
         "매 스텝 변경\nz[step % T]",
         "고정 (100스텝 유지)",
         "고정 (episode_len)"],
        ["velocity 정보",
         "포함",
         "제거 (multiplier=0)",
         "포함"],
        ["외부 데이터 필요",
         "모션 라이브러리",
         "모션 라이브러리 +\ngoal JSON",
         "학습된 버퍼 데이터"],
        ["출력 파일",
         "zs_{id}.pkl",
         "goal_reaching.pkl",
         "reward_locomotion.pkl"],
        ["핵심 코드",
         "tracking_inference.py\nlines 69-74",
         "goal_inference.py\nline 113",
         "reward_inference.py\nlines 159-182"],
    ]

    for row in rows:
        print(f"  [{row[0]}]")
        lines_t = row[1].split("\n")
        lines_g = row[2].split("\n")
        lines_r = row[3].split("\n")
        max_lines = max(len(lines_t), len(lines_g), len(lines_r))
        for i in range(max_lines):
            t = lines_t[i] if i < len(lines_t) else ""
            g = lines_g[i] if i < len(lines_g) else ""
            r = lines_r[i] if i < len(lines_r) else ""
            print(f"    Tracking: {t:<25s}  Goal: {g:<22s}  Reward: {r}")
        print()

    print("  공통점: 모두 Backward Map B를 통해 z를 계산하고, project_z로 정규화")
    print(f"         ||z|| = sqrt(z_dim) = sqrt(256) = 16.0")


# ─────────────────────────────────────────────────────────────
# 유틸: 모델/환경 로드
# ─────────────────────────────────────────────────────────────
def _load_model_and_env(model_folder, device="cpu"):
    """모델과 환경을 로드하는 공통 유틸"""
    import json
    import torch
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

    model = load_model_from_checkpoint_dir(
        os.path.join(model_folder, "checkpoint"), device=device
    )
    model.eval()

    with open(os.path.join(model_folder, "config.json")) as f:
        config = json.load(f)

    # macOS 오버라이드
    config["env"]["device"] = device
    overrides = config["env"].get("hydra_overrides", [])
    overrides = [o for o in overrides if "simulator=" not in o]
    overrides.append("simulator=mujoco")
    config["env"]["hydra_overrides"] = overrides

    from humanoidverse.agents.envs.humanoidverse_isaac import HumanoidVerseIsaacConfig

    env_cfg = HumanoidVerseIsaacConfig(**config["env"])
    wrapped_env, env_info = env_cfg.build(num_envs=1)

    return model, wrapped_env, config


def _patch_mujoco_if_needed():
    """macOS quat_rotate 패치"""
    if sys.platform != "darwin":
        return
    try:
        from humanoidverse.simulator.mujoco import mujoco as mujoco_module
        from humanoidverse.utils.torch_utils import quat_rotate
        import torch

        MuJoCoClass = mujoco_module.MuJoCo

        original_robot = MuJoCoClass.robot_root_states.fget
        original_all = MuJoCoClass.all_root_states.fget

        def _patched_robot_root_states(self):
            qpos = torch.tensor(self.mj_data.qpos, dtype=torch.float32).unsqueeze(0)
            qvel = torch.tensor(self.mj_data.qvel, dtype=torch.float32).unsqueeze(0)
            base_pos = qpos[:, :3]
            base_quat = qpos[:, 3:7]
            base_lin_vel = qvel[:, :3]
            base_ang_vel = quat_rotate(base_quat, qvel[:, 3:6], w_last=True)
            return torch.cat([base_pos, base_quat, base_lin_vel, base_ang_vel], dim=-1)

        def _patched_all_root_states(self):
            qpos = torch.tensor(self.mj_data.qpos, dtype=torch.float32).unsqueeze(0)
            qvel = torch.tensor(self.mj_data.qvel, dtype=torch.float32).unsqueeze(0)
            base_pos = qpos[:, :3]
            base_quat = qpos[:, 3:7]
            base_lin_vel = qvel[:, :3]
            base_ang_vel = quat_rotate(base_quat, qvel[:, 3:6], w_last=True)
            return torch.cat([base_pos, base_quat, base_lin_vel, base_ang_vel], dim=-1)

        MuJoCoClass.robot_root_states = property(_patched_robot_root_states)
        MuJoCoClass.all_root_states = property(_patched_all_root_states)
        print("  [패치] quat_rotate w_last=True 적용 완료")
    except Exception as e:
        print(f"  [패치 스킵] {e}")


# ─────────────────────────────────────────────────────────────
# Step 2: Tracking Inference - z 시퀀스 생성
# ─────────────────────────────────────────────────────────────
def step2_tracking_inference(model_folder, motion_id=25):
    """Tracking Inference의 z 계산 과정을 단계별로 보여줌"""
    import torch

    print("=" * 75)
    print(f"Step 2: Tracking Inference (Motion ID={motion_id})")
    print("=" * 75)

    _patch_mujoco_if_needed()
    print("\n모델/환경 로딩 중...")
    model, env, config = _load_model_and_env(model_folder)

    # 1. 모션 설정
    env.set_is_evaluating(motion_id)
    motion_name = env.env.motion_lib.motion_names[motion_id]
    print(f"\n[1] 모션 설정: ID={motion_id}, 이름='{motion_name}'")

    # 2. Backward observation 추출
    obs_dict = env.get_backward_observation()
    print(f"\n[2] Backward observation 추출:")
    for k, v in obs_dict.items():
        print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}")

    # 3. Backward Map으로 z 계산
    with torch.no_grad():
        raw_z = model.backward_map(obs_dict)

    print(f"\n[3] Backward Map 출력:")
    print(f"    raw_z shape: {list(raw_z.shape)}")
    print(f"    raw_z norm (처음 5프레임): {raw_z[:5].norm(dim=-1).tolist()}")

    # 4. project_z (L2 정규화)
    z_projected = model.project_z(raw_z)
    print(f"\n[4] project_z 후:")
    print(f"    z shape: {list(z_projected.shape)}")
    print(f"    z norm (처음 5프레임): {[f'{n:.2f}' for n in z_projected[:5].norm(dim=-1).tolist()]}")
    print(f"    기대 norm: sqrt(256) = {256**0.5:.2f}")

    # 5. z 통계
    print(f"\n[5] z 벡터 통계:")
    print(f"    총 프레임 수: {z_projected.shape[0]}")
    print(f"    z_dim: {z_projected.shape[1]}")

    # 프레임 간 코사인 유사도
    cos_sim = torch.nn.functional.cosine_similarity
    if z_projected.shape[0] > 1:
        consecutive_sim = cos_sim(z_projected[:-1], z_projected[1:])
        print(f"    연속 프레임 간 코사인 유사도:")
        print(f"      평균: {consecutive_sim.mean():.4f}")
        print(f"      최소: {consecutive_sim.min():.4f}")
        print(f"      최대: {consecutive_sim.max():.4f}")

    # 처음과 마지막 비교
    first_last = cos_sim(z_projected[0:1], z_projected[-1:])
    print(f"    첫 프레임 vs 마지막 프레임: {first_last.item():.4f}")

    print(f"\n  → Tracking: 매 스텝 z[step % {z_projected.shape[0]}]로 z가 변화")

    return z_projected, motion_name


# ─────────────────────────────────────────────────────────────
# Step 3: Goal Inference - 단일 프레임 z
# ─────────────────────────────────────────────────────────────
def step3_goal_inference(model_folder, motion_id=25, frame_idx=100):
    """Goal Inference의 z 계산 과정을 단계별로 보여줌"""
    import torch

    print("=" * 75)
    print(f"Step 3: Goal Inference (Motion ID={motion_id}, Frame={frame_idx})")
    print("=" * 75)

    _patch_mujoco_if_needed()
    print("\n모델/환경 로딩 중...")
    model, env, config = _load_model_and_env(model_folder)

    # 1. 모션 설정
    env.set_is_evaluating(motion_id)
    motion_name = env.env.motion_lib.motion_names[motion_id]
    print(f"\n[1] 모션 설정: ID={motion_id}, 이름='{motion_name}'")

    # 2. Backward observation (velocity=0)
    obs_with_vel = env.get_backward_observation()
    obs_no_vel = env.get_backward_observation(velocity_multiplier=0)

    print(f"\n[2] Backward observation (velocity 비교):")
    for k in obs_with_vel:
        v_vel = obs_with_vel[k]
        v_no = obs_no_vel[k]
        diff = (v_vel - v_no).abs().sum().item()
        print(f"    {k}: diff={diff:.4f}" + (" ← velocity 제거됨!" if diff > 0 else ""))

    # 3. 단일 프레임 추출
    total_frames = obs_no_vel[list(obs_no_vel.keys())[0]].shape[0]
    actual_frame = min(frame_idx, total_frames - 1)
    goal_obs = {k: v[actual_frame:actual_frame+1] for k, v in obs_no_vel.items()}
    print(f"\n[3] 목표 프레임 추출: frame {actual_frame}/{total_frames}")
    for k, v in goal_obs.items():
        print(f"    {k}: shape={list(v.shape)}")

    # 4. Backward Map → z
    with torch.no_grad():
        raw_z = model.backward_map(goal_obs)
        z_goal = model.project_z(raw_z)

    print(f"\n[4] Goal z 벡터:")
    print(f"    shape: {list(z_goal.shape)}")
    print(f"    norm: {z_goal.norm().item():.2f}")

    # 5. velocity 유무 비교
    with torch.no_grad():
        goal_obs_vel = {k: v[actual_frame:actual_frame+1] for k, v in obs_with_vel.items()}
        z_with_vel = model.project_z(model.backward_map(goal_obs_vel))

    cos_sim = torch.nn.functional.cosine_similarity(z_goal, z_with_vel)
    print(f"\n[5] velocity=0 vs velocity 유지:")
    print(f"    코사인 유사도: {cos_sim.item():.4f}")
    print(f"    → Goal은 velocity=0으로 '자세'만 인코딩 (속도 무관)")

    print(f"\n  → Goal: 단일 z 벡터로 고정, 100스텝마다 다음 goal로 전환")

    return z_goal, motion_name


# ─────────────────────────────────────────────────────────────
# Step 4: 3가지 모드 z 벡터 비교
# ─────────────────────────────────────────────────────────────
def step4_compare_all(model_folder, motion_id=25, frame_idx=100):
    """3가지 모드의 z 벡터를 비교 분석"""
    import torch

    print("=" * 75)
    print("Step 4: 3가지 모드 z 벡터 비교 분석")
    print("=" * 75)

    _patch_mujoco_if_needed()
    print("\n모델/환경 로딩 중...")
    model, env, config = _load_model_and_env(model_folder)

    env.set_is_evaluating(motion_id)
    motion_name = env.env.motion_lib.motion_names[motion_id]
    print(f"모션: ID={motion_id}, '{motion_name}'")

    cos_sim = torch.nn.functional.cosine_similarity

    # Tracking z (전체 시퀀스)
    with torch.no_grad():
        obs_full = env.get_backward_observation()
        z_tracking_all = model.project_z(model.backward_map(obs_full))

    total_frames = z_tracking_all.shape[0]
    actual_frame = min(frame_idx, total_frames - 1)
    z_tracking_frame = z_tracking_all[actual_frame:actual_frame+1]

    # Goal z (velocity=0, 단일 프레임)
    with torch.no_grad():
        obs_no_vel = env.get_backward_observation(velocity_multiplier=0)
        goal_obs = {k: v[actual_frame:actual_frame+1] for k, v in obs_no_vel.items()}
        z_goal = model.project_z(model.backward_map(goal_obs))

    # Reward z는 버퍼가 필요하므로 균등 분포로 대체 설명
    with torch.no_grad():
        z_random = model.sample_z(1)

    print(f"\n[비교 결과] (frame {actual_frame})")
    print(f"  z_tracking  norm: {z_tracking_frame.norm().item():.2f}")
    print(f"  z_goal      norm: {z_goal.norm().item():.2f}")
    print(f"  z_random    norm: {z_random.norm().item():.2f}")

    # 코사인 유사도 행렬
    sim_tg = cos_sim(z_tracking_frame, z_goal).item()
    sim_tr = cos_sim(z_tracking_frame, z_random).item()
    sim_gr = cos_sim(z_goal, z_random).item()

    print(f"\n[코사인 유사도 행렬]")
    print(f"                  Tracking    Goal      Random")
    print(f"  Tracking        1.0000      {sim_tg:.4f}    {sim_tr:.4f}")
    print(f"  Goal            {sim_tg:.4f}      1.0000    {sim_gr:.4f}")
    print(f"  Random          {sim_tr:.4f}      {sim_gr:.4f}    1.0000")

    print(f"\n[분석]")
    print(f"  Tracking vs Goal: {sim_tg:.4f}")
    if sim_tg > 0.8:
        print(f"    → 높은 유사도: 같은 프레임 기준, velocity 차이가 작음")
    elif sim_tg > 0.5:
        print(f"    → 중간 유사도: velocity 정보가 z에 영향을 줌")
    else:
        print(f"    → 낮은 유사도: velocity 제거가 z를 크게 변화시킴")

    print(f"  Tracking vs Random: {sim_tr:.4f}")
    print(f"    → 학습된 z vs 랜덤 z는 일반적으로 낮은 유사도")

    # 시간에 따른 z 변화 (Tracking 전용)
    print(f"\n[Tracking z의 시간적 변화]")
    sample_frames = [0, total_frames//4, total_frames//2, 3*total_frames//4, total_frames-1]
    print(f"  프레임 간 코사인 유사도 (frame 0 기준):")
    for f in sample_frames:
        sim = cos_sim(z_tracking_all[0:1], z_tracking_all[f:f+1]).item()
        print(f"    frame {f:>6d}: {sim:.4f}")

    print(f"\n[핵심 차이 요약]")
    print(f"  ┌────────────┬──────────────────────────────────────┐")
    print(f"  │ Tracking   │ z가 매 스텝 변화 → 모션 추종         │")
    print(f"  │ Goal       │ z 고정 (vel=0) → 자세 도달           │")
    print(f"  │ Reward     │ z 고정 (가중합) → 보상 최대화        │")
    print(f"  └────────────┴──────────────────────────────────────┘")
    print(f"  공통: z = project_z( B(observation) ), ||z|| = 16.0")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="12. 3가지 Inference 모드 심층 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model_folder", type=str, default="model/",
                        help="모델 폴더 경로 (기본: model/)")
    parser.add_argument("--step", type=int, default=0,
                        help="실행할 스텝 (1-4). 0이면 전부 (기본: 0)")
    parser.add_argument("--motion_id", type=int, default=25,
                        help="모션 ID (기본: 25)")
    parser.add_argument("--frame_idx", type=int, default=100,
                        help="Goal에 사용할 프레임 인덱스 (기본: 100)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║        12. 3가지 Inference 모드 심층 비교 스크립트                       ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    steps = {
        1: ("구조 비교 (모델 불필요)", lambda: step1_compare_structure()),
        2: ("Tracking Inference", lambda: step2_tracking_inference(args.model_folder, args.motion_id)),
        3: ("Goal Inference", lambda: step3_goal_inference(args.model_folder, args.motion_id, args.frame_idx)),
        4: ("3가지 모드 비교 분석", lambda: step4_compare_all(args.model_folder, args.motion_id, args.frame_idx)),
    }

    targets = list(steps.keys()) if args.step == 0 else [args.step]

    for i in targets:
        if i not in steps:
            print(f"유효한 스텝: 1-{len(steps)}")
            sys.exit(1)
        label, func = steps[i]
        print(f"\n▶ Step {i}: {label}\n")
        try:
            func()
        except Exception as e:
            print(f"  [오류] {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 75)
    print("완료!")
    print("=" * 75)


if __name__ == "__main__":
    main()
