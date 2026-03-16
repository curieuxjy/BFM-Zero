"""
explore_fbcpr.py - FB-CPR 알고리즘 탐색 스크립트

에이전트 계층(FBAgent → FBcprAgent → FBcprAuxAgent)의 핵심 메서드를
단계별로 실행하며 FB 표현 학습의 원리를 확인합니다.

실행:
  uv run python study/scripts/explore_fbcpr.py
  uv run python study/scripts/explore_fbcpr.py --step 1   # 특정 스텝만
  uv run python study/scripts/explore_fbcpr.py --step 5   # 모델 로드 필요

Step 1: 에이전트 update() 흐름 분석 (모델 불필요)
Step 2: 손실 함수 구조 (모델 불필요)
Step 3: Z 분포 샘플링 & project_z (모델 불필요)
Step 4: Discriminator & Critic 구조 (모델 불필요)
Step 5: 실제 모델로 Forward/Backward 연산 확인 (모델 필요)
"""

import argparse
import sys


# ─────────────────────────────────────────────────────────────
# Step 1: 에이전트 update() 흐름 분석
# ─────────────────────────────────────────────────────────────
def step1_update_flow():
    """3단계 에이전트의 update() 메서드 호출 순서 분석"""
    from humanoidverse.agents.fb.agent import FBAgent
    from humanoidverse.agents.fb_cpr.agent import FBcprAgent
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent
    import inspect

    print("=" * 70)
    print("Step 1: 에이전트 update() 흐름 분석")
    print("=" * 70)

    # FBAgent.update 소스에서 호출 순서 추출
    print("\n[FBAgent.update() 흐름]")
    print("  1. 배치 샘플링 (expert + train)")
    print("  2. obs 정규화")
    print("  3. z 샘플링 (goal 인코딩 or uniform)")
    print("  4. update_fb(obs, action, discount, next_obs, goal, z)")
    print("  5. update_td3_actor(obs, z)")
    print("  6. soft_update: F_target, B_target")

    print("\n[FBcprAgent.update() 추가 흐름]  ← FBAgent 상속 + 확장")
    print("  1. 배치 샘플링 (expert + train)")
    print("  2. obs 정규화")
    print("  3. expert_z = encode_expert(expert_next_obs)")
    print("  4. update_discriminator(expert_obs, expert_z, train_obs, train_z)")
    print("  5. z = sample_mixed_z(train_goal, expert_z)")
    print("  6. z 리라벨링 (relabel_ratio=0.8)")
    print("  7. update_fb(obs, action, discount, next_obs, goal, z)")
    print("  8. update_critic(obs, action, discount, next_obs, z)")
    print("  9. update_actor(obs, action, z)  ← Q_fb + Q_disc")
    print("  10. soft_update: F_target, B_target, Critic_target")

    print("\n[FBcprAuxAgent.update() 추가 흐름]  ← FBcprAgent 상속 + 확장")
    print("  1~7. FBcprAgent와 동일")
    print("  8. update_critic(obs, action, discount, next_obs, z)")
    print("  9. update_aux_critic(obs, action, discount, aux_reward, next_obs, z)")
    print("  10. update_actor(obs, action, z)  ← Q_fb + Q_disc + Q_aux")
    print("  11. soft_update: F_target, B_target, Critic_target, AuxCritic_target")

    # 메서드 오버라이드 표
    print("\n[메서드 정의 위치]")
    methods = [
        ("update", "전체 학습 루프"),
        ("update_fb", "Forward-Backward 맵 업데이트"),
        ("update_discriminator", "GAN 판별자 업데이트"),
        ("update_critic", "Critic TD 학습"),
        ("update_aux_critic", "보조 Critic 업데이트"),
        ("update_actor", "Actor 정책 업데이트"),
        ("encode_expert", "전문가 z 인코딩"),
        ("sample_mixed_z", "혼합 z 분포 샘플링"),
    ]
    print(f"  {'메서드':<25s} {'FBAgent':<12s} {'FBcprAgent':<12s} {'FBcprAux':<12s} 설명")
    print("  " + "-" * 80)
    for m, desc in methods:
        cells = []
        for cls in [FBAgent, FBcprAgent, FBcprAuxAgent]:
            has = m in cls.__dict__
            cells.append("● 정의" if has else "  상속")
        print(f"  {m:<25s} {cells[0]:<12s} {cells[1]:<12s} {cells[2]:<12s} {desc}")


# ─────────────────────────────────────────────────────────────
# Step 2: 손실 함수 구조
# ─────────────────────────────────────────────────────────────
def step2_loss_functions():
    """각 손실 함수의 수학적 구조와 역할"""
    import torch

    print("=" * 70)
    print("Step 2: 손실 함수 구조")
    print("=" * 70)

    # FB Loss 시뮬레이션
    print("\n[1] FB Loss (update_fb)")
    print("    M = F(s,z,a) · B(s')^T  →  batch×batch 행렬")
    print()

    batch = 4
    z_dim = 8
    # 시뮬레이션
    torch.manual_seed(42)
    F_out = torch.randn(batch, z_dim)
    B_out = torch.randn(batch, z_dim)
    M = torch.matmul(F_out, B_out.T)

    print(f"    예시 M 행렬 (batch={batch}):")
    for i in range(batch):
        row = "    " + "".join(f"{M[i, j]:>7.2f}" for j in range(batch))
        if i == 0:
            row += "  ← 대각=자기자신 방문(↑), 비대각=타인 방문(→0)"
        print(row)

    off_diag = 1 - torch.eye(batch)
    fb_offdiag = 0.5 * (M * off_diag).pow(2).sum() / off_diag.sum()
    fb_diag = -M.diag().mean()
    print(f"\n    fb_offdiag = {fb_offdiag:.4f}  (비대각 → 0으로 MSE)")
    print(f"    fb_diag    = {fb_diag:.4f}  (대각 최대화 → 음수)")
    print(f"    fb_loss    = {fb_offdiag + fb_diag:.4f}")

    # Orthonormality Loss
    print("\n[2] Orthonormality Loss")
    Cov = torch.matmul(B_out, B_out.T)
    orth_diag = -Cov.diag().mean()
    orth_offdiag = 0.5 * (Cov * off_diag).pow(2).sum() / off_diag.sum()
    print(f"    B·B^T 대각 평균:  {Cov.diag().mean():.4f}  (목표: 1.0)")
    print(f"    B·B^T 비대각 MSE: {orth_offdiag:.4f}  (목표: 0.0)")
    print(f"    orth_loss = {orth_offdiag + orth_diag:.4f}")
    print(f"    ortho_coef = 100.0  →  fb_loss += 100 × orth_loss")

    # Discriminator Loss
    print("\n[3] Discriminator Loss (update_discriminator)")
    print("    GAN 스타일 이진 분류 + WGAN Gradient Penalty")
    print()
    expert_logits = torch.randn(batch)
    policy_logits = torch.randn(batch)
    expert_loss = -torch.nn.functional.logsigmoid(expert_logits).mean()
    policy_loss = torch.nn.functional.softplus(policy_logits).mean()
    print(f"    expert_loss = -log(σ(D(expert))) = {expert_loss:.4f}  (전문가 → 1)")
    print(f"    policy_loss = log(1+exp(D(policy))) = {policy_loss:.4f}  (정책 → 0)")
    print(f"    + gradient_penalty × 10.0  (WGAN-GP 안정화)")

    # Critic Loss
    print("\n[4] Critic Loss (update_critic)")
    print("    TD Learning with Discriminator Reward")
    print("    r = D.compute_reward(s, z)")
    print("    target_Q = r + γ × Q_target(s', z, π(s',z))")
    print("    critic_loss = MSE(Q(s,z,a), target_Q)")

    # Actor Loss
    print("\n[5] Actor Loss (update_actor in FBcprAuxAgent)")
    print("    actor_loss = - Q_fb.mean()")
    print("                 - Q_disc.mean()  × reg_coeff(0.05)  × weight")
    print("                 - Q_aux.mean()   × reg_coeff_aux(0.02) × weight")
    print()
    print("    weight = |Q_fb| / |Q_disc|  (scale_reg=True일 때 동적 스케일링)")
    print()
    print("    → 세 종류의 보상 신호를 결합:")
    print("      Q_fb   : FB 표현에서 유도 (successor feature)")
    print("      Q_disc : 전문가 모방 보상 (GAN)")
    print("      Q_aux  : 안전/자연스러움 보상 (환경 페널티)")


# ─────────────────────────────────────────────────────────────
# Step 3: Z 분포 샘플링 & project_z
# ─────────────────────────────────────────────────────────────
def step3_z_distribution():
    """Z 벡터의 샘플링, 정규화, 리라벨링 과정"""
    import torch
    import torch.nn.functional as F
    import math

    print("=" * 70)
    print("Step 3: Z 분포 샘플링 & project_z")
    print("=" * 70)

    z_dim = 256
    batch_size = 8

    # 1. sample_z (균등 분포)
    print("\n[1] sample_z() - 균등 분포")
    z_uniform = torch.randn(batch_size, z_dim)
    print(f"    z ~ N(0,1): shape={list(z_uniform.shape)}")
    print(f"    norm (정규화 전): {z_uniform.norm(dim=-1).mean():.2f}")

    # 2. project_z
    z_proj = math.sqrt(z_dim) * F.normalize(z_uniform, dim=-1)
    print(f"\n[2] project_z() - L2 정규화")
    print(f"    z_proj = sqrt({z_dim}) × z/||z||")
    print(f"    norm (정규화 후): {z_proj.norm(dim=-1).mean():.2f}")
    print(f"    기대값: sqrt({z_dim}) = {math.sqrt(z_dim):.2f}")

    # 3. sample_mixed_z 비율
    print(f"\n[3] sample_mixed_z() - 혼합 분포 (train_bfm_zero 기준)")
    print("    ┌─────────────────────────────────────────────────┐")
    print("    │  source          ratio   설명                    │")
    print("    ├─────────────────────────────────────────────────┤")
    print("    │  goal encoding   0.2     B(train_next_obs)      │")
    print("    │  expert ASM      0.6     B(expert_next_obs)     │")
    print("    │  uniform         0.2     랜덤 z                  │")
    print("    └─────────────────────────────────────────────────┘")
    print()

    # 비율 시뮬레이션
    torch.manual_seed(0)
    n = 10000
    probs = torch.tensor([0.2, 0.6, 0.2])
    mix_idxs = torch.multinomial(probs.expand(n, -1), 1).squeeze()
    counts = torch.bincount(mix_idxs, minlength=3)
    labels = ["goal", "expert", "uniform"]
    for i in range(3):
        bar = "█" * (counts[i].item() // 200)
        print(f"    {labels[i]:<10s}: {counts[i]:>5d}/{n} ({counts[i]/n*100:.1f}%)  {bar}")

    # 4. relabel
    print(f"\n[4] Z 리라벨링 (relabel_ratio=0.8)")
    print(f"    mask = rand({batch_size}) <= 0.8")
    mask = torch.rand(batch_size) <= 0.8
    print(f"    mask = {mask.tolist()}")
    print(f"    리라벨 수: {mask.sum()}/{batch_size}")
    print(f"    → 80%의 z를 새로 샘플링한 z로 교체")
    print(f"    → 다양한 z에 대해 학습하여 일반화 능력 향상")

    # 5. 코사인 유사도
    print(f"\n[5] 서로 다른 z 벡터 간 코사인 유사도")
    z1 = math.sqrt(z_dim) * F.normalize(torch.randn(100, z_dim), dim=-1)
    cos_matrix = F.cosine_similarity(z1.unsqueeze(0), z1.unsqueeze(1), dim=-1)
    off_diag_cos = cos_matrix[~torch.eye(100, dtype=bool)]
    print(f"    100개 랜덤 z의 쌍별 코사인 유사도:")
    print(f"    평균: {off_diag_cos.mean():.4f}  (기대: ~0)")
    print(f"    표준편차: {off_diag_cos.std():.4f}")
    print(f"    범위: [{off_diag_cos.min():.4f}, {off_diag_cos.max():.4f}]")
    print(f"    → 고차원({z_dim}D)에서 랜덤 벡터는 거의 직교")


# ─────────────────────────────────────────────────────────────
# Step 4: Discriminator & Critic 구조
# ─────────────────────────────────────────────────────────────
def step4_discriminator_critic():
    """Discriminator와 Critic의 역할과 상호작용"""

    print("=" * 70)
    print("Step 4: Discriminator & Critic 구조")
    print("=" * 70)

    print("""
[Discriminator D(s, z)]
  역할: 전문가 vs 정책 판별 (GAN의 Discriminator)
  입력: 관측값(state, privileged_state) + z 벡터
  출력: logit (스칼라)
  구조: MLP hidden_dim=1024, 3 hidden layers
  학습: BCE loss + WGAN gradient penalty (λ=10)

  보상 생성:
    reward = -log(1 - σ(D(s, z)))   (AIRL 스타일)
    → 전문가처럼 행동할수록 높은 보상

[Critic Q(s, z, a)]
  역할: Discriminator 보상의 가치 함수
  입력: 관측값 + z + action
  출력: Q-value (스칼라)
  구조: residual MLP hidden_dim=2048, 6 layers, num_parallel=2
  학습: TD learning (target = r_disc + γ × Q_target)

[Aux Critic Q_aux(s, z, a)]
  역할: 보조 보상(환경 페널티)의 가치 함수
  입력: 관측값 + z + action
  출력: Q-value (스칼라)
  구조: Critic과 동일
  학습: TD learning (target = r_aux + γ × Q_aux_target)
""")

    print("[Actor 학습 시 3가지 Q-value 결합 구조]")
    print()
    print("  ┌──────────────┐")
    print("  │   Actor π    │")
    print("  │  (obs, z)    │──→ action")
    print("  └──────┬───────┘")
    print("         │ action을 세 곳에 전달")
    print("    ┌────┼──────────────────┐")
    print("    ▼    ▼                  ▼")
    print("  Q_fb  Q_disc          Q_aux")
    print("  F·z   Critic          AuxCritic")
    print("  ×1.0  ×0.05×w        ×0.02×w")
    print("    └────┼──────────────────┘")
    print("         ▼")
    print("   actor_loss = -(Q_fb + 0.05w·Q_disc + 0.02w·Q_aux)")
    print()
    print("  w = |Q_fb.mean()| / |Q_disc.mean()|  (동적 스케일링)")
    print("  → Q_fb gradient와 비슷한 크기로 다른 Q값 조절")

    # 보조 보상 목록
    print("\n[보조 보상 (aux_rewards) 스케일링]")
    aux = [
        ("limits_dof_pos", -10.0, "관절 한계 초과 페널티"),
        ("penalty_ankle_roll", -4.0, "발목 롤 페널티"),
        ("penalty_slippage", -2.0, "발 미끄러짐 페널티"),
        ("penalty_undesired_contact", -1.0, "원치않는 접촉"),
        ("penalty_feet_ori", -0.4, "발 방향 페널티"),
        ("penalty_action_rate", -0.1, "액션 변화율"),
        ("penalty_torques", 0.0, "(비활성)"),
        ("limits_torque", 0.0, "(비활성)"),
    ]
    print(f"  {'보상 이름':<30s} {'스케일':>7s}  설명")
    print("  " + "-" * 60)
    for name, scale, desc in aux:
        bar = "▓" * int(abs(scale) * 2) if scale != 0 else ""
        print(f"  {name:<30s} {scale:>7.1f}  {desc} {bar}")


# ─────────────────────────────────────────────────────────────
# Step 5: 실제 모델로 Forward/Backward 연산
# ─────────────────────────────────────────────────────────────
def step5_model_forward_backward():
    """체크포인트를 로드하여 F, B, Actor, Critic의 실제 연산 확인"""
    import torch
    import math
    from pathlib import Path
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir

    model_path = Path("model/checkpoint")
    if not model_path.exists():
        print("!!! model/checkpoint 폴더가 없습니다.")
        print("    모델 다운로드 후 다시 실행하세요.")
        return

    print("=" * 70)
    print("Step 5: 실제 모델로 Forward/Backward 연산")
    print("=" * 70)

    print("\n모델 로딩 중...")
    model = load_model_from_checkpoint_dir(str(model_path), device="cpu")
    model.eval()

    # obs 공간 확인
    print("\n[1] 입력 공간 확인")
    # 더미 obs 생성 (obs_normalizer의 키 구조 확인)
    obs_keys = {}
    for name, norm in model._obs_normalizer._normalizers.items():
        # BatchNorm의 입력 차원 추출
        if hasattr(norm, 'num_features'):
            dim = norm.num_features
        elif hasattr(norm, '_normalizer') and hasattr(norm._normalizer, 'num_features'):
            dim = norm._normalizer.num_features
        else:
            dim = "?"
        obs_keys[name] = dim
        print(f"    {name}: dim={dim}")

    # 더미 데이터 생성
    batch = 4
    z_dim = model.cfg.archi.z_dim
    action_dim = model._actor._trunk[0].in_features - z_dim  # actor 입력에서 z_dim 제외

    print(f"\n    z_dim = {z_dim}")

    # 더미 obs 딕셔너리
    dummy_obs = {}
    for k, dim in obs_keys.items():
        if isinstance(dim, int):
            dummy_obs[k] = torch.randn(batch, dim)

    # z 벡터
    z = model.sample_z(batch)
    z = model.project_z(z)

    print(f"\n[2] Backward Map: B(obs) → z")
    with torch.no_grad():
        b_out = model._backward_map(dummy_obs)
    b_proj = model.project_z(b_out)
    print(f"    입력: obs dict ({len(dummy_obs)} keys)")
    print(f"    출력: shape={list(b_out.shape)}")
    print(f"    norm (raw):  {b_out.norm(dim=-1).tolist()}")
    print(f"    norm (proj): {[f'{n:.2f}' for n in b_proj.norm(dim=-1).tolist()]}")

    print(f"\n[3] Forward Map: F(obs, z, action) → successor feature")
    dummy_action = torch.randn(batch, 29)  # 29 DOF
    with torch.no_grad():
        f_out = model._forward_map(dummy_obs, z, dummy_action)
    print(f"    입력: obs + z({z_dim}) + action(29)")
    print(f"    출력: shape={list(f_out.shape)}  (num_parallel × batch × z_dim)")

    # M = F · B^T
    M = torch.matmul(f_out, b_proj.T)
    print(f"\n    M = F · B^T: shape={list(M.shape)}")
    print(f"    M 대각 평균: {M.diagonal(dim1=-2, dim2=-1).mean():.4f}")
    print(f"    M 비대각 평균: {(M * (1 - torch.eye(batch))).sum() / (batch*(batch-1)):.4f}")

    print(f"\n[4] Actor: π(obs, z) → action")
    with torch.no_grad():
        dist = model._actor(dummy_obs, z, model.cfg.actor_std)
        action = dist.mean
    print(f"    출력: action shape={list(action.shape)}")
    print(f"    action 범위: [{action.min():.3f}, {action.max():.3f}]")
    print(f"    actor_std = {model.cfg.actor_std}")

    print(f"\n[5] Q-value 계산")
    # Q_fb = F · z
    with torch.no_grad():
        Fs = model._forward_map(dummy_obs, z, action)
    Q_fb = (Fs * z).sum(dim=-1)
    print(f"    Q_fb = F·z: shape={list(Q_fb.shape)}, mean={Q_fb.mean():.4f}")

    # Discriminator reward
    with torch.no_grad():
        disc_logits = model._discriminator.compute_logits(dummy_obs, z)
    print(f"    D(obs,z) logits: {[f'{l:.3f}' for l in disc_logits.squeeze().tolist()]}")

    # Critic Q
    with torch.no_grad():
        Q_critic = model._critic(dummy_obs, z, action)
    print(f"    Q_critic: shape={list(Q_critic.shape)}, mean={Q_critic.mean():.4f}")

    print(f"\n[요약]")
    print(f"    Forward Map:    obs → F(s,z,a) ∈ R^{z_dim}")
    print(f"    Backward Map:   obs → B(s) ∈ R^{z_dim}")
    print(f"    Actor:          obs,z → action ∈ R^29")
    print(f"    Discriminator:  obs,z → logit ∈ R^1")
    print(f"    Critic:         obs,z,a → Q ∈ R^1")
    print(f"    Q_fb = F(s,z,a)·z  (내적으로 즉시 계산)")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="03. FB-CPR 알고리즘 탐색",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="실행할 스텝 (1-5). 0이면 전부 (기본: 0)",
    )
    args = parser.parse_args()

    steps = {
        1: ("에이전트 update() 흐름", step1_update_flow),
        2: ("손실 함수 구조", step2_loss_functions),
        3: ("Z 분포 샘플링 & project_z", step3_z_distribution),
        4: ("Discriminator & Critic 구조", step4_discriminator_critic),
        5: ("실제 모델 Forward/Backward 연산 (모델 로드)", step5_model_forward_backward),
    }

    if args.step == 0:
        targets = list(steps.keys())
    elif args.step in steps:
        targets = [args.step]
    else:
        print(f"유효한 스텝: 1-{len(steps)}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          03. FB-CPR 알고리즘 탐색 스크립트                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    for i in targets:
        label, func = steps[i]
        print(f"\n▶ Step {i}: {label}\n")
        try:
            func()
        except Exception as e:
            print(f"  [오류] {e}")
            import traceback
            traceback.print_exc()
        print()

    print("=" * 70)
    print("완료! 다음 스터디: 04_environment_config (환경 설정)")
    print("  → 관련 스크립트: uv run python study/scripts/mujoco_viewer_test.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
