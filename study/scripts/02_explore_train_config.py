"""
02_explore_train_config.py - train.py 구조 탐색 스크립트

학습 파이프라인의 핵심 구성요소를 단계별로 확인합니다:
  Step 1: TrainConfig 전체 필드 & 기본값
  Step 2: train_bfm_zero() 실제 설정값 비교
  Step 3: 에이전트 클래스 계층 구조 & 메서드
  Step 4: 모델 구성요소 & 파라미터 수 (체크포인트 필요)
  Step 5: 학습 하이퍼파라미터 상세 (lr, loss coefficients 등)

실행:
  uv run python study/scripts/02_explore_train_config.py
  uv run python study/scripts/02_explore_train_config.py --step 1   # 특정 스텝만
  uv run python study/scripts/02_explore_train_config.py --step 4   # 모델 로드 (느림)
"""

import argparse
import sys


# ─────────────────────────────────────────────────────────────
# Step 1: TrainConfig 필드 & 기본값
# ─────────────────────────────────────────────────────────────
def step1_train_config_fields():
    """TrainConfig에 정의된 모든 필드와 기본값을 출력"""
    from humanoidverse.train import TrainConfig

    print("=" * 70)
    print("Step 1: TrainConfig 필드 & 기본값")
    print("=" * 70)
    print(f"{'필드명':<35s} {'타입':<30s} {'기본값'}")
    print("-" * 95)

    for name, field in TrainConfig.model_fields.items():
        ftype = str(field.annotation).replace("typing.", "")
        # 긴 타입명 축약
        if len(ftype) > 28:
            ftype = ftype[:25] + "..."
        default = field.default
        if default is None and field.default_factory:
            default = "(factory)"
        print(f"  {name:<33s} {ftype:<30s} {default}")

    print(f"\n총 {len(TrainConfig.model_fields)}개 필드")


# ─────────────────────────────────────────────────────────────
# Step 2: 기본값 vs 실제 학습 설정 비교
# ─────────────────────────────────────────────────────────────
def step2_compare_default_vs_actual():
    """TrainConfig 기본값과 train_bfm_zero() 실제 설정값을 비교"""
    from humanoidverse.train import TrainConfig, train_bfm_zero
    import inspect

    print("=" * 70)
    print("Step 2: 기본값 vs train_bfm_zero() 실제 설정")
    print("=" * 70)

    # train_bfm_zero의 소스에서 cfg를 추출하지 않고,
    # 주요 숫자형 파라미터만 직접 비교
    defaults = {
        "online_parallel_envs": 50,
        "num_env_steps": 30_000_000,
        "num_seed_steps": 50_000,
        "update_agent_every": 500,
        "num_agent_updates": 50,
        "buffer_size": 5_000_000,
        "checkpoint_every_steps": 5_000_000,
        "eval_every_steps": 1_000_000,
        "seed": 0,
        "use_trajectory_buffer": False,
        "prioritization": False,
        "use_wandb": False,
    }

    actuals = {
        "online_parallel_envs": 1024,
        "num_env_steps": 384_000_000,
        "num_seed_steps": 10_240,
        "update_agent_every": 1024,
        "num_agent_updates": 16,
        "buffer_size": 5_120_000,
        "checkpoint_every_steps": 9_600_000,
        "eval_every_steps": 9_600_000,
        "seed": 4728,
        "use_trajectory_buffer": True,
        "prioritization": True,
        "use_wandb": False,
    }

    print(f"{'파라미터':<30s} {'기본값':>15s} {'실제값':>15s}  변경")
    print("-" * 75)
    for key in defaults:
        d = defaults[key]
        a = actuals[key]
        changed = " <<<" if d != a else ""
        d_str = f"{d:,}" if isinstance(d, int) else str(d)
        a_str = f"{a:,}" if isinstance(a, int) else str(a)
        print(f"  {key:<28s} {d_str:>15s} {a_str:>15s} {changed}")

    print()
    print("주요 차이점:")
    print("  - 병렬 환경: 50 → 1024 (20배)")
    print("  - 총 스텝: 3000만 → 3.84억 (12.8배)")
    print("  - 시드 스텝: 5만 → 1만 (감소, 병렬 환경이 크므로)")
    print("  - 업데이트 주기: 500 → 1024 (병렬 환경 수에 맞춤)")
    print("  - Gradient step: 50 → 16 (줄이고 자주 업데이트)")
    print("  - Trajectory buffer & Prioritized sampling 활성화")


# ─────────────────────────────────────────────────────────────
# Step 3: 에이전트 클래스 계층 구조
# ─────────────────────────────────────────────────────────────
def step3_agent_class_hierarchy():
    """FBAgent → FBcprAgent → FBcprAuxAgent 상속 체인과 메서드 분석"""
    from humanoidverse.agents.fb.agent import FBAgent
    from humanoidverse.agents.fb_cpr.agent import FBcprAgent
    from humanoidverse.agents.fb_cpr_aux.agent import FBcprAuxAgent

    print("=" * 70)
    print("Step 3: 에이전트 클래스 계층 구조")
    print("=" * 70)

    # 상속 체인
    print("\n[상속 체인] (MRO)")
    for i, cls in enumerate(FBcprAuxAgent.__mro__):
        indent = "  " * i
        print(f"  {indent}└── {cls.__module__}.{cls.__name__}")

    # 각 클래스가 직접 정의하는 메서드
    print("\n[각 클래스가 정의하는 메서드]")
    for cls in [FBAgent, FBcprAgent, FBcprAuxAgent]:
        own_methods = sorted([
            m for m in cls.__dict__
            if callable(getattr(cls, m, None)) and not m.startswith("_")
        ])
        print(f"\n  {cls.__name__} ({len(own_methods)}개):")
        for m in own_methods:
            # 간단한 docstring 첫 줄 추출
            func = getattr(cls, m)
            doc = (func.__doc__ or "").strip().split("\n")[0][:50]
            print(f"    - {m}()" + (f"  # {doc}" if doc else ""))

    # 핵심 update 메서드 오버라이드 관계
    print("\n[update 관련 메서드 오버라이드]")
    update_methods = [
        "update", "update_fb", "update_discriminator",
        "update_critic", "update_actor", "update_aux_critic",
    ]
    print(f"  {'메서드':<25s} {'FBAgent':<12s} {'FBcprAgent':<12s} {'FBcprAuxAgent'}")
    print("  " + "-" * 60)
    for m in update_methods:
        cells = []
        for cls in [FBAgent, FBcprAgent, FBcprAuxAgent]:
            has = m in cls.__dict__
            cells.append("●" if has else "·")
        print(f"  {m:<25s} {cells[0]:<12s} {cells[1]:<12s} {cells[2]}")
    print("\n  ● = 이 클래스에서 정의/오버라이드  · = 상위 클래스에서 상속")


# ─────────────────────────────────────────────────────────────
# Step 4: 모델 구성요소 & 파라미터 수
# ─────────────────────────────────────────────────────────────
def step4_model_components():
    """체크포인트를 로드하여 모델 내부 구조와 파라미터 수 확인"""
    import torch
    from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
    from pathlib import Path

    model_path = Path("model/checkpoint")
    if not model_path.exists():
        print("!!! model/checkpoint 폴더가 없습니다.")
        print("    모델 다운로드 후 다시 실행하세요.")
        return

    print("=" * 70)
    print("Step 4: 모델 구성요소 & 파라미터 수")
    print("=" * 70)
    print("\n모델 로딩 중...")

    model = load_model_from_checkpoint_dir(str(model_path), device="cpu")
    model.eval()

    # 전체 구성요소
    print("\n[모델 구성요소]")
    print(f"  {'이름':<27s} {'파라미터 수':>12s}  설명")
    print("  " + "-" * 65)

    descriptions = {
        "_forward_map": "F(s,z,a) → s' 예측 (Successor Feature)",
        "_backward_map": "B(s) → z 인코딩",
        "_actor": "π(s,z) → action 정책",
        "_critic": "Q(s,z,a) → 판별자 보상 Q값",
        "_discriminator": "D(s,z) → expert/policy 판별",
        "_aux_critic": "Q_aux(s,z,a) → 보조 보상 Q값",
        "_obs_normalizer": "관측값 정규화 (BatchNorm)",
        "_target_forward_map": "F target (soft update τ=0.01)",
        "_target_backward_map": "B target (soft update τ=0.01)",
        "_target_critic": "Q target (soft update τ=0.005)",
        "_target_aux_critic": "Q_aux target (soft update τ=0.005)",
    }

    total = 0
    target_total = 0
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        total += param_count
        is_target = name.startswith("_target")
        if is_target:
            target_total += param_count
        desc = descriptions.get(name, "")
        marker = "(T)" if is_target else "   "
        print(f"  {marker} {name:<24s} {param_count:>12,}  {desc}")

    print("  " + "-" * 65)
    print(f"      {'전체 합계':<24s} {total:>12,}")
    print(f"      {'학습 파라미터 (target 제외)':<24s} {total - target_total:>12,}")
    print(f"      {'Target 파라미터 (복사본)':<24s} {target_total:>12,}")

    # z_dim 확인
    print(f"\n[핵심 하이퍼파라미터]")
    print(f"  z_dim       = {model.cfg.archi.z_dim}")
    print(f"  actor_std   = {model.cfg.actor_std}")
    print(f"  seq_length  = {model.cfg.seq_length}")
    print(f"  norm_z      = {model.cfg.archi.norm_z}")

    # 각 네트워크의 hidden dim
    print(f"\n[네트워크 아키텍처]")
    archi = model.cfg.archi
    configs = {
        "Forward Map (F)": archi.f,
        "Backward Map (B)": archi.b,
        "Actor (π)": archi.actor,
        "Critic (Q)": archi.critic,
        "Discriminator (D)": archi.discriminator,
        "Aux Critic (Q_aux)": archi.aux_critic,
    }
    print(f"  {'네트워크':<22s} {'hidden_dim':>10s} {'layers':>7s} {'model':<10s}")
    print("  " + "-" * 55)
    for label, cfg in configs.items():
        h = getattr(cfg, "hidden_dim", "?")
        l = getattr(cfg, "hidden_layers", "?")
        m = getattr(cfg, "model", "-")
        print(f"  {label:<22s} {str(h):>10s} {str(l):>7s} {str(m):<10s}")


# ─────────────────────────────────────────────────────────────
# Step 5: 학습 하이퍼파라미터 상세
# ─────────────────────────────────────────────────────────────
def step5_training_hyperparams():
    """학습률, 손실 계수, z 샘플링 비율 등 상세 하이퍼파라미터"""

    print("=" * 70)
    print("Step 5: 학습 하이퍼파라미터 상세 (train_bfm_zero 기준)")
    print("=" * 70)

    # Learning rates
    print("\n[학습률 (Learning Rates)]")
    lrs = {
        "lr_f (Forward Map)": 3e-4,
        "lr_b (Backward Map)": 1e-5,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "lr_aux_critic": 3e-4,
        "lr_discriminator": 1e-5,
    }
    for name, lr in lrs.items():
        bar = "█" * int(lr * 100000)
        print(f"  {name:<28s} {lr:<10.1e} {bar}")

    print("\n  ※ B와 Discriminator는 lr이 30배 낮음 (안정적 학습을 위해)")

    # Loss coefficients
    print("\n[손실 계수 (Loss Coefficients)]")
    coeffs = [
        ("ortho_coef", 100.0, "B 직교정규화 가중치"),
        ("reg_coeff", 0.05, "판별자 Q → Actor 가중치"),
        ("reg_coeff_aux", 0.02, "보조 Q → Actor 가중치"),
        ("grad_penalty_discriminator", 10.0, "WGAN gradient penalty"),
        ("discount (γ)", 0.98, "미래 보상 할인율"),
    ]
    print(f"  {'계수':<32s} {'값':>8s}  설명")
    print("  " + "-" * 65)
    for name, val, desc in coeffs:
        print(f"  {name:<32s} {val:>8.2f}  {desc}")

    # Target network
    print("\n[Target Network Soft Update (τ)]")
    taus = [
        ("fb_target_tau (F, B)", 0.01),
        ("critic_target_tau (Q, Q_aux)", 0.005),
    ]
    for name, tau in taus:
        print(f"  {name:<35s} τ = {tau}")
    print(f"  target = τ * online + (1-τ) * target  (매 gradient step)")

    # Z sampling
    print("\n[Z 분포 샘플링 비율]")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  train_goal_ratio   = 0.2  (20%) 목표 인코딩     │")
    print("  │  expert_asm_ratio   = 0.6  (60%) 전문가 인코딩   │")
    print("  │  uniform            = 0.2  (20%) 균등 분포        │")
    print("  └──────────────────────────────────────────────────┘")
    print("  relabel_ratio = 0.8 → 80% 확률로 z를 새로 리라벨링")
    print("  update_z_every_step = 100 → 100 스텝마다 rollout z 갱신")

    # Rollout
    print("\n[전문가 궤적 Rollout]")
    print("  rollout_expert_trajectories = True")
    print("  rollout_expert_trajectories_percentage = 0.5")
    print("    → 1024 환경 중 512개는 전문가 z 사용")
    print("  rollout_expert_trajectories_length = 250")
    print("    → 전문가 z를 250 스텝 동안 유지")

    # Aux rewards
    print("\n[보조 보상 스케일링]")
    aux = [
        ("penalty_action_rate", -0.1),
        ("penalty_feet_ori", -0.4),
        ("penalty_ankle_roll", -4.0),
        ("limits_dof_pos", -10.0),
        ("penalty_slippage", -2.0),
        ("penalty_undesired_contact", -1.0),
        ("penalty_torques", 0.0),
        ("limits_torque", 0.0),
    ]
    aux.sort(key=lambda x: x[1])
    for name, scale in aux:
        bar = "▓" * int(abs(scale) * 3) if scale != 0 else "(비활성)"
        print(f"  {name:<30s} {scale:>6.1f}  {bar}")

    # Actor loss 구조
    print("\n[Actor Loss 구조]")
    print("  actor_loss = - Q_fb.mean()")
    print("               - Q_disc.mean()  × reg_coeff(0.05)  × weight")
    print("               - Q_aux.mean()   × reg_coeff_aux(0.02) × weight")
    print()
    print("  weight = scale_reg가 True이면 |Q_fb| / |Q_disc|로 자동 스케일링")
    print("  → 세 Q값의 gradient 크기가 비슷해지도록 동적 조절")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="02. train.py 구조 탐색",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="실행할 스텝 번호 (1-5). 0이면 전부 실행 (기본: 0)",
    )
    args = parser.parse_args()

    steps = {
        1: ("TrainConfig 필드 & 기본값", step1_train_config_fields),
        2: ("기본값 vs 실제 학습 설정", step2_compare_default_vs_actual),
        3: ("에이전트 클래스 계층 구조", step3_agent_class_hierarchy),
        4: ("모델 구성요소 & 파라미터 수 (모델 로드)", step4_model_components),
        5: ("학습 하이퍼파라미터 상세", step5_training_hyperparams),
    }

    if args.step == 0:
        targets = list(steps.keys())
    elif args.step in steps:
        targets = [args.step]
    else:
        print(f"유효한 스텝: 1-{len(steps)}")
        sys.exit(1)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           02. train.py 구조 탐색 스크립트                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    for i in targets:
        label, func = steps[i]
        print(f"\n▶ Step {i}: {label}")
        print()
        try:
            func()
        except Exception as e:
            print(f"  [오류] {e}")
        print()

    print("=" * 70)
    print("완료! 다음 스터디: 03_fbcpr_algorithm (FB-CPR 알고리즘)")
    print("  uv run python study/scripts/03_explore_fbcpr.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
