"""
실습 2: 학습 과정 시각화

학습 로그(train_log.txt)를 읽어서 메트릭 그래프를 생성합니다.

사용법:
    uv run python study/scripts/visualize_training.py --log_dir results/bfmzero-isaac
    uv run python study/scripts/visualize_training.py --log_dir results/bfmzero-isaac --no-save_png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless 렌더링
import matplotlib.pyplot as plt
import tyro


def main(
    log_dir: str,
    save_png: bool = True,
):
    """
    학습 로그 시각화

    Args:
        log_dir: 학습 결과 디렉토리 (train_log.txt 포함)
        save_png: PNG 파일 저장 여부
    """
    log_dir = Path(log_dir)
    log_file = log_dir / "train_log.txt"

    if not log_file.exists():
        print(f"Error: {log_file} not found")
        print(f"Available files in {log_dir}:")
        if log_dir.exists():
            for f in sorted(log_dir.iterdir()):
                print(f"  {f.name}")
        return

    df = pd.read_csv(log_file)
    print(f"Loaded {len(df)} entries")
    print(f"Columns: {list(df.columns)}")
    print()

    # 주요 학습 메트릭
    train_metrics = [
        ("fb_loss", "FB Loss", "감소 후 안정화"),
        ("ortho_loss", "Ortho Loss", "감소 후 안정화"),
        ("disc_loss", "Disc Loss", "0.5~2.0 부근 안정"),
        ("critic_loss", "Critic Loss", "감소 후 안정화"),
        ("actor_loss", "Actor Loss", "점진적 감소"),
        ("mean_disc_reward", "Disc Reward", "음수→양수 전환"),
    ]

    # 존재하는 메트릭만 필터
    available = [(col, title, desc) for col, title, desc in train_metrics if col in df.columns]

    if not available:
        print("Warning: No known training metrics found in log")
        print(f"Available columns: {list(df.columns)}")
        return

    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 4 * nrows))
    fig.suptitle("BFM-Zero Training Metrics", fontsize=14)

    if nrows == 1:
        axes = axes.reshape(1, -1)

    x_col = "timestep" if "timestep" in df.columns else None

    for idx, (col, title, desc) in enumerate(available):
        ax = axes[idx // 2, idx % 2]
        x = df[x_col] if x_col else range(len(df))
        ax.plot(x, df[col], linewidth=0.8)
        ax.set_title(f"{title} ({desc})")
        ax.set_xlabel("Step" if x_col else "Entry")
        ax.grid(True, alpha=0.3)

    # 빈 subplot 숨기기
    for idx in range(len(available), nrows * 2):
        axes[idx // 2, idx % 2].set_visible(False)

    plt.tight_layout()

    if save_png:
        out_path = log_dir / "training_curves.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
    else:
        plt.show()

    # 평가 로그도 시각화
    eval_file = log_dir / "tracking_eval_log.csv"
    if eval_file.exists():
        eval_df = pd.read_csv(eval_file)
        print(f"\nEval log: {len(eval_df)} entries")

        if "emd" in eval_df.columns or "obs_state_emd" in eval_df.columns:
            emd_col = "emd" if "emd" in eval_df.columns else "obs_state_emd"
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(eval_df[emd_col], linewidth=1.0, marker="o", markersize=3)
            ax2.axhline(y=0.75, color="r", linestyle="--", label="Target (0.75)")
            ax2.set_title("Tracking Evaluation: EMD")
            ax2.set_ylabel("EMD (lower is better)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_png:
                out_path2 = log_dir / "eval_emd_curve.png"
                plt.savefig(out_path2, dpi=150)
                print(f"Saved to {out_path2}")

    # 최종 통계 출력
    print("\n=== 최근 학습 통계 ===")
    last_n = min(10, len(df))
    for col, title, _ in available:
        recent = df[col].tail(last_n)
        print(f"  {title}: mean={recent.mean():.4f}, std={recent.std():.4f}")


if __name__ == "__main__":
    tyro.cli(main)
