"""
실습 4: 보상별 z 벡터 비교

reward_inference 결과에서 z 벡터를 로드하여 보상 함수별 코사인 유사도를 분석합니다.
reward_inference 실행 후 생성된 pkl 파일을 분석합니다.

사용법:
    # 먼저 reward_inference 실행:
    # uv run python -m humanoidverse.reward_inference --model_folder model/ --simulator mujoco --device cpu --headless

    # 그 다음 z 벡터 비교:
    uv run python study/scripts/compare_reward_z.py --result_dir model/reward_inference/

    # 또는 직접 pkl 파일 경로 지정:
    uv run python study/scripts/compare_reward_z.py --pkl_files model/reward_inference/z1.pkl model/reward_inference/z2.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import tyro


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도를 계산합니다."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main(
    result_dir: str = "model/reward_inference/",
    pkl_files: list[str] | None = None,
):
    """
    보상별 z 벡터를 비교합니다.

    Args:
        result_dir: reward_inference 결과 디렉토리
        pkl_files: 직접 pkl 파일 경로 지정 (선택)
    """
    try:
        import joblib
    except ImportError:
        print("joblib required. Install: uv add joblib")
        return

    result_dir = Path(result_dir)

    # pkl 파일 찾기
    if pkl_files:
        files = [Path(f) for f in pkl_files]
    else:
        files = list(result_dir.glob("*.pkl"))
        if not files:
            files = list(result_dir.glob("**/*.pkl"))

    if not files:
        print(f"No pkl files found in {result_dir}")
        print("\nreward_inference를 먼저 실행하세요:")
        print("  uv run python -m humanoidverse.reward_inference \\")
        print("      --model_folder model/ --simulator mujoco --device cpu --headless")
        return

    print(f"Found {len(files)} pkl files:")
    z_dict = {}
    for f in sorted(files):
        try:
            data = joblib.load(f)
            if isinstance(data, np.ndarray):
                z_dict[f.stem] = data
                print(f"  {f.stem}: shape={data.shape}, ||z||={np.linalg.norm(data, axis=-1).mean():.2f}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 1:
                        z_dict[f"{f.stem}/{k}"] = v
                        print(f"  {f.stem}/{k}: shape={v.shape}")
            else:
                print(f"  {f.stem}: type={type(data)} (skipped)")
        except Exception as e:
            print(f"  {f.stem}: ERROR - {e}")

    if len(z_dict) < 2:
        print("\n비교하려면 최소 2개의 z 벡터가 필요합니다.")
        return

    # 코사인 유사도 행렬
    print("\n" + "=" * 60)
    print("=== 보상별 z 벡터 코사인 유사도 ===")
    print("=" * 60)

    keys = list(z_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            zi = z_dict[keys[i]]
            zj = z_dict[keys[j]]

            # 첫 번째 z 벡터끼리 비교
            if zi.ndim > 1:
                zi = zi[0]
            if zj.ndim > 1:
                zj = zj[0]

            cos = cosine_similarity(zi, zj)
            print(f"  {keys[i]} vs {keys[j]}: cos_sim = {cos:.4f}")

    print("\n해석:")
    print("  cos_sim > 0.5: 유사한 행동 의도")
    print("  cos_sim ~0.0: 무관한 행동 의도")
    print("  cos_sim < -0.3: 반대 행동 의도")


if __name__ == "__main__":
    tyro.cli(main)
