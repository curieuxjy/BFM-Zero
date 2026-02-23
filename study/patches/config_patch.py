"""
Hydra Config 패치

study/configs의 설정 파일들을 humanoidverse/config에 복사합니다.
메인 소스코드를 직접 수정하지 않고 필요한 config를 추가합니다.
"""

import shutil
from pathlib import Path


def apply_config_patches():
    """
    study/configs의 파일들을 humanoidverse/config로 복사

    이 함수는 실행 시점에 필요한 config 파일들을 복사합니다.
    원본 저장소에 없는 config 파일들 (예: mujoco.yaml)을 추가합니다.
    """
    study_root = Path(__file__).parent.parent
    hv_config_root = study_root.parent / "humanoidverse" / "config"
    study_config_root = study_root / "configs"

    copied_files = []

    if study_config_root.exists():
        for src_file in study_config_root.rglob("*.yaml"):
            # 상대 경로 계산
            rel_path = src_file.relative_to(study_config_root)
            dst_file = hv_config_root / rel_path

            # 대상 디렉토리 생성
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # 파일이 없거나 다른 경우에만 복사
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                copied_files.append(str(rel_path))
                print(f"[config] 복사됨: {rel_path}")

    if copied_files:
        print(f"[config] {len(copied_files)}개 config 파일 적용됨")
    else:
        print("[config] 모든 config 파일이 이미 존재함")

    return copied_files


def remove_config_patches():
    """
    apply_config_patches로 복사된 파일들을 제거

    원본 상태로 되돌릴 때 사용합니다.
    """
    study_root = Path(__file__).parent.parent
    hv_config_root = study_root.parent / "humanoidverse" / "config"
    study_config_root = study_root / "configs"

    removed_files = []

    if study_config_root.exists():
        for src_file in study_config_root.rglob("*.yaml"):
            rel_path = src_file.relative_to(study_config_root)
            dst_file = hv_config_root / rel_path

            if dst_file.exists():
                dst_file.unlink()
                removed_files.append(str(rel_path))
                print(f"[config] 제거됨: {rel_path}")

    if removed_files:
        print(f"[config] {len(removed_files)}개 config 파일 제거됨")

    return removed_files
