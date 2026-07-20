# env_copy.py
from __future__ import annotations
import os
import sys
import shutil
import sysconfig
import site
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_SRC = Path("hps/bpe_simple_vocab_16e6.txt.gz")
DEFAULT_DEST_REL = Path("hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz")

__all__ = ["copy_bpe_vocab_to_open_clip", "find_open_clip_dir", "site_packages_candidates"]

def site_packages_candidates() -> list[Path]:
    """현재 실행 중인 파이썬(가상환경)의 site-packages 후보 경로들을 추정."""
    cand: set[Path] = set()
    for key in ("purelib", "platlib"):
        p = sysconfig.get_paths().get(key)
        if p:
            cand.add(Path(p))
    try:
        for p in site.getsitepackages():
            cand.add(Path(p))
    except Exception:
        pass
    try:
        cand.add(Path(site.getusersitepackages()))
    except Exception:
        pass
    # 일반적인 백업 경로
    cand.add(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    return [p for p in cand if p.exists()]

def find_open_clip_dir(dest_rel: Path = DEFAULT_DEST_REL) -> Path:
    """
    site-packages들 중에 dest_rel의 상위 디렉토리(hpsv2/src/open_clip)를 찾아 반환.
    없으면 가장 그럴듯한 첫 후보에 생성할 경로를 돌려준다.
    """
    bases = site_packages_candidates()
    # 정확 경로 탐색 (빠름)
    for base in bases:
        direct = base / dest_rel.parent
        if direct.exists():
            return direct
    # 패턴 탐색 (폴더 구조가 살짝 다른 경우)
    for base in bases:
        matches = list(base.rglob(str(dest_rel.parent).replace("\\", "/")))
        if matches:
            return matches[0]
    # 아무것도 없으면 첫 후보에 생성할 경로 지정
    if not bases:
        raise RuntimeError("site-packages 경로를 찾지 못했습니다. 가상환경이 올바른지 확인하세요.")
    return bases[0] / dest_rel.parent

def copy_bpe_vocab_to_open_clip(
    src: os.PathLike | str = DEFAULT_SRC,
    *,
    dest_rel: os.PathLike | str = DEFAULT_DEST_REL,
    overwrite: bool = True
) -> Path:
    """
    현재 가상환경의 site-packages 아래 `hpsv2/src/open_clip/`에 BPE vocab 파일을 복사.
    반환값: 실제로 복사된 대상 파일의 절대 경로(Path).
    """
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"소스 파일이 없습니다: {src_path.resolve()}")

    dest_dir = find_open_clip_dir(Path(dest_rel))
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / Path(dest_rel).name
    if dest_path.exists() and not overwrite:
        # 이미 있으면 그대로 반환
        return dest_path.resolve()

    shutil.copy2(src_path, dest_path)
    return dest_path.resolve()

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Copy BPE vocab into current env's hpsv2/src/open_clip/")
    ap.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="Source vocab path")
    ap.add_argument("--dest-rel", type=str, default=str(DEFAULT_DEST_REL), help="Relative dest under site-packages")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite if exists")
    args = ap.parse_args()

    out = copy_bpe_vocab_to_open_clip(args.src, dest_rel=args.dest_rel, overwrite=not args.no_overwrite)
    print(json.dumps({"copied_to": str(out)}, ensure_ascii=False))
