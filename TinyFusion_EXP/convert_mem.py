# pack_to_memmap.py
import os, numpy as np, glob
from tqdm import tqdm

features_dir = "/data/hyuns/imagenet_encoded/imagenet_encoded/imagenet256_features"
labels_dir   = "/data/hyuns/imagenet_encoded/imagenet_encoded/imagenet256_labels"
out_dir      = "/data/hyuns/imagenet_encoded/imagenet_encoded"; os.makedirs(out_dir, exist_ok=True)

feat_files = sorted(os.listdir(features_dir))
labl_files = sorted(os.listdir(labels_dir))
assert len(feat_files) == len(labl_files)

# 첫 샘플로 shape/dtype 파악
f0 = np.load(os.path.join(features_dir, feat_files[0]), mmap_mode="r")
l0 = np.load(os.path.join(labels_dir,   labl_files[0]), mmap_mode="r")
N  = len(feat_files)

F = np.memmap(os.path.join(out_dir, "features.npy"), dtype=f0.dtype, mode="w+", shape=(N,)+f0.shape)
L = np.memmap(os.path.join(out_dir, "labels.npy"),   dtype=l0.dtype, mode="w+", shape=(N,)+l0.shape)

for i,(ff,lf) in tqdm(enumerate(zip(feat_files, labl_files))):
    F[i] = np.load(os.path.join(features_dir, ff), mmap_mode="r")
    L[i] = np.load(os.path.join(labels_dir,   lf), mmap_mode="r")

F.flush(); L.flush()

# 원래 파일 이름 순서를 기록(재현성 검증/역참조용)
np.save(os.path.join(out_dir, "index_mapping.npy"),
        np.array(list(zip(feat_files, labl_files)), dtype=object))
print("done:", N, "samples")
