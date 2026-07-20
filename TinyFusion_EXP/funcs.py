# multiconv in Random-Conditioning https://github.com/dohyun-as/Random-Conditioning/blob/main/src/funcs.py
import os
import torch
import torch.nn as nn

@torch.no_grad()
def build_projector(student_model, teacher_model, student_layers, teacher_layers, x_t, t, model_kwargs):
    teacher_model.eval()
    # _ = teacher_model(sample_x, sample_c)  # features 채우기
    # _ = student_model(sample_x, sample_c)
    _, cfg_drop_ids = student_model(x_t, t, return_cfg_drop=True, **model_kwargs)
    
    with torch.no_grad():
        _ = teacher_model(x_t, t, cfg_drop_ids=cfg_drop_ids, **model_kwargs)

    s_feats = [student_model.module.blocks[i].feature for i in student_layers]
    t_feats = [teacher_model.blocks[j].feature for j in teacher_layers]

    # 두 모델 모양이 동일하다고 가정(토큰↔토큰, 맵↔맵)
    if s_feats[0].dim() == 3:  # (B,N,C) 토큰
        in_list  = [f.shape[-1] for f in s_feats]
        out_list = [f.shape[-1] for f in t_feats]
        projector = MultiLinear1x1(in_list, out_list)        # ← Linear 사용
    elif s_feats[0].dim() == 4:  # (B,C,H,W) 맵
        in_list  = [f.shape[1] for f in s_feats]
        out_list = [f.shape[1] for f in t_feats]
        projector = MultiConv1x1(in_list, out_list)           # ← 네가 만든 Conv1×1 사용
    else:
        raise RuntimeError(f"Unexpected feature shape: {s_feats[0].shape}")
    return projector

class MultiLinear1x1(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(cin, cout)
            for cin, cout in zip(in_channels_list, out_channels_list)
        ])
    def forward(self, x_list):
        return [p(x) for p, x in zip(self.projs, x_list)]


class MultiConv1x1(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        """
        여러 개의 입력 및 출력 채널에 맞는 1x1 Conv 레이어들을 정의하는 클래스.
        
        Args:
            in_channels_list (list of int): 각 입력에 대한 채널 리스트.
            out_channels_list (list of int): 각 출력에 대한 채널 리스트.
        """
        super(MultiConv1x1, self).__init__()
        assert len(in_channels_list) == len(out_channels_list), "Input and output channel lists must have the same length."
        
        # 여러 개의 1x1 conv 레이어를 nn.ModuleList로 정의
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
            for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])
        
    def forward(self, x_list):
        """
        입력으로 들어온 여러 텐서에 대해 1x1 Conv 레이어들을 적용
        
        Args:
            x_list (list of tensors): 각 입력 텐서의 리스트.
            
        Returns:
            list of tensors: 각 입력 텐서에 대해 변환된 결과 텐서 리스트.
        """
        assert len(x_list) == len(self.convs), "Number of inputs must match the number of convolution layers."
        
        # 각 텐서에 대응되는 conv 레이어 적용
        output_list = [conv(x) for conv, x in zip(self.convs, x_list)]
        return output_list
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), save_path)
        # 필요한 경우 config 저장
        # 예: torch.save(self.config, os.path.join(save_directory, 'config.bin'))

    @classmethod
    def from_pretrained(cls, load_directory, student_channels_list, teacher_channels_list):
        model = cls(student_channels_list, teacher_channels_list)
        load_path = os.path.join(load_directory, 'pytorch_model.bin')
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict)
        return model
    
    
def get_layer_output_channels(model, mapping_layers):
    channels_list = []
    for layer_name in mapping_layers:
        layer = dict(model.named_modules())[layer_name]  # 해당 레이어 가져오기
        # print(f"Layer '{layer_name}': {layer}")
        # print(f"Available attributes: {dir(layer)}")
        if isinstance(layer, nn.Conv2d):  # Conv 레이어일 경우
            channels_list.append(layer.out_channels)
        elif isinstance(layer, nn.BatchNorm2d):  # BatchNorm일 경우, num_features는 채널 크기
            channels_list.append(layer.num_features)
        elif hasattr(layer, 'out_channels'):  # 혹시 다른 커스텀 레이어일 경우
            channels_list.append(layer.out_channels)
        else:
            raise ValueError(f"Layer {layer_name} does not have 'out_channels' or similar property.")
    return channels_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())



from typing import Optional, Union
def load_or_build_imagefolder_dataset(data_dir, split="train", save_root=None):
    import os, time, hashlib
    from datasets import load_dataset, load_from_disk, Image
    from filelock import FileLock

    # ① 저장 위치를 데이터 폴더 안으로
    if save_root is None:
        save_root = os.path.join(os.path.abspath(data_dir), ".hf_arrow")
    os.makedirs(save_root, exist_ok=True)

    key = hashlib.md5((os.path.abspath(data_dir) + ":" + split).encode()).hexdigest()[:8]
    save_dir = os.path.join(save_root, f"imagefolder_{key}_{split}")

    t0 = time.time()
    if os.path.isdir(save_dir):
        print(f"*** load_from_disk: {save_dir}")
        ds = load_from_disk(save_dir)
        print(f"*** load_from_disk done in {time.time()-t0:.3f} sec")
        return ds

    # ② 첫 1회만 빌드 (동시 실행 대비 락)
    lock_path = save_dir + ".lock"
    with FileLock(lock_path):
        if os.path.isdir(save_dir):
            ds = load_from_disk(save_dir); return ds

        print(f"*** building dataset → {save_dir}")
        ds = load_dataset("imagefolder", data_dir=data_dir, split=split,
                          ignore_verifications=True)
        ds = ds.cast_column("image", Image(decode=False))
        ds.save_to_disk(save_dir)

    return load_from_disk(save_dir)
