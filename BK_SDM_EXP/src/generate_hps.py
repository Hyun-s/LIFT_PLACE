# ------------------------------------------------------------------------------------
# Copyright 2023–2024 Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import time
from utils.inference_pipeline import InferencePipeline
from utils.misc import get_file_list_from_csv, change_img_size
import hpsv2

from hps.mv_vocab import copy_bpe_vocab_to_open_clip

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-small")    
    parser.add_argument("--save_dir", type=str, default="./results/bk-sdm-small",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--unet_path", type=str, default=None)   
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=8)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device)
    pipeline.set_pipe_and_generator()    
    # except:
    #     if 'v2' in args.model_id:
    #         mod = "stabilityai/stable-diffusion-2-1-base"
    #     else:
    #         mod = "CompVis/stable-diffusion-v1-4"
    #     pipeline = StableDiffusionPipeline.from_pretrained(
    #                 mod,
    #                 safety_checker=None,
    #                 revision=args.revision,
    #             )
    #     pipeline.set_pipe_and_generator()    

    if args.unet_path is not None: # use a separate trained unet for generation        
        from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet_ema')
        pipeline.pipe.unet = unet.half().to(args.device)
        print(f"** load unet from {args.unet_path}")        

    save_dir_src = os.path.join(args.save_dir, f'hpsv2') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)    

    file_list = hpsv2.benchmark_prompts('all') 
    params_str = pipeline.get_sdm_params()
    
    for style, prompts in file_list.items():
        for idx, prompt in enumerate(prompts):
            imgs = pipeline.generate(prompt = prompt,
                                     n_steps = args.num_inference_steps,
                                     img_sz = args.img_sz)[0]
            # print(len(imgs), imgs)
            os.makedirs(os.path.join(save_dir_src, style),exist_ok=True)
            imgs.save(os.path.join(save_dir_src, style, f"{idx:05d}.jpg")) 
            imgs.close()
            
    dst = copy_bpe_vocab_to_open_clip()
    print("copied to:", dst)
    hpsv2.evaluate(save_dir_src) 

    pipeline.clear()
    
    print(f"{(time.perf_counter()-t0):.2f} sec elapsed")
