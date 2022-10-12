"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os, sys
from functools import partial

import numpy as np
import torch as th
import torchvision.utils as tvu
from torchvision.transforms import functional as TF
import torch.distributed as dist
import torch.nn.functional as F

sys.path.append('')
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from config import create_config


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    ## Change your output path here
    out_dir = os.path.join('symlink/output/',args.model_name,args.cond_name,args.method)
    if dist.get_rank() == 0: 
        print(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    config, model_config0, class_config = create_config(args.model_name, args.timestep_rp)

    batch_size = config['batch_size']
    logger.log("creating model and diffusion...")
    model_config = model_and_diffusion_defaults()
    model_config.update(model_config0)
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict( dist_util.load_state_dict(config['model_path'], map_location="cpu") )
    model.requires_grad_(False).eval().cuda()
    model.to(dist_util.dev())
    if model_config['use_fp16']:
        model.convert_to_fp16()

    logger.log("loading classifier...")
    classifier_config = classifier_defaults()
    classifier_config.update(class_config)
    classifier = create_classifier(**classifier_config)
    classifier.load_state_dict( dist_util.load_state_dict(config['classifier_path'], map_location="cpu") )
    classifier.to(dist_util.dev())
    classifier.requires_grad_(False).eval().cuda()
    if classifier_config['classifier_use_fp16']:
        classifier.convert_to_fp16()

    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in)[0]
            return  grad * config['classifier_scale']

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if model_config['class_cond'] else None)

    fixseed = True
    if fixseed: # setting seed
        worker_seed = 10
        np.random.seed(worker_seed)
        th.manual_seed(worker_seed)

    if args.method == 'ddim':
        sample_fn = diffusion.ddim_sample_loop
    elif args.method[:4] in ['plms','pndm']:
        sample_fn = partial(diffusion.plms_sample_loop, order=int(args.method[4]))
    elif args.method[:4] in ['ltsp','ours','ltts']:
        sample_fn = partial(diffusion.ltsp_sample_loop, order=int(args.method[4]))
    elif args.method[:4] in ['stsp','bchf']:
        sample_fn = partial(diffusion.stsp_sample_loop, order=int(args.method[4]))   
    else:
        sample_fn = diffusion.p_sample_loop 

    if args.cond_name == 'cond1': cond_fn0 = cond_fn
    else: cond_fn0 = None

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample = sample_fn(
            model_fn,
            (batch_size, 3,config['image_size'], config['image_size']),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn0,
            impu_fn=None,
            progress=True,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * batch_size} samples")


    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(out_dir, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():

    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        use_ddim=True,
        model_name = "u256",
        method = "ddim",
        cond_name = "cond1",
        timestep_rp = 25,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
