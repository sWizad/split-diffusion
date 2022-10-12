
def create_config(model_name='u256', timestep_rp=50):
    if model_name == 'c64':
        #batch_size = 6  
        #use_ddim=False
        #clip_denoised=True
        #classifier_scale=0.1
        
        ## Change your model path here
        config = {
            'model_path' :'symlink/pretrained/64x64_diffusion.pt',
            'classifier_path':'symlink/pretrained/64x64_classifier.pt',
            'image_size': 64,
            'batch_size' : 64,
            'use_ddim':False,
            'clip_denoised':True,
            'classifier_scale': 0.1,
        }
        model_config = {
            'attention_resolutions': '32, 16, 8',
            'class_cond': True,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': str(timestep_rp),
            'dropout': 0.1,
            'image_size': 64,
            'learn_sigma': True,
            'noise_schedule': 'cosine',
            'num_channels': 192,
            'num_head_channels': 64,
            'num_res_blocks': 3,
            'resblock_updown': True,
            'use_new_attention_order': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        }
        class_config = {'classifier_depth':4 }
        #model_path = 'symlink/pretrained/64x64_diffusion.pt'
        #classifier_path ='symlink/pretrained/64x64_classifier.pt'
    elif model_name == 'c128':
        ## Change your model path here
        config = {
            'model_path' :'symlink/pretrained/128x128_diffusion.pt',
            'classifier_path':'symlink/pretrained/128x128_classifier.pt',
            'image_size': 128,
            'batch_size' : 25,
            'use_ddim':False,
            'clip_denoised':True,
            'classifier_scale': 1.25,
        }
        model_config = {
            'attention_resolutions': '32, 16, 8',
            'class_cond': True,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': str(timestep_rp),
            'image_size': 128,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_heads': 4,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        }
        class_config = {
            'image_size': 128,
        }
    elif model_name == 'c256':
        #batch_size = 6
        #use_ddim=False
        #clip_denoised=True
        #classifier_scale=2.5

        ## Change your model path here
        config = {
            'model_path' :'symlink/pretrained/256x256_diffusion.pt',
            'classifier_path':'symlink/pretrained/256x256_classifier.pt',
            'image_size': 256,
            'batch_size' : 6,
            'use_ddim':False,
            'clip_denoised':True,
            'classifier_scale': 2.5,
        }
        model_config = {
            'attention_resolutions': '32, 16, 8',
            'class_cond': True,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': str(timestep_rp),
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        }
        class_config = {
            'image_size': 256,
        }
    elif model_name == 'u256':
        ## Change your model path here
        config = {
            'model_path' :'symlink/pretrained/256x256_diffusion_uncond.pt',
            'classifier_path':'symlink/pretrained/256x256_classifier.pt',
            'image_size': 256,
            'batch_size' : 6,
            'use_ddim':False,
            'clip_denoised':True,
            'classifier_scale': 10.0,
        }
        model_config = {
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': str(timestep_rp),
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': True,
            'use_scale_shift_norm': True,
        }
        class_config = {
            'image_size': 256,
        }
        #model_path = 'symlink/pretrained/256x256_diffusion_uncond.pt'
        #classifier_path ='symlink/pretrained/256x256_classifier.pt'
    elif model_name == 'c512':
        ## Change your model path here
        config = {
            'model_path' :'symlink/pretrained/512x512_diffusion.pt',
            'classifier_path':'symlink/pretrained/512x512_classifier.pt',
            'image_size': 512,
            'batch_size' : 9,
            'use_ddim':False,
            'clip_denoised':True,
            'classifier_scale': 9.0,
        }
        model_config = {
            'attention_resolutions': '32, 16, 8',
            'class_cond': True,
            'diffusion_steps': 1000,
            'rescale_timesteps': True,
            'timestep_respacing': str(timestep_rp),
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': False,
            'use_scale_shift_norm': True,
        }
        class_config = {
            'image_size': 512,
        }
    
    return config, model_config, class_config