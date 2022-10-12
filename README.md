# Accelerating Guided Diffusion Sampling with Splitting Numerical Methods
The implementation for Accelerating Guided Diffusion Sampling with Splitting Numerical Methods (2023)

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion) and [crowsonkb/guided-diffusion](https://github.com/crowsonkb/guided-diffusion), with modifications on sampling method.

# Installation
Clone this repository and run:
```
pip install -e .
```
This should install the python package that the scripts depend on.

# Download pre-trained models
All checkpoints of diffusion and classifier models are provided in [this](https://github.com/openai/guided-diffusion#download-pre-trained-models).

# Classifier guidance
For this code version, user need to download pretrain models and change the models' location in `config.py`.
The output directly can be cange in `scripts/classifier_sample.py`

```
python scripts/classifier_sample.py --model=u256 --method=stps4 --timestep_rp=20
```

Some example of `--method` options are ```stsp4, stsp2, ltsp4, ltsp2, plms4, plms2, ddim ```

- 128x128 model: `--model=c128`
- 256x256 model: `--model=c256`
- 256x256 model (unconditional): `--model=u256`
- 512x512 model: `--model=c512`

# Other tasks
For detailed usage example, see the note books directory.

- [![][colab]][SD-text2im] This notebook shows how to use Splitting Numerical Methods with CLIP-guided Stable Diffusion.

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[SD-text2im]: <https://colab.research.google.com/drive/1uDArGUikVwuNVPX6KRVnSxIjfd6vJeZ1?usp=sharing>