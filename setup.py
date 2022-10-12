from setuptools import setup

setup(
    name="split-diffusion",
    py_modules=["split_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "fastprogress", "mpi4py"],
)