from setuptools import setup, find_packages

setup(
    name="aerotrack_core",
    version="0.1.0",
    description="AeroTrack: UAV Open-Vocabulary Tracking and Segmentation Core",
    packages=find_packages(),
    python_requires=">=3.10",
    # Only lightweight common deps; heavy framework deps (torch, mmcv, mmdet, etc.)
    # must be installed manually following INSTALL.md staged pipeline.
    install_requires=[
        "numpy>=1.26,<2",
        "tqdm",
        "ftfy==6.1.1",
        "regex",
        "iopath>=0.1.10",
        "huggingface_hub",
        "opencv-python<=4.11.0.86",
        "supervision>=0.19.0",
        "psutil",
        "einops",
        "timm>=1.0.17",
    ],
)
