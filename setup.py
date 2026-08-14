#!/usr/bin/env python
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


setup(
    name='karyoflow',
    version='0.1.0',
    description='Few-step box transport for chromosome detection',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=find_packages(
        include=('ldmdet', 'ldmdet.*', 'experiments', 'experiments.*')
    ),
    python_requires='>=3.8',
    install_requires=[
        'einops>=0.7',
        'matplotlib>=3.7',
        'mmcv>=2.0.0,<2.2.0',
        'mmdet==3.3.0',
        'mmengine>=0.10.0,<1.0.0',
        'numpy>=1.24,<2.0',
        'opencv-python>=4.8',
        'pycocotools>=2.0.7',
        'PyYAML>=6.0',
        'scipy>=1.10',
        'swanlab>=0.7',
        'torch>=2.1',
        'torchvision>=0.16',
    ],
    extras_require={
        'dev': ['pre-commit>=3.0', 'pytest>=8.0', 'ruff>=0.9'],
    },
    include_package_data=True,
    zip_safe=False,
)
