from setuptools import find_packages, setup

PACKAGE_NAME = "vmoba"
VERSION = "0.0.0"
AUTHOR = "JianzongWu"
DESCRIPTION = "VMoBA: Mixture-of-Block Attention for Video Diffusion Models"
URL = "https://github.com/KwaiVGI/VMoBA"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    url=URL,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires='>=3.12',
    install_requires=[
        "torch>=2.7.1",
        "flash_attn>=2.7.4"
    ]
)