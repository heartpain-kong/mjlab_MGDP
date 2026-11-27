"""Installation script for the mjlab_MGDP python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]

# Installation operation
setup(
    name="mjlab_mgdp",
    packages=find_packages(include=["src", "src.*"]),
    version="0.1.0",
    install_requires=INSTALL_REQUIRES,
)
