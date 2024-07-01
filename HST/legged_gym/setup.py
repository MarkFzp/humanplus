from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym',
    version='1.0.0',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for Legged Robots',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib',
                      'numpy==1.20',]
)