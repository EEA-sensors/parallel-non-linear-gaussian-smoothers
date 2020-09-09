# Author: Adrien Corenflos

"""Install pekf."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='pekf',
    version='0.1',
    description='Parallel Extended Kalman Filter.',
    author='Adrien Corenflos',
    author_email='adrien.corenflos@gmail.com',
    url='https://github.com/AdrienCorenflos/parallelEKF',
    packages=find_packages(),
    install_requires=requirements,
)
