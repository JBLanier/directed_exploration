import os

from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

install_requires = [
    "numpy",
    "gym_boxpush",
]

setup(
    name='directed_exploration',
    version='0.0.1',
    description='',
    classifiers=[
        "Intended Audience :: Science/Research"
    ],
    url='https://github.com/JBLanier/exploration_seach',
    author='J.B. Lanier',
    author_email='jblanier@uci.edu',
    keywords='',
    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
    install_requires=install_requires
)
