from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
import os

ext_modules = [
    Pybind11Extension(
        "ih_prep._core",
        ["src/bindings.cpp"],
        include_dirs=[],
        cxx_std=17,
    ),
]

setup(
    name="ih-prep",
    version="0.1.0",
    author="Your Name",
    description="Data preparation for information-theoretic analysis",
    ext_modules=ext_modules,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
    ],
)