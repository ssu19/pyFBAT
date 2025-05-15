from setuptools import setup, find_packages

setup(
    name="pyFBAT",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "scipy",
        "tqdm"
    ],
    description="A package for Python-based Family Based Association Tests",
    url="https://github.com/ssu19/pyFBAT"
)