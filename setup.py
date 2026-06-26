from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
VERSION = {}
exec((ROOT / "averagers" / "_version.py").read_text(encoding="utf-8"), VERSION)

setup(
    name="averagers",
    version=VERSION["__version__"],
    description="Tools for mean temperature estimation",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Kenji Fukushima",
    author_email="kfuku52@gmail.com",
    url="https://github.com/kfuku52/averagers.git",
    keywords="temperature meteorology photoperiod",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy", "pandas", "ephem", "matplotlib"],
    extras_require={
        "test": ["pytest"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
