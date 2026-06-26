from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fast_kmeans",
    version="1.0.0",
    author="Ivan Kholodilo",
    author_email="iholodilo2008@gmail.com",
    description="High-performance, GPU-accelerated Scalable Prototype-based Machine Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Icold21/FastKMeans",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scikit-learn",
        "tqdm"
    ],
    extras_require={
        "full": ["faiss-cpu", "matplotlib", "ipython"]
    }
)