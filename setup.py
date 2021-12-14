from setuptools import setup, find_packages

requirements = [
    "transformers>=4.12.5",
    "datasets>=1.15.1",
    "torch>=1.0"
]

setup(
    name="question_generation",
    version="0.0.3",
    author="algolet",
    author_email="wei.cai@algolet.com",
    description="Question Generation and Question Answering Pipeline",
    long_description=open("README_b.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/algolet/question_generation",
    package_dir={"": "question_generation"},
    packages=find_packages("question_generation"),
    install_requires=requirements,
    python_requires=">=3.6.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
  


