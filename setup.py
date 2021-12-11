from setuptools import setup, find_packages

setup(
    name="question_generation",
    version="0.0.1",
    author="algolet",
    author_email="wei.cai@algolet.com",
    description="Question Generation and Question Answering Pipeline",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/algolet/question_generation",
    package_dir={"": "qg"},
    packages=find_packages("qg"),
    python_requires=">=3.6.0",
)
  


