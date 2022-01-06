from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='glp',
    version="0.0.1",
    description="Genomic Language Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Katarina Gresova",
    author_email="gresova11@gmail.com",
    extras_require=dict(tests=['pytest']),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    keywords=["bioinformatics", "genomics", "nlp"],
    license="Apache License 2.0",
    url="https://github.com/katarinagresova/GLP",
    install_requires=[
        "biopython>=1.79",
        "genomic_benchmarks>=0.0.6",
        "torch>=1.10.0",
        "torchtext>=0.11.1"
    ],
    classifiers=[
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Development Status :: 3 - Alpha",
        # Define that your audience are developers
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        # Specify which pyhton versions that you want to support
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)