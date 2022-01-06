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
    install_requires=[
        "biopython>=1.79",
        "genomic_benchmarks>=0.0.6",
        "torch>=1.10.0"
    ],
)