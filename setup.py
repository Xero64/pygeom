import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygeom",
    version="0.0.12",
    author="Xero64",
    author_email="xero64@gmail.com",
    description="Geometry Definition and Calculation Package for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xero64/pygeom",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires='>=3.7'
)
