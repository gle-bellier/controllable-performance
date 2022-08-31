import setuptools

setuptools.setup(
    name="controllable-performance",
    version="0.0.1",
    author="Le Bellier Georges",
    author_email="georges.lebellier@sony.com",
    description=
    "Package for expressive musical performances generation with diffusion models.",
    project_urls={
        "Bug Tracker":
        "https://github.com/gle-bellier/controllable-performance/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)