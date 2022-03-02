import setuptools
import subprocess
import os

rgc_version = (
    subprocess.run(["git", "describe", "--tags"], stdout=subprocess.PIPE)
    .stdout.decode("utf-8")
    .strip()
)

assert "." in rgc_version

assert os.path.isfile("src/version.py")
with open("src/VERSION", "w", encoding="utf-8") as fh:
    fh.write(f"{rgc_version}\n")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rgc",
    version=rgc_version,
    author="Astronomy Research Group, IUB (ARGI)",
    author_email="arg@iub.edu.bd",
    description="RGC: Radio Galaxy Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/argiub/rgc",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={"rgc": ["VERSION"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    project_urls={
        "Bug Tracker": "https://github.com/argiub/rgc/issues",
    },
    install_requires=[

    ],
)