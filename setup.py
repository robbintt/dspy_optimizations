from setuptools import setup, find_packages

setup(
    name="mdap_lib",
    version="0.1.0",
    packages=find_packages(where="lib"),
    package_dir={"": "lib"},
    install_path=".",
    install_requires=[
        "flask",
    ],
)
