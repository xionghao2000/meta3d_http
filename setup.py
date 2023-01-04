from setuptools import setup

setup(
    name="meta3d",
    packages=[
        "meta3d",
        "meta3d.common",
        "meta3d.services",
    ],
    install_requires=[
        "torch"
    ],
    author="meta3d_model",
    description="meta3d_model",
    long_description=open("README.md").read(),
    version="1.0.0"
)
