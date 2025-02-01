from setuptools import setup, find_packages

setup(
    name="document_wrapper_adamllryan",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "yt-dlp",
        "rouge"
    ]
)
