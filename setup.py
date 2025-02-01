from setuptools import setup, find_packages

setup(
    name="document_wrapper_adamllryan",
    version="0.1",
    packages="document_wrapper_adamllryan",
    install_requires=[
        "yt-dlp",
        'pyannote.audio',
        'ujson',
        'sentence_transformers',
        'moviepy',
        'spacy',
        'torch',
    ]
)
