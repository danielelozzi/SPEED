# setup.py
from setuptools import setup, find_packages

setup(
    name="speed-analyzer",
    version="3.7.0", # Versione aggiornata
    author="Daniele Lozzi, LabSCoC",
    description="A package for processing and extracting eye-tracking data.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",

    url="https://github.com/danielelozzi/SPEED",
    project_urls={
        "Homepage": "https://github.com/danielelozzi/SPEED",
        "Source": "https://github.com/danielelozzi/SPEED",
        "Bug Tracker": "https://github.com/danielelozzi/SPEED/issues",
    },

    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "pandas", "numpy", "matplotlib", "opencv-python",
        "scipy", "tqdm", "moviepy", "ultralytics", "Pillow"
    ]
)
