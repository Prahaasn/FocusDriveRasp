"""
FocusDrive setup script.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = []

setup(
    name="focusdrive",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-time driver distraction detection system powered by LFM2-VL-1.6B",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/focusdrive",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "focusdrive-train=train:main",
            "focusdrive-demo=demo:main",
            "focusdrive-download=src.data.download_dataset:main",
            "focusdrive-preprocess=src.data.preprocess:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
