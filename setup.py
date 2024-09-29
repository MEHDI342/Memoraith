from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="memoraith",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight model profiler for deep learning frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/memoraith",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.7.0',
        'tensorflow>=2.4.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pdfkit>=0.6.0',
        'aiofiles>=0.6.0',
        'jinja2>=2.11.0',
        'pynvml>=8.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-asyncio>=0.14.0',
            'black>=20.8b1',
            'isort>=5.7.0',
            'flake8>=3.8.0',
        ],
    },
)