from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="memoraith",
    version="0.5.0",  # Updated to match __init__.py
    author="Mehdi El Jouhfi",
    author_email="midojouhfi@gmail.com",
    description="Advanced lightweight model profiler for deep learning frameworks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mehdi342/Memoraith",
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.7.0',
        'tensorflow>=2.4.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'plotly>=4.14.0',
        'pandas>=1.2.0',
        'jinja2>=2.11.0',
        'pdfkit>=0.6.0',
        'psutil>=5.8.0',
        'pynvml>=8.0.0',
        'colorama>=0.4.4',
        'tqdm>=4.60.0',
        'aiofiles>=0.6.0',
        'asyncio>=3.4.3',
        'networkx>=2.5',
        'python-dotenv>=0.19.0',
        'pyyaml>=5.4.0'
    ],
    extras_require={
        'full': [
            'torch>=1.7.0',
            'tensorflow>=2.4.0',
            'tensorboard>=2.4.0',
            'optuna>=2.3.0',
            'ray>=1.2.0',
        ],
        'dev': [
            'pytest>=6.2.0',
            'pytest-asyncio>=0.14.0',
            'black>=20.8b1',
            'isort>=5.7.0',
            'flake8>=3.8.0',
            'mypy>=0.800',
            'tox>=3.20.0',
            'sphinx>=3.4.3',
            'sphinx-rtd-theme>=0.5.1',
        ],
    },
    entry_points={
        'console_scripts': [
            'memoraith=memoraith.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        'Bug Reports': 'https://github.com/mehdi342/Memoraith/issues',
        'Source': 'https://github.com/mehdi342/Memoraith/',
    },
)
