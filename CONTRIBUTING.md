# Contributing to Memoraith

We welcome contributions to Memoraith! This document provides guidelines for contributing to the project.

## Setting up the development environment

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/memoraith.git
   cd memoraith
   ```
3. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

## Making Changes

1. Create a new branch for your changes:
   ```
   git checkout -b your-feature-branch
   ```
2. Make your changes and commit them with a clear commit message.
3. Push your changes to your fork on GitHub:
   ```
   git push origin your-feature-branch
   ```
4. Open a pull request from your fork to the main Memoraith repository.

## Coding Standards

- Follow PEP 8 guidelines for Python code.
- Use type hints for all function arguments and return values.
- Write clear, concise docstrings for all classes and functions.
- Add unit tests for new functionality.

## Running Tests

Run the test suite using pytest:

```
pytest
