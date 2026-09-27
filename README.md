# pytemplate

## Introduction
A modern Python + Cython project template, suitable for developing high-performance Python packages with C/C++ extensions.

## Installation
```bash
pip install -e .
```

## Build Cython Extensions
```bash
python setup.py build_ext --inplace
```

## Run Tests
```bash
pytest
```

## Dependencies
- pandas >= 1.4.0
- Cython

## Build Documentation (if doc/ exists)
```bash
cd doc
make html
```

## Project Structure
```
src/python/pytemplate/        # Main package and Cython extensions
src/python/pytemplate/tests/  # Test cases
setup.py                      # Build script
pyproject.toml                # Build configuration
requirements.txt              # Development dependencies
README.md                     # Project documentation
```

## Contributing
Contributions are welcome! Please open issues or pull requests.

