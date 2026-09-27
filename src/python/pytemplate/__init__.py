"""
PyTemplate - A modern Python template repository with Cython extensions.

This package provides a solid foundation for Python packages that need to integrate
with C++ code through Cython for high-performance computing.
"""

__version__ = "0.1.0"
__author__ = "Panda Pan"
__email__ = "panchongdan@gmail.com"
__license__ = "MIT"

# Import main classes and functions
try:
    from .template_class import TemplateClass
    from .cy import CythonClass
    from .vector import vec
except ImportError as e:
    # Handle cases where Cython extensions haven't been built
    import warnings
    warnings.warn(f"Could not import Cython extensions: {e}. "
                  "Run 'python setup.py build_ext --inplace' to build them.")

# Define what gets imported with "from pytemplate import *"
__all__ = [
    "TemplateClass",
    "CythonClass", 
    "vec",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Package metadata
__package_info__ = {
    "name": "pytemplate",
    "version": __version__,
    "description": "A modern Python template repository with Cython extensions",
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "python_requires": ">=3.11",
    "keywords": ["python", "template", "cython", "cpp"],
}
