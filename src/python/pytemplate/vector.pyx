# cython: cpp_locals=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++20

from libcpp.vector cimport vector
from libcpp.string cimport string


cpdef void vec():
    """
    Example C++ vector function that demonstrates libcpp usage.
    
    Creates a C++ vector of strings, adds an element, and prints it.
    """
    cdef vector[string] v = vector[string]()
    v.push_back(string(b'vec'))
    print(v[0].c_str())
