cdef class CythonClass:
    cdef readonly double x

    cdef void cprint(self) noexcept

    cpdef double python_print(self)