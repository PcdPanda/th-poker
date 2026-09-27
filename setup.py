from __future__ import annotations

from   Cython.Build             import cythonize
import glob
import os
from   setuptools               import Extension, find_packages, setup

package_name = "pytemplate"
package_dir = "src/python"
pyx_files = glob.glob("src/python/**/*.pyx", recursive=True)

# 合并 .pyx 和 .pxd 文件
ext_modules = [
    Extension(
        name=file.replace("src/python/", "").replace("/", ".").replace("\\", ".").replace(".pyx", ""),
        sources=[file],
        language="c++",
        extra_compile_args=["-std=c++20"],
        extra_link_args=["-std=c++20"],
    )
    for file in pyx_files
]
os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"


setup(
    name=package_name,
    version="0.0.1",
    description="Python Template",
    author="Panda Pan",
    author_email="panchongdan@gmail.com",
    packages=find_packages(where=package_dir),
    py_modules=[package_name],
    package_dir={package_name: f"{package_dir}/{package_name}"},
    package_data={package_name: ["lib/*.so"]},
    install_requires=["pandas>=1.4.0", "Cython"],  # some install requires example
    tests_require=[
        "pytest>=3.3.1",
        "pytest-cov>=2.5.1",
        "pytest-sugar>=1.0.0",
    ],
    extras_require=dict(dev=["pytest>=8.0", "pytest-cov>=2.5.1", "pytest-sugar>=1.0.0"]),
    python_requires=">=3.11",
    ext_modules=cythonize(ext_modules, compiler_directives=dict(language_level=3, cpp_locals=True)),
)
