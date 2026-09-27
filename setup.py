from   setuptools               import find_packages, setup


package_name = "thpoker"
package_dir = "./src/python"

setup(
    name=package_name,
    version="0.0.1",
    description="Texas Hold'em Poker Game",
    author="Panda Pan",
    author_email="panchongdan@gmail.com",
    packages=find_packages(where=package_dir),
    scripts=["bin/run_thpoker.py"],
    py_modules=[package_name],
    package_dir={package_name: f"{package_dir}/{package_name}"},
    package_data={package_name: ["data/*.json.gz", "web_static/*"]},
    install_requires=[
        "numpy>=2.5.0",
    ],  # some install requires example
    extras_require={"test": ["pytest>=8.3", "pytest-cov>=2.5.1", "pytest-sugar>=1.0.0"]},
    python_requires=">=3.14",
)
