from setuptools import setup, find_packages

# Doing it as suggested here:
# https://packaging.python.org/guides/single-sourcing-package-version/
# (number 3)

version = {}
with open("mala/version.py") as fp:
    exec(fp.read(), version)

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

extras = {
    'dev': ['bump2version'],
    'opt': ['oapackage'],
    'test': ['pytest'],
    'doc': open('docs/requirements.txt').read().splitlines(),
    'experimental': ['asap3', 'dftpy', 'minterpy']
}

setup(
    name="materials-learning-algorithms",
    version=version["__version__"],
    description=("Materials Learning Algorithms. "
                 "A framework for machine learning materials properties from "
                 "first-principles data."),
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/mala-project/mala",
    author="MALA developers",
    license=license,
    packages=find_packages(exclude=("test", "docs", "examples", "install",
                                    "ml-dft-sandia")),
    zip_safe=False,
    install_requires=open('requirements.txt').read().splitlines(),
    extras_require=extras,
)
