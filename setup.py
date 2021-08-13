from setuptools import setup

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

setup(
    name="mala",
    version=version["__version__"],
    description="Framework for Electronic Structure Learning",
    long_description=readme,
    url="https://gitlab.com/hzdr/mala/mala",
    author="Lenz Fiedler et al.",
    author_email="l.fiedler@hzdr.de",
    license=license,
    packages=["mala"],
    zip_safe=False,
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='<3.9',
)
