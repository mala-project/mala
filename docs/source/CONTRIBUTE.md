# Contributing to MALA

MALA is open source software and as such always welcomes additions and
improvements. However, we kindly ask any contributor to adhere to the following
suggestions in order to keep the overall quality and maintainability of the
code high.

## Versioning and releases

MALA has a versioning system. The version number is only updated when merging
on `master`. This constitues a release. Please note that not all changes
to code constitute such a release and generally, merges will be directed
to the `develop` branch
(see [branching strategy](#branching-strategy)). The version number has
the form `MAJOR.MINOR.FIX`:

* `MAJOR`: Big changes to the code, that fundamentally change the way it
  functions or wrap up a longer development.
* `MINOR`: new features have been added to the code.
* `FIX`: A bug in a feature has been fixed.

Every new version should be accompanied by a changelog. Please include the
version of the test data repository with which this version is supposed to be
used in this changelog.

## Branching strategy

In general, contributors should develop on branches based off of `develop` and
merge requests should be to `develop`. Please choose a descriptive branch name.
Branches from `develop` to `master` will be done after prior consultation of
the core development team.

## Developing code

* Regularly check your code for PEP8 compliance
* Make sure all your classes, functions etc. are documented properly,
  follow the
  [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
  for docstrings
* Keep your code object-oriented, modular, and easily reusable
* If you're adding code that should be tested, add tests
* If you're adding or modifying examples, make sure to add them to `test_examples.py`


## Pull Requests
We actively welcome pull requests.
1. Fork the repo and create your branch from `develop`
2. During development, make sure that you follow the guidelines for [developing code](#developing-code)
3. Rebase your branch onto `develop` before submitting a merge request
4. Ensure the test suite passes before submitting a pull request

## Issues

* Use issues to document potential enhancements, bugs and such
* Please tag your issues, and consider setting up a deadline
* Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue

## License
By contributing to MALA, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

