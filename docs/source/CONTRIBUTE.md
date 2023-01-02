# Contributing to MALA

MALA is open source software and as such always welcomes additions and
improvements. However, we kindly ask any contributor to adhere to the following
suggestions in order to keep the overall quality and maintainability of the
code high.

## Versioning and releases

MALA has a versioning system. The version number is only updated when merging
on `master`. This constitutes a release. Please note that not all changes
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

### Creating a release

In order to correctly update the MALA version, we use 
[bumpversion](https://github.com/peritus/bumpversion). The actual release 
process is very straightforward:

1. Create a PR from `develop` to `master`.
2. Merge the PR.
3. Update the `date-released: ...` entry in `CITATION.cff` (on `master`).
4. Create a tagged (and signed) commit on `master` with `bumpversion minor --allow-dirty` (check changes with `git show` or `git diff HEAD^`). Use either `major`, `minor` or `fix`, depending on what this release updates.
5. Check out `develop` and do a `git merge master --ff`
6. Push `master` and `develop` including tags (`--tags`). 
7. Create a new release out of the tag on GitHub (https://github.com/mala-project/mala/releases/new) and add release notes/change log.
8. Check if release got published to PyPI.

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

### Adding dependencies

If you add additional dependencies, make sure to add them to `requirements.txt`
if they are required or to `setup.py` under the appropriate `extras` tag if 
they are not. 
Further, in order for them to be available during the CI tests, make sure to 
add them in the `Dockerfile` for the `conda` environment build.


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

