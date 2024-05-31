# Contributions

MALA is an open-source software and is built upon the collaborative efforts of
many contributors. The MALA team warmly welcomes additional contributions and
kindly requests potential contributors to follow the suggested guidelines below
to ensure the code's overall quality and maintainability.

## MALA contributors

Many people have made valuable contributions to MALA, and we are immensely
grateful for their support.
If you decide to contribute to MALA, please add your name to the following
alphabetically ordered list of contributors and include a note of the
nature of your contribution:


- Bartosz Brzoza (Bugfixes, GNN implementation)
- Timothy Callow (Grid-size transferability)
- Attila Cangi (Scientific supervision)
- Austin Ellis (General code infrastructure)
- Omar Faruk (Training parallelization via horovod)
- Lenz Fiedler (General code development and maintenance)
- James Fox (GNN implementation)
- Nils Hoffmann (NASWOT method)
- Kyle Miller (Data shuffling)
- Daniel Kotik (Documentation and CI)
- Somashekhar Kulkarni (Uncertainty quantification)
- Normand Modine (Total energy module and parallelization)
- Parvez Mohammed (OAT method)
- Vladyslav Oles (Hyperparameter optimization)
- Gabriel Popoola (Parallelization)
- Franz Pöschel (OpenPMD interface)
- Siva Rajamanickam (Scientific supervision)
- Josh Romero (GPU usage improvement for model tuning)
- Steve Schmerler (Uncertainty quantification)
- Adam Stephens (Uncertainty quantification work)
- Hossein Tahmasbi (Minterpy descriptors)
- Aidan Thompson (Descriptor calculation)
- Sneha Verma (Tensorboard interface)
- Jon Vogel (Inference parallelization)

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

### Formatting code

* MALA uses [`black`](https://github.com/psf/black) for code formatting
* The `black` configuration is located in `pyproject.toml`, the `black` version
  is specified in `.pre-commit-config.yaml`
* Currently, no automatic code reformatting will be done in the CI, thus
  please ensure that your code is properly formatted before creating a pull
  request. We suggest to use [`pre-commit`](https://pre-commit.com/). You can

  * manually run `pre-commit run -a` at any given time
  * configure it to run before each commit by executing `pre-commit install`
    once locally

  Without `pre-commit`, please install the `black` version named in
  `.pre-commit-config.yaml` and run `find -name "*.py" | xargs black` or just
  `black my_modified_file.py`.

### Adding dependencies

If you add additional dependencies, make sure to add them to `requirements.txt`
if they are required or to `setup.py` under the appropriate `extras` tag if
they are not.
Further, in order for them to be available during the CI tests, make sure to
add _required_ dependencies to the appropriate environment files in folder `install/` and _extra_ requirements directly in the `Dockerfile` for the `conda` environment build.

## Pull Requests
We actively welcome pull requests.
1. Fork the repo and create your branch from `develop`
2. During development, make sure that you follow the guidelines for [developing code](#developing-code)
3. Rebase your branch onto `develop` before submitting a pull request
4. Ensure the test suite passes before submitting a pull request

```{note}
The test suite workflows are not triggered for draft pull requests in order to avoid expensive multiple runs.
As soon as a pull request is marked as *ready to review*, the test suite is run through.
If the pipeline fails, one should return to a draft pull request, fix the problems, mark it as ready again
and repeat the steps if necessary.
```

## Issues

* Use issues to document potential enhancements, bugs and such
* Please tag your issues, and consider setting up a deadline
* Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue

## License
By contributing to MALA, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.

