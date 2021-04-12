## Contributing to MALA

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
* `MINOR`: new features have beend added to the code.
* `FIX`: A bug in a feature has been fixed. 


## Branching strategy

In general, contributors should develop on branches based off of `develop` and
merge requests should be to `develop`. Please choose a descriptive branch name,
ideally incorporating some identifying information (such as your initials)
or the starting date of your developments. Branches from `develop` to `master`
will be done after prior consultation of the core development team.

## Developing code

* Regularly check your code for PEP8 compliance
* Make sure all your classes, functions etc. are documented properly, 
  follow the 
  [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) 
  for docstrings
* Keep your code object-oriented, modular, and easily reusable
* Add tests and examples for sizable new features

## Submitting a merge request

* Ensure that you followed the guidelines for [developing code](#developing-code)
* Rebase your branch onto `develop` before submitting a merge request
* Only assign yourself to a merge request when time does not permit an 
  external check

## Issues

* Use issues to document potential enhancements, bugs and such
* Please tag your issues, and consider setting up a deadline 

