import os

# This env var should point to /path/to/test-data which is a checkout of
# https://github.com/mala-project/test-data . Only used in unit tests and
# examples.
name = "MALA_DATA_REPO"
assert name in os.environ, f"Environment variable {name} not set."
data_repo_path = os.environ[name]
