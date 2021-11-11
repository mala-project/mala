"""
Compatibility layer for tests and examples.

Provide access to the MALA_DATA_REPO env var via the module level
`data_repo_path` attribute.

MALA_DATA_REPO should point to /path/to/test-data which is a checkout of
https://github.com/mala-project/test-data .
"""

import os

name = "MALA_DATA_REPO"
assert name in os.environ, f"Environment variable {name} not set."
data_repo_path = os.environ[name]
