"""
Compatibility layer for tests and examples.

Provide access to the ``MALA_DATA_REPO`` var via the module level
`data_repo_path` attribute.

``MALA_DATA_REPO`` should point to /path/to/test-data which is a checkout of
https://github.com/mala-project/test-data.
"""

import os
import warnings

name = "MALA_DATA_REPO"
if name in os.environ:
    data_repo_path = os.environ[name]
else:
    warnings.warn(
        f"Environment variable {name} not set. You won't be able "
        "to run all examples and tests."
    )
    data_repo_path = None
