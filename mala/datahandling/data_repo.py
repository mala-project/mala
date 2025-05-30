"""
Compatibility layer for tests and examples.

Provide access to the ``MALA_DATA_REPO`` var via the module level
`data_repo_path` attribute.

``MALA_DATA_REPO`` should point to /path/to/test-data which is a checkout of
https://github.com/mala-project/test-data.
"""

import os
from mala.common.parallelizer import parallel_warn

name = "MALA_DATA_REPO"
if name in os.environ:
    data_repo_path = os.environ[name]
    data_path_be = os.path.join(data_repo_path, "Be2")
    data_path_bao = os.path.join(data_repo_path, "Ba4O4")

    # For legacy purposes.
    data_path = data_path_be

else:
    parallel_warn(
        f"Environment variable {name} not set. You won't be able "
        "to run all examples and tests."
    )
    data_repo_path = None
    data_path_be = None
    data_path_bao = None
