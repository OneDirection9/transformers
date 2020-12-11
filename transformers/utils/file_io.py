from __future__ import absolute_import, division, print_function

from foundation.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathManagerFactory

__all__ = ["PathManager"]

# Get per-project PathManager
PathManager = PathManagerFactory.get("transformers_path_manager")

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
