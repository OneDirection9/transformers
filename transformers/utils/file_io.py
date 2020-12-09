from __future__ import absolute_import, division, print_function

from foundation.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathManagerBase

__all__ = ["PathManager"]

PathManager = PathManagerBase()

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
