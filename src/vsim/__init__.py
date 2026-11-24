"""vsim: visual similarity search and clustering over a folder of images."""

__version__ = "0.1.0"

from .clustering import Assignment, assign, preprocess
from .search import Result, build_index, search

__all__ = ["Assignment", "Result", "__version__", "assign", "build_index", "preprocess", "search"]
