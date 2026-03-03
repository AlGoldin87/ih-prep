"""ih-prep: Data preparation for information-theoretic analysis.

This library prepares pandas DataFrames for use with ih-lib.
It handles categorical encoding and quantitative discretization.
"""

from .core import prepare_data

__all__ = ["prepare_data"]
__version__ = "0.1.0"