import numpy as np

from .lib.autowoe import AutoWoE
from .lib.report.report import ReportDeco


__all__ = ["AutoWoE", "ReportDeco"]

__version__ = "1.2.4"

np.random.seed(42)
