# from . import config
# import confuse
# config = confuse.Configuration('wmodel', __name__)
import numpy as np

from .lib.autowoe import AutoWoE
from .lib.report.report import ReportDeco

__all__ = ["AutoWoE", "ReportDeco"]

__version__ = '1.1'

np.random.seed(42)


