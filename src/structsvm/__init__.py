from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("structsvm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"


from .bundle_method import BundleMethod
from .hamming_costs import HammingCosts
from .soft_margin_loss import SoftMarginLoss

__all__ = ["BundleMethod", "HammingCosts", "SoftMarginLoss", "__version__"]
