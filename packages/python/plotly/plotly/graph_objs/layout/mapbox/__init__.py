import sys

if sys.version_info < (3, 7):
    from ._center import Center
    from ._domain import Domain
    from ._layers import Layers
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__, [], ["._center.Center", "._domain.Domain", "._layers.Layers"]
    )
