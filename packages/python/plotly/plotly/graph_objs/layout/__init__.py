import sys

if sys.version_info < (3, 7):
    from ._activeshape import Activeshape
    from ._annotations import Annotations
    from ._coloraxis import Coloraxis
    from ._colorscale import Colorscale
    from ._font import Font
    from ._geo import Geo
    from ._grid import Grid
    from ._hoverlabel import Hoverlabel
    from ._images import Images
    from ._legend import Legend
    from ._mapbox import Mapbox
    from ._margin import Margin
    from ._modebar import Modebar
    from ._newshape import Newshape
    from ._polar import Polar
    from ._scene import Scene
    from ._shapes import Shapes
    from ._sliders import Sliders
    from ._template import Template
    from ._ternary import Ternary
    from ._title import Title
    from ._transition import Transition
    from ._uniformtext import Uniformtext
    from ._updatemenus import Updatemenus
    from ._xaxis import XAxis
    from ._yaxis import YAxis
    from . import coloraxis
    from . import geo
    from . import grid
    from . import hoverlabel
    from . import legend
    from . import mapbox
    from . import newshape
    from . import polar
    from . import scene
    from . import template
    from . import ternary
    from . import title
    from . import xaxis
    from . import yaxis
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__,
        [
            ".coloraxis",
            ".geo",
            ".grid",
            ".hoverlabel",
            ".legend",
            ".mapbox",
            ".newshape",
            ".polar",
            ".scene",
            ".template",
            ".ternary",
            ".title",
            ".xaxis",
            ".yaxis",
        ],
        [
            "._activeshape.Activeshape",
            "._annotations.Annotations",
            "._coloraxis.Coloraxis",
            "._colorscale.Colorscale",
            "._font.Font",
            "._geo.Geo",
            "._grid.Grid",
            "._hoverlabel.Hoverlabel",
            "._images.Images",
            "._legend.Legend",
            "._mapbox.Mapbox",
            "._margin.Margin",
            "._modebar.Modebar",
            "._newshape.Newshape",
            "._polar.Polar",
            "._scene.Scene",
            "._shapes.Shapes",
            "._sliders.Sliders",
            "._template.Template",
            "._ternary.Ternary",
            "._title.Title",
            "._transition.Transition",
            "._uniformtext.Uniformtext",
            "._updatemenus.Updatemenus",
            "._xaxis.XAxis",
            "._yaxis.YAxis",
        ],
    )
