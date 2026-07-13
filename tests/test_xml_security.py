"""Regression tests for hardened SVG XML parsing."""

import pytest
from defusedxml.common import DefusedXmlException

from svgsmith.postprocess import postprocess, snap_background_layer, svg_bbox
from svgsmith.report import svg_stats
from svgsmith.smooth import smooth_svg

_SVG_WITH_ENTITY = """<!DOCTYPE svg [<!ENTITY unsafe "blocked">]>
<svg xmlns="http://www.w3.org/2000/svg"><text>&unsafe;</text></svg>"""


@pytest.mark.parametrize(
    "parse_svg",
    [postprocess, snap_background_layer, svg_bbox, svg_stats, smooth_svg],
)
def test_runtime_svg_parsers_reject_entity_declarations(parse_svg):
    with pytest.raises(DefusedXmlException):
        parse_svg(_SVG_WITH_ENTITY)
