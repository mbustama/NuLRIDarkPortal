# Land outlines for the schematic of the Earth

`land_110m_decimated.json`: the outlines of the land, as `[[[longitude, latitude], ...], ...]` in degrees, one ring
per polygon: Natural Earth's 1:110m land polygons (<https://www.naturalearthdata.com>, public domain), reduced to
their outer rings and decimated to 30 polygons and 1993 points. Copied unchanged on 2026-10-14 from the user's
GeoneutrinoDecay repository (`input/naturalearth/`, itself copied from the Mag(nu)s repository), so that the globe in
this paper's schematic matches theirs. Nothing was downloaded.

Read by `tools/schematic_dark_portal.py` (`globe_rings`).
