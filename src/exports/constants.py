"""Constants used for for EXPORTS inversions."""
from numpy import diff

MLD = 30  # mixed layer depth
ZG = 100  # grazing zone depth
GRID = (30, 50, 100, 150, 200, 330, 500)
UMZ_START = GRID.index(ZG) + 1
LAYERS = tuple(range(len(GRID)))
ZONE_LAYERS = ('EZ', 'UMZ') + LAYERS
THICK = diff((0,) + GRID)
