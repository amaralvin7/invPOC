#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

ZG = 100  # grazing zone depth
MLD = 30  # mixed layer depth
MMC = 12.011  # molar mass of carbon
DPY = 365.24  # days per year 
GRID = (30, 50, 100, 150, 200, 330, 500)
LAYERS = tuple(range(len(GRID)))
THICK = np.diff((0,) + GRID)
RE = 0.5  # relative error for (some) priors
GAMMA = 0.5