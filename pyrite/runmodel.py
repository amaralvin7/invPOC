#!/usr/bin/env python3
from constants import ZG, MLD, MMC, DPY, GRID
import data.data as data

all_data = data.load_data()

poc_data = data.process_poc_data(all_data['POC'])

print(poc_data)