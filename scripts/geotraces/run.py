#!/usr/bin/env python3
import time

import src.geotraces.data as data

start_time = time.time()

poc_data = data.load_poc_data()

Lp_priors, Lp_error = data.get_Lp_priors(poc_data)
Po_priors = data.get_Po_priors(Lp_priors, Lp_error)

