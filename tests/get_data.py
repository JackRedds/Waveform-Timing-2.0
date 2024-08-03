#!/usr/bin/env python3

from wave_timing.utils import get_data
import pandas as pd

start_date = '2021-01-19'
end_date = '2021-01-20'

data = get_data(start_date, end_date)

vac_data = data.get_vac_data()

print(vac_data)

mag_data = data.get_mag_data()

print(mag_data)
