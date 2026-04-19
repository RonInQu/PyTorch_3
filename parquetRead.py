# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:52:12 2026

@author: RonaldKurnik
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dfx2 = pd.read_parquet('test2.parquet')

plt.figure()
# plt.plot(dfx2.timeInMS/1000, dfx2.label)
# plt.plot(dfx2.time_sec, dfx2.label)
plt.plot(dfx2.time_sec, dfx2.curr_led_state)
plt.title("label")
plt.show()

plt.figure()
# plt.plot(dfx2.timeInMS/1000, dfx2.da_label)
plt.plot(dfx2.time_sec, dfx2.event_type_1)
plt.title("da_label")
plt.show()

plt.figure()
# plt.plot(dfx2.timeInMS/1000, dfx2.magRLoadAdjusted)
plt.plot(dfx2.time_sec, dfx2.imp)
plt.title("magRLoadAdjusted")
plt.show()


dfx = pd.read_parquet('test.parquet')

plt.figure()
plt.plot(dfx.time_sec, dfx.curr_led_state)
plt.title("curr_led_state")
plt.show()

plt.figure()
plt.plot(dfx.time_sec, dfx.event_type_1)
plt.title("event_type_1")
plt.show()

plt.figure()
plt.plot(dfx.time_sec, dfx.imp-dfx.blood_baseline)
plt.show()
