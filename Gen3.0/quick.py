# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:52:12 2026

@author: RonaldKurnik
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
name = "2026-05-12 206-104 Promedica_state.parquet"
# name = "20260430_LOG4_solo.parquet"
# df = pd.read_csv("LOG4_state.csv")

# Save as a Parquet file
# df.to_parquet("name", index=False)

dfx2 = pd.read_parquet(name)
time = dfx2.timestamp_ms/1000
# im_0 = dfx2.imp_mag_adj_0_ohm
# im_1 = dfx2.imp_mag_adj_1_ohm
# im_2 = dfx2.imp_mag_adj_2_ohm

# plt.figure(0)
# plt.plot(time,im_0)
# plt.show()

# plt.figure(1)
# plt.plot(time,im_1)
# plt.show()

# plt.figure(2)
# plt.plot(time,im_2)
# plt.show()

lbl_0 = dfx2.cms_led_state_i
plt.figure()
plt.plot(time,lbl_0)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout='tight')

# 3. Plot on the first subplot (Row 0)
axs[0].plot(time, dfx2.han_state_i, color='blue')
axs[0].set_title('Label')
axs[0].set_xlabel('time')
axs[0].set_ylabel('label')

# 4. Plot on the second subplot (Row 1)
axs[1].plot(time, dfx2.han_pressure_mmhg, color='red')
axs[1].set_title('Pressure')
axs[1].set_xlabel('time')
axs[1].set_ylabel('Pressure, mmHg')

# 5. Display the plot
plt.show()