# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:52:12 2026

@author: RonaldKurnik
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfx1 = pd.read_parquet('ResistorOnCatheter.parquet')
if 'timestamp_ms' in dfx1.columns:
    dfx1 = dfx1[dfx1.timestamp_ms >= 1500]
    
dfx2 = dfx1.copy()
dfx3 = dfx1.copy()
col_1 = 'imp_mag_adj_1_ohm' 
col_2 = 'imp_mag_adj_2_ohm'

if col_1 in dfx2.columns:
    dfx2 = dfx2[dfx2[col_1] > 0]

if col_2 in dfx3.columns:
    dfx3 = dfx3[dfx3[col_2] > 0]

plt.figure()
plt.plot(dfx1.timestamp_ms/1000, dfx1.imp_mag_adj_0_ohm)
plt.title("Resistor Impedance at 50 kHz")
plt.show()

plt.figure()
plt.plot(dfx2.timestamp_ms/1000, dfx2.imp_mag_adj_1_ohm)
plt.title("Resistor Impedance at 100 kHz")
plt.show()

plt.figure()
plt.plot(dfx3.timestamp_ms/1000, dfx3.imp_mag_adj_2_ohm)
plt.title("Resistor Impedance at 12.5 kHz")
plt.show()

