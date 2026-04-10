# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:52:12 2026

@author: RonaldKurnik
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfx = pd.read_parquet('33CFB812_labeled_segment.parquet')
dfx2 = pd.read_parquet('33CFB812_labeled_segment_denoised.parquet')

plt.plot(dfx.timeInMS, dfx.magRLoadAdjusted)
plt.show()

plt.plot(dfx2.timeInMS, dfx2.magRLoadAdjusted)
plt.show()
