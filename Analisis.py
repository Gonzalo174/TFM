# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:08:50 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import oiffile as of
import os

#%%

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = "Times New Roman"

#%%

cs10 = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 22, 25 ] 
ss10 = [14, 15, 17, 18, 26, 27, 28, 29]

cs7 =  [ (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss7 =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]

#%%

pre, post, mascara, mascara10, pixel_size = celula( cs7[1], 'MCF7', place = 'home' )
# plt.imshow(pre)































