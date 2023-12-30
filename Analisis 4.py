# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 16:54:56 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
import oiffile as of
import os

import TFM
import auxi

from matplotlib.colors import ListedColormap

cm_crimson = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255)) for i in range(1000)] )
cm_green = ListedColormap( [(0,128*i/(999*255),0) for i in range(1000)] )
cm_yellow = ListedColormap( [( (220*i/(999*255)),128*i/(999*255),0) for i in range(1000)] )
cm_y = ListedColormap( [(1, 1, 1), (1, 1, 0)] )   # Blanco - Amarillo
cm_ar = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (0.839, 0.152, 0.157)] ) 
cm_aa = ListedColormap( [(0.122, 0.467, 0.706), (1, 1, 1), (1.000, 0.498, 0.055)] ) 
cm_aa2 = ListedColormap( [(0.122, 0.467, 0.706), (0, 0, 0), (1.000, 0.498, 0.055)] ) 

c0 = (0.122, 0.467, 0.706)
c1 = (1.000, 0.498, 0.055)
c2 = (0.173, 0.627, 0.173)
c3 = (0.839, 0.152, 0.157)
colores = [c2, c3, c0, c1]

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )
color_maps = [cm2, cm3, cm0, cm1]

#%% Parámetros de ploteo

plt.rcParams['figure.figsize'] = [7,7]
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = "Times New Roman"

#%% Células

cs10 = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 22, 25 ] 
ss10 = [14, 15, 17, 18, 26, 27, 28, 29]
cs7 =  [ (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss7 =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
conjuntos = [cs10, ss10, cs7, ss7]


muestra = [ 9, 25, (11,4), (10,5) ]
linea_muestra = [ 'MCF10', 'MCF10', 'MCF7', 'MCF7' ]

# valores de exploración?

#%%

data0 = auxi.celula( muestra[0], linea_muestra[0], place = 'home' )
data1 = auxi.celula( muestra[1], linea_muestra[1], place = 'home' )
data2 = auxi.celula( muestra[2], linea_muestra[2], place = 'home' )
data3 = auxi.celula( muestra[3], linea_muestra[3], place = 'home' )
data_cels = [data0, data1, data2, data3]

#%%

for i in range(4):
    
    pre, post, mascara, mascara10, mascara20, ps = data_cels[i]
    
    ws, exp = 2.5, 0.6
    if i == 1:
        exp = 1
    
    A1 = auxi.busca_A( pre, 0.75, ps, win = 2.5, A0 = 0.85 )
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
    X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )

    E, nu, lam = 31.6, 0.5, -1   
    # uy, ux = Y_s, X_s
    uy, ux = TFM.smooth( Y_nmt*auxi.reshape_mask(mascara20, x, y), 3 ), TFM.smooth( X_nmt*auxi.reshape_mask(mascara20, x, y), 3 )
    x_plot, y_plot = x, y
    
    ty, tx, vy0, vx0 = TFM.traction(uy, ux, ps*1e-6, ws*1e-6, E*1e3, nu, lam, Lcurve = True)
    ty, tx = TFM.smooth(ty,3), TFM.smooth(tx,3)
    vy0, vx0 = TFM.deformation( np.real(ty), np.real(tx), ws*1e-6, E*1e3, nu )

    duvy0, duvx0 = uy*ps*1e-6 - vy0, ux*ps*1e-6 - vx0

    plt.figure( figsize = [7,7], layout = 'compressed' )

    # Solucion T
    plt.subplot(2,2,1)
    # plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
    plt.quiver(x_plot, y_plot, tx*auxi.reshape_mask(mascara10, x, y), -ty*auxi.reshape_mask(mascara10, x, y), scale = 20000)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '12', color = 'k', more_text = 'T' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Comprobacion V
    plt.subplot(2,2,2)
    plt.quiver(x_plot, y_plot, vx0, -vy0, scale = 0.00001)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'V' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Entrada U
    plt.subplot(2,2,3)
    plt.quiver(x_plot, y_plot, ux*ps*1e-6, -uy*ps*1e-6, scale = 0.00001)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'U'  )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Diferencia U - V
    plt.subplot(2,2,4)
    plt.quiver(x_plot, y_plot, duvx0, -duvy0, scale = 0.00001)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'U-V' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    plt.show()

















