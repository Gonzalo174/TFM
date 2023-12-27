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

import TFM
import aux

#%%

plt.rcParams['figure.figsize'] = [10,10]
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = "Times New Roman"

#%%
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
colores = [c0, c1, c2, c3]

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )
color_maps = [cm0, cm1, cm2, cm3]

#%% 

cs10 = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 22, 25 ] 
ss10 = [14, 15, 17, 18, 26, 27, 28, 29]
cs7 =  [ (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss7 =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
listas = [cs10, ss10, cs7, ss7]

muestra = [ (11,4), (10,5), 9, 25 ]
linea_muestra = [ 'MCF7', 'MCF7', 'MCF10', 'MCF10' ]

# valores de A?
# valores de exploración?

#%%
pre, post, mascara, mascara10, ps = aux.celula( (1,1), 'MCF7', place = 'home' )

ws = 2.5
bordes_extra = 4

dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), it, exploration = bordes_extra, mode = "Smooth3", A = 0.75)
Y_0, X_0 = deformacion 
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)    # NMT para el ángulo?
X_s, Y_s = TFM.smooth(  X_nmt*reshape_mask(mascara10, x, y)  ,3), TFM.smooth(  Y_nmt*reshape_mask(mascara10, x, y)  ,3)


plt.figure(figsize = [6,4] )
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )

# plt.quiver(x,y,X_0,-Y_0, res, cmap = cm_crimson, scale = 100, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, scale = 100, pivot='tail')
plt.quiver(x,y,X_s,-Y_s, scale = 100, pivot='tail')

aux.barra_de_escala( 10, sep = 1.5,  font_size = 11, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.show()

#%% Resolucion del problema inverso
nu = 0.5
E = 31.6      # kPa

# lam = 5*1e-20
# uX, uY = X_s, Y_s
# x_plot, y_plot = x, y

lam = 0
uX, uY = TFM.smooth(TFM.four_core(X_s*ps),2), TFM.smooth(TFM.four_core(Y_s*ps),2) 
x_plot, y_plot = np.meshgrid( np.arange( x[0,0], x[-1,-1] + (x[0,1] - x[0,0]), (x[0,1] - x[0,0])/2 ), np.arange( x[0,0], x[-1,-1] + (x[0,1] - x[0,0]), (x[0,1] - x[0,0])/2 )  )

# ty, tx = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam)
ty, tx, vy0, vx0 = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam, Lcurve = True)

duvy0, duvx0 = uY*ps*1e-6 - vy0, uX*ps*1e-6 - vx0

#%%

plt.quiver(x_plot, y_plot, TFM.smooth(tx,5), -TFM.smooth(ty,5), scale = 2000)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = '11', color = 'k', more_text = 'T' )
plt.xlim([0,1023])
plt.ylim([1023,0])


#%%
plt.figure( figsize = [7,7], layout = 'compressed' )

# Solucion
plt.subplot(2,2,1)
plt.quiver(x_plot, y_plot, tx, -ty, scale = 2000)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = '11', color = 'k', more_text = 'T' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Comprobacion
plt.subplot(2,2,2)
plt.quiver(x_plot, y_plot, vx0, -vy0, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = '11', color = 'k', more_text = 'V' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Entrada
plt.subplot(2,2,3)
plt.quiver(x_plot, y_plot, uX*ps*1e-6, -uY*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = '11', color = 'k', more_text = 'U'  )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Diferencia
plt.subplot(2,2,4)
plt.quiver(x_plot, y_plot, duvx0, -duvy0, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
barra_de_escala( 10, sep = 1.5,  font_size = '11', color = 'k', more_text = 'U-V' )
plt.xlim([0,1023])
plt.ylim([1023,0])

plt.show()

#%% Exploracion del espacio de lambda
# norm = np.linalg.norm()

nu = 0.5
E = 31.6      # kPa

# uX, uY = smooth(four_core(X_s*ps),2), smooth(four_core(Y_s*ps),2) 
uX, uY = X_s, Y_s

N = 1000
lam_list = np.logspace(-30, -10, N )
tr_list = np.zeros( N )
duvr0_list = np.zeros( N )

for i in range(N):
    ty, tx, vy0, vx0 = traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam_list[i], Lcurve = True)
    duvy0, duvx0 = uY*ps*1e-6 - vy0, uX*ps*1e-6 - vx0

    tr = np.abs( np.sqrt( tx**2 + ty**2 ) )
    duvr0 =  np.sqrt( np.abs(duvx0)**2 + np.abs(duvy0)**2  )

    # tr_list[i] = np.real( np.sum(tr) )
    # duvr0_list[i] = np.real( np.sum(duvr0) )

    tr_list[i] = np.linalg.norm(tr)
    duvr0_list[i] = np.linalg.norm(duvr0)

N0 = np.argmin( np.abs(lam_list - 2*1e-20) )
plt.plot( duvr0_list, tr_list )
plt.plot( [duvr0_list[N0]], [tr_list[N0]], 'o', c ='r' )
plt.xscale('log')
plt.yscale('log')
plt.xlabel('||u-Kt||')
plt.ylabel('||t||')
plt.grid()














pre_0, post_0, mascara_0, mascara10_0, pixel_size_0 = aux.celula( muestra[0], linea_muestra[0], place = 'home' )



























