# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 21:14:26 2024

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
colores = [c3, c2, c0, c1]

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )
color_maps = [cm0, cm3, cm2, cm1]

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

muestra = [ 25, 9, (11,4), (10,5) ]
As = [  0.75, 0.75, 0.85, 0.8]
linea_muestra = [ 'MCF10', 'MCF10', 'MCF7', 'MCF7' ]


muestra2 = [ 25, 27, (11,4), (7,2) ]
linea_muestra2 = [ 'MCF10', 'MCF10', 'MCF7', 'MCF7' ]

#MCF7 D30 R4 cel9 del 1/9   0
#MCF7 C30 R5 cel5 del 1/9   1
#MCF10 D04 R9 5/10          2
#MCF10 G18 R25 del 19/10    3

#%%

celulas = []
bordes = []
mas00 = []
mas10 = []
listaX = []
listaY = []
listaXt = []
listaYt = []
listax = []
listay = []
ps_list = []

for iterador in range(4):
    cel = iterador
    pre, post, celula, mascara, mascara10, mascara20, ps = auxi.celula( muestra2[cel], linea_muestra2[cel], place = 'home', trans = True, D_pp = 0 )
    b = auxi.border(mascara, k = 11)

    ws, exp = 2.5, 0.7
    E, nu = 31.6, 0.5  # kPa, adim

    A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
    X_s, Y_s = TFM.smooth(X_nmt,3), TFM.smooth(Y_nmt, 3)
    x, y = dominio
    
    lam = auxi.busca_lambda( Y_s, X_s, ps, solo_lambda = True )
    ty, tx = TFM.traction( Y_s, X_s, ps*1e-6, ws*1e-6, E*1e3, nu, lam )
    uy, ux = Y_s*ps*1e-6, X_s*ps*1e-6

    fondo = auxi.median_blur(celula, 50)
    celulas.append( celula - fondo )
    # celulas.append( celula )
    bordes.append( b )
    mas00.append( mascara )
    mas10.append( mascara10 )
    listaX.append( ux )
    listaY.append( uy )
    listaXt.append( tx )
    listaYt.append( ty )
    listax.append( x )
    listay.append( y )
    ps_list.append( ps )

#%%
esc = 20
separacion = 1.5
fs = 'x-small'
plt.figure( figsize = [5, 11], layout = 'compressed' )
plt.subplots_adjust(wspace=0.04, hspace=0.0005) 

cel = 0
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y, tx, ty = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel], listaXt[cel], listaYt[cel]
scale0 = 0.00001
scale1 = 20000

plt.subplot(4,3,1)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed', lw = 0.75  )
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,3,2)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], sep = separacion, img_len = 990,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])

plt.subplot(4,3,3)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.7 )
plt.quiver(x,y,tx,-ty, scale = scale1, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], sep = separacion, img_len = 990,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])


cel = 1
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y, tx, ty = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel], listaXt[cel], listaYt[cel]

plt.subplot(4,3,4)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed', lw = 0.75    )
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,3,5)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990, sep = separacion,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])

plt.subplot(4,3,6)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.7 )
plt.quiver(x,y,tx,-ty, scale = scale1, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], sep = separacion, img_len = 990,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])

cel = 2
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y, tx, ty = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel], listaXt[cel], listaYt[cel]

plt.subplot(4,3,7)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed', lw = 0.75    )
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,3,8)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990, sep = separacion,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])


plt.subplot(4,3,9)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.7 )
plt.quiver(x,y,tx,-ty, scale = scale1, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], sep = separacion, img_len = 990,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])


cel = 3
celula_pre, b, mascara, mascara10, X_s, Y_s, x, y, tx, ty = celulas[cel], bordes[cel], mas00[cel], mas10[cel], listaX[cel], listaY[cel], listax[cel], listay[cel], listaXt[cel], listaYt[cel]

plt.subplot(4,3,10)

plt.imshow( celula_pre, cmap = 'gray' )
plt.plot( b[1] ,b[0], c = 'w', ls = 'dashed' , lw = 0.75   )
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990,  sep = separacion,  font_size = fs, color = 'w' )

plt.subplot(4,3,11)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[cel], alpha = 0.5 )
plt.quiver(x,y,X_s,-Y_s, scale = scale0, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], img_len = 990, sep = separacion,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])


plt.subplot(4,3,12)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.7 )
plt.quiver(x,y,tx,-ty, scale = scale1, pivot='tail')
auxi.barra_de_escala( esc, pixel_size = ps_list[cel], sep = separacion, img_len = 990,  font_size = fs, color = 'k', text = False  )
plt.xlim([0,1023])
plt.ylim([1023,0])



















#%%

cel = 0
pre, post, celula, mascara, mascara10, mascara20, ps = auxi.celula( muestra[cel], linea_muestra[cel], place = 'home', trans = True, D_pp = 0 )
b = auxi.border(mascara)


#%%

plt.figure(figsize = [7,7], layout = "compressed")

plt.subplot(1,3,1)
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 700 )
# plt.imshow( post, cmap = cm_green, vmin = 150, vmax = 300, alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'w', more_text = 'pre' )

plt.subplot(1,3,2)
# plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 400 )
plt.imshow( post, cmap = cm_green, vmin = 150, vmax = 700 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'w', more_text = 'post' )

plt.subplot(1,3,3)
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 450 )
plt.imshow( post, cmap = cm_green, vmin = 150, vmax = 400, alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'w' )


#%%











