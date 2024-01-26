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

traccionan = [ 1, 4, 5, 6, 7, 8, 9, 11, 12, 20, 22, 25, 14, 15, 17, 18, 26, 27, 29, (11,3), (11,4), (10,2), (10,5), (1,1)  ]

# valores de exploración?

#%%

data0 = auxi.celula( muestra[0], linea_muestra[0], place = 'home' )
data1 = auxi.celula( muestra[1], linea_muestra[1], place = 'home' )
data2 = auxi.celula( muestra[2], linea_muestra[2], place = 'home' )
data3 = auxi.celula( muestra[3], linea_muestra[3], place = 'home' )
data_cels = [data0, data1, data2, data3]

#%% Visualizador

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
    uy, ux = Y_s, X_s
    # uy, ux = TFM.smooth( Y_nmt*auxi.reshape_mask(mascara20, x, y), 3 ), TFM.smooth( X_nmt*auxi.reshape_mask(mascara20, x, y), 3 )
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



#%% Campo y celula
i = 3
pre, post, transmision, mascara, mascara10, mascara20, ps = auxi.celula( muestra[2], linea_muestra[2], place = 'home', trans = True )
# bor = auxi.border(mascara)
ws, exp = 2.5, 0.7


A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Fit", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
x_plot, y_plot = x, y

# fondo = auxi.median_blur(transmision, 50)
# celula = transmision - fondo
# bor = auxi.border(mascara)
i = 2
plt.figure( figsize = [8,8], layout = 'compressed' )

# Trans 
plt.subplot(1,3,1)
plt.imshow( celula, cmap = 'gray' )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'w' )
plt.plot( bor[1], bor[0], c = 'w', ls = 'dashed', lw = 0.7 )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U
plt.subplot(1,3,2)
# plt.quiver(x_plot, y_plot, X_nmt*ps*1e-6, -Y_nmt*ps*1e-6, scale = 0.00001)
plt.quiver(x_plot, y_plot, X_s*ps*1e-6, -Y_s*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])

E, nu = 31.6, 0.5  # kPa, adim
# ty, tx, vy, vx = TFM.traction(Y_nmt, X_nmt, ps*1e-6, ws*1e-6, E*1e3, nu, -1, Lcurve = True)
ty, tx, vy, vx = TFM.traction(Y_s, X_s, ps*1e-6, ws*1e-6, E*1e3, nu, -1, Lcurve = True)
plt.subplot(1,3,3)
plt.quiver(x_plot, y_plot, tx, -ty, scale = 40000)
# plt.quiver(x_plot, y_plot, tx*auxi.reshape_mask(mascara20, x, y), -ty*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])


#%% Resolucion directa suavizando
i = 2
pre, post, mascara, mascara10, mascara20, ps = data_cels[i]
ws, exp = 2.5, 0.7
E, nu = 31.6, 0.5  # kPa, adim

A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
x_plot, y_plot = x, y
uY, uX = Y_s, X_s
# uY, uX = Y_s*auxi.reshape_mask(mascara20, x, y), X_s*auxi.reshape_mask(mascara20, x, y)

ty, tx, vy, vx = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, -1, Lcurve = True)
ty_s, tx_s = TFM.smooth(ty,3), TFM.smooth(tx,3)
vy_s, vx_s = TFM.deformation( np.real(ty_s), np.real(tx_s), ws*1e-6, E*1e3, nu )
duvy, duvx = uY*ps*1e-6 - vy, uX*ps*1e-6 - vx
duvy_s, duvx_s = uY*ps*1e-6 - vy_s, uX*ps*1e-6 - vx_s


plt.figure( figsize = [7,7], layout = 'compressed' )

# Solucion T 
plt.subplot(3,2,1)
plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
# plt.quiver(x_plot, y_plot, tx*auxi.reshape_mask(mascara20, x, y), -ty*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "t Crudo", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )


# Comprobacion V
plt.subplot(3,2,3)
plt.quiver(x_plot, y_plot, vx, -vy, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Diferencia U - V
plt.subplot(3,2,5)
plt.quiver(x_plot, y_plot, duvx, -duvy, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Solucion T suave
plt.subplot(3,2,2)
plt.quiver(x_plot, y_plot, tx_s, -ty_s, scale = 20000)
# plt.quiver(x_plot, y_plot, tx_s*auxi.reshape_mask(mascara20, x, y), -ty_s*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "t Suave", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )


# Comprobacion V suave
plt.subplot(3,2,4)
plt.quiver(x_plot, y_plot, vx_s, -vy_s, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Diferencia U - V suave
plt.subplot(3,2,6)
plt.quiver(x_plot, y_plot, duvx_s, -duvy_s, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

plt.show()


#%% Regularización para distintos mapas de deformación

i = 2
pre, post, mascara, mascara10, mascara20, ps = data_cels[i]
ws, exp = 2.5, 1#0.7
E, nu = 31.6, 0.5  # kPa, adim

dominio1, deformacion1 = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = 0)
Y_1, X_1 = deformacion 

A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
x_plot, y_plot = x, y

Y_2, X_2 = Y_nmt, X_nmt
Y_3, X_3 = Y_s, X_s


N = 100
expos = [-21, -20, -19, -18]

NlL1, l_list1, tr_list1, duvr_list1 = auxi.busca_lambda(5e-22, 5e-18, N, Y_1, X_1, ps*1e-6, ws*1e-6, E*1e3, nu, norma = True)
D1 = np.diff( tr_list1 )/np.diff( duvr_list1 )
m11, m21 = np.mean( D1[:int(N/10)] ) , np.mean( D1[-int(N/10 + 1):] )
b11, b21 = np.mean( tr_list1[:int(N/10)] - m11*duvr_list1[:int(N/10)] ) , np.mean( tr_list1[-int(N/10 + 1):] - m21*duvr_list1[-int(N/10 + 1):] ) 
N01 = [ np.argmin( np.abs(l_list1 - 10.0**expo ) ) for expo in expos ] 

NlL2, l_list2, tr_list2, duvr_list2 = auxi.busca_lambda(5e-22, 5e-18, N, Y_2, X_2, ps*1e-6, ws*1e-6, E*1e3, nu, norma = True)
D2 = np.diff( tr_list2 )/np.diff( duvr_list2 )
m12, m22 = np.mean( D2[:int(N/10)] ) , np.mean( D2[-int(N/10 + 1):] )
b12, b22 = np.mean( tr_list2[:int(N/10)] - m12*duvr_list2[:int(N/10)] ) , np.mean( tr_list2[-int(N/10 + 1):] - m22*duvr_list2[-int(N/10 + 1):] ) 
N02 = [ np.argmin( np.abs(l_list2 - 10.0**expo ) ) for expo in expos ] 

NlL3, l_list3, tr_list3, duvr_list3 = auxi.busca_lambda(5e-22, 5e-18, N, Y_3, X_3, ps*1e-6, ws*1e-6, E*1e3, nu, norma = True)
D3 = np.diff( tr_list3 )/np.diff( duvr_list3 )
m13, m23 = np.mean( D3[:int(N/10)] ) , np.mean( D3[-int(N/10 + 1):] )
b13, b23 = np.mean( tr_list3[:int(N/10)] - m13*duvr_list3[:int(N/10)] ) , np.mean( tr_list3[-int(N/10 + 1):] - m23*duvr_list3[-int(N/10 + 1):] ) 
N03 = [ np.argmin( np.abs(l_list3 - 10.0**expo ) ) for expo in expos ] 



ty1, tx1, vy1, vx1 = TFM.traction(Y_1, X_1, ps*1e-6, ws*1e-6, E*1e3, nu, l_list1[NlL1], Lcurve = True)
duvy1, duvx1 = Y_1*ps*1e-6 - vy1, X_1*ps*1e-6 - vx1

ty2, tx2, vy2, vx2 = TFM.traction(Y_2, X_2, ps*1e-6, ws*1e-6, E*1e3, nu, l_list2[NlL2], Lcurve = True)
duvy2, duvx2 = Y_2*ps*1e-6 - vy2, X_2*ps*1e-6 - vx2

ty3, tx3, vy3, vx3 = TFM.traction(Y_3, X_3, ps*1e-6, ws*1e-6, E*1e3, nu, l_list3[NlL3], Lcurve = True)
duvy3, duvx3 = Y_3*ps*1e-6 - vy3, X_3*ps*1e-6 - vx3

#%% Vertical

plt.figure( figsize = [10,10], layout = 'compressed' )

# U1
plt.subplot(4,3,1)
plt.quiver(x_plot, y_plot, X_1*ps*1e-6, -Y_1*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "u Crudo", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )

# T1 
plt.subplot(4,3,4)
plt.quiver(x_plot, y_plot, tx1, -ty1, scale = 20000)
# plt.quiver(x_plot, y_plot, tx1*auxi.reshape_mask(mascara20, x, y), -ty1*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U-V 1
plt.subplot(4,3,7)
plt.quiver(x_plot, y_plot, duvx1, -duvy1, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# lambda plot 1
plt.subplot(4,3,10)
duvr_plot1 = np.linspace( np.min(duvr_list1), np.max(duvr_list1), int(N*10) )
plt.plot( duvr_plot1, m11*duvr_plot1 + b11, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot1, m21*duvr_plot1 + b21, c = 'k', ls = 'dashed' )
plt.plot( duvr_list1, tr_list1, c = colores[i], ms = 4  )
plt.plot( duvr_list1[N01], tr_list1[N01], '.', c = 'k', ms = 6 )
plt.plot( [duvr_list1[NlL1]], [tr_list1[NlL1]], '.', c = 'k', ms = 6 )
plt.xlabel('||u-v||')
plt.ylabel('||t||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list1)-0.1*np.max(tr_list1), 1.1*np.max(tr_list1)] )

for j in range(4):
    plt.text( duvr_list1[N01[j]]+0.01, tr_list1[N01[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list1[N01[j]]+0.09, tr_list1[N01[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL1 = int( np.log10(l_list1[NlL1]) - 1 )
valL1 = np.round(l_list1[NlL1]/10**expoL1,1)
plt.text( duvr_list1[NlL1]+0.01, tr_list1[NlL1]+0.01, "λ = " + str(valL1) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list1[NlL1]+0.31, tr_list1[NlL1]+0.035, str(expoL1) , ha='left', va = 'bottom', fontsize = 7 )

# U2
plt.subplot(4,3,2)
plt.quiver(x_plot, y_plot, X_2*ps*1e-6, -Y_2*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "u NMT", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )

# T2 
plt.subplot(4,3,5)
plt.quiver(x_plot, y_plot, tx2, -ty2, scale = 20000)
# plt.quiver(x_plot, y_plot, tx2*auxi.reshape_mask(mascara20, x, y), -ty2*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U-V 2
plt.subplot(4,3,8)
plt.quiver(x_plot, y_plot, duvx2, -duvy2, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# lambda plot 2
plt.subplot(4,3,11)
duvr_plot2 = np.linspace( np.min(duvr_list2), np.max(duvr_list2), int(N*10) )
plt.plot( duvr_plot2, m12*duvr_plot2 + b12, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot2, m22*duvr_plot2 + b22, c = 'k', ls = 'dashed' )
plt.plot( duvr_list2, tr_list2, c = colores[i], ms = 4  )
plt.plot( duvr_list2[N02], tr_list2[N02], '.', c = 'k', ms = 6  )
plt.plot( [duvr_list2[NlL2]], [tr_list2[NlL2]], '.', c = 'k', ms = 6  )
plt.xlabel('||u-v||')
# plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list2)-0.1*np.max(tr_list2), 1.1*np.max(tr_list2)] )

for j in range(4):
    plt.text( duvr_list2[N02[j]]+0.01, tr_list2[N02[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list2[N02[j]]+0.09, tr_list2[N02[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL2 = int( np.log10(l_list2[NlL2]) - 1 )
valL2 = np.round(l_list2[NlL2]/10**expoL2,1)
plt.text( duvr_list2[NlL2]+0.01, tr_list2[NlL2]+0.01, "λ = " + str(valL2) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list2[NlL2]+0.31, tr_list2[NlL2]+0.035, str(expoL2) , ha='left', va = 'bottom', fontsize = 7 )

# U3
plt.subplot(4,3,3)
plt.quiver(x_plot, y_plot, X_3*ps*1e-6, -Y_3*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "u Suave", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )
plt.text(1150, -20, ".", c = 'w',  ha='center', va = 'bottom', fontsize = 14 )

# T3 
plt.subplot(4,3,6)
plt.quiver(x_plot, y_plot, tx3, -ty3, scale = 20000)
# plt.quiver(x_plot, y_plot, tx3*auxi.reshape_mask(mascara20, x, y), -ty3*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U-V 3
plt.subplot(4,3,9)
plt.quiver(x_plot, y_plot, duvx3, -duvy3, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# lambda plot 3
plt.subplot(4,3,12)
duvr_plot3 = np.linspace( np.min(duvr_list3), np.max(duvr_list3), int(N*10) )
plt.plot( duvr_plot3, m13*duvr_plot3 + b13, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot3, m23*duvr_plot3 + b23, c = 'k', ls = 'dashed' )
plt.plot( duvr_list3, tr_list3, c = colores[i], ms = 4  )
plt.plot( duvr_list3[N03], tr_list3[N03], '.', c = 'k', ms = 6  )
plt.plot( [duvr_list3[NlL3]], [tr_list3[NlL3]], '.', c = 'k', ms = 6  )
plt.xlabel('||u-v||')
# plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list3)-0.1*np.max(tr_list3), 1.1*np.max(tr_list3)] )

for j in range(4):
    plt.text( duvr_list3[N03[j]]+0.01, tr_list3[N03[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list3[N03[j]]+0.09, tr_list3[N03[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL3 = int( np.log10(l_list3[NlL3]) - 1 )
valL3 = np.round(l_list3[NlL3]/10**expoL3,1)
plt.text( duvr_list3[NlL3]+0.01 + 0.05, tr_list3[NlL3]+0.01 - 0.05, "λ = " + str(valL3) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list3[NlL3]+0.3 + 0.06, tr_list3[NlL3]+0.035 - 0.05, str(expoL3) , ha='left', va = 'bottom', fontsize = 7 )
#%% Horizontal
plt.figure( figsize = [10,10], layout = 'compressed' )

# U1
plt.subplot(3,4,1)
plt.quiver(x_plot, y_plot, X_1*ps*1e-6, -Y_1*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# T1 
plt.subplot(3,4,2)
plt.quiver(x_plot, y_plot, tx1*auxi.reshape_mask(mascara20, x, y), -ty1*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U-V 1
plt.subplot(3,4,3)
plt.quiver(x_plot, y_plot, duvx1, -duvy1, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# lambda plot 1
plt.subplot(3,4,4)
duvr_plot1 = np.linspace( np.min(duvr_list1), np.max(duvr_list1), int(N*10) )
plt.plot( duvr_plot1, m11*duvr_plot1 + b11, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot1, m21*duvr_plot1 + b21, c = 'k', ls = 'dashed' )
plt.plot( duvr_list1, tr_list1, c = colores[i], ms = 4  )
plt.plot( duvr_list1[N01], tr_list1[N01], '.', c = 'k', ms = 6 )
plt.plot( [duvr_list1[NlL1]], [tr_list1[NlL1]], '.', c = 'k', ms = 6 )
# plt.xlabel('||U-V||')
plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list1)-0.1*np.max(tr_list1), 1.1*np.max(tr_list1)] )

for j in range(4):
    plt.text( duvr_list1[N01[j]]+0.01, tr_list1[N01[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list1[N01[j]]+0.08, tr_list1[N01[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL1 = int( np.log10(l_list1[NlL1]) - 1 )
valL1 = np.round(l_list1[NlL1]/10**expoL1,1)
plt.text( duvr_list1[NlL1]+0.01, tr_list1[NlL1]+0.01, "λ = " + str(valL1) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list1[NlL1]+0.3, tr_list1[NlL1]+0.035, str(expoL1) , ha='left', va = 'bottom', fontsize = 7 )

# U2
plt.subplot(3,4,1+4)
plt.quiver(x_plot, y_plot, X_2*ps*1e-6, -Y_2*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# T2 
plt.subplot(3,4,2+4)
plt.quiver(x_plot, y_plot, tx2*auxi.reshape_mask(mascara20, x, y), -ty2*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U-V 2
plt.subplot(3,4,3+4)
plt.quiver(x_plot, y_plot, duvx2, -duvy2, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# lambda plot 2
plt.subplot(3,4,4+4)
duvr_plot2 = np.linspace( np.min(duvr_list2), np.max(duvr_list2), int(N*10) )
plt.plot( duvr_plot2, m12*duvr_plot2 + b12, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot2, m22*duvr_plot2 + b22, c = 'k', ls = 'dashed' )
plt.plot( duvr_list2, tr_list2, c = colores[i], ms = 4  )
plt.plot( duvr_list2[N02], tr_list2[N02], '.', c = 'k', ms = 6  )
plt.plot( [duvr_list2[NlL2]], [tr_list2[NlL2]], '.', c = 'k', ms = 6  )
# plt.xlabel('||U-V||')
plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list2)-0.1*np.max(tr_list2), 1.1*np.max(tr_list2)] )

for j in range(4):
    plt.text( duvr_list2[N02[j]]+0.01, tr_list2[N02[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list2[N02[j]]+0.08, tr_list2[N02[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL2 = int( np.log10(l_list2[NlL2]) - 1 )
valL2 = np.round(l_list2[NlL2]/10**expoL2,1)
plt.text( duvr_list2[NlL2]+0.01, tr_list2[NlL2]+0.01, "λ = " + str(valL2) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list2[NlL2]+0.3, tr_list2[NlL2]+0.035, str(expoL2) , ha='left', va = 'bottom', fontsize = 7 )

# U3
plt.subplot(3,4,1+8)
plt.quiver(x_plot, y_plot, X_3*ps*1e-6, -Y_3*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# T3 
plt.subplot(3,4,2+8)
plt.quiver(x_plot, y_plot, tx3*auxi.reshape_mask(mascara20, x, y), -ty3*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# U-V 3
plt.subplot(3,4,3+8)
plt.quiver(x_plot, y_plot, duvx3, -duvy3, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# lambda plot 3
plt.subplot(3,4,4+8)
duvr_plot3 = np.linspace( np.min(duvr_list3), np.max(duvr_list3), int(N*10) )
plt.plot( duvr_plot3, m13*duvr_plot3 + b13, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot3, m23*duvr_plot3 + b23, c = 'k', ls = 'dashed' )
plt.plot( duvr_list3, tr_list3, c = colores[i], ms = 4  )
plt.plot( duvr_list3[N03], tr_list3[N03], '.', c = 'k', ms = 6  )
plt.plot( [duvr_list3[NlL3]], [tr_list3[NlL3]], '.', c = 'k', ms = 6  )
plt.xlabel('||U-V||')
plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list3)-0.1*np.max(tr_list3), 1.1*np.max(tr_list3)] )

for j in range(4):
    plt.text( duvr_list3[N03[j]]+0.01, tr_list3[N03[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list3[N03[j]]+0.08, tr_list3[N03[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL3 = int( np.log10(l_list3[NlL3]) - 1 )
valL3 = np.round(l_list3[NlL3]/10**expoL3,1)
plt.text( duvr_list3[NlL3]+0.01 + 0.05, tr_list3[NlL3]+0.01 - 0.05, "λ = " + str(valL3) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list3[NlL3]+0.3 + 0.05, tr_list3[NlL3]+0.035 - 0.05, str(expoL3) , ha='left', va = 'bottom', fontsize = 7 )

#%% Regularización para distintos valores de lambda

i = 2
pre, post, mascara, mascara10, mascara20, ps = data_cels[i]
ws, exp = 2.5, 0.7
E, nu = 31.6, 0.5  # kPa, adim

A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
x_plot, y_plot = x, y

Y_2, X_2 = Y_nmt, X_nmt
Y_3, X_3 = Y_s, X_s

N = 100
expos = [-21, -20, -19, -18]

NlL2, l_list2, tr_list2, duvr_list2 = auxi.busca_lambda(5e-22, 5e-18, N, Y_2, X_2, ps*1e-6, ws*1e-6, E*1e3, nu, norma = True)
D2 = np.diff( tr_list2 )/np.diff( duvr_list2 )
m12, m22 = np.mean( D2[:int(N/10)] ) , np.mean( D2[-int(N/10 + 1):] )
b12, b22 = np.mean( tr_list2[:int(N/10)] - m12*duvr_list2[:int(N/10)] ) , np.mean( tr_list2[-int(N/10 + 1):] - m22*duvr_list2[-int(N/10 + 1):] ) 
N02 = [ np.argmin( np.abs(l_list2 - 10.0**expo ) ) for expo in expos ] 

# a L0, b L0/10 (más chico), c L0*10 (más grande)
ty2a, tx2a, vy2a, vx2a = TFM.traction(Y_2, X_2, ps*1e-6, ws*1e-6, E*1e3, nu, l_list2[NlL2], Lcurve = True)
ty2b, tx2b, vy2b, vx2b = TFM.traction(Y_2, X_2, ps*1e-6, ws*1e-6, E*1e3, nu, l_list2[NlL2]*1e-1, Lcurve = True)
ty2c, tx2c, vy2c, vx2c = TFM.traction(Y_2, X_2, ps*1e-6, ws*1e-6, E*1e3, nu, l_list2[NlL2]*1e1, Lcurve = True)

#%%
plt.figure( figsize = [7,7], layout = 'compressed' )

# Ta
plt.subplot(1,3,1)
plt.quiver(x_plot, y_plot, tx2b*auxi.reshape_mask(mascara20, x, y), -ty2b*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 'λ ' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(990, 85, "0", c = 'k',  ha='center', va = 'bottom', fontsize = '6' )
plt.text(910, 50, "1", c = 'k',  ha='center', va = 'bottom', fontsize = '6' )
plt.text(910, 50, "_", c = 'k',  ha='center', va = 'bottom', fontsize = '8' )
plt.text(910, 88, "10", c = 'k',  ha='center', va = 'bottom', fontsize = '6' )


# Tb 
plt.subplot(1,3,2)
plt.quiver(x_plot, y_plot, tx2a*auxi.reshape_mask(mascara20, x, y), -ty2a*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'λ ' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(990, 85, "0", c = 'k',  ha='center', va = 'bottom', fontsize = '6' )

# Tc
plt.subplot(1,3,3)
plt.quiver(x_plot, y_plot, tx2c*auxi.reshape_mask(mascara20, x, y), -ty2c*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = '10λ ' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(990, 85, "0", c = 'k',  ha='center', va = 'bottom', fontsize = '6' )


#%% Interpolación

i = 2
pre, post, mascara, mascara10, mascara20, ps = data_cels[i]
ws, exp = 2.5, 0.7
E, nu = 31.6, 0.5  # kPa, adim

A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
x_plot, y_plot = x, y
uY, uX = Y_s, X_s
# uY, uX = Y_s*auxi.reshape_mask(mascara20, x, y), X_s*auxi.reshape_mask(mascara20, x, y)

ty, tx, vy, vx = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, -1, Lcurve = True)
duvy, duvx = uY*ps*1e-6 - vy, uX*ps*1e-6 - vx


uY_i, uX_i = auxi.interpolate(Y_s), auxi.interpolate(X_s)
ty_i0, tx_i0, vy_i0, vx_i0 = TFM.traction(uY_i, uX_i, ps*1e-6, ws*1e-6/2, E*1e3, nu, -1, Lcurve = True)
ty_i, tx_i = auxi.anti_interpolate(ty_i0), auxi.anti_interpolate(tx_i0)
vy_i, vx_i = TFM.deformation( ty_i, tx_i )
duvy_i, duvx_i = uY*ps*1e-6 - vy_i, uX*ps*1e-6 - vx_i


plt.figure( figsize = [7,7], layout = 'compressed' )

# Solucion T 
plt.subplot(3,2,1)
plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
# plt.quiver(x_plot, y_plot, tx*auxi.reshape_mask(mascara20, x, y), -ty*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "u Suave", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )


# Comprobacion V
plt.subplot(3,2,3)
plt.quiver(x_plot, y_plot, vx, -vy, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Diferencia U - V
plt.subplot(3,2,5)
plt.quiver(x_plot, y_plot, duvx, -duvy, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Solucion T suave
plt.subplot(3,2,2)
plt.quiver(x_plot, y_plot, tx_i, -ty_i, scale = 20000)
# plt.quiver(x_plot, y_plot, tx_s*auxi.reshape_mask(mascara20, x, y), -ty_s*auxi.reshape_mask(mascara20, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.text(512, -20, "u Suave e Interpolado", c = 'k',  ha='center', va = 'bottom', fontsize = 14 )


# Comprobacion V suave
plt.subplot(3,2,4)
plt.quiver(x_plot, y_plot, vx_i, -vy_i, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Diferencia U - V suave
plt.subplot(3,2,6)
plt.quiver(x_plot, y_plot, duvx_i, -duvy_i, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u-v' )
plt.xlim([0,1023])
plt.ylim([1023,0])

plt.show()

#%%

duvr_i = auxi.R(duvx_i, duvy_i)
ur_i = auxi.R(uY, uX)*ps*1e-6

print(np.max(duvr_i))
print(np.max(ur_i))
print(np.max(duvr_i)/np.max(ur_i))


#%%

t_directa = []
t_renormalizada = []
t_interpolada = [] 

for j in range(len(traccionan)):
   
    N = traccionan[j]
    
    if type(N) == int:
        linea = 'MCF10'
    elif type(N) == tuple:
        linea = 'MCF7'

    pre, post, mascara, mascara10, mascara20, ps = auxi.celula( N, linea )

    ws, exp = 2.5, 0.7
    E, nu = 31.6, 0.5  # kPa, adim

    if linea == 'MCF10':
        if N == 22 or N == 25:
            exp = 1
    
    A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
    X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
    x_plot, y_plot = x, y
    
    uY_d, uX_d = Y_s, X_s
    uY_r, uX_r = Y_nmt, X_nmt
    # uY_r, uX_r = Y_s, X_s
    uY_i, uX_i = auxi.interpolate(Y_s), auxi.interpolate(X_s)
    ty_d, tx_d = TFM.traction( uY_d, uX_d, ps*1e-6 )

    NlL, l_list, tr_list, duvr_list = auxi.busca_lambda( uY_r, uX_r, ps*1e-6)
    ty_r, tx_r = TFM.traction(uY_r, uX_r, ps*1e-6, lam = l_list[NlL] )

    ty_i0, tx_i0 = TFM.traction(uY_i, uX_i, ps*1e-6, ws = ws*1e-6/2)
    ty_i, tx_i = auxi.anti_interpolate(ty_i0), auxi.anti_interpolate(tx_i0)
    
    tx_d, ty_d = TFM.smooth(  tx_d, 3 ), TFM.smooth(  ty_d, 3 )
    tx_r, ty_r = TFM.smooth(  tx_r, 3 ), TFM.smooth(  ty_r, 3 )
    tx_i, ty_i = TFM.smooth(  tx_i, 3 ), TFM.smooth(  ty_i, 3 )
    
    tr_d = auxi.R(tx_d, ty_d)
    tr_r = auxi.R(tx_r, ty_r)
    tr_i = auxi.R(tx_i, ty_i)
    
    M = auxi.reshape_mask(mascara, x, y)
    
    t_directa.append( np.sum( tr_d*M )/np.sum(M) )
    t_renormalizada.append( np.sum( tr_r*M )/np.sum(M) )
    t_interpolada.append( np.sum( tr_i*M )/np.sum(M) )
    
    plt.figure( layout = "compressed" )
    
    plt.subplot(1,3,1)
    plt.quiver(x, y, tx_d, -ty_d, scale = 20000)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't_d' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    plt.subplot(1,3,2)
    plt.quiver(x, y, tx_r, -ty_r, scale = 20000)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't_r' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    plt.subplot(1,3,3)
    plt.quiver(x, y, tx_i, -ty_i, scale = 20000)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'k', more_text = 't_i' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    plt.show()

print( 'directo: ', np.mean(t_directa), np.std(t_directa) )
print( 'renormalizando: ', np.mean(t_renormalizada ), np.std(t_renormalizada ) )
print( 'interpolando: ', np.mean(t_interpolada ), np.std(t_interpolada ) ) 

#%%

import seaborn as sns

conjuntos = [t_directa, t_renormalizada, t_interpolada ]

plt.figure( figsize = [5,5] )
sns.boxplot(  conjuntos, color = colores[2]  )
plt.grid(True)
plt.xticks([0,1,2], ['Inversión Directa', 'Renormalización', 'Interpolación'])#['10CS','10SS','7CS','7SS'] )
delta0 = 0.05
# plt.text(0-delta0, -0.05, 'MCF10', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(0+delta0, -0.05, 'con suero', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(1-delta0, -0.05, 'MCF10', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(1+delta0, -0.05, 'sin suero', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(2-delta0, -0.05, 'MCF7', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(2+delta0, -0.05, 'con suero', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(3-delta0, -0.05, 'MCF7', rotation=90, va='top', ha='center', color='k', fontsize=11)
# plt.text(3+delta0, -0.05, 'sin suero', rotation=90, va='top', ha='center', color='k', fontsize=11)

# plt.xticks([0,1,2,3], ['MCF7SS','MCF7CS','MCF10SS','MCF10CS'])
plt.ylabel( "Tracción promedio [Pa]" )


#%%

NlL, l_list, tr_list, duvr_list = auxi.busca_lambda( Y_nmt, X_nmt, ps*1e-6)

# lambda plot
D = np.diff( tr_list )/np.diff( duvr_list )
m1, m2 = np.mean( D[:int(10)] ) , np.mean( D[-int(11):] )
b1, b2 = np.mean( tr_list[:int(10)] - m1*duvr_list[:int(10)] ) , np.mean( tr_list[-int(11):] - m2*duvr_list[-int(11):] ) 
N0 = [ np.argmin( np.abs(l_list - 10.0**expo ) ) for expo in expos ] 


duvr_plot = np.linspace( np.min(duvr_list), np.max(duvr_list), int(10) )
plt.plot( duvr_plot, m1*duvr_plot + b1, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot, m2*duvr_plot + b2, c = 'k', ls = 'dashed' )
plt.plot( duvr_list, tr_list, c = colores[i], ms = 4  )
plt.plot( duvr_list[N0], tr_list[N0], '.', c = 'k', ms = 6  )
plt.plot( [duvr_list[NlL]], [tr_list[NlL]], '.', c = 'k', ms = 6  )
plt.xlabel('||U-V||')
plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
plt.ylim( [np.min(tr_list)-0.1*np.max(tr_list), 1.1*np.max(tr_list)] )

for j in range(4):
    plt.text( duvr_list[N0[j]]+0.01, tr_list[N0[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list[N0[j]]+0.08, tr_list[N0[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL = int( np.log10(l_list3[NlL]) - 1 )
valL = np.round(l_list3[NlL3]/10**expoL3,1)
plt.text( duvr_list[NlL]+0.01 + 0.05, tr_list[NlL]+0.01 - 0.05, "λ = " + str(valL) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list[NlL]+0.3 + 0.05, tr_list[NlL]+0.035 - 0.05, str(expoL) , ha='left', va = 'bottom', fontsize = 7 )

















#%%
#%%
pre, post, mascara, mascara10, mascara20, ps = data_cels[2]
ws, exp = 2.5, 0.7
E, nu = 31.

# A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
A1 = 0
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )


#%%
N = 100
NlL, l_list, tr_list, duvr_list = auxi.busca_lambda(5e-22, 5e-18, N, Y_0, X_0, ps*1e-6, ws*1e-6, E*1e3, nu, norma = True)
#%%
D = np.diff( tr_list )/np.diff( duvr_list )
m1, m2 = np.mean( D[:int(N/10)] ) , np.mean( D[-int(N/10 + 1):] )
b1, b2 = np.mean( tr_list[:int(N/10)] - m1*duvr_list[:int(N/10)] ) , np.mean( tr_list[-int(N/10 + 1):] - m2*duvr_list[-int(N/10 + 1):] ) 
expos = [-21, -20, -19, -18]
N0 = [ np.argmin( np.abs(l_list - 10.0**expo ) ) for expo in expos ] 

duvr_plot = np.linspace( np.min(duvr_list), np.max(duvr_list), int(N*10) )
plt.plot( duvr_plot, m1*duvr_plot + b1, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot, m2*duvr_plot + b2, c = 'k', ls = 'dashed' )
plt.plot( duvr_list[N0], tr_list[N0], '.', c = 'k', ms = 10  )
plt.plot( [duvr_list[NlL]], [tr_list[NlL]], '.', c = 'k', ms = 10  )
plt.plot( duvr_list, tr_list, '.', c = colores[2], ms = 4  )
plt.xlabel('||U-V||')
plt.ylabel('||T||')
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
# plt.yaxis.set_label_position('right')
# plt.tick_params(axis='y', which='both', right=True, left = False)
plt.ylim( [np.min(tr_list)-0.1*np.max(tr_list), 1.1*np.max(tr_list)] )
# plt.xscale('log')
# plt.yscale('log')

for i in range(4):
    plt.text( duvr_list[N0[i]]+0.01, tr_list[N0[i]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list[N0[i]]+0.04, tr_list[N0[i]]+0.03, str(expos[i]) , ha='left', va = 'bottom', fontsize = 7 )

expoL = int( np.log10(l_list[NlL]) - 1 )
valL = np.round(l_list[NlL]/10**expoL,1)
plt.text( duvr_list[NlL]+0.01, tr_list[NlL]+0.01, "λ = " + str(valL) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list[NlL]+0.14, tr_list[NlL]+0.03, str(expoL) , ha='left', va = 'bottom', fontsize = 7 )



print(l_list[NlL])



# esquina 2.7e-20 parece mejor
# minimo 6.9e-20



#%%

duvr_plot2 = np.linspace( 0.1, 0.9, int(N*10) )
plt.plot( duvr_plot2, m12*duvr_plot2 + b12, c = 'k', ls = 'dashed' )
plt.plot( duvr_plot2, m22*duvr_plot2 + b22, c = 'k', ls = 'dashed' )
plt.plot( duvr_list2[1:-1], tr_list2[1:-1], c = colores[i], ms = 4  )
plt.plot( duvr_list2[N02], tr_list2[N02], '.', c = 'k', ms = 6  )
plt.plot( [duvr_list2[NlL2]], [tr_list2[NlL2]], '.', c = 'k', ms = 6  )
plt.xlabel('||U-V||')
plt.ylabel('||T||')
# plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
# plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],['','','','','',''])
plt.grid()
# plt.ylim( [np.min(tr_list2)-0.1*np.max(tr_list2), 1.1*np.max(tr_list2)] )
# plt.xscale('log')
# plt.yscale('log')

for j in range(4):
    plt.text( duvr_list2[N02[j]]+0.01, tr_list2[N02[j]]+0.01, "10" , ha='left', va = 'bottom', fontsize = 11 )
    plt.text( duvr_list2[N02[j]]+0.08, tr_list2[N02[j]]+0.035, str(expos[j]) , ha='left', va = 'bottom', fontsize = 7 )

expoL2 = int( np.log10(l_list2[NlL2]) - 1 )
valL2 = np.round(l_list2[NlL2]/10**expoL2,1)
plt.text( duvr_list2[NlL2]+0.01, tr_list2[NlL2]+0.01, "λ = " + str(valL2) + " 10" , ha='left', va = 'bottom', fontsize = 11 )
plt.text( duvr_list2[NlL2]+0.3, tr_list2[NlL2]+0.035, str(expoL2) , ha='left', va = 'bottom', fontsize = 7 )
