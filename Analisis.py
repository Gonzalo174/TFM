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
colores = [c0, c1, c2, c3]

cm0 = ListedColormap( [(1, 1, 1), (0.122, 0.467, 0.706) ] )
cm1 = ListedColormap( [(1, 1, 1), (1.000, 0.498, 0.055) ] )
cm2 = ListedColormap( [(1, 1, 1), (0.173, 0.627, 0.173) ] )
cm3 = ListedColormap( [(1, 1, 1), (0.839, 0.152, 0.157) ] )
color_maps = [cm0, cm1, cm2, cm3]

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

# valores de A?
# valores de exploración?

#%% PIV

pre, post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home' )

ws = 2.5
bordes_extra = 8

A1 = auxi.busca_A( pre, 0.75, ps, win = 2.5, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = bordes_extra, mode = "Smooth3", A = A1)
Y_0, X_0 = deformacion 
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)    # NMT para el ángulo?


X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
X_s2, Y_s2 = TFM.smooth(  X_nmt*auxi.reshape_mask(mascara10, x, y)  ,3), TFM.smooth(  Y_nmt*auxi.reshape_mask(mascara10, x, y)  ,3)


plt.figure(figsize = [6,4] )
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )

# plt.quiver(x,y,X_0,-Y_0, res, cmap = cm_crimson, scale = 100, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, scale = 100, pivot='tail')
plt.quiver(x,y,X_s,-Y_s, scale = 100, pivot='tail')

auxi.barra_de_escala( 10, sep = 1.5,  font_size = 11, color = 'k' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.show()

#%% TFM
nu = 0.5
E = 31.6      # kPa

lam = -1#3*1e-20
# uX, uY = X_nmt, Y_nmt
uX, uY = X_s, Y_s
# uX, uY = X_s2, Y_s2
x_plot, y_plot = x, y

# lam = 0
# uX, uY = auxi.interpolate(X_s), auxi.interpolate(Y_s) 
# x_plot, y_plot = np.meshgrid( np.arange( x[0,0], x[-1,-1] + (x[0,1] - x[0,0])/2, (x[0,1] - x[0,0])/2 ), np.arange( x[0,0], x[-1,-1] + (x[0,1] - x[0,0])/2, (x[0,1] - x[0,0])/2 )  )

ty, tx, vy0, vx0 = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam, Lcurve = True)
# ty, tx = TFM.smooth(ty,3), TFM.smooth(tx,3)
vy0, vx0 = TFM.deformation( np.real(ty), np.real(tx), ws*1e-6, E*1e3, nu )


# ty, tx = anti_interpolate( ty ), anti_interpolate( tx )
# vy0, vx0 = anti_interpolate( vy0 ), anti_interpolate( vx0 )
# uX, uY = X_s, Y_s
# x_plot, y_plot = x, y

duvy0, duvx0 = uY*ps*1e-6 - vy0, uX*ps*1e-6 - vx0

plt.figure( figsize = [7,7], layout = 'compressed' )

# Solucion T
plt.subplot(2,2,1)
# plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
plt.quiver(x_plot, y_plot, tx*auxi.reshape_mask(mascara10, x, y), -ty*auxi.reshape_mask(mascara10, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, font_size = '12', color = 'k', more_text = 'T' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Comprobacion V
plt.subplot(2,2,2)
plt.quiver(x_plot, y_plot, vx0, -vy0, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'V' )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Entrada U
plt.subplot(2,2,3)
plt.quiver(x_plot, y_plot, uX*ps*1e-6, -uY*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'U'  )
plt.xlim([0,1023])
plt.ylim([1023,0])

# Diferencia U - V
plt.subplot(2,2,4)
plt.quiver(x_plot, y_plot, duvx0, -duvy0, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'U-V' )
plt.xlim([0,1023])
plt.ylim([1023,0])

plt.show()

#%%
R = np.sqrt( np.abs( TFM.smooth(ty,3) )**2 + np.abs( TFM.smooth(tx,3))**2 )
b = auxi.border(mascara, 600, k = 3)

plt.imshow( R, extent = [0,1024,1024,0] )
plt.colorbar()
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75 )

#%% Exploracion del espacio de lambda

ws, exp, A1 = 2.5, 10, 0#auxi.busca_A( pre, 0.75, ps, win = 2.5, A0 = 0.85 )

dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = exp, mode = "Smooth3", A = A1)
Y_0, X_0 = deformacion 
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)    # NMT para el ángulo?
X_s, Y_s = TFM.smooth( X_nmt, 3 ), TFM.smooth( Y_nmt, 3 )
# X_s, Y_s = TFM.smooth( X_nmt*auxi.reshape_mask(mascara10, x, y), 3 ), TFM.smooth( Y_nmt*auxi.reshape_mask(mascara10, x, y), 3 )
x_plot, y_plot = x, y

E, nu = 31.6, 0.5  # kPa, adim

uX, uY = X_0, Y_0
# uX, uY = X_nmt, Y_nmt
# uX, uY = X_s, Y_s

plt.figure()
plt.quiver(x_plot, y_plot, uX*ps*1e-6, -uY*ps*1e-6, scale = 0.00001 )
plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  font_size = '11', color = 'k', more_text = 'T' )
plt.xlim([0,1023])
plt.ylim([1023,0])
plt.show()

a1, b1 = -21, -18
N = 100
lam_list = np.logspace(a1, b1, N )
# lam_list = np.linspace(10**a1, 10**b1, N )
tr_list = np.zeros( N )
duvr0_list = np.zeros( N )

for i in range(N):
    ty, tx, vy0, vx0 = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam_list[i], Lcurve = True)
    duvy0, duvx0 = uY*ps*1e-6 - vy0, uX*ps*1e-6 - vx0

    # uR = np.sqrt( np.real(uX*ps*1e-6)**2 + np.real(uY*ps*1e-6)**2 ) 
    tr = np.sqrt( np.real(tx)**2 + np.real(ty)**2 ) 
    duvr0 =  np.sqrt( np.real(duvx0)**2 + np.real(duvy0)**2  )

    tr_list[i] = np.sum(tr)
    duvr0_list[i] = np.sum(duvr0)

N0 = [ np.argmin( np.abs(lam_list - 10.0**expo ) ) for expo in np.arange( a1+1, b1, 1) ] 

tr_list, duvr0_list = auxi.normalizar(tr_list), auxi.normalizar( duvr0_list )

#%%

plt.figure(figsize = [6,6])
plt.plot( duvr0_list, tr_list, '.' )

plt.plot( duvr0_list[N0], tr_list[N0], 'o', c ='r' )
for i, N0_ in enumerate(N0):
    plt.text( 0.01+duvr0_list[N0_], 0.01+tr_list[N0_], "10^(" + str(np.arange( a1+1, b1, 1)[i]) + ")" , ha='left', va = 'bottom', fontsize = 11 )
    # if i == 2:
    #     plt.arrow(1.3*duvr0_list[N0[i]], 1.3*tr_list[N0[i]], 1.3*(duvr0_list[N0[i]] - duvr0_list[N0[i-1]]), 1.3*(tr_list[N0[i]] - tr_list[N0[i-1]]) , head_width=0.000010, head_length=200000, width = 0.000001, fc='red', ec='red', length_includes_head = False)

# plt.xscale('log')
# plt.yscale('log')
plt.xlabel('||u-Kt|| [U.A.]')
plt.ylabel('||t|| [U.A.]')
plt.grid()

m1, m2 = (tr_list[0] - tr_list[1])/(duvr0_list[0] - duvr0_list[1]), (tr_list[-2] - tr_list[-1])/(duvr0_list[-2] - duvr0_list[-1])
b1, b2 = tr_list[0] - m1*duvr0_list[0], tr_list[-1] - m2*duvr0_list[-1]
uKt_plot = np.linspace(np.min(duvr0_list), np.max(duvr0_list), 1000)
t1_plot, t2_plot = m1*uKt_plot + b1, m2*uKt_plot + b2

plt.plot(uKt_plot, t1_plot, c = 'k', ls = 'dashed')
plt.plot(uKt_plot, t2_plot, c = 'k', ls = 'dashed')
plt.ylim( [-0.05, 1.05] )


#%%
Dtr_list = np.diff( tr_list )/np.diff( duvr0_list )
duvr0_list2 = (duvr0_list[:-1] + duvr0_list[1:])/2
plt.figure(figsize = [6,6])
plt.plot( duvr0_list2, Dtr_list)

#%%
DDtr_list = np.diff( Dtr_list )/np.diff( duvr0_list2 )
duvr0_list3 = (duvr0_list2[:-1] + duvr0_list2[1:])/2
plt.figure(figsize = [6,6])
DDtr_list3 = np.convolve(DDtr_list, [1/5]*5, mode = 'same')

plt.plot( duvr0_list3, DDtr_list3)
#%%

plt.plot( duvr0_list3, DDtr_list3/3e8 )
plt.plot( duvr0_list, tr_list )

xR = -(b1 - b2)/(m1 - m2)
yR = m1*xR + b1
ND = np.argmax( DDtr_list3 )
NR = np.argmin( np.sqrt( (tr_list - yR)**2 + (duvr0_list - xR)**2 ) )


plt.plot( [0.000089]*2, [0,1.5e6] )


plt.plot(uKt_plot, t1_plot)
plt.plot(uKt_plot, t2_plot)
plt.ylim( [np.min(tr_list)-1e5, np.max(tr_list)+1e5] )

plt.plot( [xR], [yR], 'o', c = 'b' )
plt.plot( duvr0_list[ND], tr_list[ND], 'o', c ='k' )
plt.plot( duvr0_list[NR], tr_list[NR], 'o', c ='b' )



#%%
'''
El criterio de la curva L es una herramienta para encontrar un punto
en el que se balancea el tamaño de la solución y el error del ajuste.


It is a convenient graphical tool for displaying the trade-off between 
the size of a regularized solution and its fit to the given data, 
as the regularization parameter varies.

'''


#%%

cel = []
suero_ = []
linea_ = []

defo = []
fuer = []
cant = []
area = []


for index in range(4):
    lista = conjuntos[index]
    if index == 0 or index == 2:
        suero = 'CS'
    if index == 1 or index == 3:
        suero = 'SS'    
    
    if index == 0 or index == 1:
        linea = 'MCF10'
    if index == 2 or index == 3:
        linea = 'MCF7'    
        
    for N in lista:
        pre, post, mascara, mascara10, mascara20, ps = auxi.celula( N, linea )

        ws, exp = 2.5, int(0.7/ps)
        A1 = auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
        if N == 22 or N == 25:
            exp = int(1/ps)
        
        dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = exp, mode = "Smooth3", A = A1)
        Y_0, X_0 = deformacion 
        x, y = dominio
        Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
        X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
        R_s = np.sqrt( X_s**2 + Y_s**2 )

        plt.figure(figsize = [6,4] )
        plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
        plt.imshow( mascara10, cmap = color_maps[0], alpha = 0.5 )

        # plt.quiver(x,y,X_0,-Y_0, res, cmap = cm_crimson, scale = 100, pivot='tail')
        # plt.quiver(x,y,X_nmt,-Y_nmt, scale = 100, pivot='tail')
        plt.quiver(x,y,X_s,-Y_s, scale = 100, pivot='tail')

        auxi.barra_de_escala( 10, sep = 1.5, pixel_size = ps, font_size = 11, color = 'k', more_text = linea, a_lot_of_text = suero )
        plt.xlim([0,1023])
        plt.ylim([1023,0])
        plt.show()
        
        
        E, nu = 31.6, 0.5      # kPa

        lam = 0
        uX, uY = X_s, Y_s
        x_plot, y_plot = x, y

        ty, tx, vy0, vx0 = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam, Lcurve = True)
        ty, tx = TFM.smooth(ty,3), TFM.smooth(tx,3)
        tr = np.sqrt( np.abs(ty)**2 + np.abs(tx)**2 )

        plt.figure(figsize = [6,4] )
        plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
        plt.imshow( mascara, cmap = color_maps[0], alpha = 0.5 )
        auxi.barra_de_escala( 10, sep = 1.5, pixel_size = ps, font_size = '11', color = 'k', more_text = 'T' )
        plt.xlim([0,1023])
        plt.ylim([1023,0])        
        
        
        cel.append(N)
        suero_.append(suero)
        linea_.append(linea)

        defo.append( np.sum( R_s*auxi.reshape_mask(mascara10, x, y) )  )
        fuer.append( np.sum( tr*auxi.reshape_mask(mascara10, x, y) )  )
        cant.append( np.sum( auxi.reshape_mask(mascara10, x, y) ) )
        area.append( np.sum( auxi.reshape_mask(mascara, x, y) )*ps**2 )
        


#%%

data = pd.DataFrame()


data["Celula"] = cel
data["Area"] = area
data["N"] = cant

data["Deformación"] = defo
data["Tracción"] = fuer
data["Suero"] = suero_
data["Linea"] = linea_


data.to_csv( "data0.csv" )



#%%

ws = 2.5
desvios_bin, limit = auxi.busca_esferas( pre, ps, ws, 0.8 )
plt.figure( figsize = [6, 6] )
plt.imshow( pre[ :limit , :limit ], cmap = cm_crimson, vmin = 150, vmax = 600 )
plt.imshow( 1-desvios_bin, cmap = 'gray', alpha = 0.09, extent = [0,limit,limit,0])
auxi.barra_de_escala( 10, sep = 2, pixel_size = ps )
print(  np.mean(desvios_bin) )







#%% 4 furiosos

pre_0, post_0, mascara_0, mascara10_0, mascara20_0, pixel_size_0 = auxi.celula( muestra[0], linea_muestra[0], place = 'home' )
pre_1, post_1, mascara_1, mascara10_1, mascara20_1, pixel_size_1 = auxi.celula( muestra[1], linea_muestra[1], place = 'home' )
pre_2, post_2, mascara_2, mascara10_2, mascara20_2, pixel_size_2 = auxi.celula( muestra[2], linea_muestra[2], place = 'home' )
pre_3, post_3, mascara_3, mascara10_3, mascara20_3, pixel_size_3 = auxi.celula( muestra[3], linea_muestra[3], place = 'home' )



























