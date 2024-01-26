# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:43:32 2024

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

#%%
path1 = r"C:\Users\gonza\1\Tesis\2023\\" + "23.10.20 - gon MCF10 7 - I18"
ps1 = 0.1007
mediciones1 = os.listdir(path1)
mediciones26 = [ medicion for medicion in mediciones1 if '26' in medicion and 'files' not in medicion]
print( mediciones26 )
#%%

stack_pre11 = of.imread( path1 + r"\\" + mediciones26[1] )[0]
stack_pre12 = of.imread( path1 + r"\\" + mediciones26[3] )[0]
stack_pre13 = of.imread( path1 + r"\\" + mediciones26[4] )[0]
stack_pre14 = of.imread( path1 + r"\\" + mediciones26[5] )[0]

stack_post11 = of.imread( path1 + r"\\" + mediciones26[7] )[0]
stack_post12 = of.imread( path1 + r"\\" + mediciones26[6] )[0]

cel_pre11 = of.imread( path1 + r"\\" + mediciones26[1] )[1,1]
cel_pre12 = of.imread( path1 + r"\\" + mediciones26[3] )[1,1]
cel_pre13 = of.imread( path1 + r"\\" + mediciones26[4] )[1,1]
cel_pre14 = of.imread( path1 + r"\\" + mediciones26[5] )[1,1]
m11 = iio.imread( path1 + r"\\" + 'cel10_m1.png' )
m12 = iio.imread( path1 + r"\\" + 'cel10_m2.png' )
m13 = iio.imread( path1 + r"\\" + 'cel10_m3.png' )
m14 = iio.imread( path1 + r"\\" + 'cel10_m4.png' )


cel_post11 = of.imread( path1 + r"\\" + mediciones26[7] )[1,1]
cel_post12 = of.imread( path1 + r"\\" + mediciones26[6] )[1,1]

#%%
p1 = stack_post11[5]
p2 = stack_post12[5]

p2_c, M, YX = TFM.correct_driff( p2, p1, 300, info = True )

lz = len(stack_post11)
post_grande = np.ones(  [ lz, 1219, 1219 ]  )*np.mean(stack_post11)

post_grande[ :, :1024, -1024: ] = stack_post11
post_grande[ :, 97:1024+97, -1024-195:-195 ] = stack_post12

pre_grande11 = np.ones(  [ len(stack_pre11), 1219, 1219 ]  )*np.mean(stack_pre11)
pre_grande11[ :, 100:1124, -1124:-100 ] = stack_pre11
pre_grande12 = np.ones(  [ len(stack_pre12), 1219, 1219 ]  )*np.mean(stack_pre11)
pre_grande12[ :, 100:1124, -1124:-100 ] = stack_pre12
pre_grande13 = np.ones(  [ len(stack_pre13), 1219, 1219 ]  )*np.mean(stack_pre11)
pre_grande13[ :, 100:1124, -1124:-100 ] = stack_pre13
pre_grande14 = np.ones(  [ len(stack_pre14), 1219, 1219 ]  )*np.mean(stack_pre11)
pre_grande14[ :, 100:1124, -1124:-100 ] = stack_pre14


post1 = post_grande[ 5 ]
pre11, ZYX1 = TFM.correct_driff_3D( pre_grande11, post1, 300, info = True )
pre12, ZYX2 = TFM.correct_driff_3D( pre_grande12, post1, 300, info = True )
pre13, ZYX3 = TFM.correct_driff_3D( pre_grande13, post1, 300, info = True )
pre14, ZYX4 = TFM.correct_driff_3D( pre_grande14, post1, 300, info = True )

mascara11 = np.zeros( [1219]*2 )
mascara11[ 100 + ZYX1[1] : 1124  + ZYX1[1] , -1124  + ZYX1[2] : -100 + ZYX1[2]  ] = m11
mascara12 = np.zeros( [1219]*2 )
mascara12[ 100 + ZYX2[1] : 1124  + ZYX2[1] , -1124  + ZYX2[2] : -100 + ZYX2[2]  ] = 1-m12
mascara13 = np.zeros( [1219]*2 )
mascara13[ 100 + ZYX3[1] : 1124  + ZYX3[1] , -1124  + ZYX3[2] : -100 + ZYX3[2]  ] = 1-m13
mascara14 = np.zeros( [1219]*2 )
mascara14[ 100 + ZYX4[1] : 1124  + ZYX4[1] , -1124  + ZYX4[2] : -100 + ZYX4[2]  ] = 1-m14

celula11 = np.ones( [1219]*2 )*np.mean(cel_pre11)
celula11[ 100 + ZYX1[1] : 1124  + ZYX1[1] , -1124  + ZYX1[2] : -100 + ZYX1[2]  ] = cel_pre11
celula12 = np.ones( [1219]*2 )*np.mean(cel_pre12)
celula12[ 100 + ZYX2[1] : 1124  + ZYX2[1] , -1124  + ZYX2[2] : -100 + ZYX2[2]  ] = cel_pre12
celula13 = np.ones( [1219]*2 )*np.mean(cel_pre13)
celula13[ 100 + ZYX3[1] : 1124  + ZYX3[1] , -1124  + ZYX3[2] : -100 + ZYX3[2]  ] = cel_pre13
celula14 = np.ones( [1219]*2 )*np.mean(cel_pre14)
celula14[ 100 + ZYX4[1] : 1124  + ZYX4[1] , -1124  + ZYX4[2] : -100 + ZYX4[2]  ] = cel_pre14

b1 = auxi.border(mascara11)
b2 = auxi.border(mascara12)
b3 = auxi.border(mascara13)
b4 = auxi.border(mascara14)

#%%
post = post1
ps = ps1
ws, exp = 2.5, 0.7
E, nu = 31.6, 0.5
A1 = 0.85

pre = pre11
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s1, Y_s1 = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
lam = auxi.busca_lambda(X_s1, Y_s1, ps*1e-6, solo_lambda = True)
ty1, tx1 = TFM.traction(Y_s1, X_s1, ps*1e-6, ws*1e-6, E*1e3, nu, lam )

pre = pre12
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s2, Y_s2 = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
lam = auxi.busca_lambda(X_s2, Y_s2, ps*1e-6, solo_lambda = True)
ty2, tx2 = TFM.traction(Y_s2, X_s2, ps*1e-6, ws*1e-6, E*1e3, nu, lam )

pre = pre13
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s3, Y_s3 = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
lam = auxi.busca_lambda(X_s3, Y_s3, ps*1e-6, solo_lambda = True)
ty3, tx3 = TFM.traction(Y_s3, X_s3, ps*1e-6, ws*1e-6, E*1e3, nu, lam )

pre = pre14
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s4, Y_s4 = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
lam = auxi.busca_lambda(X_s4, Y_s4, ps*1e-6, solo_lambda = True)
ty4, tx4 = TFM.traction(Y_s4, X_s4, ps*1e-6, ws*1e-6, E*1e3, nu, lam )

#%%

fon, L = 12, 20
plt.figure( figsize = [7,7], layout = 'compressed' )

X_s, Y_s, tx, ty, mascara, celula, b = X_s1, Y_s1, tx1, ty1, mascara11, celula11, b1

plt.subplot( 4, 3, 3 )
plt.quiver(x, y, tx, -ty, scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
cm = auxi.center_of_mass(mascara)
tx_cm, ty_cm = np.sum( tx*auxi.reshape_mask(mascara, x, y) ), np.sum( ty*auxi.reshape_mask(mascara, x, y) )
plt.quiver(cm[1], cm[0], -tx_cm, ty_cm, scale = 20000)
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = '0 minutos' )
plt.xlim([0,1219])
plt.ylim([1219,0])
plt.text(609, -20, "Tracción", c = 'k',  ha='center', va = 'bottom', fontsize = 12 )


plt.subplot( 4, 3, 2 )
plt.quiver(x, y, X_s*ps*1e-6, -Y_s*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])
plt.text(609, -20, "Deformación", c = 'k',  ha='center', va = 'bottom', fontsize = 12 )

plt.subplot( 4, 3, 1 )
plt.imshow( celula, cmap = 'gray' )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.7 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])
plt.text(609, -20, "Transmisión", c = 'k',  ha='center', va = 'bottom', fontsize = 12 )

X_s, Y_s, tx, ty, mascara, celula, b = X_s2, Y_s2, tx2, ty2, mascara12, celula12, b2

plt.subplot( 4, 3, 6 )
plt.quiver(x, y, tx, -ty, scale = 20000)
cm = auxi.center_of_mass(mascara)
tx_cm, ty_cm = np.sum( tx*auxi.reshape_mask(mascara, x, y) ), np.sum( ty*auxi.reshape_mask(mascara, x, y) )
plt.quiver(cm[1], cm[0], -tx_cm, ty_cm, scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = '31 minutos' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 5 )
plt.quiver(x, y, X_s*ps*1e-6, -Y_s*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 4 )
plt.imshow( celula, cmap = 'gray' )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.7 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])

X_s, Y_s, tx, ty, mascara, celula, b = X_s3, Y_s3, tx3, ty3, mascara13, celula13, b3

plt.subplot( 4, 3, 9 )
plt.quiver(x, y, tx, -ty, scale = 20000)
cm = auxi.center_of_mass(mascara)
tx_cm, ty_cm = np.sum( tx*auxi.reshape_mask(mascara, x, y) ), np.sum( ty*auxi.reshape_mask(mascara, x, y) )
plt.quiver(cm[1], cm[0], -tx_cm, ty_cm, scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = '50 minutos' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 8 )
plt.quiver(x, y, X_s*ps*1e-6, -Y_s*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 7 )
plt.imshow( celula, cmap = 'gray' )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.7 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])

X_s, Y_s, tx, ty, mascara, celula, b = X_s4, Y_s4, tx4, ty4, mascara14, celula14, b4

plt.subplot( 4, 3, 12 )
plt.quiver(x, y, tx, -ty, scale = 20000)
cm = auxi.center_of_mass(mascara)
tx_cm, ty_cm = np.sum( tx*auxi.reshape_mask(mascara, x, y) ), np.sum( ty*auxi.reshape_mask(mascara, x, y) )
plt.quiver(cm[1], cm[0], -tx_cm, ty_cm, scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = '78 minutos' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 11 )
plt.quiver(x, y, X_s*ps*1e-6, -Y_s*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 10 )
plt.imshow( celula, cmap = 'gray' )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.7 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k' )
plt.xlim([0,1219])
plt.ylim([1219,0])












#%%
# X_s, Y_s, tx, ty, mascara, celula, b = X_s1, Y_s1, tx1, ty1, mascara11, celula11, b1
X_s, Y_s, tx, ty, mascara, celula, b = X_s2, Y_s2, tx2, ty2, mascara12, celula12, b2
# X_s, Y_s, tx, ty, mascara, celula, b = X_s3, Y_s3, tx3, ty3, mascara13, celula13, b3

plt.figure(figsize=[3,3])
plt.quiver(x, y, tx, -ty, auxi.R(tx, -ty), scale = 20000)
cm = auxi.center_of_mass(mascara)
tx_cm, ty_cm = np.sum( np.real(tx)*auxi.reshape_mask(mascara, x, y) ), np.sum( np.real(ty)*auxi.reshape_mask(mascara, x, y) )
plt.quiver([cm[1]], [cm[0]], [tx_cm], [-ty_cm], scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( L, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = '31 minutos' )
plt.xlim([0,1219])
plt.ylim([1219,0])
plt.show()

#%%
# X_s, Y_s, tx, ty, mascara, celula, b = X_s1, Y_s1, tx1, ty1, mascara11, celula11, b1
# X_s, Y_s, tx, ty, mascara, celula, b = X_s2, Y_s2, tx2, ty2, mascara12, celula12, b2
X_s, Y_s, tx, ty, mascara, celula, b = X_s3, Y_s3, tx3, ty3, mascara13, celula13, b3

plt.imshow(auxi.R(tx, -ty))









#%%

# post, pre, mascara = post1, pre11, mascara11
# post, pre, mascara = post1, pre12, mascara12
# post, pre, mascara = post1, pre13, mascara13
post, pre, mascara = post1, pre14, mascara14
ps = ps1
ws, exp = 2.5, 0.7

A1 = auxi.busca_A( pre, 0.75, ps, win = 2.5, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )

E, nu, lam = 31.6, 0.5, -1  
lam = auxi.busca_lambda(X_s, Y_s, ps*1e-6, solo_lambda = True)

uy, ux = Y_s, X_s
x_plot, y_plot = x, y

ty, tx, vy0, vx0 = TFM.traction(uy, ux, ps*1e-6, ws*1e-6, E*1e3, nu, lam, Lcurve = True)
ty, tx = TFM.smooth(ty,3), TFM.smooth(tx,3)
vy0, vx0 = TFM.deformation( np.real(ty), np.real(tx), ws*1e-6, E*1e3, nu )

duvy0, duvx0 = uy*ps*1e-6 - vy0, ux*ps*1e-6 - vx0

plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = 'T' )
plt.xlim([0,1219])
plt.ylim([1219,0])

#%%

plt.figure( figsize = [7,7], layout = 'compressed' )

# Solucion T
plt.subplot(2,2,1)
plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
# plt.quiver(x_plot, y_plot, tx*auxi.reshape_mask(mascara10, x, y), -ty*auxi.reshape_mask(mascara10, x, y), scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '12', color = 'k', more_text = 'T' )
plt.xlim([0,1219])
plt.ylim([1219,0])

# Comprobacion V
plt.subplot(2,2,2)
plt.quiver(x_plot, y_plot, vx0, -vy0, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '12', color = 'k', more_text = 'V' )
plt.xlim([0,1219])
plt.ylim([1219,0])

# Entrada U
plt.subplot(2,2,3)
plt.quiver(x_plot, y_plot, ux*ps*1e-6, -uy*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '12', color = 'k', more_text = 'U'  )
plt.xlim([0,1219])
plt.ylim([1219,0])

# Diferencia U - V
plt.subplot(2,2,4)
plt.quiver(x_plot, y_plot, duvx0, -duvy0, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '12', color = 'k', more_text = 'U-V' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.show()





#%%

plt.figure( figsize = [7,7], layout = 'compressed' )

X_s, Y_s, tx, ty, mascara, celula = X_s1, Y_s1, tx1, ty1, mascara11, celula 

plt.subplot( 4, 3, 1 )
plt.quiver(x, y, tx, -ty, scale = 20000)
plt.imshow( mascara, cmap = color_maps[2], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps, img_len = 1200, font_size = '7', color = 'k', more_text = 'T' )
plt.xlim([0,1219])
plt.ylim([1219,0])

plt.subplot( 4, 3, 2 )
plt.quiver(x, y, X_s*ps*1e-6, -Y_s*ps*1e-6, scale = 0.00001)
plt.imshow( mascara, cmap = color_maps[i], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'u' )
plt.xlim([0,1023])
plt.ylim([1023,0])





















#%%
plt.figure()
plt.title('P1')
plt.imshow( pre_grande14[5] , cmap = 'gray', vmin = 80, vmax = 700)

pre_list = [3, 5, 5, 5]

#%%
post1 = post_grande[ 5 ]
pre1, ZYX = TFM.correct_driff_3D( pre_grande11, post1, 300, info = True )
c = cel_pre11
mascara1 = np.zeros( [1219]*2 )
mascara1[ 100 + ZYX[1] : 1124  + ZYX[1] , -1124  + ZYX[2] : -100 + ZYX[2]  ] = m11
N = 1

m1_grande = np.copy(mascara1)
#%%
celula_pre = np.ones( [1219]*2 )*np.mean(c)
celula_pre[ 100 + ZYX[1] : 1124  + ZYX[1] , -1124  + ZYX[2] : -100 + ZYX[2]  ] = c

# PIV + NMT + Suavizado
vi = int( int( 3/ps1 )*4 )
it = 3
# bordes_extra = 10 # px
bordes_extra = int(np.round(vi/12))


Noise_for_NMT = 0.2
Threshold_for_NMT = 2.5
modo = "Smooth5"
# modo = "No control"
mapas = False
suave0 = 3

dominio, deformacion = TFM.n_iterations( pre, post, vi, it, exploration = bordes_extra, mode = modo, A = 0.8)
Y_nmt, X_nmt, res = nmt(*deformacion, Noise_for_NMT, Threshold_for_NMT)
X_s, Y_s = smooth(X_nmt,suave0), smooth(Y_nmt, suave0)
# mascara = 1 - iio.imread( "mascara_3.png" )

x, y = dominio

inf = 120
a = np.mean(post)/np.mean(pre)
pre_plot = np.copy( (pre+5)*a - inf )
post_plot = np.copy(post - inf )
pre_plot[ pre < 0 ] = 0
pre_plot[ post < 0 ] = 0

scale0 = 100
scale_length = 10  # Length of the scale bar in pixels
scale_pixels = scale_length/ps
scale_unit = 'µm'  # Unit of the scale bar

wind = vi/( 2**(it-1) )
d = int( ( resolution - len(Y_nmt)*wind )/2   )

# Add the scale bar
scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
start_x = d + 100  # Starting x-coordinate of the scale bar
start_y = resolution -( 2*wind ) + 10# Starting y-coordinate of the scale bar


plt.figure(figsize=(20,20), tight_layout=True)

plt.subplot(1,3,1)

plt.imshow( celula_pre , cmap = 'gray' )
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
plt.text(start_x + scale_pixels/2, start_y - 35, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center', fontsize = "xx-large")
plt.text(start_x, 100, str(N), color='black', weight='bold', ha='center', fontsize = "xx-large")


plt.subplot(1,3,2)

plt.imshow(np.zeros(pre.shape), cmap = ListedColormap([(1,1,1)]))
plt.imshow( mascara, cmap = cm_orange2, alpha = 0.4 )
plt.quiver(x,y,-X_s,Y_s, scale = scale0, pivot='tail')
# plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0, pivot='tail')

# plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)

plt.xticks([])
plt.yticks([])
plt.xlim([0,resolution])
plt.ylim([resolution,0])

plt.subplot(1,3,3)
plt.imshow( np.zeros(pre.shape), cmap = cm0 )
plt.imshow( pre_plot, cmap = cm1, vmin = 0, vmax = 170, alpha = 1)
plt.imshow( post_plot, cmap = cm2, vmin = 0, vmax = 170, alpha = 0.5)
plt.xticks([])
plt.yticks([])
for i in range(20):
    plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='white', linewidth = 1)


# plt.savefig(name + '_figura.png')
plt.show()



#%%
plt.figure()
plt.title('P1')
plt.imshow( pre_grande4[5] , cmap = 'gray', vmin = 80, vmax = 700)

pre_list = [3, 5, 5, 5]





























