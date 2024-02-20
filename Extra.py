# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 11:38:08 2024

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
color_maps = [cm3, cm2, cm0, cm1]

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

#%% Hacer figura con: Bien adherida: G18 R22 | Mal adherida: A30 R1

pre, post, celula_B, mascara_B, mascara10, mascara20, ps_B = auxi.celula( 22, 'MCF10', place = 'home', trans = True, D_pp = 0 )
b_B = auxi.border(mascara_B)

pre, post, celula_M, mascara_M, mascara10, mascara20, ps_M = auxi.celula( (9,1), 'MCF7', place = 'home', trans = True, D_pp = 0 )
mascara_M = plt.imread("M1.png")[:,:,0]
b_M = auxi.border(mascara_M,k=11)


#%%

plt.figure( figsize = [6,6], layout = 'compressed' )
plt.subplot( 1, 2, 1 )
plt.imshow(celula_B, cmap = 'gray')
auxi.barra_de_escala( 10, sep = 1.3,  pixel_size = ps_B,  font_size = '10', color = 'w' )
# plt.plot( b_B[1], b_B[0], c = 'w', ls = 'dashed', lw = 0.5  )

plt.subplot( 1, 2, 2 )
plt.imshow(celula_M, cmap = 'gray')
auxi.barra_de_escala( 10, sep = 1.3,  pixel_size = ps_M,  font_size = '10', color = 'w' )
# plt.plot( b_M[1], b_M[0], c = 'w', ls = 'dashed', lw = 0.5  )

#%%
cm_crimson2 = ListedColormap( [(220*i/(999*255),20*i/(999*255),60*i/(999*255), 0.5) for i in range(1000)] )
cm_green2 = ListedColormap( [(0,128*i/(999*255),0,0.5) for i in range(1000)] )

pre0, post0, celula, celula_post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True, cel_post = True, D_pp = -2 )
pre1, post1, celula, celula_post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True, cel_post = True, D_pp = -1 )
pre2, post2, celula, celula_post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True, cel_post = True, D_pp = 0 )
pre3, post3, celula, celula_post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True, cel_post = True, D_pp = 1 )
pre4, post4, celula, celula_post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True, cel_post = True, D_pp = 2 )
pre5, post5, celula, celula_post, mascara, mascara10, mascara20, ps = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True, cel_post = True, D_pp = 3 )
b = auxi.border(mascara)
#%%
dl = 128
imagenes = [pre0[:dl,:dl], pre1[:dl,:dl], pre2[:dl,:dl], pre3[:dl,:dl], pre4[:dl,:dl], pre5[:dl,:dl]]
# imagenes = [pre1[:dl,:dl], pre3[:dl,:dl], pre5[:dl,:dl]]
# imagenes = [pre0[:dl,:dl], pre2[:dl,:dl], pre4[:dl,:dl]]
imagenes = [pre2[:dl,:dl]]

fig = plt.figure(figsize = [8,8])
z_offset = 0
z_disp = 1
ax = fig.add_subplot(111, projection='3d')
for imagen in imagenes:
    X, Y = np.meshgrid(np.arange(imagen.shape[1]), np.arange(imagen.shape[0]))
    Z = np.zeros(imagen.shape) + z_offset 
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = cm_crimson2(imagen), shade=False )
    z_offset += z_disp

plt.axis('off')
# ax.text( -2, -2, 2, "Z = -1 µm", va = 'bottom', ha = "right", weight='bold', fontsize = 18 )
# ax.text( -2, -2, 1, "Z = -2 µm", va = 'bottom', ha = "right", weight='bold', fontsize = 18 )
# ax.text( -2, -2, 0, "Z = -3 µm", va = 'bottom', ha = "right", weight='bold', fontsize = 18 )

plt.show()

#%%
fondo1 = auxi.median_blur(celula, 50)
fondo2 = auxi.median_blur(celula_post, 50)
#%%
plt.figure( figsize = [7,7], layout = 'compressed' )
plt.subplot( 1, 2, 1 )
plt.imshow(celula - fondo1, cmap = 'gray')
auxi.barra_de_escala( 10, sep = 1.3,  pixel_size = ps,  font_size = '10', color = 'w' )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )

plt.subplot( 1, 2, 2 )
plt.imshow(celula_post - fondo2, cmap = 'gray')
auxi.barra_de_escala( 10, sep = 1.3,  pixel_size = ps,  font_size = '10', color = 'w' )
# plt.plot( b_M[1], b_M[0], c = 'w', ls = 'dashed', lw = 0.5  )



#%% no anda
dl = 128
imagenes = [pre1[:dl,:dl], pre3[:dl,:dl], pre5[:dl,:dl]]
fig = plt.figure(figsize = [14,7], layout = "compressed")
z_offset = 0
z_disp = 1
ax1 = fig.add_subplot(111, projection='3d')
for imagen in imagenes:
    X, Y = np.meshgrid(np.arange(imagen.shape[1]), np.arange(imagen.shape[0]))
    Z = np.zeros(imagen.shape) + z_offset 
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = cm_crimson2(imagen), shade=False, vmin = 70, vmax = 600 )

    z_offset += z_disp
ax1.axis('off')

ax2 = plt.subplot( 122 )
ax2.imshow(celula, cmap = 'gray' )
auxi.barra_de_escala( 10, sep = 1.3,  pixel_size = ps,  font_size = '10', color = 'w' )
ax2.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.5  )

plt.show()

#%%


dl = 128
# imagenes = [pre0[:dl,:dl], pre1[:dl,:dl], pre2[:dl,:dl], pre3[:dl,:dl], pre4[:dl,:dl], pre5[:dl,:dl]]
# imagenes = [pre1[:dl,:dl], pre3[:dl,:dl], pre5[:dl,:dl]]
imagenes = []
vmin, vmax = 70, 600
for imagen in imagenes0:
    imagen[ imagen < 150 ] = 150
    imagen[ imagen > 700 ] = 700
    imagenes.append( imagen )

fig = plt.figure()
z_offset = 0
z_disp = 10000
ax = fig.add_subplot(111, projection='3d')
for imagen in imagenes:
    X, Y = np.meshgrid(np.arange(imagen.shape[1]), np.arange(imagen.shape[0]))
    Z = np.zeros(imagen.shape) + z_offset 
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = cm_crimson2(imagen), shade=False, vmin = 70, vmax = 600 )

    z_offset += z_disp
ax.axis('off')


#%%


cel = 0
pre, post, celula, mascara, mascara10, mascara20, ps = auxi.celula( muestra[cel], linea_muestra[cel], place = 'home', trans = True, D_pp = 0 )
b = auxi.border(mascara)

post_corrido = np.ones([1024]*2)*np.mean(post)
post_corrido[10:1024, 10:1024] = post[:1014,:1014]

#%%

plt.figure(figsize = [7,7], layout = "compressed")

plt.subplot(1,3,1)
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 450 )
plt.imshow( post_corrido, cmap = cm_green, vmin = 150, vmax = 400, alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'w', more_text = 'con corrimiento' )

plt.subplot(1,3,2)
plt.imshow( pre, cmap = cm_crimson, vmin = 150, vmax = 450 )
plt.imshow( post, cmap = cm_green, vmin = 150, vmax = 400, alpha = 0.5 )
plt.plot( b[1], b[0], c = 'w', ls = 'dashed', lw = 0.75  )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'w', more_text = 'corrimiento corregido')

#%%




#%%
data0 = auxi.celula( muestra[0], linea_muestra[0], place = 'home' )
data1 = auxi.celula( muestra[1], linea_muestra[1], place = 'home' )
data2 = auxi.celula( muestra[2], linea_muestra[2], place = 'home' )
data3 = auxi.celula( muestra[3], linea_muestra[3], place = 'home' )
data = [data0, data1, data2, data3]
#%%
plt.figure( figsize=[10,10], tight_layout = True )

pre, post, mascara, mascara10, mascara20, ps = data[0]
plt.subplot(1,4,4)
plt.imshow( mascara, cmap = cm3 )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k' )

pre, post, mascara, mascara10, mascara20, ps = data[1]
plt.subplot(1,4,3)
plt.imshow( mascara, cmap = cm2 )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k' )

pre, post, mascara, mascara10, mascara20, ps = data[2]
plt.subplot(1,4,1)
plt.imshow( mascara, cmap = cm0 )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k' )

pre, post, mascara, mascara10, mascara20, ps = data[3]
plt.subplot(1,4,2)
plt.imshow( mascara, cmap = cm1 )
auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k' )




























#%%












import matplotlib.pyplot as plt
import numpy as np

# Generar datos para las imágenes (reemplaza esto con tus propias imágenes)
imagenes = [np.random.rand(10, 10) for _ in range(5)]

fig = plt.figure()

# Definir la posición inicial y el desplazamiento
x_offset = 0
y_offset = 0
z_offset = 0

# Parámetros para el ajuste de la perspectiva
x_disp = 0
y_disp = 0
z_disp = 5  # Aumentar para más separación entre imágenes

ax = fig.add_subplot(111, projection='3d')

# Graficar cada imagen con un desplazamiento
for imagen in imagenes:
    X, Y = np.meshgrid(np.arange(imagen.shape[1]), np.arange(imagen.shape[0]))
    Z = np.zeros(imagen.shape) + z_offset
    ax.plot_surface(X + x_offset, Y + y_offset, Z, rstride=1, cstride=1, facecolors=plt.cm.viridis(imagen), shade=False)
    
    # Actualizar los desplazamientos para la siguiente imagen
    z_offset += z_disp

# Ocultar los ejes
ax.axis('off')

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Aunque no se use directamente, es necesario importarlo

# Definir el colormap personalizado cm_crimson2
# Asegúrate de tener definido cm_crimson2 correctamente antes de usarlo.
# Por ejemplo, usando LinearSegmentedColormap como se mostró en ejemplos anteriores.

dl = 128
imagenes = [np.random.rand(dl, dl), np.random.rand(dl, dl), np.random.rand(dl, dl)]  # Ejemplo de datos

fig = plt.figure(figsize=[14, 7])  # Ajusta el tamaño de la figura según necesites

# Primer subplot (3D)
ax1 = fig.add_subplot(121, projection='3d')  # 1 fila, 2 columnas, primer subplot
z_offset = 0
z_disp = 1
for imagen in imagenes:
    X, Y = np.meshgrid(np.arange(imagen.shape[1]), np.arange(imagen.shape[0]))
    Z = np.zeros(imagen.shape) + z_offset
    ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=plt.cm.viridis(imagen), shade=False, vmin=70, vmax=600)  # Asumiendo que cm_crimson2 es similar a plt.cm.viridis
    z_offset += z_disp
ax1.axis('off')

# Segundo subplot (Ejemplo de un gráfico de líneas)
ax2 = fig.add_subplot(122)  # 1 fila, 2 columnas, segundo subplot
ax2.plot([0, 1, 2], [3, 2, 1])  # Gráfico de líneas de ejemplo
ax2.set_title('Gráfico de Líneas')

plt.show()









