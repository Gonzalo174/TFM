# -*- coding: utf-8 -*-
"""
Created on Thu Jan  25 15:17:34 2023

@author: Usuario
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

#%% NMT polar

cel = 3
pre, post, celula, mascara, mascara10, mascara20, ps = auxi.celula( muestra[cel], linea_muestra[cel], place = 'home', trans = True, D_pp = 0 )
b = auxi.border(mascara)


#%%
ws, exp = 2, 0.7
E, nu = 31.6, 0.5  # kPa, adim

A1 = 0.85 #auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )
dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
x, y = dominio
Y_0, X_0 = deformacion 
Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5, polar = False)
X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )
#%%
# plt.quiver(x, y, X_0, -Y_0, res, cmap = cm_crimson, scale = 100)
# plt.quiver(x, y, X_nmt, -Y_nmt, scale = 100)
plt.quiver(x, y, X_s, -Y_s, scale = 100)

plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '10', color = 'k', more_text = 'v' )
plt.xlim([0,1023])
plt.ylim([1023,0])



#%% Axial
cel = 3
ws, exp = 2.5, 0.7
A1 = 0.85#auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )

image_filenames = []
zf_list = [9, 15, 11, 17]
zf = zf_list[cel]
z0 = 5

for i in range(-3, zf - z0, 1):
    pre, post, celula, mascara, mascara10, mascara20, ps = auxi.celula( muestra[cel], linea_muestra[cel], place = 'home', trans = True, D_pp = i )
    
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5, polar = False)
    X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )

    plt.figure(figsize = [4,4])
    # plt.title( "w = "  + str(int(vi/4)) + " px, Deformación promedio = " + str( np.round(r_mean,3)) + ' µm'  )
    # plt.quiver(x, y, X_0, -Y_0, res, cmap = cm_crimson, scale = 100)
    # plt.quiver(x, y, X_nmt, -Y_nmt, scale = 100)
    plt.quiver(x, y, X_s, -Y_s, scale = 100)
    plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '11', color = 'k', more_text = str((i+3)/2) + ' µm' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])
    name =  r"C:\Users\gonza\1\Tesis\Tesis\Defensa\Gif" + '\img_' + str(i) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )

    plt.show()
    
#%% gif maker

from PIL import Image

# Open the images and convert them to the same format
images = []
for filename in image_filenames:
    image = Image.open(filename)
    # image = image.convert('RGBA')  # Convert to RGBA format (optional, for transparency support)
    images.append(image)

# Save the images as an animated GIF
output_filename = r"C:\Users\gonza\1\Tesis\Tesis\Defensa\Gif" + '\profundidad4.gif'  # Specify the output filename
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=500, loop=0)






#%% Ventana

exp = 0.7
A1 = 0.85#auxi.busca_A( pre, 0.75, ps, win = ws, A0 = 0.85 )

image_filenames = []


ventanas = np.arange( int(4/ps), int(1/ps) - 1, -1 )

for i in range(len(ventanas)):
    ws = ventanas[i]
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5, polar = False)
    X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )

    plt.figure(figsize = [4,4])
    # plt.title( "w = "  + str(int(vi/4)) + " px, Deformación promedio = " + str( np.round(r_mean,3)) + ' µm'  )
    plt.quiver(x, y, X_0, -Y_0, res, cmap = cm_crimson, scale = 100)
    # plt.quiver(x, y, X_nmt, -Y_nmt, scale = 300)
    plt.imshow( mascara, cmap = color_maps[cel], alpha = 0.5 )
    auxi.barra_de_escala( 10, sep = 1.5,  pixel_size = ps,  font_size = '11', color = 'k', more_text = str(np.round(ventanas[i]*ps,1)) + ' µm' )
    plt.xlim([0,1023])
    plt.ylim([1023,0])
    name =  r"C:\Users\gonza\1\Tesis\Tesis\Defensa\Gif" + '\img_' + str(i) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )

    plt.show()
    
#%% gif maker

from PIL import Image

# Open the images and convert them to the same format
images = []
for filename in image_filenames:
    image = Image.open(filename)
    # image = image.convert('RGBA')  # Convert to RGBA format (optional, for transparency support)
    images.append(image)

# Save the images as an animated GIF
output_filename = r"C:\Users\gonza\1\Tesis\Tesis\Defensa\Gif" + '\prueba3.gif'  # Specify the output filename
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=500, loop=0)





































#%% gif

n = 0
vi = 100
# windows = [200, 220, 256, 280, 300]
it = 3
exploration = 7 # px
scale0 = 100
modo = "Fit"

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

image_filenames = []

for n in range(len(stack_pre)):
    pre1 = stack_pre[ n ]
    post0 = centrar_referencia( stack_post[ n ] , pre1, 250)

    Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
    Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
    suave0 = 3
    X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)
    
    l = len(Y_nmt)
    wind = vi/( 2**(it-1) )
    field_length = int( l*wind )
    image_length = len( celula )
    d = (image_length - field_length)/2
    r_plot = np.arange(l)*wind + wind/2 + d
    
    x,y = np.meshgrid( r_plot , r_plot )
    
    plt.figure()
    plt.title("Mapa de deformación - z = -" + str(n/2 + 3) + ' µm')
    # plt.title("Mapa de deformación - w = " + str(int(vi/4)) + ' px')
    
    plt.imshow( 1-mascara , cmap = 'Reds', alpha = alfa, vmax = 0.1 )
    # plt.imshow( 1-mascara1 , cmap = 'Greens', alpha = 0.2 )
    # plt.imshow( 1-mascara2 , cmap = 'Reds', alpha = 0.2 )
    # plt.imshow( 1-mascara3 , cmap = 'Blues', alpha = 0.2 )
    # plt.imshow( 1-mascara4 , cmap = 'Oranges', alpha = 0.2 )
    # plt.imshow( 1-mascara5 , cmap = 'Purples', alpha = 0.2 )
    
    # plt.quiver(x,y,X_nmt,-Y_nmt, scale = scale0)
    plt.quiver(x,y,X_s,-Y_s, scale = scale0)
    
    scale_length = 10  # Length of the scale bar in pixels
    scale_pixels = scale_length/pixel_size
    scale_unit = 'µm'  # Unit of the scale bar
    scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    start_x = d + wind  # Starting x-coordinate of the scale bar
    start_y = image_length -( d + wind )# Starting y-coordinate of the scale bar

    plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
    for i in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i - 10, start_y + i - 10], color='black', linewidth = 1)
    plt.text(start_x + scale_pixels/2, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold', ha='center')

    plt.xticks([])
    plt.yticks([])
    # plt.xlim([d,image_length-d])
    # plt.ylim([image_length-d,d])
    name =  str(n) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )


#%% gif

vi = 100
it = 3
exploration = 5 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

image_filenames = []

for vi in windows:
    pre1 = stack_pre[ n ]
    post0 = centrar_referencia( stack_post[ n ] , pre1, 250)

    Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
    Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
    suave0 = 3
    X_s,Y_s = suavizar(X_nmt,suave0),suavizar(Y_nmt, suave0)


    r = np.sqrt(Y_s.flatten()**2 + X_s.flatten()**2)*pixel_size
    r_mean = np.mean( r )
    plt.figure()
    plt.title( "w = "  + str(int(vi/4)) + " px, Deformación promedio = " + str( np.round(r_mean,3)) + ' µm'  )
    plt.xlabel('Desplazamiento [µm]')
    plt.ylabel('Densidad')
    plt.grid(True)
    plt.hist(r, bins = np.arange(-0.04, 0.6 , 0.08) , density=True )
    plt.plot( [r_mean,r_mean], [0,4], color = "r", linestyle = "dashed", lw = 4 )
    plt.xlim([-0.06,0.66])
    plt.ylim([0,4.1])
    name =  str(vi) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )
    
#%% gif maker

from PIL import Image

# Open the images and convert them to the same format
images = []
for filename in image_filenames:
    image = Image.open(filename)
    # image = image.convert('RGBA')  # Convert to RGBA format (optional, for transparency support)
    images.append(image)

# Save the images as an animated GIF
output_filename = 'defo32_fit.gif'  # Specify the output filename
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=500, loop=0)














#%% Fuerza en la celula

vi = 256
it = 3
exploration = 5 # px

Noise_for_NMT = 0.2
Threshold_for_NMT = 5

Y, X = n_iteraciones( post0, pre1, vi, it, bordes_extra = exploration)
Y_nmt, X_nmt, res = nmt(Y, X, Noise_for_NMT, Threshold_for_NMT)
suave0 = 3
X_s,Y_s = suavizar(X_nmt, suave0),suavizar(Y_nmt, suave0)
X_work, Y_work = np.copy(X_s), np.copy(Y_s)
X_work[:3,:4], Y_work[:3,:4] = np.zeros([3,4]), np.zeros([3,4])

l = len(Y_nmt)
scale0 = 50
wind = vi/( 2**(it-1) )
field_length = int( l*wind )
image_length = len( celula )
d = (image_length - field_length)/2
r_plot = np.arange(l)*wind + wind/2 + d
x,y = np.meshgrid( r_plot , r_plot )
    

#%% Desplacement at restricted areas

cell_area = 1 - mascara1 
cell_area_down = np.copy( cell_area )
ps = pixel_size
il = len(cell_area)
ks = 40
th = 0.9
print( ks*(0.5 - th)*ps )

n_r = 10
x_a = np.zeros([n_r*2-1])
y_a = np.zeros([n_r*2-1])
dx_a = np.zeros([n_r*2-1])
dy_a = np.zeros([n_r*2-1])
a = np.zeros([n_r*2-1, il, il])

for n in range(n_r):
    # Para afuera
    if n !=0:
        cell_area = area_upper( cell_area, ks, th)

    Y_cell = []
    X_cell = []

    for j in range(l):
        for i in range(l):
            if cell_area[ int(x[j,i]), int(y[j,i]) ] == 1:
                Y_cell.append(Y_work[j,i])
                X_cell.append(X_work[j,i])

    x_a[n+n_r-1] = np.mean( X_cell )
    y_a[n+n_r-1] = np.mean( Y_cell ) 
    dx_a[n+n_r-1] = np.std( X_cell )
    dy_a[n+n_r-1] = np.std( Y_cell ) 
    a[n+n_r-1] = cell_area
    
    # Para adentro
    cell_area_down = area_upper( cell_area_down, ks, 1-th)
    
    Y_cell = []
    X_cell = []

    for j in range(l):
        for i in range(l):
            if cell_area_down[ int(x[j,i]), int(y[j,i]) ] == 1:
                Y_cell.append(Y_work[j,i])
                X_cell.append(X_work[j,i])
    
    print( n+1 )
    x_a[-n+n_r-2] = np.mean( X_cell )
    y_a[-n+n_r-2] = np.mean( Y_cell ) 
    dx_a[-n+n_r-2] = np.std( X_cell )
    dy_a[-n+n_r-2] = np.std( Y_cell ) 
    a[-n+n_r-2] = cell_area_down
    
r_a = np.sqrt( x_a**2 + y_a**2 )
d_ra = np.sqrt(  (x_a**2)*( r_a )**3  +  (y_a**2)*( r_a )**3   )

iio.imwrite("a.tiff",a)

#%%
r_plot = -(np.arange(len(r_a)) - n_r )
dr_plot = np.ones(len(r_a))*ps*3

plt.title("Deformación resultante")
plt.errorbar(x = -r_plot, y = r_a, yerr=d_ra/2, xerr=dr_plot, fmt = '.')
plt.grid(True)
plt.xlabel("dr [µm]")
plt.ylabel("Deformación resultante [µm]")

#%%
CM = center_of_mass(mascara)
mapita = np.sum( a, 0 )
    
plt.imshow( mapita , cmap = "plasma")
plt.plot( [CM[1]],[CM[0]] , 'o')
plt.xlim([0,1000])
plt.ylim([1000,0])

#%% gif

CM = center_of_mass(mascara)
scale1 = 3
scale0 = 30
image_filenames = []

for i in range(len(a)):
    alpha = 0.2
    # if i - n_r  == 0:
    #     alpha = 0.5
    plt.figure()
    plt.title("Deformación resultante - dr = " + str( -(i - n_r ) ) + ' µm')
    plt.imshow( a[i-1] , cmap = 'Blues', alpha=alpha )
    plt.imshow( 1-mascara1 , cmap = 'Blues', alpha=0.3)
    plt.quiver([CM[1]],[CM[0]],x_a[-i-1],-y_a[-i-1], scale = scale1)
    plt.quiver(x,y,X_s,-Y_s, scale = scale0)
    
    scale_length = 10  # Length of the scale bar in pixels
    scale_pixels = scale_length/pixel_size
    scale_unit = 'µm'  # Unit of the scale bar
    scale_bar_length = int(scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    start_x = 800 # Starting x-coordinate of the scale bar
    start_y = 950 # Starting y-coordinate of the scale bar
    plt.plot([start_x+20, start_x + scale_pixels-20], [start_y-25, start_y-25], color='white', linewidth = 40)
    for i0 in range(20):
        plt.plot([start_x, start_x + scale_pixels], [start_y + i0 - 10, start_y + i0 - 10], color='black', linewidth = 1)
    plt.text(start_x + scale_bar_length, start_y-25, f'{scale_length} {scale_unit}', color='black', weight='bold')#, ha='center')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,1000])
    plt.ylim([1000,0])
    # plt.xlim([d,image_length-d])
    # plt.ylim([image_length-d,d])
    name =  str( -(i - n_r ) ) + ".png" 
    plt.savefig(  name  )
    image_filenames.append( name )





#%% gif maker

from PIL import Image

# Open the images and convert them to the same format
image_filenames = ['t1.png','t2.png','t3.png','t4.png','t5.png','t5.png','t5.png']
images = []
for filename in image_filenames:
    image = Image.open(filename)
    # fondo = np.zeros
    # image = image.convert('RGBA')  # Convert to RGBA format (optional, for transparency support)
    images.append(image)

# Save the images as an animated GIF
output_filename = 'piv.gif'  # Specify the output filename
images[0].save(output_filename, save_all=True, append_images=images[1:], duration=1000, loop=0)


































