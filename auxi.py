# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 11:06:36 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import oiffile as of
import TFM

def barra_de_escala( scale_length, pixel_size = 0.1007, scale_unit = 'µm', loc = 'lower right', sep = 1, img_len = 1000, font_size = "x-large", color = 'white', text = True, more_text = '', a_lot_of_text = '' ):
    scale_pixels = scale_length/pixel_size
    scale_bar_length = int( scale_pixels / plt.rcParams['figure.dpi'])  # Convert scale length to figure units
    
    for i in np.arange(0, -sep/pixel_size, -0.1):
        plt.plot([ img_len - sep/pixel_size - scale_pixels, img_len - sep/pixel_size ], [img_len - sep/pixel_size + i, img_len - sep/pixel_size + i], color=color, linewidth = 2)
    if text:
        plt.text(img_len - sep/pixel_size - scale_pixels/2, img_len - 3*sep/pixel_size , f'{scale_length} {scale_unit}', color=color, weight='bold', ha='center', va = 'bottom', fontsize = font_size )

    if more_text != '':    
        plt.text(img_len - sep/pixel_size/2, sep/pixel_size , more_text, color=color, weight='bold', ha='right', va = 'top', fontsize = font_size )
    if a_lot_of_text != '':
        plt.text(sep/pixel_size, img_len - sep/pixel_size/2 , a_lot_of_text, color=color, weight='bold', ha='left', va = 'bottom', fontsize = font_size )

    plt.xticks([])
    plt.yticks([])


def busca_esferas( img, ps = 0.1007, win = 2.5, A = 1, std_img = 0 ):
    if std_img == 0:
        std_img = np.std( img )
    
    ws = int( np.round( win/ps ) )
    l = int( len(img)/ws )
    std_matrix = np.zeros( [l]*2 )
    
    for j in range(l):
        for i in range(l):
            img_ = img[ int(ws*j) : int(ws*(j+1)), int(ws*i) : int(ws*(i+1)) ]
            std_matrix[j,i] = np.std( img_ )
    
    win_with_sph = np.zeros( [l]*2 )
    win_with_sph[ std_matrix/std_img > A ] = 1
    
    return win_with_sph, int(ws*l)

def busca_A( img, f0, ps = 0.1007, win = 3, A0 = 0.85, std_img = 0 ):
    it = 0
    fraccion = 0
    while np.abs( f0 - fraccion ) > 0.01 and it < 30:
        ventanas_con_esferas, dim = busca_esferas( img, ps, win, A = A0, std_img = 0 )
        fraccion = np.mean( ventanas_con_esferas )
        
        if fraccion > f0:
            A0 = A0 + 0.01
        elif fraccion < f0:
            A0 = A0 - 0.01
        it += 1    
            
    return A0

def busca_manchas(img, th = 1000):
    sat = np.zeros([1024]*2)
    sat[ img > th ] = 1
    sat = area_upper(sat, kernel_size = 20, threshold = 0.1)
    return sat

def round_pro(array2D):
    """
    Parameters
    ----------
    array2D : numpy.2darray
        2 dimentional array.

    Returns
    -------
    round_array2D : numpy.2darray
        The same 2 dimentional array but each element is an intiger.

    """
    round_array2D = array2D
    if type( array2D ) == np.ndarray:
        round_array2D = np.round( array2D )
    return round_array2D

def smooth(img, kernel_size):
    """
    Parameters
    ----------
    img : numpy.2darray like
        2 dimentional array - image.
    kernel_size : int
        lenght of the kernel used to blur the image

    Returns
    -------
    smooth_img : numpy.2darray
        2 dimentional array - image, each pixel value is the mean of the pixels values at kernel´s area, cetered in that pixel.

    """
    kernel = np.ones([kernel_size]*2)/kernel_size**2
    smooth_img = signal.convolve2d(img, kernel, mode='same')
    return smooth_img

def median_blur(img, kernel_size):
    """
    Parameters
    ----------
    img : numpy.2darray like
        2 dimentional array - image.
    kernel_size : int
        lenght of the kernel used to blur the image
    
    Returns
    -------
    blur : numpy.2darray like
        2 dimentional array - image, each pixel value is the median of the pixels values at kernel´s area, cetered in that pixel.

    """
    L, k = len(img), kernel_size
    img0 = np.ones([L + k//2, L + k//2])*np.mean( img.flatten() )
    img0[k//2:L + k//2, k//2:L + k//2] = img
    blur = np.zeros([L,L])
    for j in range(L):
        for i in range(L):
            muestra = img0[ j: j+k , i: i+k ].flatten()
            media = np.median(muestra)
            blur[j,i] = media
    return blur 


def area_upper(binary_map, kernel_size = 50, threshold = 0.1):
    """
    Parameters
    ----------
    binary_map : numpy.2darray like
        2 dimentional array - binary image pixels take 1 value inside cell area and 0 outside.
    kernel_size : int, optional
        lenght of the kernel used to blur the image. The default is 50.
    threshold : float, optional
        Value between 0 and 1, if it is more than 0.5 area increase, if it is less, area decrease. The default is 0.1.

    Returns
    -------
    mask : numpy.2darray like
        Smoothed and rebinarized the cell mask. Now pixels take 1 value inside cell and a border arround it.

    """
    l = len(binary_map)
    mask = np.zeros([l,l])
    smooth_map = TFM.smooth( binary_map, kernel_size)
    mask[ smooth_map > threshold ] = 1
    return mask


def contour(img):
    """
    Parameters
    ----------
    img : numpy.2darray like
        2 dimentional array - image.

    Returns
    -------
    bordes : numpy.2darray
        Edges of the input.

    """
    s = [[ 1,  2,  1],  
         [ 0,  0,  0], 
         [-1, -2, -1]]

    HR = signal.convolve2d(img, s)
    VR = signal.convolve2d(img, np.transpose(s))
    bordes = (HR**2 + VR**2)**0.5
    return bordes


def center_of_mass(binary_map):
    """
    Parameters
    ----------
    binary_map : numpy.2darray like
        2 dimentional array - binary image pixels take 1 value inside cell area and 0 outside.

    Returns
    -------
    CM : tuple
        Coordinates of the center of mass of the cell mask.

    """
    CMy = 0
    CMx = 0
    mass = np.sum(binary_map)
    l = len(binary_map)
    for j in range(l):
        for i in range(l):
            CMy += j*binary_map[j,i]/mass
            CMx += i*binary_map[j,i]/mass
    CM = ( int(CMy), int(CMx) )
    return CM 

def normalizar(array):
    array = np.array(array)
    return (array - min(array))/max(array - min(array))


def desvio_por_plano(stack):
    desvios = np.zeros(len(stack))
    for z in range(len(stack)):
        desvios[z] = np.std( stack[z] )
    return normalizar( desvios )

def border(img, y0, k = 3):
    vecinos = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    mascara = np.zeros(img.shape)
    img_s = smooth( img, k )
    mascara[ img_s > 0.5 ] = 1

    y_borde = [ y0 ]
    x_borde = [    ]
    j = 0
    while len(x_borde) == 0:
        if mascara[600,j] == 1:
            x_borde.append(j-1)
        j += 1    

    seguir = True
    while seguir:
        x0 = x_borde[-1] 
        y0 = y_borde[-1]
        for j in range(8):
            v0 = mascara[ y0 + vecinos[j-1][0], x0 + vecinos[j-1][1] ]
            v1 = mascara[   y0 + vecinos[j][0],   x0 + vecinos[j][1] ]
            if v0 == 0 and v1 == 1:
                x_borde.append( x0 + vecinos[j-1][1] )
                y_borde.append( y0 + vecinos[j-1][0] )
        if ( x_borde[-1] == x_borde[0] and y_borde[-1] == y_borde[0] and len(x_borde) > 1 ) or len(x_borde) > 10000:
            seguir = False

    borde = np.concatenate( (  np.reshape( np.array( y_borde ), [1, len(y_borde)]) , np.reshape( np.array( x_borde ) ,  [1, len(y_borde)] ) ) , axis = 0 )

    return borde

def reshape_mask(mask, dom_x, dom_y):
    l = len(dom_x)
    new_mask = np.zeros( [l,l] )
    for j in range(l):
        for i in range(l):
            if  0 < int( np.round(dom_x[j,i]) ) < 1024 and 0 < int( np.round(dom_y[j,i]) ) < 1024 and int(mask[ int(dom_y[j,i]), int(dom_x[j,i]) ]) == 1:
                new_mask[j,i] = 1
    return new_mask

def interpolate( array ):
    l = len(array)
    l_ = int(2*l-1)
    array_ = np.zeros([l_,l_])
    for j in range(l):
        for i in range(l_):
            if i%2 == 0:
                array_[2*j,i] = array[j,i//2]
            else:
                array_[2*j,i] = (array[j,i//2] + array[j,i//2+1])/2
    for j in range(l-1):
        array_[2*j+1] = (array_[2*j] + array_[2*j+2])/2             
    return array_

def anti_interpolate( array_ ):
    l_ = len(array_)
    l = int((l_+1)/2)
    array = np.zeros([l,l])
    for j in range(l):
        for i in range(l):
            array[j,i] = array_[2*j,2*i]    
    return array


def celula( N, line, place = 'home' ):
    if place == 'home':
        path = r"C:\Users\gonza\1\Tesis\2023\\"
    else:    
        path = r"D:\Gonzalo\\"
    
    if line == 'MCF7':
        carpetas = ["23.08.17 - gon MCF7 1 - C16", "23.08.18 - gon MCF7 2 - B16", "23.08.18 - gon MCF7 3 - A16", "23.08.24 - gon MCF7 4 - A23", "23.08.24 - gon MCF7 5 - B23", "23.08.25 - gon MCF7 6 - D23", "23.08.25 - gon MCF7 7 - C23", "23.08.31 - gon MCF7 8 - B30", "23.08.31 - gon MCF7 9 - A30", "23.09.01 - gon MCF7 10 - C30", "23.09.01 - gon MCF7 11 - D30"]
        pre_post7 = {(11,2):(6,5), (11,3):(4,3), (11,4):(5,5), (10,1):(4,3), (10,2):(8,4), (10,5):(4,4), (9,1):(3,3), (1,1):(6,5),    (8,2):(4,4), (8,3):(5,5), (7,1):(6,4), (7,2):(5,4), (6,2):(6,5), (6,3):(5,4), (6,4):(4,5), (5,4):(3,4), (4,1):(7,7), (3,3):(6,6)   }
        full_path1 = path + carpetas[ N[0] - 1 ]

        name = carpetas[ N[0] - 1 ][-3:] + "_R0" + str( N[1] )
        print('invocando',name)
        metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
        metadata_region = metadata.loc[ metadata["Región"] == N[1] ]    
        

        zoom = metadata_region["Zoom"].values[0]
        pixel_size = 1/(4.97*zoom)

        stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
        stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
        celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
        celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[1, 2 + pre_post7[N][0] - pre_post7[N][1] ]
        mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_00um.png")
        if mascara[1,1] == 1:
            mascara = 1 - mascara
        mascara10 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_10um.png")
        if mascara10[1,1] == 1:
            mascara10 = 1 - mascara10        
        mascara20 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF7\\" + name + "_m_20um.png")
        if mascara20[1,1] == 1:
            mascara20 = 1 - mascara20        
        
        pre = stack_pre[ pre_post7[N][0] ]
        
        if N == (8,3):  
            post = TFM.correct_driff( stack_post[ pre_post7[N][1] ], pre, 300 )
        elif N == (7,1):  
            post = TFM.unrotate( stack_post[ pre_post7[N][1] ], pre, 50, exploration_angle = 1)
        elif N == (4,1):
            pre = np.concatenate( ( pre, np.ones([4, 1024])*np.mean(pre) ), axis = 0  )
            post = TFM.correct_driff( stack_post[ pre_post7[N][1] ], pre, 50 )
        elif N == (3,3):  
            post = TFM.unrotate( stack_post[ pre_post7[N][1] ], pre, 50, exploration_angle = 1)    
        else:
            post = TFM.correct_driff( stack_post[ pre_post7[N][1] ], pre, 50 )
            
        if N == (1,1):
            delta = 4
            pre_profundo = stack_pre[ pre_post7[N][0] + delta ]
            post_profundo = TFM.correct_driff( stack_post[ pre_post7[N][1] + delta ], pre_profundo, 50 )
            
            sat = busca_manchas(pre)
            
            pre = pre*(1-sat) + pre_profundo*sat
            post = post*(1-sat) + post_profundo*sat
    

    if line == 'MCF10':
        carpetas = ["23.10.05 - gon MCF10 1 - A04", "23.10.05 - gon MCF10 2 - D04", "23.10.05 - gon MCF10 3 - E04", "23.10.06 - gon MCF10 4 - C04", "23.10.19 - gon MCF10 6 - G18", "23.10.20 - gon MCF10 7 - I18" ]
        distribucion = [ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5 ]
        pre10 =  [ 4, 4, 5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 5, 2, 4, 4, 4, 4, 4,  4, 6, 5, 6, 5, 5,  3, 4, 5, 5, 3 ]
        post10 = [ 4, 4, 6, 3, 2, 5, 3, 3, 4, 2, 5, 4, 4, 4, 4, 5, 5, 4, 4,  4, 5, 7, 8, 4, 6,  4, 5, 6, 4, 4 ]

        full_path1 = path + carpetas[ distribucion[N-1] ]
        
        name = carpetas[ distribucion[N-1] ][-3:] + "_R" + str(int( 100 +N ))[1:]
        print('invocando',name)
        metadata = pd.read_csv( full_path1 + "\Data.csv", delimiter = ',', usecols=np.arange(3,15,1))
        metadata_region = metadata.loc[ metadata["Región"] == N ]
        
        zoom = metadata_region["Zoom"].values[0]
        pixel_size = 1/(4.97*zoom)
        
        stack_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[0]
        stack_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[-1]+".oif" )[0]
        celula_pre = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'PRE' ]["Archivo"].values[0]+".oif" )[1,2]
        celula_post = of.imread( full_path1 + r"\\" + metadata_region.loc[ metadata_region["Tipo"] == 'POST' ]["Archivo"].values[0]+".oif" )[1,2+pre10[N-1]-post10[N-1]]
        mascara = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_00um.csv")
        mascara10 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_10um.csv")
        mascara20 = np.loadtxt( path[:-6] + r"PIV\Mascaras MCF10\\" + name + "_m_20um.csv")

        pre = stack_pre[ pre10[N-1] ]
        post = TFM.correct_driff( stack_post[ post10[N-1] ], pre, 50 )
        
        if N == 1 or N == 4 or N == 8 or N == 14 or N == 15 or N == 17 or N == 18:
            delta = 4
            if N == 1 or N == 17:
                delta = 6
            pre_profundo = stack_pre[ pre10[N-1] + delta ]
            post_profundo = TFM.correct_driff( stack_post[ post10[N-1] + delta ], pre_profundo, 50 )
            sat = busca_manchas(pre, 700)
            pre = pre*(1-sat) + pre_profundo*sat
            post = post*(1-sat) + post_profundo*sat    

    return pre, post, mascara, mascara10, mascara20, pixel_size




