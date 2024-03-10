# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:00:04 2024

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit

from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import ttest_ind as tt
from scipy.stats import ranksums as rt


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

c0 = (0.839, 0.152, 0.157)
c1 = (0.580, 0.000, 0.827)
c2 = (0.122, 0.467, 0.706)
c3 = (0.173, 0.627, 0.173)
c4 = (1.000, 0.498, 0.055)

colores = [c0, c1, c2, c3, c4]

cm0 = ListedColormap( [(1, 1, 1), c0 ] )
cm1 = ListedColormap( [(1, 1, 1), c1 ] )
cm2 = ListedColormap( [(1, 1, 1), c2 ] )
cm3 = ListedColormap( [(1, 1, 1), c3 ] )
cm4 = ListedColormap( [(1, 1, 1), c4 ] )

color_maps = [cm0, cm1, cm2, cm3, cm4]

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

#%%
# 10 CS: 18, 6, 5   SS: 29, 27, 17, 14
data0 = auxi.celula( 6, 'MCF10', place = 'home', trans = True ) # 10 CS
data1 = auxi.celula( 29, 'MCF10', place = 'home', trans = True  ) # 10 SS

data2 = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True  ) # 7 CS
data3 = auxi.celula( (7,2), 'MCF7', place = 'home', trans = True  ) # 7 SS

data_cels = [ data0, data1, data2, data3 ]

#%%
celulas = [ data_cels[i][2] - auxi.median_blur(data_cels[i][2], 50)  for  i in range(4)  ]
# celulas = [ data_cels[i][2]  for  i in range(4)  ]
#%%

data_plot = []

for i in range(4):
    pre, post, trans, mascara, mascara10, mascara20, ps = data_cels[i]
    # plt.imshow(mascara, cmap = color_maps[i])
    # plt.show()
    
    ws, exp = 2.5, 0.7
   
    A1 = auxi.busca_A( pre, 0.75, ps, win = 2.5, A0 = 0.85 )
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
    X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )

    E, nu = 31.6, 0.5   
    # uy, ux = Y_s, X_s
    uy, ux = Y_nmt, X_nmt
    x_plot, y_plot = x, y

    if i == 0:
        uy, ux = Y_s*0.8, X_s*0.8

    if i == 1:
        uy, ux = Y_s*1.2, X_s*1.2


    lam = auxi.busca_lambda( uy, ux, ps, solo_lambda = True )    
    ty, tx = TFM.traction(uy, ux, ps*1e-6, ws*1e-6, E*1e3, nu, lam )

    celula = celulas[i]

    data_i = celula, mascara, mascara10, mascara20, uy, ux, ty, tx, y_plot, x_plot, ps

    data_plot.append(data_i)

#%%

plt.figure( figsize = [7,7], layout = 'compressed' )

# orden = [1,4,7,10,2,5,8,11,3,6,9,12]
orden = [1,2,3,4,5,6,7,8,9,10,11,12]
tipo = ['MCF10A Con Suero','MCF10A Sin Suero','MCF7 Con Suero','MCF7 Sin Suero']


for i in range(4):    
     
    celula, mascara, mascara10, mascara20, uy, ux, ty, tx, y_plot, x_plot, ps = data_plot[i]
    M = auxi.smooth( auxi.reshape_mask(mascara20, x_plot, y_plot), 10 )


    # Celula
    bor = auxi.border(mascara, k = 7)
    plt.subplot(4,3, orden[3*i] )
    plt.imshow( celula, cmap = 'gray' )
    auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'w', more_text = tipo[i] )
    plt.plot( bor[1], bor[0], c = 'w', ls = 'dashed', lw = 0.7 )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Deformacion
    plt.subplot(4,3, orden[3*i+1] )
    plt.quiver(x_plot, y_plot, ux*ps*1e-6*M, -uy*ps*1e-6*M, scale = 0.00001)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.3 )
    plt.imshow( mascara10, cmap = color_maps[i], alpha = 0.3 )
    auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'u', text = False  )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Traccion
    plt.subplot(4,3,orden[3*i+2] )
    # plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
    plt.quiver(x_plot, y_plot, tx*M, -ty*M, scale = 20000)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.6 )
    auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps, font_size = '12', color = 'k', more_text = 't', text = False )
    plt.xlim([0,1023])
    plt.ylim([1023,0])


#%%


#%%
# 10 CS: 18, 6, 5   SS: 29, 27, 17, 14
data0 = auxi.celula( 6, 'MCF10', place = 'home', trans = True ) # 10 CS
data1 = auxi.celula( 29, 'MCF10', place = 'home', trans = True  ) # 10 SS

data2 = auxi.celula( (11,4), 'MCF7', place = 'home', trans = True  ) # 7 CS
data3 = auxi.celula( (7,2), 'MCF7', place = 'home', trans = True  ) # 7 SS

data_cels = [ data0, data1, data2, data3 ]

#%%
celulas = [ data_cels[i][2] - auxi.median_blur(data_cels[i][2], 50)  for  i in range(4)  ]
# celulas = [ data_cels[i][2]  for  i in range(4)  ]
#%%

data_plot = []

for i in range(4):
    pre, post, trans, mascara, mascara10, mascara20, ps = data_cels[i]
    # plt.imshow(mascara, cmap = color_maps[i])
    # plt.show()
    
    ws, exp = 2.5, 0.7
   
    A1 = auxi.busca_A( pre, 0.75, ps, win = 2.5, A0 = 0.85 )
    dominio, deformacion = TFM.n_iterations( post, pre, int( np.round(ws/ps)*4 ), 3, exploration = int(exp/ps), mode = "Smooth3", A = A1)
    x, y = dominio
    Y_0, X_0 = deformacion 
    Y_nmt, X_nmt, res = TFM.nmt(*deformacion, 0.2, 5)
    X_s, Y_s = TFM.smooth(  X_nmt, 3 ), TFM.smooth(  Y_nmt, 3 )

    E, nu = 31.6, 0.5   
    # uy, ux = Y_s, X_s
    uy, ux = Y_nmt, X_nmt
    x_plot, y_plot = x, y

    if i == 0:
        uy, ux = Y_s*0.8, X_s*0.8

    if i == 1:
        uy, ux = Y_s*1.2, X_s*1.2


    lam = auxi.busca_lambda( uy, ux, ps, solo_lambda = True )    
    ty, tx = TFM.traction(uy, ux, ps*1e-6, ws*1e-6, E*1e3, nu, lam )

    celula = celulas[i]

    data_i = celula, mascara, mascara10, mascara20, uy, ux, ty, tx, y_plot, x_plot, ps

    data_plot.append(data_i)

#%%

plt.figure( figsize = [7,7], layout = 'compressed' )

# orden = [1,4,7,10,2,5,8,11,3,6,9,12]
orden = [1,2,3,4,5,6,7,8,9,10,11,12]
tipo = ['MCF10A Con Suero','MCF10A Sin Suero','MCF7 Con Suero','MCF7 Sin Suero']


for i in range(4):    
     
    celula, mascara, mascara10, mascara20, uy, ux, ty, tx, y_plot, x_plot, ps = data_plot[i]
    M = auxi.smooth( auxi.reshape_mask(mascara20, x_plot, y_plot), 10 )


    # Celula
    bor = auxi.border(mascara, k = 7)
    plt.subplot(4,3, orden[3*i] )
    plt.imshow( celula, cmap = 'gray' )
    auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps, font_size = '10', color = 'w', more_text = tipo[i] )
    plt.plot( bor[1], bor[0], c = 'w', ls = 'dashed', lw = 0.7 )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Deformacion
    plt.subplot(4,3, orden[3*i+1] )
    plt.quiver(x_plot, y_plot, ux*ps*1e-6*M, -uy*ps*1e-6*M, scale = 0.00001)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.3 )
    plt.imshow( mascara10, cmap = color_maps[i], alpha = 0.3 )
    auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps,  font_size = '12', color = 'k', more_text = 'u', text = False  )
    plt.xlim([0,1023])
    plt.ylim([1023,0])

    # Traccion
    plt.subplot(4,3,orden[3*i+2] )
    # plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
    plt.quiver(x_plot, y_plot, tx*M, -ty*M, scale = 20000)
    plt.imshow( mascara, cmap = color_maps[i], alpha = 0.6 )
    auxi.barra_de_escala( 20, sep = 1.5,  pixel_size = ps, font_size = '12', color = 'k', more_text = 't', text = False )
    plt.xlim([0,1023])
    plt.ylim([1023,0])




























#%%



#%%

# df = pd.read_csv( r"C:\Users\gonza\1\Tesis\TFM\data22.01.csv", index_col = 'Celula')
df = pd.read_csv( r"C:\Users\gonza\1\Tesis\TFM\data22.01_v2.txt", index_col = 'Celula')

df['1/N'] = 1/df['N']
df['1/Area'] = 1/df['Area']
df10 = df.loc[ df["Linea"] == 'MCF10' ]
df7 = df.loc[ df["Linea"] == 'MCF7' ]

#%% Conjuntos

key1 = 'Deformación'
key2 = '1/N'
fac = 1e6
valoresD = [ df10.loc[ df10["Suero"] == 'CS' ][key1].values*df10.loc[ df10["Suero"] == 'CS' ][key2].values*fac  , df10.loc[ df10["Suero"] == 'SS' ][key1].values*df10.loc[ df10["Suero"] == 'SS' ][key2].values*fac, df7.loc[ df7["Suero"] == 'CS' ][key1].values*df7.loc[ df7["Suero"] == 'CS' ][key2].values*fac, df7.loc[ df7["Suero"] == 'SS' ][key1].values*df7.loc[ df7["Suero"] == 'SS' ][key2].values*fac ]

key1 = 'Tracción'
key2 = '1/N'
fac = 1#e6
valoresT = [ df10.loc[ df10["Suero"] == 'CS' ][key1].values*df10.loc[ df10["Suero"] == 'CS' ][key2].values*fac  , df10.loc[ df10["Suero"] == 'SS' ][key1].values*df10.loc[ df10["Suero"] == 'SS' ][key2].values*fac, df7.loc[ df7["Suero"] == 'CS' ][key1].values*df7.loc[ df7["Suero"] == 'CS' ][key2].values*fac, df7.loc[ df7["Suero"] == 'SS' ][key1].values*df7.loc[ df7["Suero"] == 'SS' ][key2].values*fac ]

key1 = 'Tracción'
key2 = 'Area'
fac = 1e6
valoresF = [ df10.loc[ df10["Suero"] == 'CS' ][key1].values*df10.loc[ df10["Suero"] == 'CS' ][key2].values*fac  , df10.loc[ df10["Suero"] == 'SS' ][key1].values*df10.loc[ df10["Suero"] == 'SS' ][key2].values*fac, df7.loc[ df7["Suero"] == 'CS' ][key1].values*df7.loc[ df7["Suero"] == 'CS' ][key2].values*fac, df7.loc[ df7["Suero"] == 'SS' ][key1].values*df7.loc[ df7["Suero"] == 'SS' ][key2].values*fac ]

key1 = 'Area'
fac = 1e12
valoresA = [ df10.loc[ df10["Suero"] == 'CS' ][key1].values*fac, df10.loc[ df10["Suero"] == 'SS' ][key1].values*fac, df7.loc[ df7["Suero"] == 'CS' ][key1].values*fac, df7.loc[ df7["Suero"] == 'SS' ][key1].values*fac ]


#%%
plt.rcParams['font.size'] = 12

plt.figure( figsize = [5,8], layout = "compressed" )

plt.subplot(3,1,1)
sns.boxplot( valoresD, palette = colores )
plt.grid( True )
plt.ylabel( "Deformación promedio [µm]" )
plt.ylim([-0.01, 0.27])
plt.xticks([0,1,2,3], ['','','',''] )
plt.text( 3, 0.15, "*", ha = 'center', fontsize = 21 )


plt.subplot(3,1,2)
sns.boxplot( valoresT, palette = colores )
plt.grid( True )
plt.ylabel( "Tracción promedio [Pa]" )
plt.ylim([-30, 430])
plt.xticks([0,1,2,3], ['','','',''] )
plt.text( 3, 200, "*", ha = 'center', fontsize = 21 )


plt.subplot(3,1,3)
sns.boxplot( valoresF, palette = colores )
plt.grid( True )
plt.ylabel( "Fuerza [µN]" )
plt.ylim([-50, 650])
plt.xticks([0,1,2,3], ['MCF10A CS','MCF10A SS','MCF7 CS','MCF7 SS'] )
plt.text( 3, 300, "*", ha = 'center', fontsize = 21 )

plt.text( 0, -170, "(N = 11)", ha = 'center' )
plt.text( 1, -170, "(N = 8)", ha = 'center' )
plt.text( 2, -170, "(N = 7)", ha = 'center' )
plt.text( 3, -170, "(N = 10)", ha = 'center' )

plt.show()

#%%

m, T = auxi.crea_tabla(valoresA, rt, label = "ttest fuerza")

print(m)
# print(T)
#%%

valoresD_todos = valoresD[0].tolist() + valoresD[1].tolist() + valoresD[2].tolist() + valoresD[3].tolist()
valoresT_todos = valoresT[0].tolist() + valoresT[1].tolist() + valoresT[2].tolist() + valoresT[3].tolist()

def rec(x, m, b):
    return m*x + b

popt, pcov = curve_fit(rec, valoresD_todos, valoresT_todos )

m, b = popt[0], popt[1]
Dm, Db = np.sqrt( pcov[0,0] ), np.sqrt( pcov[1,1] )


D_plot = np.array([0,0.25])
T_plot = popt[0]*D_plot + popt[1]

plt.figure(figsize = [6.44,4])
plt.plot( valoresD_todos, valoresT_todos, 'o', color = colores[0])
plt.plot( D_plot, T_plot, c = 'k', ls = 'dashed' )


# m = (1440+30)Pa/µm
# b = (20+6)Pa



#%%

plt.figure(figsize = [10,6.25], layout = "compressed")

plt.subplot( 2,2,1 )
plt.plot( D_plot, T_plot, c = 'k', ls = 'dashed', lw = 0.9 )
plt.plot( valoresD[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75)
plt.legend()
plt.grid()
plt.ylabel("Tracción promedio [Pa]")
# plt.xlabel("Deformación promedio [µm]")
plt.xlim([-0.03, 0.28])
plt.ylim([-10,410])
plt.xticks([0,0.05,0.1,0.15,0.2,0.25],['']*6)
plt.yticks([0,100,200,300,400], [0,100,200,300,400])
plt.text( 0.257, 15, '(a)', fontsize = 15)


plt.subplot( 2,2,2 )
plt.plot( valoresA[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
# plt.ylabel("Tracción promedio [Pa]")
# plt.xlabel("Area [µm²]")
plt.xlim([-10, 3010])
plt.ylim([-10,410])
plt.xticks([0,500,1000,1500,2000,2500,3000], ['']*7)
plt.yticks([0,100,200,300,400], ['']*5)
plt.text( 2770, 15, '(b)', fontsize = 15)


plt.subplot( 2,2,3 )
plt.plot( valoresD[0], valoresF[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresF[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresF[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresF[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
plt.ylabel("Fuerza [mN]")
plt.xlabel("Deformación promedio [µm]")
plt.xlim([-0.03, 0.28])
plt.ylim([-0.05,0.65])
plt.xticks([0,0.05,0.1,0.15,0.2,0.25],[0,0.05,0.1,0.15,0.2,0.25])
plt.yticks([0,0.2,0.4,0.6], [0,0.2,0.4,0.6])
plt.text( 0.257, 0, '(c)', fontsize = 15)


# plt.xlim([-0.02,0.272])
# plt.ylim([-0.2,1.2])


plt.subplot( 2,2,4 )
plt.plot( valoresA[0], valoresF[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresF[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresF[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresF[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
# plt.ylabel("Fuerza [mN]")
plt.xlabel("Area [µm²]")
plt.xlim([-10, 3010])
plt.ylim([-0.05,0.65])
plt.xticks([0,500,1000,1500,2000,2500,3000],[0,500,1000,1500,2000,2500,3000])
plt.yticks([0,0.2,0.4,0.6], ['']*4)
plt.text( 2770, 0, '(d)', fontsize = 15)



plt.show()


#%% En funcion del área

plt.figure(figsize = [6,4], layout = "compressed")

plt.subplot( 2,1,1 )
plt.plot( valoresA[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
plt.ylabel("Tracción promedio [Pa]")
# plt.xlabel("Area [µm²]")
plt.xlim([200, 2800])
plt.ylim([-10,410])
plt.xticks([0,500,1000,1500,2000,2500,3000], ['']*7)
plt.yticks([0,100,200,300,400])
# plt.text( 2770, 15, '(b)', fontsize = 15)


plt.subplot( 2,1,2 )
plt.plot( valoresA[0], valoresF[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresF[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresF[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresF[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
plt.ylabel("Fuerza [µN]")
plt.xlabel("Área [µm²]")
plt.xlim([200, 2800])
plt.ylim([-15,615])
plt.xticks([0,500,1000,1500,2000,2500,3000],[0,500,1000,1500,2000,2500,3000])
# plt.yticks([0,0.2,0.4,0.6], [0,0.2,0.4,0.6])
# plt.text( 2770, 0, '(d)', fontsize = 15)
plt.legend(loc = "lower right")


plt.show()


#%%

plt.figure(figsize = [6,4], layout = "compressed")

plt.plot( valoresA[0], valoresF[0]**0.5, 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresF[1]**0.5, 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresF[2]**0.5, 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresF[3]**0.5, 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
plt.legend()
plt.grid()
plt.ylabel("Fuerza^0.5 [µN^0.5]")
plt.xlabel("Área [µm²]")
plt.xlim([200, 2800])
# plt.ylim([-15,615])
plt.xticks([0,500,1000,1500,2000,2500,3000],[0,500,1000,1500,2000,2500,3000])


#%% R de pearson

area_todos = df['Area'].values
fuerza_todos = df['Tracción'].values*df['Area'].values
traccion_todos = df['Tracción'].values*df['1/N'].values
defor_todos = df['Deformación'].values*df['1/N'].values


R_trac, p_valorT = pearsonr(area_todos, traccion_todos) 
R_fuer, p_valorF = pearsonr(area_todos, fuerza_todos)
R, p = pearsonr(defor_todos, traccion_todos )

K_trac, p_valorTK = kendalltau(area_todos, traccion_todos) 
K_fuer, p_valorFK = kendalltau(area_todos, fuerza_todos)
K, pK = kendalltau(defor_todos, traccion_todos )


print(K_trac, K_fuer)
print(R_trac, R_fuer)
#%%

area_todos2 = np.concatenate( (valoresA[0],valoresA[1],valoresA[2]) )
traccion_todos2 = np.concatenate( (valoresT[0],valoresT[1],valoresT[2]) )
fuerza_todos2 = np.concatenate( (valoresF[0],valoresF[1],valoresF[2]) )

R_trac, p_valorT = pearsonr(area_todos2, traccion_todos2) 
R_fuer, p_valorF = pearsonr(area_todos2, fuerza_todos2)
# R, p = pearsonr(defor_todos, traccion_todos )

K_trac, p_valorTK = kendalltau(area_todos2, traccion_todos2) 
K_fuer, p_valorFK = kendalltau(area_todos2, fuerza_todos2)
# K, pK = kendalltau(defor_todos, traccion_todos )

print(K_trac, K_fuer)
print(R_trac, R_fuer)


#%%
plt.plot(area_todos, fuerza_todos**(0.5), "o")

#%% Defo y trac

plt.figure(figsize = [6,4], layout = "compressed")

plt.subplot( 2,1,1 )
plt.plot( D_plot, T_plot, c = 'k', ls = 'dashed', lw = 1.5 )
plt.plot( valoresD[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75)
plt.legend()
plt.grid()
plt.ylabel("Tracción promedio [Pa]")
# plt.xlabel("Deformación promedio [µm]")
plt.xlim([-0.03, 0.28])
plt.ylim([-10,410])
plt.xticks([0,0.05,0.1,0.15,0.2,0.25],['']*6)
plt.yticks([0,100,200,300,400], [0,100,200,300,400])
# plt.text( 0.257, 15, '(a)', fontsize = 15)

plt.subplot( 2,1,2 )
plt.plot( D_plot, [0]*len(D_plot), c = 'k', ls = 'dashed', lw = 1.5 )
plt.plot( valoresD[0], valoresT[0] - m*valoresD[0] - b, 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresT[1] - m*valoresD[1] - b, 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresT[2] - m*valoresD[2] - b, 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresT[3] - m*valoresD[3] - b, 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75)
# plt.legend()
plt.grid()
plt.ylabel("Residuos [Pa]")
plt.xlabel("Deformación promedio [µm]")
plt.xlim([-0.03, 0.28])
plt.ylim([-55,55])
plt.yticks([-50,-25,0,25,50])
plt.xticks([0,0.05,0.1,0.15,0.2,0.25])



#%%

# plt.plot( [0]*len(valoresA[0]), valoresA[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
# plt.plot( [1]*len(valoresA[1]), valoresA[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
# plt.plot( [2]*len(valoresA[2]), valoresA[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
# plt.plot( [3]*len(valoresA[3]), valoresA[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )

plt.plot( [0]*len(valoresA[0]), valoresA[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( [0]*len(valoresA[1]), valoresA[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( [1]*len(valoresA[2]), valoresA[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( [1]*len(valoresA[3]), valoresA[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
plt.xlim([-4,5])


#%%
plt.figure( figsize = [5,3], layout = "compressed" )
sns.boxplot( valoresA, palette = colores )
plt.grid( True )
plt.ylabel( "Área [µm²]" )
plt.ylim([300, 2100])
plt.xticks([0,1,2,3], ['MCF10A CS','MCF10A SS','MCF7 CS','MCF7 SS'] )
# plt.text( 3, 300, "*", ha = 'center', fontsize = 21 )




#%%

plt.figure(figsize = [6.44,4])
plt.plot( D_plot, T_plot, c = 'k', ls = 'dashed', lw = 0.9 )
plt.plot( valoresD[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75)

plt.legend()
plt.grid()
plt.ylabel("Tracción promedio [Pa]")
plt.xlabel("Deformación promedio [µm]")

#%%

plt.figure(figsize = [6.44,4])
plt.plot( valoresD[0], valoresF[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresF[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresF[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresF[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
plt.legend()
plt.grid()
plt.ylabel("Fuerza [mN]")
plt.xlabel("Deformación promedio [µm]")

# plt.xlim([-0.02,0.272])
# plt.ylim([-0.2,1.2])

#%%

plt.figure(figsize = [6.44,4])
plt.plot( valoresA[0], valoresF[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresF[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresF[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresF[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
plt.legend()
plt.grid()
plt.ylabel("Fuerza [mN]")
plt.xlabel("Area [µm²]")

#%%

plt.figure(figsize = [6.44,4])
plt.plot( valoresA[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
plt.legend()
plt.grid()
plt.ylabel("Tracción promedio [Pa]")
plt.xlabel("Area [µm²]")









































