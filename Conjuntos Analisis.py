# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:08:50 2023

@author: gonza
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from scipy.optimize import curve_fit

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

c0 = (0.122, 0.467, 0.706)
c1 = (1.000, 0.498, 0.055)
c2 = (0.173, 0.627, 0.173)
c3 = (0.839, 0.152, 0.157)
# colores = [c2, c3, c0, c1]
colores = [c0, c3, c2, c1]

#%% Parámetros de ploteo

plt.rcParams['figure.figsize'] = [7,7]
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = "Times New Roman"

#%% Células

cs10 = [ 1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 20, 22, 25 ] 
ss10 = [ 14, 15, 17, 18, 26, 27, 28, 29]
cs7 =  [ (11,3), (11,4),  (10,1), (10,2), (10,5), (9,1), (1,1) ]
ss7 =  [ (8,2), (8,3), (7,1), (7,2), (6,2), (6,3), (6,4), (5,4), (4,1), (3,3) ]
conjuntos = [cs10, ss10, cs7, ss7]


muestra = [ 9, 25, (11,4), (10,5) ]
linea_muestra = [ 'MCF10', 'MCF10', 'MCF7', 'MCF7' ]

# valores de A?
# valores de exploración?

#%%

cel = []
suero_ = []
linea_ = []

defo = [] # m
fuer = [] # kPa
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
        R_s = np.sqrt( X_s**2 + Y_s**2 )*ps*1e-6 # m

        plt.figure(figsize = [6,4] )
        plt.imshow( mascara, cmap = 'Reds', alpha = 0.5 )
        plt.imshow( mascara10, cmap = 'Reds', alpha = 0.5 )

        # plt.quiver(x,y,X_0,-Y_0, res, cmap = cm_crimson, scale = 100, pivot='tail')
        # plt.quiver(x,y,X_nmt,-Y_nmt, scale = 100, pivot='tail')
        plt.quiver(x,y,X_s,-Y_s, scale = 100, pivot='tail')

        # auxi.barra_de_escala( 10, sep = 1.5, pixel_size = ps, font_size = 11, color = 'k', more_text = linea, a_lot_of_text = suero )
        auxi.barra_de_escala( 10, sep = 1.5, pixel_size = ps, font_size = 11, color = 'k', more_text = linea, a_lot_of_text = str(N) )

        plt.xlim([0,1023])
        plt.ylim([1023,0])
        plt.show()
        
        
        E, nu = 31.6, 0.5      # kPa
        uX, uY = X_s, Y_s
        x_plot, y_plot = x, y

        # lam = 0
        lam = auxi.busca_lambda( uY, uX, ps*1e-6, solo_lambda = True )

        ty, tx, vy0, vx0 = TFM.traction(uY, uX, ps*1e-6, ws*1e-6, E*1e3, nu, lam, Lcurve = True)
        # ty, tx = TFM.smooth(ty,3), TFM.smooth(tx,3)
        tr = np.sqrt( np.real(ty)**2 + np.real(tx)**2 )

        plt.figure(figsize = [6,4] )
        plt.quiver(x_plot, y_plot, tx, -ty, scale = 20000)
        plt.imshow( mascara, cmap = 'Reds', alpha = 0.5 )
        auxi.barra_de_escala( 10, sep = 1.5, pixel_size = ps, font_size = '11', color = 'k', more_text = 'T' )
        plt.xlim([0,1023])
        plt.ylim([1023,0])        
        plt.show()
        
        cel.append(N)
        suero_.append(suero)
        linea_.append(linea)

        defo.append( np.sum( R_s*auxi.reshape_mask(mascara10, x, y) )  )
        fuer.append( np.sum( tr*auxi.reshape_mask(mascara10, x, y) )  )
        cant.append( np.sum( auxi.reshape_mask(mascara10, x, y) ) )
        area.append( np.sum( mascara)*(ps*1e-6)**2 )
        


#%%

data = pd.DataFrame()


data["Celula"] = cel
data["Area"] = area
data["N"] = cant

data["Deformación"] = defo
data["Tracción"] = fuer
data["Suero"] = suero_
data["Linea"] = linea_


data.to_csv( "data23.01.csv" )


#%%

df = pd.read_csv( r"C:\Users\gonza\1\Tesis\TFM\data22.01.csv", index_col = 'Celula')
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
fac = 1e3
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

plt.subplot(3,1,2)
sns.boxplot( valoresT, palette = colores )
plt.grid( True )
plt.ylabel( "Tracción promedio [Pa]" )
plt.ylim([-10, 410])
plt.xticks([0,1,2,3], ['','','',''] )

plt.subplot(3,1,3)
sns.boxplot( valoresF, palette = colores )
plt.grid( True )
plt.ylabel( "Fuerza [mN]" )
plt.ylim([-0.05, 0.65])
plt.xticks([0,1,2,3], ['MCF10A CS','MCF10A SS','MCF7 CS','MCF7 SS'] )

plt.text( 0, -0.17, "(N = 13)", ha = 'center' )
plt.text( 1, -0.17, "(N = 8)", ha = 'center' )
plt.text( 2, -0.17, "(N = 7)", ha = 'center' )
plt.text( 3, -0.17, "(N = 10)", ha = 'center' )

plt.show()

#%%

m, T = auxi.crea_tabla(valoresF, tt, label = "ttest fuerza")
print(m)


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
plt.xlim([-0.03, 0.73])
plt.ylim([-10,1010])
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],['']*8)
plt.text( 0.68, 30, '(a)', fontsize = 15)

plt.subplot( 2,2,3 )
plt.plot( valoresD[0], valoresF[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresD[1], valoresF[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresD[2], valoresF[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75)
plt.plot( valoresD[3], valoresF[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
plt.ylabel("Fuerza [mN]")
plt.xlabel("Deformación promedio [µm]")
plt.xlim([-0.03, 0.73])
plt.ylim([-0.1,2.9])
plt.text( 0.68, 0.03, '(c)', fontsize = 15)

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
plt.xlim([-10, 3310])
plt.ylim([-0.1,2.9])
plt.yticks([0,0.5,1,1.5,2,2.5], ['']*6)
plt.text( 3070, 0.03, '(d)', fontsize = 15)


plt.subplot( 2,2,2 )
plt.plot( valoresA[0], valoresT[0], 'o', color = colores[0], label = 'MCF10A CS', alpha = 0.75)
plt.plot( valoresA[1], valoresT[1], 'o', color = colores[1], label = 'MCF10A SS', alpha = 0.75)
plt.plot( valoresA[2], valoresT[2], 'o', color = colores[2], label = 'MCF7 CS', alpha = 0.75 )
plt.plot( valoresA[3], valoresT[3], 'o', color = colores[3], label = 'MCF7 SS', alpha = 0.75 )
# plt.legend()
plt.grid()
# plt.ylabel("Tracción promedio [Pa]")
# plt.xlabel("Area [µm²]")
plt.xlim([-10, 3310])
plt.ylim([-10,1010])
plt.xticks([0,500,1000,1500,2000,2500,3000], ['']*7)
plt.yticks([0,200,400,600,800,1000], ['']*6)
plt.text( 3070, 30, '(b)', fontsize = 15)

plt.show()

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




#%%

valoresD_todos = valoresD[0].tolist() + valoresD[1].tolist() + valoresD[2].tolist() + valoresD[3].tolist()
valoresT_todos = valoresT[0].tolist() + valoresT[1].tolist() + valoresT[2].tolist() + valoresT[3].tolist()

def rec(x, m, b):
    return m*x + b

popt, pcov = curve_fit(rec, valoresD_todos, valoresT_todos )

m, b = popt[0], popt[1]
Dm, Db = np.sqrt( pcov[0,0] ), np.sqrt( pcov[1,1] )


D_plot = np.array([0,0.7])
T_plot = popt[0]*D_plot + popt[1]

plt.figure(figsize = [6.44,4])
plt.plot( valoresD_todos, valoresT_todos, 'o', color = colores[0])
plt.plot( D_plot, T_plot, c = 'k', ls = 'dashed' )


# m = (1440+30)Pa/µm
# b = (20+6)Pa

















