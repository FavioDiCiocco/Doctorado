# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:32:31 2023

@author: favio
"""
#############################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
import time
from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, Data, RealData
from scipy.spatial.distance import jensenshannon

import warnings
warnings.filterwarnings('ignore')
#############################################################################################

def data_processor(x):
    if(isinstance(x, int)):
        return x
    elif(isinstance(x, float)):
        return int(x)
    elif(isinstance(x, str)):
        return int(x[0]) if(x[0]!="-" and int(x[0])<9) else 0
    elif(x.isnan()):
        return 0
    else:
        print("Error, no se ha identificado el tipo: {}".format(type(x)))
        
#--------------------------------------------------------------------------------
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo(t0):
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))

#############################################################################################

# Cargo el archivo de datos total

filename = "../Anes_2016/anes_timeseries_2016.dta"
df_raw_data = pd.read_stata(filename)

#############################################################################################

t0 = time.time()

# Brief description of the codes
dict_labels = {'V161114x':'Obamacare','V161151x':'Voting as duty or choice','V161158x':'Party identification', 
               'V161194x':'Birthright citizenship','V161195x':'Illegal-coming children sent back',
               'V161196x':'Build wall with Mexico','V161198':'Government assistance to blacks',
               'V161204x':'Affirmative action in the universities','V161213x':'Send troops to fight ISIS',
               'V161214x':'Allow Syrian refugees','V161225x':'Government action about rising temp.',
               'V161226x':'Employers to offer paid leave to parents','V161227x':'Allow to refuse service to same-sex couples',
               'V161228x':'Transgender bathroom use', 'V162136x':'Econ. mobility compared to 20 yrs ago',
               'V162147x':'Vaccines at school','V162150x':'Equal pay','V162162x':'Benefits of vaccin. outweight risks',
               'V162171':'Liberal/conservative self-placement (post)','V162176x':'Free-trade agreements',
               'V162180x':'Government should regulate banks','V162193x':'Government spending for healthcare','V162295x':'Torture for terrorists'
               }

# Questions from the pre-election survey
labels_removidos = ['V161082x','V161083x','V161084x','V161085x','V161229x','V161233x']

labels_pre = ['V161114x','V161151x','V161158x','V161194x','V161195x',
               'V161196x','V161198','V161204x','V161213x','V161214x','V161225x','V161226x','V161227x','V161228x']
# Questions from the post-election survey
labels_post = ['V162136x','V162147x','V162150x','V162162x','V162171','V162176x','V162180x','V162193x','V162295x']
        
labels = labels_pre + labels_post

# Questions with 7 boxes
labels_7 = ['V161114x','V161151x','V161158x','V161194x','V161196x','V161198','V161204x','V161213x','V161214x',
            'V161225x','V161226x','V162136x','V162147x','V162150x','V162162x','V162176x',
            'V162180x','V162193x','V162295x']
# Questions with 6 boxes
labels_6 = ['V161195x','V161227x', 'V161228x']

# Weights:
# V160101 for pre-election data
# V160102 for post-election data

#############################################################################################

df_data_aux = df_raw_data[labels]
df_data = pd.DataFrame()

for code in labels:
    df_data[code] = df_data_aux[code].apply(data_processor)
    
df_data[['V160101','V160102']] = df_raw_data[['V160101','V160102']]

#############################################################################################

# Graficando con Seaborn

# Gráfico de una pregunta

"""
code = 'V162162x' # Birthright Citizenship
weights = 'V160101'

sns.histplot(df_data.loc[df_data[code]>0], x=code, weights=weights, discrete=True)
plt.xlabel(dict_labels[code])
plt.show()
"""

# Gráfico de dos preguntas simultáneas

"""
code_1 = 'V161114x' # Wall with Mexico
code_2 = 'V162162x' # Birthright citizenship
weights = 'V160101'

df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]

sns.histplot(df_aux, x=code_1, y=code_2, weights=weights, discrete=True, cbar=True, cmap='inferno')
plt.xlabel(dict_labels[code_1])
plt.ylabel(dict_labels[code_2])
plt.show()
"""

#############################################################################################

# Gráfico de dos preguntas simultáneas con distribuciones individuales en los ejes
"""

weights = 'V160101'
for i,code_1 in enumerate(labels_pre):
    for code_2 in labels_pre[i+1::]:
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        plt.rcParams.update({'font.size': 28})
        # plt.figure(figsize=(40,21))
        sns.jointplot(df_aux, x=code_1, y=code_2, kind="hist", vmin=0, cmap='inferno', height = 15,
                      joint_kws={'discrete': True, 'weights': df_aux[weights]}, 
                      marginal_kws={'discrete': True, 'weights': df_aux[weights]})
        
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Datos/{}vs{}.png".format(code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()

# plt.gca().invert_yaxis()

weights = 'V160102'
for i,code_1 in enumerate(labels_post):
    for code_2 in labels_post[i+1::]:
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        plt.rcParams.update({'font.size': 28})
        # plt.figure(figsize=(40,21))
        sns.jointplot(df_aux, x=code_1, y=code_2, kind="hist", vmin=0, cmap='inferno', height = 15,
                      joint_kws={'discrete': True, 'weights': df_aux[weights]}, 
                      marginal_kws={'discrete': True, 'weights': df_aux[weights]})
        
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Datos/{}vs{}.png".format(code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()

"""
#############################################################################################

# Voy a armar los gráficos de las distribuciones de mis preguntas, cosa de tener
# visto cómo se distribuyen los histogramas de cada una. También los voy a separar
# en políticas y no políticas
"""
politicas_pre = ['V161114x','V161151x','V161158x','V161194x','V161195x','V161196x','V161198','V161204x',
             'V161213x','V161214x','V161225x','V161226x','V161227x','V161228x']

politicas_post = ['V162150x','V162171','V162176x','V162180x','V162193x']

no_politicas_post = ['V162136x','V162147x','V162162x']


weights = 'V160101'
plt.rcParams.update({'font.size': 28})
for code in politicas_pre:
    
    # Set the figure size
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    sns.histplot(df_data.loc[df_data[code]>0], x=code, weights=weights, discrete=True)
    plt.xlabel(dict_labels[code])
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Distribucion preguntas/Politicas/Histograma {}.png".format(code))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()

weights = 'V160102'
for code in politicas_post:
    
    # Set the figure size
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    sns.histplot(df_data.loc[df_data[code]>0], x=code, weights=weights, discrete=True)
    plt.xlabel(dict_labels[code])
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Distribucion preguntas/Politicas/Histograma {}.png".format(code))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()


for code in no_politicas_post:
    
    # Set the figure size
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    sns.histplot(df_data.loc[df_data[code]>0], x=code, weights=weights, discrete=True)
    plt.xlabel(dict_labels[code])
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Distribucion preguntas/No politicas/Histograma {}.png".format(code))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()
"""
#############################################################################################
"""
weights = 'V160101'
for i,code_1 in enumerate(politicas_pre):
    for code_2 in politicas_pre[i+1::]:
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        plt.rcParams.update({'font.size': 28})
        # plt.figure(figsize=(40,21))
        sns.jointplot(df_aux, x=code_1, y=code_2, kind="hist", vmin=0, cmap='inferno', height = 15,
                      joint_kws={'discrete': True, 'weights': df_aux[weights]}, 
                      marginal_kws={'discrete': True, 'weights': df_aux[weights]})
        
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Distribucion preguntas/Politicas/{}vs{}.png".format(code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()


weights = 'V160102'
for i,code_1 in enumerate(politicas_post):
    for code_2 in (politicas_post+politicas_pre)[i+1::]:
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        plt.rcParams.update({'font.size': 28})
        # plt.figure(figsize=(40,21))
        sns.jointplot(df_aux, x=code_1, y=code_2, kind="hist", vmin=0, cmap='inferno', height = 15,
                      joint_kws={'discrete': True, 'weights': df_aux[weights]}, 
                      marginal_kws={'discrete': True, 'weights': df_aux[weights]})
        
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Distribucion preguntas/Politicas/{}vs{}.png".format(code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()


weights = 'V160102'
for i,code_1 in enumerate(no_politicas_post):
    for code_2 in no_politicas_post[i+1::]:
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        plt.rcParams.update({'font.size': 28})
        # plt.figure(figsize=(40,21))
        sns.jointplot(df_aux, x=code_1, y=code_2, kind="hist", vmin=0, cmap='inferno', height = 15,
                      joint_kws={'discrete': True, 'weights': df_aux[weights]}, 
                      marginal_kws={'discrete': True, 'weights': df_aux[weights]})
        
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/Distribucion preguntas/No politicas/{}vs{}.png".format(code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()
"""
#############################################################################################

# Voy a calcular la distancia Jensen-Shannon para ver que tengo entendido cómo
# hacer esto. Voy a comparar dos distribuciones de polarización ideológica y una
# de polarización descorrelacionada. Esperaría que las primeras dos tengan menor
# distancia entre ellas que con la tercera.

# Polarización ideológica 1

code_1 = 'V161194x' # 'Birthright citizenship'
code_2 = 'V161196x' # 'Build wall with Mexico'
weights = 'V160101'

# Extraigo la distribución en hist2d
df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_1], y=df_aux[code_2], weights=df_aux[weights], vmin=0,
         bins=[np.arange(df_aux[code_1].min()-0.5, df_aux[code_1].max()+1.5, 1), np.arange(df_aux[code_2].min()-0.5, df_aux[code_2].max()+1.5, 1)])

PId1 = np.reshape(hist2d,(1,hist2d.shape[0]*hist2d.shape[1]))

# Polarización ideológica 2

code_1 = 'V161114x' # 'Obamacare'
code_2 = 'V161214x' # 'Allow Syrian refugees'
weights = 'V160101'

# Extraigo la distribución en hist2d
df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_1], y=df_aux[code_2], weights=df_aux[weights], vmin=0,
         bins=[np.arange(df_aux[code_1].min()-0.5, df_aux[code_1].max()+1.5, 1), np.arange(df_aux[code_2].min()-0.5, df_aux[code_2].max()+1.5, 1)])

PId2 = np.reshape(hist2d,(1,hist2d.shape[0]*hist2d.shape[1]))

# Polarización ideológica 3

code_1 = 'V161114x' # 'Obamacare'
code_2 = 'V161158x' # 'Party Identification'
weights = 'V160101'

# Extraigo la distribución en hist2d
df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_1], y=df_aux[code_2], weights=df_aux[weights], vmin=0,
         bins=[np.arange(df_aux[code_1].min()-0.5, df_aux[code_1].max()+1.5, 1), np.arange(df_aux[code_2].min()-0.5, df_aux[code_2].max()+1.5, 1)])

PId3 = np.reshape(hist2d,(1,hist2d.shape[0]*hist2d.shape[1]))

# Polarización descorrelacionada

code_1 = 'V161114x' # 'Obamacare'
code_2 = 'V161151x' # 'Voting as duty or choice'
weights = 'V160101'

# Extraigo la distribución en hist2d
df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_1], y=df_aux[code_2], weights=df_aux[weights], vmin=0,
         bins=[np.arange(df_aux[code_1].min()-0.5, df_aux[code_1].max()+1.5, 1), np.arange(df_aux[code_2].min()-0.5, df_aux[code_2].max()+1.5, 1)])

Pdesc = np.reshape(hist2d,(1,hist2d.shape[0]*hist2d.shape[1]))

#############################################################################################

# Ahora que tengo mis distribuciones, calculo las distancias Jensen-Shannon

print("La distancia Jensen-Shannon entre las distribuciones polarizadas ideológicamente es: ",jensenshannon(PId1, PId3, axis=1))

print("La distancia Jensen-Shannon entre la primer distribución ideológica y la descorrelacionada es: ",jensenshannon(PId1, Pdesc, axis=1))

print("La distancia Jensen-Shannon entre la segunda distribución ideológica y la descorrelacionada es: ",jensenshannon(PId2, Pdesc, axis=1))

print("La distancia Jensen-Shannon de una distribución de Consenso con una polarizada ideológicamente es: ",jensenshannon(PId1, PId2, axis=1))

Tiempo(t0)