#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:37:42 2024

@author: favio
"""

#############################################################################################
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from matplotlib.gridspec import GridSpec
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

t0 = time.time()

#############################################################################################

# Cargo el archivo de datos total

filename = "../Anes_2020/anes_timeseries_2020.dta"
df_raw_data = pd.read_stata(filename)

#############################################################################################

# Brief description of the codes
dict_labels = {'V201200':'Liberal-Conservative self Placement', 'V201225x':'Voting as duty or choice','V201231x':'Party Identity',
               'V201246':'Spending & Services', 'V201249':'Defense Spending', 'V201252':'Gov-private Medical Insurance',
               'V201255':'Guaranteed job Income', 'V201258':'Gov Assistance to Blacks', 'V201262':'Environment-Business Tradeoff',
               'V201342x':'Abortion Rights Supreme Court', 'V201345x':'Death Penalty','V201356x':'Vote by mail',
               'V201362x':'Allowing Felons to vote', 'V201372x':'Pres didnt worry Congress',
               'V201375x':'Restricting Journalist access', 'V201382x':'Corruption increased or decreased since Trump',
               'V201386x':'Impeachment', 'V201405x':'Require employers to offer paid leave to parents',
               'V201408x':'Service to same sex couples', 'V201411x':'Transgender Policy', 'V201420x':'Birthright Citizenship',
               'V201423x':'Should children brought illegally be sent back','V201426x':'Wall with Mexico',
               'V201429':'Urban Unrest','V201605x':'Political Violence compared to 4 years ago',
               'V202236x':'Allowing refugees to come to US','V202239x':'Effect of Illegal inmigration on crime rate',
               'V202242x':'Providing path to citizenship','V202245x':'Returning unauthorized immigrants to native country',
               'V202248x':'Separating children from detained immigrants','V202255x':'Less or more Government',
               'V202256':'Good for society to have more government regulation',
               'V202259x':'Government trying to reduce income inequality','V202276x':'People in rural areas get more/less from Govt.',
               'V202279x':'People in rural areas have too much/too little influence','V202282x':'People in rural areas get too much/too little respect',
               'V202286x':'Easier/Harder for working mother to bond with child','V202290x':'Better/Worse if man works and woman takes care of home',
               'V202320x':'Economic Mobility','V202328x':'Obamacare','V202331x':'Vaccines in Schools',
               'V202336x':'Regulate Greenhouse Emissions','V202341x':'Background checks for guns purchases',
               'V202344x':'Banning Rifles','V202347x':'Government buy back of "Assault-Style" Rifles',
               'V202350x':'Govt. action about opiods','V202361x':'Free trade agreements with other countries',
               'V202376x':'Federal program giving 12K a year to citizens','V202380x':'Government spending to help pay for health care',
               'V202383x':'Benefits of vaccination','V202390x':'Trasgender people serve in military',
               'V202490x':'Government treats whites or blacks better','V202493x':'Police treats whites or blacks better',
               'V202542':'Use Facebook','V202544':'Use Twitter'}

labels = list(dict_labels.keys())

labels_pre = list()
labels_post = list()

for label in labels:
    if label[3] == "1":
        labels_pre.append(label)
    elif label[3] == "2":
        labels_post.append(label)


# labels_politicos = ['V201200','V201231x','V201342x','V201345x','V201372x','V201382x','V201386x','V201408x','V201411x',
#                     'V201420x','V201426x','V201605x','V202255x','V202256','V202259x','V202328x','V202336x','V202390x']

# labels_apoliticos = ['V201405x','V201423x','V201429','V202236x','V202239x','V202276x','V202279x','V202282x','V202286x',
#                      'V202290x','V202320x','V202331x','V202341x','V202344x','V202347x','V202350x','V202361x','V202376x',
#                      'V202383x','V202542','V202544']

# labels_dudosos = ['V201225x','V201246','V201249','V201252','V201255','V201258','V201262','V201356x','V201362x','V201375x',
#                   'V202242x','V202245x','V202248x','V202380x','V202490x','V202493x']

# Primer Filtro

labels_politicos = ['V201200','V201231x','V201372x','V201386x','V201408x',
                    'V201411x','V201420x','V201426x','V202255x','V202328x','V202336x']

labels_apoliticos = ['V201429','V202320x','V202331x','V202341x','V202344x',
                     'V202350x','V202383x']

# labels_dudosos = ['V201225x','V201246','V201249','V201252','V201255','V201258',
#                   'V201262','V202242x','V202248x']

#--------------------------------------------------------------------------------

# Segundo filtro
# Revisé las distribuciones unidimensionales y me quedé con las preguntas de siete
# respuestas que tienen distribuciones bimodales o unimodales que podrían resultar
# más útiles.

#labels_politicos = ['V201372x','V201386x','V201426x','V201408x','V201411x']
#
#labels_apoliticos = ['V202331x','V202341x']
#
#labels_dudosos = ['V201225x','V201262','V202242x','V202248x']


labels_filtrados = labels_politicos + labels_apoliticos # + labels_dudosos

#############################################################################################

df_data_aux = df_raw_data[labels]
df_data = pd.DataFrame()

for code in labels_filtrados:
    df_data[code] = df_data_aux[code].apply(data_processor)
    
df_data[['V200010a','V200010b']] = df_raw_data[['V200010a','V200010b']]

Df_preguntas = pd.read_csv("Tabla pares de preguntas.csv")
Beta = 1.1
Cosd = 0.02
Df_cluster = Df_preguntas.loc[(Df_preguntas["Beta_100"]==Beta) & (Df_preguntas["Cosd_100"]==Cosd)]
# carpeta = "B11C02Cluster"


#############################################################################################
"""
# Gráfico de dos preguntas simultáneas con distribuciones individuales en los ejes

for i,code_1 in enumerate(labels_politicos):
    for code_2 in labels_politicos[i+1::]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
        
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        # Filter out rows where either code_1 or code_2 is 3
        if np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 7:
            df_filtered = df_aux[(df_aux[code_1] != 4) & (df_aux[code_2] != 4)] # Saca la cruz
        elif np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 6:
            df_filtered = df_aux[df_aux[code_1] != 4] # Saca el centro de la pregunta con siete resupuestas
        elif np.unique(df_aux[code_1]).shape[0] == 6 and np.unique(df_aux[code_2]).shape[0] == 7:
            df_filtered = df_aux[df_aux[code_2] != 4] # Saca el centro de la pregunta con siete resupuestas
        
        carpeta = "Sin Cruz"
        
        # Set up the figure and grid layout
        plt.rcParams.update({'font.size': 28})
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 5, figure=fig, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 1, 1, 0.1])
        
        # Main plot: 2D histogram
        ax_main = fig.add_subplot(gs[1:, :-2])  # 3x3 space for the main plot
        hist2d, xedges, yedges, im = ax_main.hist2d(
            x=df_filtered[code_1], 
            y=df_filtered[code_2], 
            weights=df_filtered[weights], 
            vmin=0, 
            cmap="binary", 
            density=True,
            bins=[np.arange(0.5, df_filtered[code_1].max()+1.5, 1), 
                  np.arange(0.5, df_filtered[code_2].max()+1.5, 1)]
        )
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax_main, cax=fig.add_subplot(gs[1:, -1]))  # Colorbar in the last column
        cbar.ax.tick_params(labelsize=28)  # Optionally, set the size of the colorbar labels
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format colorbar ticks to 2 decimal places
        
        # Top histogram (1D)
        ax_top = fig.add_subplot(gs[0, :-2], sharex=ax_main)
        ax_top.hist(df_filtered[code_1], bins=np.arange(0.5, df_filtered[code_1].max()+1.5, 1), weights=df_filtered[weights], color='tab:blue', edgecolor='black')
        ax_top.axis('off')  # Optionally turn off axis labels
        
        # Right histogram (1D)
        ax_right = fig.add_subplot(gs[1:, -2], sharey=ax_main)
        ax_right.hist(df_filtered[code_2], bins=np.arange(0.5, df_filtered[code_2].max()+1.5, 1), weights=df_filtered[weights], color='tab:blue', edgecolor='black', orientation='horizontal')
        ax_right.axis('off')  # Optionally turn off axis labels
        
        # Set labels
        ax_main.set_xlabel(dict_labels[code_1])
        ax_main.set_ylabel(dict_labels[code_2])

        # Save the figure
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/{}/Politicos/{}vs{}.png".format(carpeta, code_1, code_2))
        plt.savefig(direccion_guardado, bbox_inches="tight")
        plt.close()


for i,code_1 in enumerate(labels_apoliticos):
    for code_2 in labels_apoliticos[i+1::]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        if np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 7:
            df_filtered = df_aux[(df_aux[code_1] != 4) & (df_aux[code_2] != 4)] # Saca la cruz
        elif np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 6:
            df_filtered = df_aux[df_aux[code_1] != 4] # Saca el centro de la pregunta con siete resupuestas
        elif np.unique(df_aux[code_1]).shape[0] == 6 and np.unique(df_aux[code_2]).shape[0] == 7:
            df_filtered = df_aux[df_aux[code_2] != 4] # Saca el centro de la pregunta con siete resupuestas
        
        # Set up the figure and grid layout
        plt.rcParams.update({'font.size': 28})
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 5, figure=fig, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 1, 1, 0.1])
        
        carpeta = "Sin Cruz"
        
        # Main plot: 2D histogram
        ax_main = fig.add_subplot(gs[1:, :-2])  # 3x3 space for the main plot
        hist2d, xedges, yedges, im = ax_main.hist2d(
            x=df_filtered[code_1], 
            y=df_filtered[code_2], 
            weights=df_filtered[weights], 
            vmin=0, 
            cmap="binary", 
            density=True,
            bins=[np.arange(0.5, df_filtered[code_1].max()+1.5, 1), 
                  np.arange(0.5, df_filtered[code_2].max()+1.5, 1)]
        )
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax_main, cax=fig.add_subplot(gs[1:, -1]))  # Colorbar in the last column
        cbar.ax.tick_params(labelsize=28)  # Optionally, set the size of the colorbar labels
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format colorbar ticks to 2 decimal places
        
        # Top histogram (1D)
        ax_top = fig.add_subplot(gs[0, :-2], sharex=ax_main)
        ax_top.hist(df_filtered[code_1], bins=np.arange(0.5, df_filtered[code_1].max()+1.5, 1), weights=df_filtered[weights], color='tab:blue', edgecolor='black')
        ax_top.axis('off')  # Optionally turn off axis labels
        
        # Right histogram (1D)
        ax_right = fig.add_subplot(gs[1:, -2], sharey=ax_main)
        ax_right.hist(df_filtered[code_2], bins=np.arange(0.5, df_filtered[code_2].max()+1.5, 1), weights=df_filtered[weights], color='tab:blue', edgecolor='black', orientation='horizontal')
        ax_right.axis('off')  # Optionally turn off axis labels
        
        # Set labels
        ax_main.set_xlabel(dict_labels[code_1])
        ax_main.set_ylabel(dict_labels[code_2])

        # Save the figure
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/{}/No Politicos/{}vs{}.png".format(carpeta, code_1, code_2))
        plt.savefig(direccion_guardado, bbox_inches="tight")
        plt.close()


for i,code_1 in enumerate(labels_dudosos):
    for code_2 in labels_dudosos[i+1::]:
        
        if code_1[3] == '1' and code_2[3] == '1':
            weights = 'V200010a'
        else:
            weights = 'V200010b'
        
        df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
        
        if np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 7:
            df_filtered = df_aux[(df_aux[code_1] != 4) & (df_aux[code_2] != 4)] # Saca la cruz
        elif np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 6:
            df_filtered = df_aux[df_aux[code_1] != 4] # Saca el centro de la pregunta con siete resupuestas
        elif np.unique(df_aux[code_1]).shape[0] == 6 and np.unique(df_aux[code_2]).shape[0] == 7:
            df_filtered = df_aux[df_aux[code_2] != 4] # Saca el centro de la pregunta con siete resupuestas
        
        plt.rcParams.update({'font.size': 28})
        plt.figure(figsize=(24,20))
        
        
        carpeta = "Sin Cruz"
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_1], y=df_filtered[code_2], weights=df_filtered[weights], vmin=0, cmap = "inferno", density = True,
                                                bins=[np.arange(0.5, df_filtered[code_1].max()+1.5, 1), np.arange(0.5, df_filtered[code_2].max()+1.5, 1)])
        
        # Add a colorbar
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=28)  # Optionally, set the size of the colorbar labels
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format colorbar ticks to 2 decimal places
        
        plt.xlabel(dict_labels[code_1])
        plt.ylabel(dict_labels[code_2])
#        plt.gca().invert_yaxis()
        direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/{}/Dudosos/{}vs{}.png".format(carpeta,code_1,code_2))
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close()
"""

####################################################################################################################
"""

plt.rcParams.update({'font.size': 28})

for code in labels_politicos:
    
    if code[3] == '1':
        weights = 'V200010a'
    elif code[3] == '2':
        weights = 'V200010b'
    
    df_aux = df_data.loc[df_data[code]>0]
    if np.unique(df_aux[code]).shape[0] == 7:
        df_aux = df_aux[df_aux[code] != 4] # Sólo saca el centro
    
    # Set the figure size
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    hist = plt.hist(x=df_aux[code], weights=df_aux[weights], density = True,
         bins=np.arange(df_aux[code].min(), df_aux[code].max()+2, 1), align='left')
    plt.xlabel(dict_labels[code])
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/Histogramas/Politicos/Histograma {}.png".format(code))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()
    
    
for code in labels_apoliticos:
    
    if code[3] == '1':
        weights = 'V200010a'
    elif code[3] == '2':
        weights = 'V200010b'
        
    df_aux = df_data.loc[df_data[code]>0]
    if np.unique(df_aux[code]).shape[0] == 7:
        df_aux = df_aux[df_aux[code] != 4] # Sólo saca el centro
    
    # Set the figure size
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    hist = plt.hist(x=df_aux[code], weights=df_aux[weights], density = True,
         bins=np.arange(df_aux[code].min(), df_aux[code].max()+2, 1), align='left')
    plt.xlabel(dict_labels[code])
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/Histogramas/No Politicos/Histograma {}.png".format(code))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()
    


for code in labels_dudosos:
    
    if code[3] == '1':
        weights = 'V200010a'
    elif code[3] == '2':
        weights = 'V200010b'
    
    df_aux = df_data.loc[df_data[code]>0]
    if np.unique(df_aux[code]).shape[0] == 7:
        df_aux = df_aux[df_aux[code] != 4] # Sólo saca el centro
    
    # Set the figure size
    plt.figure(figsize=(20, 15))  # Adjust width and height as needed
    hist = plt.hist(x=df_aux[code], weights=df_aux[weights], density = True,
         bins=np.arange(df_aux[code].min(), df_aux[code].max()+2, 1), align='left')
    plt.xlabel(dict_labels[code])
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/Histogramas/Dudosos/Histograma {}.png".format(code))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()
"""


####################################################################################################################
"""

# Veamos si puedo hacer un poco esto que me decían de revisar que los gráficos de las distribuciones
# estén normalizados. Arranquemos revisando un gráfico en particular, cualquiera.

tuplas_datos = [('V201429','V202341x','V200010b')]# ('V201255','V201258','V200010a'),('V201200','V201420x','V200010a'),('V202320x','V202350x','V200010b')]

code_1 = 'V201258'
code_2 = 'V201255'
weights = 'V200010a'

df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]

plt.rcParams.update({'font.size': 28})
plt.figure(figsize=(20,20))
hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_1], y=df_aux[code_2], weights=df_aux[weights], vmin=0, cmap = "inferno",density = True,
                                        bins=[np.arange(df_aux[code_1].min()-0.5, df_aux[code_1].max()+1.5, 1), np.arange(df_aux[code_2].min()-0.5, df_aux[code_2].max()+1.5, 1)])
plt.xlabel(dict_labels[code_1])
plt.ylabel(dict_labels[code_2])
plt.show()

print(hist2d)

"""

####################################################################################################################

for code_1,code_2 in zip(Df_preguntas["código x"],Df_preguntas["código y"]):
        
    if code_1[3] == '1' and code_2[3] == '1':
        weights = 'V200010a'
    else:
        weights = 'V200010b'
    
    df_aux = df_data.loc[(df_data[code_1]>0) & (df_data[code_2]>0)]
    
    # Filter out rows where either code_1 or code_2 is 3
    if np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 7:
        df_filtered = df_aux[(df_aux[code_1] != 4) & (df_aux[code_2] != 4)] # Saca la cruz
    elif np.unique(df_aux[code_1]).shape[0] == 7 and np.unique(df_aux[code_2]).shape[0] == 6:
        df_filtered = df_aux[df_aux[code_1] != 4] # Saca el centro de la pregunta con siete resupuestas
    elif np.unique(df_aux[code_1]).shape[0] == 6 and np.unique(df_aux[code_2]).shape[0] == 7:
        df_filtered = df_aux[df_aux[code_2] != 4] # Saca el centro de la pregunta con siete resupuestas
    
    # Set up the figure and grid layout
    plt.rcParams.update({'font.size': 28})
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 5, figure=fig, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 1, 1, 0.1])
    
    
    # Main plot: 2D histogram
    ax_main = fig.add_subplot(gs[1:, :-2])  # 3x3 space for the main plot
    hist2d, xedges, yedges, im = ax_main.hist2d(
        x=df_filtered[code_1], 
        y=df_filtered[code_2], 
        weights=df_filtered[weights], 
        vmin=0, 
        cmap="binary", 
        density=True,
        bins=[np.arange(0.5, df_filtered[code_1].max()+1.5, 1), 
              np.arange(0.5, df_filtered[code_2].max()+1.5, 1)]
    )
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax_main, cax=fig.add_subplot(gs[1:, -1]))  # Colorbar in the last column
    cbar.ax.tick_params(labelsize=28)  # Optionally, set the size of the colorbar labels
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Format colorbar ticks to 2 decimal places
    
    # Top histogram (1D)
    ax_top = fig.add_subplot(gs[0, :-2], sharex=ax_main)
    ax_top.hist(df_filtered[code_1], bins=np.arange(0.5, df_filtered[code_1].max()+1.5, 1), weights=df_filtered[weights], color='tab:blue', edgecolor='black')
    ax_top.axis('off')  # Optionally turn off axis labels
    
    # Right histogram (1D)
    ax_right = fig.add_subplot(gs[1:, -2], sharey=ax_main)
    ax_right.hist(df_filtered[code_2], bins=np.arange(0.5, df_filtered[code_2].max()+1.5, 1), weights=df_filtered[weights], color='tab:blue', edgecolor='black', orientation='horizontal')
    ax_right.axis('off')  # Optionally turn off axis labels
    
    # Set labels
    ax_main.set_xlabel(dict_labels[code_1])
    ax_main.set_ylabel(dict_labels[code_2])

    # Save the figure
    direccion_guardado = Path("../../../Imagenes/Distribucion_ANES/2020/Conjunto_total/{}vs{}.png".format(code_2, code_1))
    plt.savefig(direccion_guardado, bbox_inches="tight")
    plt.close()


Tiempo(t0)