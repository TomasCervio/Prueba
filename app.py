import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats

# Cargar el dataset
@st.cache
def cargar_datos():
    df_FW = pd.read_csv('df_FW.csv')  # Usa la ruta completa del archivo
    return df_FW

df_FW = cargar_datos()

# Calcular percentiles
columnas = ['stats_G-PK.1', 'stats_npxG.1', 'stats_PrgC', 'stats_xAG.1', 'stats_npxG+xAG.1', 
            'shooting_Sh/90', 'shooting_SoT/90', 'gca_SCA90', 'gca_GCA90', 'passing_Att', 
            'passing_1/3', 'passing_PPA', 'passing_CrsPA', 'passing_PrgP', 'defense_Att 3rd', 
            'defense_TklW', 'possession_Mid 3rd', 'possession_Att 3rd', 'possession_Att Pen', 
            'possession_Live', 'possession_PrgDist', 'possession_PrgR', 'possession_Succ%', 'misc_Won%']

for columna in columnas:
    df_FW['Percentil_' + columna.split('_')[1]] = df_FW[columna].rank(pct=True) * 100

df_FW.replace([np.inf, -np.inf], np.nan, inplace=True)
df_FW.fillna(0, inplace=True)

# Evitar divisiones por 0 en 'possession_90s'
df_FW['possession_90s'].replace(0, np.nan, inplace=True)
columnas_a_dividir = ['possession_Mid 3rd', 'possession_Att 3rd', 'passing_Att', 'passing_1/3', 'passing_PPA', 
                      'passing_CrsPA', 'passing_PrgP', 'possession_Att Pen', 'possession_Live', 
                      'possession_PrgDist', 'possession_PrgR','defense_TklW','defense_Att 3rd']
df_FW[columnas_a_dividir] = df_FW[columnas_a_dividir].div(df_FW['possession_90s'], axis=0)

# Función para encontrar jugadores parecidos
def encontrar_jugadores_parecidos(jugador, df):
    columnas_percentiles = [col for col in df.columns if 'Percentil' in col]
    
    # Calcula los percentiles del jugador propuesto
    percentiles_jugador = df[df['Player'] == jugador][columnas_percentiles].values.flatten()
    
    # Calcula la distancia euclidiana
    def calcular_distancia(jugador_row):
        percentiles_comparar = jugador_row[columnas_percentiles].values.flatten()
        return np.sqrt(np.sum((percentiles_jugador - percentiles_comparar) ** 2))
    
    df['Distancia'] = df.apply(calcular_distancia, axis=1)
    df_sorted = df.sort_values(by='Distancia')
    
    return df_sorted

# Función para crear el gráfico radial
def crear_grafico_radial(jugadores, df):
    columnas_percentiles = [col for col in df.columns if 'Percentil' in col]
    etiquetas = [col.replace('Percentil_', '') for col in columnas_percentiles]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colores = ['blue', 'green']
    
    for i, jugador in enumerate(jugadores):
        if jugador in df['Player'].values:
            datos_jugador = df[df['Player'] == jugador][columnas_percentiles].values.flatten()
            indices_no_cero = np.where(datos_jugador != 0)[0]
            
            if len(indices_no_cero) == 0:
                st.write(f"El jugador '{jugador}' no tiene datos válidos.")
                return
            
            datos_jugador = datos_jugador[indices_no_cero]
            etiquetas_filtradas = [etiquetas[j] for j in indices_no_cero if j < len(etiquetas)]
            
            # Repetir el primer valor para cerrar el gráfico radial
            datos_jugador = np.concatenate((datos_jugador, [datos_jugador[0]]))
            angulos = np.linspace(0, 2 * np.pi, len(datos_jugador) - 1, endpoint=False).tolist()
            angulos += angulos[:1]
            
            ax.fill(angulos, datos_jugador, color=colores[i], alpha=0.25, label=jugador)
            ax.plot(angulos, datos_jugador, color=colores[i], linewidth=2)
        else:
            st.write(f"Jugador '{jugador}' no encontrado en el dataframe.")
    
    ax.set_yticklabels([])
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(etiquetas_filtradas)
    plt.title(f"Comparación de percentiles", size=15, y=1.1)

    # Agregar leyenda personalizada fuera del gráfico
    handles = [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='green', lw=2)]
    labels = [f"{jugadores[0]}", f"{jugadores[1]}"]
    plt.figlegend(handles, labels, loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    return fig


# Interfaz de usuario en Streamlit
st.title("Comparador de Jugadores de Fútbol")

jugador_a_comparar = st.text_input("Ingresa el nombre del jugador a comparar:", "Marco Ruben")

if jugador_a_comparar:
    top_jugadores = encontrar_jugadores_parecidos(jugador_a_comparar, df_FW)
    top_5_jugadores = top_jugadores.iloc[1:6]['Player']  # Top 5 similares excluyendo al propio jugador
    
    if not top_5_jugadores.empty:
        # Mostrar el gráfico radial
        top_1_jugador = top_5_jugadores.iloc[0]
        fig = crear_grafico_radial([jugador_a_comparar, top_1_jugador], df_FW)
        st.pyplot(fig)
        
        st.write(f"El jugador más parecido a '{jugador_a_comparar}' es '{top_1_jugador}'.")
        st.write("Top 5 jugadores similares:")
        st.write(top_5_jugadores)
    else:
        st.write(f"No se encontraron jugadores similares a '{jugador_a_comparar}'.")
