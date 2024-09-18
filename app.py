import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import io

# Función para cargar el dataset
def cargar_datos():
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# Función para crear gráfico radial
def crear_grafico_radial(jugadores, df):
    columnas_percentiles = [col for col in df.columns if 'Percentil' in col]
    etiquetas = [col.replace('Percentil_', '') for col in columnas_percentiles]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colores = ['blue', 'green']
    
    for i, jugador in enumerate(jugadores):
        if jugador in df['Player'].values:
            datos_jugador = df[df['Player'] == jugador][columnas_percentiles].values.flatten()
            indices_no_cero = np.where(datos_jugador != 0)[0]
            datos_jugador = datos_jugador[indices_no_cero]
            etiquetas_filtradas = [etiquetas[j] for j in indices_no_cero]
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
    st.pyplot(fig)

# Función para encontrar jugadores parecidos
def encontrar_jugadores_parecidos(jugador, df):
    df = df.drop_duplicates(subset='Player', keep='first')
    columnas_percentiles = [col for col in df.columns if 'Percentil' in col]
    
    if jugador not in df['Player'].values:
        return f"Jugador '{jugador}' no encontrado en el dataframe."
    
    vector_jugador = df[df['Player'] == jugador][columnas_percentiles].values
    vectores_todos_jugadores = df[columnas_percentiles].values
    distancias = euclidean_distances(vector_jugador, vectores_todos_jugadores)
    
    df['Distancia'] = distancias.flatten()
    df_filtrado = df[df['Player'] != jugador]
    top_5_parecidos = df_filtrado.nsmallest(5, 'Distancia')
    
    st.write("Top 5 jugadores más parecidos:")
    st.write(top_5_parecidos[['Player', 'Distancia']])

    top_1_jugador = top_5_parecidos.iloc[0]['Player']
    crear_grafico_radial([jugador, top_1_jugador], df)

# Interfaz de usuario de Streamlit
def main():
    st.title("Comparación de Jugadores de Fútbol")
    df = cargar_datos()

    if df is not None:
        jugador_a_comparar = st.text_input("Ingresa el nombre del jugador a comparar:", 'Marco Ruben')
        if st.button("Comparar"):
            encontrar_jugadores_parecidos(jugador_a_comparar, df)

if __name__ == "__main__":
    main()
