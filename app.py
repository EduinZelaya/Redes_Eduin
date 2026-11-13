import os
import json
import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title='Clasificación de Conexiones', layout='centered')
st.title('Detección de Conexiones Sospechosas - Eduin Fabricio Zelaya')

@st.cache_resource
def cargar_modelo():
    # Rutas correspondientes a los archivos guardados en el entrenamiento
    pkl_modelos = 'models/modelos_clasificacion_conexion.pkl' if os.path.exists('models/modelos_clasificacion_conexion.pkl') else 'modelos_clasificacion_conexion.pkl'
    meta = 'models/model_metadata.json' if os.path.exists('models/model_metadata.json') else 'model_metadata.json'
    
    modelos = joblib.load(pkl_modelos)
    with open(meta, encoding='utf-8') as f:
        metadata = json.load(f)
    
    return modelos, metadata

@st.cache_data
def cargar_base():
    csv = 'conexiones_red.csv'
    if os.path.exists(csv):
        return pd.read_csv(csv)
    return None

try:
    modelos, metadata = cargar_modelo()
    df = cargar_base()
    modelo_seleccionado_defecto = metadata.get('mejor_modelo_cv', 'SVM RBF')
except Exception as e:
    st.error(f"Error al cargar los recursos: {e}")
    st.stop()

st.caption(metadata.get('nombre_modelo', 'Clasificador de Tráfico de Red'))

# Formulario para ingresar nuevos datos de red
with st.form('datos_conexion'):
    st.subheader('Parámetros de la Conexión')
    
    col1, col2 = st.columns(2)
    
    # Entradas numéricas basadas en el rango del dataset original
    trafico = col1.number_input('Tráfico (Mbps)', min_value=0.0, max_value=1000.0, value=50.0, step=1.0)
    latencia = col2.number_input('Latencia (ms)', min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
    
    # Selección del modelo a usar para la predicción
    algoritmo = st.selectbox('Seleccione el Algoritmo de Predicción', list(modelos.keys()), index=list(modelos.keys()).index(modelo_seleccionado_defecto))
    
    enviar = st.form_submit_button('Evaluar Conexión')

if enviar:
    # Crear DataFrame de entrada con los mismos nombres de variables
    cliente = pd.DataFrame([{
        'trafico_mbps': trafico,
        'latencia_ms': latencia
    }])
    
    # Obtener la predicción del modelo seleccionado
    modelo_actuante = modelos[algoritmo]
    prediccion = int(modelo_actuante.predict(cliente)[0])
    
    resultado_clase = "Sospechosa (Requiere Bloqueo/Análisis)" if prediccion == 1 else "Normal (Segura)"
    color_alerta = "red" if prediccion == 1 else "green"
    
    st.markdown(f"### Diagnóstico: :{color_alerta}[{resultado_clase}]")
    st.write(f"Algoritmo utilizado: **{algoritmo}**")
    
    # Integración opcional con Groq API / OpenAI para dar explicación experta en seguridad de TI
    api_key = st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', ''))
    
    if api_key:
        prompt = f'''
        Actúa como un experto en Seguridad Informática y Administrador de Redes.
        
        Explica brevemente por qué una conexión con las siguientes métricas puede ser catalogada como {resultado_clase}.
        Proporciona 3 recomendaciones de seguridad inmediatas para el equipo de TI.
        
        Métricas actuales:
        - Tráfico de Red: {trafico} Mbps
        - Latencia: {latencia} ms
        
        Resultado del modelo:
        - Clasificación: {resultado_clase} (0=Normal, 1=Sospechosa)
        - Modelo actuante: {algoritmo}
        '''
        
        try:
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            
            respuesta = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un asesor experto en ciberseguridad y monitorización de tráfico de red."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            texto = respuesta.choices[0].message.content
            st.info(texto)
            
        except Exception as e:
            st.warning(f'Error al conectar con el servicio de IA: {e}')
    else:
        st.info('Para obtener recomendaciones automatizadas de seguridad mediante IA, configure su GROQ_API_KEY en los secretos de Streamlit.')

if df is not null:
    st.divider()
    st.write('Historial Reciente de Tráfico de Red (Dataset de Entrenamiento)')
    st.dataframe(df.head(15), use_container_width=True)