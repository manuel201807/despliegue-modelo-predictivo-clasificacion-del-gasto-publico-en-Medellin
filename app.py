import pickle
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import sklearn
from collections import UserList
import sklearn.compose._column_transformer as ct

PALETA_MEDELLIN = ["#173D59", "#438C32", "#529FE3", "#A64DD4", "#DC8B5C", "#204D3B", "#F8D9B5"]
BASE = Path(__file__).resolve().parent

st.set_page_config(page_title='SECOP Analytics', page_icon='🏛️', layout='wide')
# Tema visual integrado en app.py: no requiere config.toml.
st.markdown('''<style>
:root {--primary-color:#438C32; --background-color:#F8F9FA; --secondary-background-color:#EAF1EC; --text-color:#173D59;}
.stApp, .stApp label, .stApp h1, .stApp h2, .stApp h3 {font-family:Arial,sans-serif;color:#173D59;}
[data-baseweb="select"] > div, [data-baseweb="input"] {background:#EAF1EC !important;color:#173D59 !important;}
[data-baseweb="select"] input, [data-baseweb="input"] input {color:#173D59 !important;}
[data-testid="stHeader"] {background:#F8F9FA;}
[data-testid="stSidebar"] {background:#EAF1EC !important;}
[data-testid="stSidebar"] * {color:#173D59 !important;}

.stApp {background: #F8F9FA; color:#173D59;}


.hero {background:linear-gradient(120deg,#173D59,#204D3B);color:white;padding:32px;border-radius:20px;margin-bottom:24px;}
.hero h1 {color:white;margin:0;font-size:36px;}.hero p {color:#d8eaf2;margin-bottom:0;}
[data-testid="stForm"] {background:white;border:1px solid #DCE7DE;border-radius:18px;padding:24px;}
[data-testid="stMetric"] {background:white;border:1px solid #DCE7DE;padding:18px;border-radius:14px;}
.stButton button, [data-testid="stFormSubmitButton"] button {background:#438C32;color:white;border:0;border-radius:10px;font-weight:600;}
.result {background:#EAF1EC;border-left:5px solid #438C32;padding:22px;border-radius:12px;color:#173D59;margin:16px 0;}
</style>''', unsafe_allow_html=True)

# Las categorías coinciden exactamente con las utilizadas en el entrenamiento.
CATEGORIAS = {
 'Orden': ['Corporacion Autonoma','Nacional','Territorial'],
 'Sector': ['Ambiente y Desarrollo Sostenible','Ciencia Tecnologia','Cultura','Educacion Nacional','Hacienda y Credito Publico','Informacion Estadistica','Ley de Justicia','No aplica/No pertenece','Planeacion','Salud y Proteccion Social','Servicio Publico','Tecnologias de la Informacion y las Comunicaciones','Trabajo','Transporte','Vivienda, Ciudad y Territorio','defensa','deportes'],
 'Rama': ['Corporacion Autonoma','Ejecutivo','Judicial','Legislativo'],
 'Tipo_de_Contrato': ['Arrendamiento de inmuebles','Arrendamiento de muebles','Comodato','Compraventa','Consultoria','Decreto 092 de 2017','Interventoria','Obra','Operaciones de Credito Publico','Otro','Prestacion de servicios','Seguros','Servicios financieros','Suministros'],
 'Obligacion_Ambiental':['No','Si'], 'Origen_de_los_Recursos':['Distribuido','Recursos Propios'], 'El_contrato_puede_ser_prorrogado':['No','Si']}
FUENTES = {'Presupuesto_General_de_la_Nacion_PGN':'Presupuesto General de la Nación (PGN)', 'Sistema_General_de_Participaciones':'Sistema General de Participaciones', 'Sistema_General_de_Regalias':'Sistema General de Regalías', 'Recursos_Propios_Alcaldias_y_Gobernaciones':'Recursos propios territoriales', 'Recursos_de_Credito':'Recursos de crédito', 'Otros_Recursos_Especie_Privados_Cooperacion_Propios_Entidades_Autonomas':'Otros recursos'}
NUMERICAS = ['Valor_del_Contrato', *FUENTES, 'Duracion_del_contrato','mes_fecha_firma']
MESES = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
CLASES = {0:'Funcionamiento',1:'Inversión'}

# Cargamos el pipeline exportado desde el notebook, solo desde un archivo local confiable.
@st.cache_resource
def cargar_modelo(ruta, version):
    # Compatibilidad con modelos guardados en scikit-learn 1.6.1.
    # Versiones recientes retiraron esta clase interna de ColumnTransformer.
    if not hasattr(ct, '_RemainderColsList'):
        class _RemainderColsList(UserList):
            pass
        ct._RemainderColsList = _RemainderColsList
    with open(ruta,'rb') as archivo:
        return pickle.load(archivo)

with st.sidebar:
    st.title('SECOP Analytics')
    st.caption('INTELIGENCIA CONTRACTUAL · MEDELLÍN')
    st.markdown('---')
    st.markdown('**Clasificación presupuestal**\n\nInversión / Funcionamiento')
    st.markdown('**Fuente histórica**\n\nSECOP II · 2026')
    st.markdown('**Entradas del modelo**\n\n16 variables predictoras')
    st.markdown('---')
    st.caption('Herramienta académica de apoyo a la planeación. No sustituye la clasificación presupuestal oficial.')

if (BASE / "medellin.png").exists():
    st.image(str(BASE / "medellin.png"), use_container_width=True)
st.markdown('<div class="hero"><small>ANALÍTICA PREDICTIVA · CONTRATACIÓN PÚBLICA</small><h1>Planea con datos, decide con información.</h1><p>Estima el destino del gasto de una nueva iniciativa contractual en Medellín.</p></div>', unsafe_allow_html=True)
ruta = BASE / 'modelo-clasificacion.pkl'
modelo = None
if ruta.exists():
    try:
        modelo = cargar_modelo(str(ruta), ruta.stat().st_mtime_ns)
        if not hasattr(modelo,'predict_proba') or set(getattr(modelo,'feature_names_in_',[])) != set(NUMERICAS + list(CATEGORIAS)):
            raise ValueError('El archivo no corresponde al pipeline de clasificación SECOP esperado.')
    except Exception as e:
        modelo = None
        st.error(f'No se pudo cargar el modelo: {e}')
else:
    st.info('Para habilitar las predicciones, coloca modelo-clasificacion.pkl junto a app.py.')

with st.form('contrato'):
    st.subheader('01 · Características de la contratación')
    datos = {}
    a,b,c = st.columns(3)
    with a:
        datos['Orden'] = st.selectbox('Orden institucional', CATEGORIAS['Orden'], index=2)
        datos['Rama'] = st.selectbox('Rama institucional', CATEGORIAS['Rama'], index=1)
    with b:
        datos['Sector'] = st.selectbox('Sector', CATEGORIAS['Sector'])
        datos['Tipo_de_Contrato'] = st.selectbox('Tipo de contrato', CATEGORIAS['Tipo_de_Contrato'], index=10)
    with c:
        datos['Obligacion_Ambiental'] = st.radio('Obligación ambiental', CATEGORIAS['Obligacion_Ambiental'], horizontal=True)
        datos['El_contrato_puede_ser_prorrogado'] = st.radio('¿Puede ser prorrogado?', CATEGORIAS['El_contrato_puede_ser_prorrogado'], horizontal=True)
    st.divider()
    st.subheader('02 · Planeación y presupuesto')
    a,b,c = st.columns(3)
    datos['Valor_del_Contrato'] = a.number_input('Valor estimado (COP)', min_value=0.01, value=25000000.0, step=1000000.0)
    datos['Duracion_del_contrato'] = b.number_input('Duración prevista (días)', min_value=0.25, max_value=36500.0, value=145.0, step=1.0)
    datos['mes_fecha_firma'] = c.selectbox('Mes previsto de firma', list(range(1,13)), format_func=lambda m: MESES[m-1])
    datos['Origen_de_los_Recursos'] = st.selectbox('Origen de los recursos', CATEGORIAS['Origen_de_los_Recursos'])
    st.divider()
    st.subheader('03 · Fuentes de financiación')
    st.caption('Ingresa los montos previstos en pesos colombianos. Usa cero cuando la fuente no aporte recursos.')
    cols = st.columns(2)
    for i,(campo, etiqueta) in enumerate(FUENTES.items()):
        datos[campo] = cols[i%2].number_input(etiqueta, min_value=0.0, value=0.0, step=1000000.0, key=campo)
    enviado = st.form_submit_button('Analizar destino del gasto', use_container_width=True, disabled=modelo is None)

if enviado and modelo is not None:
    st.session_state['contrato_analizado'] = datos.copy()

if 'contrato_analizado' in st.session_state and modelo is not None:
    datos = st.session_state['contrato_analizado']
    total = sum(datos[c] for c in FUENTES)
    if abs(total-datos['Valor_del_Contrato']) > 0.01:
        st.warning('La suma de las fuentes de financiación no coincide con el valor estimado. Revisa la información antes de interpretar el resultado.')
    if datos['mes_fecha_firma']==12:
        st.warning('Diciembre no estaba representado en la base histórica de entrenamiento.')
    # El pipeline aplica automáticamente las transformaciones aprendidas durante el entrenamiento.
    df = pd.DataFrame([datos], columns=list(modelo.feature_names_in_))
    try:
        pred = int(modelo.predict(df)[0])
        probs = dict(zip(modelo.classes_, modelo.predict_proba(df)[0]))
        st.subheader('Resultado del análisis')
        st.markdown(f'<div class="result"><small>DESTINO DEL GASTO ESTIMADO</small><h2>{CLASES[pred]}</h2>Clasificación generada a partir de las características ingresadas.</div>', unsafe_allow_html=True)
        a,b = st.columns(2)
        for col, clase in [(a,1),(b,0)]:
            col.metric(CLASES[clase], f'{probs[clase]:.1%}')
            col.progress(float(probs[clase]))
        grafica = px.bar(pd.DataFrame({'Destino': [CLASES[1], CLASES[0]], 'Probabilidad': [probs[1], probs[0]]}), x='Probabilidad', y='Destino', color='Destino', orientation='h', color_discrete_map={CLASES[1]: PALETA_MEDELLIN[1], CLASES[0]: PALETA_MEDELLIN[0]}, color_discrete_sequence=PALETA_MEDELLIN, text_auto='.1%')
        grafica.update_layout(showlegend=False, height=250, paper_bgcolor='#F8F9FA', plot_bgcolor='#EAF1EC', font_color='#173D59', margin=dict(l=10,r=25,t=15,b=15))
        grafica.update_xaxes(range=[0,1], tickformat='.0%')
        st.plotly_chart(grafica, use_container_width=True)
        st.caption('Las probabilidades son estimaciones del modelo, no garantías de acierto ni una decisión presupuestal oficial.')
        salida = df.assign(Destino_Gasto_Estimado=CLASES[pred], Probabilidad_Inversion=probs[1], Probabilidad_Funcionamiento=probs[0])
        with st.expander('Consultar información analizada'):
            st.dataframe(salida, hide_index=True)
        st.download_button('Descargar resultado CSV', salida.to_csv(index=False).encode('utf-8-sig'), 'prediccion_secop.csv','text/csv')
    except Exception as e:
        st.error(f'No fue posible realizar la predicción: {e}')
st.divider()
st.caption('SECOP Analytics · Proyecto académico CRISP-DM · Medellín, Colombia')
