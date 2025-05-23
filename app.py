# ------------ Librerías -----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------ Configuración página -----------------
st.set_page_config(
    page_title="Análisis Bianual de Resultados: Campañas de Marketing",
    page_icon="🚀",
    layout="wide"
)

# ------------ Títulos y descripción -----------------
st.title("🚀 Dashboard Interactivo: Análisis campañas de marketing")
st.markdown("""
Este dashboard nos permite analizar los resultados de las campañas de marketing de la empresa en los últimos dos años, teniendo en cuenta Tipo de Campaña, Canal y Audiencia para establecer relaciones entre el gasto, las ganancias, el ROI, la tasa de conversión y el beneficio neto.
""")

# ------------ Variables de iniciaciación segura -----------------
df_filtrado = None
df = None

# ------------ Carga de datos -----------------
@st.cache_data(ttl=3600)
def cargar_datos():
    try:
        df = pd.read_csv('/Users/n.arcos89/Documents/GitHub/upgrade-hub-marketing-analysis/Preprocesamiento/pre-marketing.csv')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

# ------------ Colores -----------------
colores_canales = {
    'Referral': '#636EFA',
    'Unknown': '#EF553B',
    'orgánico': '#00CC96',
    'pagado': '#AB63FA',
    'promoción': '#FFA15A'
}
colores_campañas = {
    'redes sociales': '#636EFA',
    'webinar': '#EF553B',
    'email': '#00CC96',
    'podcast': '#AB63FA',
    'Desconocido': '#FFA15A',
    'B2B': '#FFA15A',
    'evento': '#FFA15A'
}
colores_audiencia = {
    'B2B': '#1f77b4',  # azul
    'B2C': '#ff7f0e'   # naranja
}

# ------------ Asegurar tamaños positivos -----------------
def asegurar_positivo(valores, tamaño_min=3):
    if isinstance(valores, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(valores), tamaño_min)
    else:
        return max(abs(valores), tamaño_min)

# ------------ Cargar datos -----------------
try:
    with st.spinner('Cargando datos...'):
        df = cargar_datos()
        
    if df is not None and not df.empty:
        # Barra lateral para filtros
        st.sidebar.header("Filtra los datos")
        
        # ------------ Filtros de datos para el usuario -----------------
        # Filtro de fecha
        min_fecha = pd.to_datetime(df['start_date']).min().date()
        max_fecha = pd.to_datetime(df['end_date']).max().date()
        
        rango_fechas = st.sidebar.date_input(
            "Selecciona la fecha de inicio y fin",
            [min_fecha, max_fecha],
            min_value=min_fecha,
            max_value=max_fecha
        )
        
        # Convertir fechas seleccionadas a datetime para filtrar
        if len(rango_fechas) == 2:
            fecha_inicio, fecha_fin = rango_fechas
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            df_filtrado = df[(df['start_date'] >= pd.to_datetime(fecha_inicio)) & (df['end_date'] <= pd.to_datetime(fecha_fin))].copy()
        else:
            df_filtrado = df.copy()
        
        # Validar si hay datos después del filtro de fecha
        if df_filtrado.empty:
            st.warning("No se encuentran campañas en el periodo seleccionado.")
        
        # Filtro por tipo de campaña
        if df_filtrado is not None and not df_filtrado.empty:
            st.sidebar.header("🏠 Filtro por tipo de campaña")
            tipos_campaña = df['type'].unique().tolist()
            tipos_seleccionados = st.sidebar.multiselect(
                "Tipo de campaña",
                options=tipos_campaña,
                default=tipos_campaña
            )
            if tipos_seleccionados:
                df_filtrado = df_filtrado[df_filtrado['type'].isin(tipos_seleccionados)]
        
        # Filtro por canal
        if df_filtrado is not None and not df_filtrado.empty:
            st.sidebar.header("🪂 Filtro por canal")
            tipos_canal = df['channel'].unique().tolist()
            canales_seleccionados = st.sidebar.multiselect(
                "Tipo de canal",
                options=tipos_canal,
                default=tipos_canal
            )
            if canales_seleccionados:
                df_filtrado = df_filtrado[df_filtrado['channel'].isin(canales_seleccionados)]
            
        # Filtro por audiencia
        if df_filtrado is not None and not df_filtrado.empty:
            st.sidebar.header("👽 Filtro por audiencia")
            tipos_audiencia = df['target_audience'].unique().tolist()
            audiencias_seleccionadas = st.sidebar.multiselect(
                "Tipo de audiencia",
                options=tipos_audiencia,
                default=tipos_audiencia
            )
            if audiencias_seleccionadas:
                df_filtrado = df_filtrado[df_filtrado['target_audience'].isin(audiencias_seleccionadas)]
        
        # Comprobar si hay datos después del filtrado
        if len(df_filtrado) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            # Organización de los datos
            pestañas_principales = st.tabs(["📊 Resumen", "🏠 Tipo de Campaña", "👽 Audiencia", "🪂 Canales", "📈 Principales KPIs", "💟 Mejores Campañas"])
            
            #------------------ 1. Resumen -----------------
            with pestañas_principales[0]:
                # Principales Métricas
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total de campañas", len(df_filtrado))
                col2.metric("Gasto medio", f"{df_filtrado['budget'].mean():.2f}")
                col3.metric("Ganancia media", f"{df_filtrado['revenue'].mean():.2f}")
                col4.metric("ROI medio", f"{df_filtrado['roi'].mean():.2f}")
                
                # Gráficos cruce variables
                col_variables1, col_variables2 = st.columns(2)
                with col_variables1:
                    st.subheader("Gasto en campañas y canales")
                    fig_gasto = px.violin(
                        df_filtrado,
                        x='type',
                        y='budget',
                        color='channel',
                        box=True,
                        points='all',
                        title='Distribución del gasto por Tipo de Campaña y Canal',
                        labels={'type': 'Tipo de Campaña', 'budget': 'Gasto', 'channel': 'Canal'},
                        color_discrete_map=colores_canales
                    )
                    fig_gasto.update_layout(title_x=0.5)
                    st.plotly_chart(fig_gasto, use_container_width=True)
                
                with col_variables2:
                    st.subheader("Ganancia")
                    fig_ganancia = px.violin(
                        df_filtrado,
                        x='type',
                        y='revenue',
                        color='channel',
                        box=True,
                        points='all',
                        title='Distribución de la ganancia por Tipo de Campaña y Canal',
                        labels={'type': 'Tipo de Campaña', 'revenue': 'Ganancia', 'channel': 'Canal'},
                        color_discrete_map=colores_canales
                    )
                    fig_ganancia.update_layout(title_x=0.5)
                    st.plotly_chart(fig_ganancia, use_container_width=True)  
                
                col_cruce1, col_cruce2 = st.columns(2)
                with col_cruce1:
                    st.subheader("ROI")   
                    fig_roi = px.violin(
                        df_filtrado,
                        x='type',
                        y='roi',
                        color='channel',
                        box=True,
                        points='all',
                        title='Distribución del ROI por Tipo de Campaña y Canal',
                        labels={'type': 'Tipo de Campaña', 'roi': 'ROI', 'channel': 'Canal'},
                        color_discrete_map=colores_canales
                    )
                    fig_roi.update_layout(title_x=0.5)
                    st.plotly_chart(fig_roi, use_container_width=True)
                
                with col_cruce2:
                    st.subheader("Tasa de conversión")   
                    fig_conversion = px.violin(
                        df_filtrado,
                        x='type',
                        y='conversion_rate',
                        color='channel',
                        box=True,
                        points='all',
                        title='Distribución de la Conversión por Tipo de Campaña y Canal',
                        labels={'type': 'Tipo de Campaña', 'conversion_rate': 'Tasa de Conversión', 'channel': 'Canal'},
                        color_discrete_map=colores_canales
                    )
                    fig_conversion.update_layout(title_x=0.5)
                    st.plotly_chart(fig_conversion, use_container_width=True)
            
                # Evolución de las métricas en el periodo
                if 'month_year' in df_filtrado.columns:
                    fig_gasto_meses = px.line(
                        df_filtrado.groupby('month_year')['budget'].mean().reset_index(),
                        x='month_year',
                        y='budget',
                        title='Gasto Medio por Meses',
                        labels={'month_year': 'Mes', 'budget': 'Gasto Medio'},
                        color_discrete_sequence=['#636EFA']
                    )
                    fig_gasto_meses.update_layout(title_text='Gasto Medio por Meses', title_x=0.5)
                    fig_gasto_meses.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_gasto_meses, use_container_width=True)
                    
                    fig_roi_meses = px.line(
                        df_filtrado.groupby('month_year')['roi'].mean().reset_index(),
                        x='month_year',
                        y='roi',
                        title='ROI Medio por Meses',
                        labels={'month_year': 'Mes', 'roi': 'ROI Medio'},
                        color_discrete_sequence=['#636EFA']
                    )
                    fig_roi_meses.update_layout(title_text='ROI Medio por Meses', title_x=0.5)
                    fig_roi_meses.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_roi_meses, use_container_width=True)

                    fig_beneficio_meses = px.line(
                        df_filtrado.groupby('month_year')['beneficio_neto'].mean().reset_index(),
                        x='month_year',
                        y='beneficio_neto',
                        title='Beneficio Neto Medio por Meses',
                        labels={'month_year': 'Mes', 'beneficio_neto': 'Beneficio Neto Medio'},
                        color_discrete_sequence=['#636EFA']
                    )
                    fig_beneficio_meses.update_layout(title_text='Beneficio Neto Medio por Meses', title_x=0.5)
                    fig_beneficio_meses.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_beneficio_meses, use_container_width=True)
                    
                    fig_conversion_meses = px.line(
                        df_filtrado.groupby('month_year')['conversion_rate'].mean().reset_index(),
                        x='month_year',
                        y='conversion_rate',
                        title='Tasa de Conversión Media por Meses',
                        labels={'month_year': 'Mes', 'conversion_rate': 'Tasa de Conversión Media'},
                        color_discrete_sequence=['#636EFA']
                    )
                    fig_conversion_meses.update_layout(title_text='Tasa de Conversión Media por Meses', title_x=0.5)
                    fig_conversion_meses.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_conversion_meses, use_container_width=True)

            #------------------ 2. Tipo de Campaña -----------------
            with pestañas_principales[1]:            
                st.subheader("Tipo de Campaña")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Campaña con más conversión")
                    tipo_mejor_conversion = df_filtrado.groupby('type')['conversion_rate'].mean().idxmax()
                    st.write(tipo_mejor_conversion)
                with col2:
                    st.subheader("Campaña con más ROI")
                    tipo_mejor_roi = df_filtrado.groupby('type')['roi'].mean().idxmax()
                    st.write(tipo_mejor_roi)
                with col3:
                    st.subheader("Campaña con más Beneficio Neto")
                    tipo_mejor_beneficio = df_filtrado.groupby('type')['beneficio_neto'].mean().idxmax()
                    st.write(tipo_mejor_beneficio)
                
                # Evolución KPIS
                camp_tab1, camp_tab2 = st.tabs([
                    "Gasto y Ganancia", 
                    "KPIS",
                ])
                with camp_tab1:
                    st.subheader("Relación Gasto y Ganancia")
                    # Aquí puedes continuar con los gráficos y análisis adicionales
except Exception as e:
    st.error(f"Error general: {e}")
