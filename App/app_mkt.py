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
    page_title="Analisis Bianual de Resultados: Campañas de Marketing",
    page_icon="🚀",
    layout="wide"
)

# ------------ Títulos y descripción -----------------
st.title("🚀 Dashboard Interactivo: Análisis campañas de marketing")
st.markdown("""
Este dashboard nos permite analizar los resultados de las campañas de marketing de la empresa en losúltimos dos años, teniendo en cuenta Tipo de Campaña, Canal y Target para establecer relaciones entre el gasto, las ganancias, el rio, la tasa de conversión y el beneficio neto.
""")

# ------------ Variables de iniciaciación segura -----------------
filtered_df = None
df = None

# ------------ Carga de datos -----------------

# Carga del dato
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv('/Users/n.arcos89/Documents/GitHub/upgrade-hub-marketing-analysis/Preprocesamiento/pre-marketing.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# ------------ Colores -----------------
# Colores para Canales
mcolores_canales = {
    'Referral': '#636EFA',
    'Unknown': '#EF553B',
    'organic': '#00CC96',
    'paid': '#AB63FA',
    'promotion': '#FFA15A'
}
# Colores para Tipo de Campaña
colores_campañas = {
    'social media': '#636EFA',
    'webinar': '#EF553B',
    'email': '#00CC96',
    'podcast': '#AB63FA',
    'Unknown': '#FFA15A',
    'B2B': '#FFA15A',
    'event': '#FFA15A'
}

#Colores para Target
colores_target = {
    'B2B': '#1f77b4',  # azul
    'B2C': '#ff7f0e'   # naranja
}

# ------------ Asegurar tamaños positivos -----------------
def ensure_positive(values, min_size=3):
    if isinstance(values, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(values), min_size)
    else:
        return max(abs(values), min_size)

# ------------ Cargar datos -----------------
try:
    with st.spinner('Loading data...'):
        df = load_data()
        
    if df is not None and not df.empty:
        # Barra lateral para filtros
        st.sidebar.header("Filtra los datos")
        
# ------------ Filtros de datos para el usuario -----------------
        # Filtro de fecha
        min_date = df['start_date'].min().date()
        max_date = df['end_date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Selecciona la fecha de inicio y fin",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        
        # Convert selected dates to datetime for filtering
        if len(date_range) == 2:
            start_date, end_date = date_range
            
            # Filter the dataframe (now both are of the same type)
            filtered_df = df[(df['start_date'] >= start_datetime) & (df['end_date'] <= end_datetime)].copy()
        else:
            filtered_df = df.copy()
        
        # Validar si hay datos después del filtro de fecha
            if filtered_df.empty:
                st.warning("No se encuentran campañas en periodo seleccionado.")
        
        # Filtro por tipo de campaña para usuario
        if filtered_df is not None and not filtered_df.empty:
            st.sidebar.header("🏠 Filtro por campaña")
        campaign_types = df['type'].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Tipo de campaña",
            options=campaign_types,
            default=campaign_types
        )
        if selected_types:
            filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]
        
        # Filtro por canal para usuario
        if filtered_df is not None and not filtered_df.empty:
            st.sidebar.header("🪂 Filtro por canal")
        channel_types = df['channel'].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Tipo de canal",
            options=channel_types,
            default=channel_types
        )
        if selected_types:
            filtered_df = filtered_df[filtered_df['channel'].isin(selected_types)]
            
        # Filtro por target
        if filtered_df is not None and not filtered_df.empty:
            st.sidebar.header("👽 Filtro por audiencia")
        target_types = df['target_audience'].unique().tolist()
        selected_types = st.sidebar.multiselect(
            "Tipo de audiencia",
            options=target_types,
            default=target_types
        )
        if selected_types:
            filtered_df = filtered_df[filtered_df['target_audiencel'].isin(selected_types)]
        
        # Comprobar si hay datos después del filtrado
        if len(filtered_df) == 0:
            st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        else:
            # Organización de los datos
            main_tabs = st.tabs(["📊 Resumen", "🏠 Tipo de Campaña", "👽 Target ", "🪂 Canales", "📈 Principales KPIs", " 💟 Mejores Campañas"])
            
#------------------ 1. Resumen -----------------

            #Tab 1: Resumen
            with main_tabs[0]:
                #Principales Métricas
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total de campañas", len(filtered_df))
                col2.metric("Gasto medio", f"{filtered_df['budget'].mean():.2f}")
                col3.metric("Ganancia media", f"{filtered_df['revenue'].mean():.2f}")
                col4.metric("Roi medio", f"{filtered_df['roi'].mean():.2f}")
                
                #Gráficos cruce variables
                col_variables1, col_variables2 = st.columns(2)
                
                with col_variables1:
                #Gráfica Gasto en Camapañas y Canales
                st.subheader("Gasto en campañas y canales")
                fig_gasto = px.violin(
                    df_sin_outlier,
                    x='type',
                    y='budget',
                    color='channel',
                    box=True,
                    points='all',
                    title='Distribución del gasto por Tipo de Campaña y Canal (sin outlier)',
                    labels={'type': 'Tipo de Campaña', 'budget': 'Gasto', 'channel': 'Canal'},
                    color_discrete_map=colores_canales
                )
                fig_gasto.update_layout(title_x=0.5)
                fig_gasto.show()
                fig_gasto.update_layout(bargap=0.1)
                st.plotly_chart(fig_gasto, use_container_width=True)
                
                with col_variables2:
                #Gráfico Ganancia en Campañas y Canales
                st.subheader("Ganancia")
                fig_ganancia = px.violin(
                    df_sin_outlier,
                    x='type',
                    y='revenue',
                    color='channel',
                    box=True,
                    points='all',
                    title='Distribución de la ganancia por Tipo de Campaña y Canal (sin outlier)',
                    labels={'type': 'Tipo de Campaña', 'revenue': 'Ganancia', 'channel': 'Canal'},
                    color_discrete_map=colores_canales
                )
                fig_ganancia.update_layout(title_x=0.5)
                fig_ganancia.show()
                fig_ganancia.update_layout(bargap=0.1)
                st.plotly_chart(fig_ganancia, use_container_width=True)  
                
                #Gráficos cruce variables
                col_cruce1, col_cruce2 = st.columns(2)
                with col_cruce1:
                #Gráfico Roi en Campañas y Canales
                st.subheader("Roi")   
                fig_roi = px.violin(
                    df_sin_outlier,
                    x='type',
                    y='roi',
                    color='channel',
                    box=True,
                    points='all',
                    itle='Distribución del ROI por Tipo de Campaña y Canal (sin outlier)',
                    labels={'type': 'Tipo de Campaña', 'roi': 'ROI', 'channel': 'Canal'},
                    color_discrete_map=colores_canales
                )
                fig_roi.update_layout(title_x=0.5)
                fig_roi.show()
                fig_roi.update_layout(bargap=0.1)
                st.plotly_chart(fig_roi, use_container_width=True)
                
                with col_cruce2:
                #Gráfico Conversión en Campañas y Canales
                st.subheader("Tasa de conversión")   
                fig_conversion = px.violin(
                    df_sin_outlier,
                    x='type',
                    y='conversion_rate',
                    color='channel',
                    box=True,
                    points='all',
                    title='Distribución de la Conversión por Tipo de Campaña y Canal (sin outlier)',
                    labels={'type': 'Tipo de Campaña', 'conversion_rate': 'Tasa de Conversión', 'channel': 'Canal'},
                    color_discrete_map=colores_canales
                )
                fig_conversion.update_layout(title_x=0.5)
                fig_conversion.show()
                fig_conversion.update_layout(bargap=0.1)
                st.plotly_chart(fig_conversion, use_container_width=True)
            
            #Evolución de las métricas en el periodo
            
            #Gráfica que muestre el budget medio por meses
            fig_gasto_meses = px.line(
                df.groupby('month_year')['budget'].mean().reset_index(),
                x='month_year',
                y='budget',
                title='Gasto Medio por Meses',
                labels={'month_year': 'Mes', 'budget': 'Gasto Medio'},
                color_discrete_sequence=['#636EFA']
            )
            fig_gasto_meses.update_layout(title_text='Gasto Medio por Meses', title_x=0.5)
            fig_gasto_meses.update_xaxes(tickangle=45)
            fig_gasto_meses.show()
            fig_gasto_meses.update_layout(bargap=0.1)
            st.plotly_chart(fig_gasto_meses, use_container_width=True)
            
            #Gráfica que muestre el roi medio por meses
            fig_roi_meses = px.line(
                df.groupby('month_year')['roi'].mean().reset_index(),
                x='month_year',
                y='roi',
                title='ROI Medio por Meses',
                labels={'month_year': 'Mes', 'roi': 'ROI Medio'},
                color_discrete_sequence=['#636EFA']
            )
            fig_roi_meses.update_layout(title_text='ROI Medio por Meses', title_x=0.5)
            fig_roi_meses.update_xaxes(tickangle=45)
            fig_roi_meses.show()
            fig_roi_meses.update_layout(bargap=0.1)
            st.plotly_chart(fig_roi_meses, use_container_width=True)

            #Gráfica que muestre el beneficio neto medio por meses
            fig_beneficio_meses = px.line(
                df.groupby('month_year')['beneficio_neto'].mean().reset_index(),
                x='month_year',
                y='beneficio_neto',
                title='Beneficio Neto Medio por Meses',
                labels={'month_year': 'Mes', 'beneficio_neto': 'Beneficio Neto Medio'},
                color_discrete_sequence=['#636EFA']
            )
            fig_beneficio_meses.update_layout(title_text='Beneficio Neto Medio por Meses', title_x=0.5)
            fig_beneficio_meses.update_xaxes(tickangle=45)
            fig_beneficio_meses.show()
            fig_beneficio_meses.update_layout(bargap=0.1)
            st.plotly_chart(fig_beneficio_meses, use_container_width=True)
            
            #Gráfica que muestre el conversion_rate medio por meses
            fig_conversion_meses = px.line(
                df.groupby('month_year')['conversion_rate'].mean().reset_index(),
                x='month_year',
                y='conversion_rate',
                title='Tasa de Conversión Media por Meses',
                labels={'month_year': 'Mes', 'conversion_rate': 'Tasa de Conversión Media'},
                color_discrete_sequence=['#636EFA']
            )
            fig_conversion_meses.update_layout(title_text='Tasa de Conversión Media por Meses', title_x=0.5)
            fig_conversion_meses.update_xaxes(tickangle=45)
            fig_conversion_meses.show()
            fig_conversion_meses.update_layout(bargap=0.1)
            st.plotly_chart(fig_conversion_meses, use_container_width=True)

#------------------ 2. Tipo de Campaña -----------------

            #Tab 2: Tipo de Campaña
            with main_tabs[1]:            
            st.subheader("Tipo de Camapaña")
            
            #Resumen campañas más exitosas
            col1, col2, col3 = st.columns(3)
            
                with col1:
                #tipo de campaña con mejor conversión
                st.subheader("Campaña con más conversión")
                tipo_mejor_conversion = df.groupby('type')['conversion_rate'].mean().idxmax()
                tipo_mejor_conversion
            
                with col2:
                #tipo de campaña con mejor ROI
                st.subheader("Campaña con más ROI")
                tipo_mejor_roi = df.groupby('type')['roi'].mean().idxmax()
                tipo_mejor_roi
            
                with col3:
                #tipo de campaña con mejor Beneficio Neto
                st.subheader("Campaña con más Beneficio Neto")
                tipo_mejor_beneficio = df.groupby('type')['beneficio_neto'].mean().idxmax()
                tipo_mejor_beneficio
        
                
            # Evolución KPIS
            camp_tab1, camp_tab2 = st.tabs([
                "Gasto y Ganancia", 
                "KPIS",
            ])
            # Tab 1: Gasto y Ganancia
                with camp_tab1:
                    st.subheader("Relación Gasto y Ganacia")
                    #Gráficas
                col1, col2 = st.columns(2)
                
                    with col1