# ------------ Librerías -----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------ Configuración página -----------------
st.set_page_config(
    page_title="Análisis Bianual de Resultados: Campañas de Marketing",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------ Estilos personalizados -----------------
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] {justify-content: center;}
    .stTabs [data-baseweb="tab"] {font-size: 1.1rem;}
    .metric-label {font-size: 1.1rem;}
    .metric-value {font-size: 1.5rem; font-weight: bold;}
    .stPlotlyChart {margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

# ------------ Títulos y descripción -----------------
st.title("🚀 Dashboard Interactivo: Análisis campañas de marketing")
st.markdown("""
<div style="font-size:1.1rem; color:#444;">
Este dashboard permite analizar los resultados de las campañas de marketing de la empresa en los últimos dos años. 
Explora el rendimiento por tipo de campaña, canal y audiencia, y descubre relaciones entre gasto, ganancias, ROI, tasa de conversión y beneficio neto.
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ------------ Variables de iniciaciación segura -----------------
df_filtrado = None
df = None

# ------------ Carga de datos -----------------
@st.cache_data(ttl=3600)
def cargar_datos():
    try:
        df = pd.read_csv('Preprocesamiento/pre-marketing.csv')
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
    'B2B': '#1f77b4',
    'B2C': '#ff7f0e'
}

# Unificación de nombres de color para audiencia
colores_target = colores_audiencia

# ------------ Asegurar tamaños positivos -----------------
def asegurar_positivo(valores, tamaño_min=3):
    if isinstance(valores, (pd.Series, np.ndarray, list)):
        return np.maximum(np.abs(valores), tamaño_min)
    else:
        return max(abs(valores), tamaño_min)

# ------------ Cargar datos -----------------
with st.spinner('Cargando datos...'):
    df = cargar_datos()

if df is not None and not df.empty:
    # Barra lateral para filtros
    st.sidebar.header("🎛️ Filtros de análisis")
    st.sidebar.markdown("Ajusta los filtros para personalizar el análisis de campañas.")

    # ------------ Filtros de datos para el usuario -----------------
    # Filtro de fecha
    min_fecha = pd.to_datetime(df['start_date']).min().date()
    max_fecha = pd.to_datetime(df['end_date']).max().date()

    rango_fechas = st.sidebar.date_input(
        "Selecciona el rango de fechas",
        [min_fecha, max_fecha],
        min_value=min_fecha,
        max_value=max_fecha,
        help="Filtra campañas por fecha de inicio y fin"
    )

    # Convertir fechas seleccionadas a datetime para filtrar
    if len(rango_fechas) == 2:
        fecha_inicio, fecha_fin = rango_fechas
        fecha_inicio = pd.to_datetime(fecha_inicio, utc=True)
        fecha_fin = pd.to_datetime(fecha_fin, utc=True)

    df['start_date'] = pd.to_datetime(df['start_date'], utc=True)
    df['end_date'] = pd.to_datetime(df['end_date'], utc=True)

    df_filtrado = df[(df['start_date'] >= fecha_inicio) & (df['end_date'] <= fecha_fin)].copy()

    # Validar si hay datos después del filtro de fecha
    if df_filtrado.empty:
        st.warning("No se encuentran campañas en el periodo seleccionado.")
        st.stop()

    # Filtro por tipo de campaña
    st.sidebar.header("🏠 Tipo de campaña")
    tipos_campaña = df['type'].unique().tolist()
    tipos_seleccionados = st.sidebar.multiselect(
        "Selecciona tipo(s) de campaña",
        options=tipos_campaña,
        default=tipos_campaña
    )
    if tipos_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['type'].isin(tipos_seleccionados)]

    # Filtro por canal
    st.sidebar.header("🪂 Canal")
    tipos_canal = df['channel'].unique().tolist()
    canales_seleccionados = st.sidebar.multiselect(
        "Selecciona canal(es)",
        options=tipos_canal,
        default=tipos_canal
    )
    if canales_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['channel'].isin(canales_seleccionados)]

    # Filtro por audiencia
    st.sidebar.header("👽 Audiencia")
    tipos_audiencia = df['target_audience'].unique().tolist()
    audiencias_seleccionadas = st.sidebar.multiselect(
        "Selecciona audiencia(s)",
        options=tipos_audiencia,
        default=tipos_audiencia
    )
    if audiencias_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['target_audience'].isin(audiencias_seleccionadas)]

    # Comprobar si hay datos después del filtrado
    if len(df_filtrado) == 0:
        st.warning("No hay datos disponibles con los filtros seleccionados. Por favor, ajusta los filtros.")
        st.stop()

    # Organización de los datos
    pestañas_principales = st.tabs([
        "📊 Resumen", 
        "🏠 Tipo de Campaña",
        "🪂 Canales",
        "👽 Audiencia", 
        "💟 Mejores Campañas",
        "🔥 Conclusiones"
    ])

    #------------------ 1. Resumen -----------------
    with pestañas_principales[0]:
        st.markdown("### 📊 Resumen general de campañas")
        st.markdown("Visualiza las métricas clave y la evolución temporal de las campañas seleccionadas.")

        # Principales Métricas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de campañas", len(df_filtrado))
        col2.metric("Gasto medio", f"{df_filtrado['budget'].mean():,.2f} €")
        col3.metric("Ganancia media", f"{df_filtrado['revenue'].mean():,.2f} €")
        col4.metric("ROI medio", f"{df_filtrado['roi'].mean():.2f}")

        st.markdown("---")

        # Gráficos cruce variables
        col_variables1, col_variables2 = st.columns(2)
        with col_variables1:
            st.subheader("Distribución del gasto por tipo de campaña y canal")
            fig_gasto = px.violin(
                df_filtrado,
                x='type',
                y='budget',
                color='channel',
                title='Resumen Gasto',
                box=True,
                points='all',
                labels={'type': 'Tipo de Campaña', 'budget': 'Gasto (€)', 'channel': 'Canal'},
                color_discrete_map=colores_canales
            )
            fig_gasto.update_layout(
                margin=dict(l=40, r=20, t=20, b=40),
                legend_title_text='Canal',
                font=dict(size=13),
                title_font=dict(size=16),
                title_x=0.0  # Centrar el título
            )
            st.plotly_chart(fig_gasto, use_container_width=True)

        with col_variables2:
            st.subheader("Distribución de la ganancia por tipo de campaña y canal")
            fig_ganancia = px.violin(
                df_filtrado,
                x='type',
                y='revenue',
                color='channel',
                title='Resumen Ganancia',
                box=True,
                points='all',
                labels={'type': 'Tipo de Campaña', 'revenue': 'Ganancia (€)', 'channel': 'Canal'},
                color_discrete_map=colores_canales
            )
            fig_ganancia.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text='Canal',
                font=dict(size=13),
                title_font=dict(size=16),
                title_x=0.0
            )
            st.plotly_chart(fig_ganancia, use_container_width=True)

        st.markdown("---")
        col_cruce1, col_cruce2 = st.columns(2)
        with col_cruce1:
            st.subheader("Distribución del ROI por tipo de campaña y canal")
            fig_roi = px.violin(
                df_filtrado,
                x='type',
                y='roi',
                color='channel',
                title='Resumen Roi',
                box=True,
                points='all',
                labels={'type': 'Tipo de Campaña', 'roi': 'ROI', 'channel': 'Canal'},
                color_discrete_map=colores_canales
            )
            fig_roi.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text='Canal',
                font=dict(size=13),
                title_font=dict(size=16),
                title_x=0.0
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col_cruce2:
            st.subheader("Distribución de la tasa de conversión por tipo de campaña y canal")
            fig_conversion = px.violin(
                df_filtrado,
                x='type',
                y='conversion_rate',
                color='channel',
                title='Resumen Conversión',
                box=True,
                points='all',
                labels={'type': 'Tipo de Campaña', 'conversion_rate': 'Tasa de Conversión', 'channel': 'Canal'},
                color_discrete_map=colores_canales
            )
            fig_conversion.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                legend_title_text='Canal',
                font=dict(size=13),
                title_font=dict(size=16),
                title_x=0.0
            )
            st.plotly_chart(fig_conversion, use_container_width=True)

        st.markdown("---")
        # Evolución de las métricas en el periodo
        st.subheader("Evolución de las métricas en el tiempo")
        if not df_filtrado.empty:
            df_filtrado['month'] = pd.to_datetime(df_filtrado['start_date']).dt.month
            df_filtrado['year'] = pd.to_datetime(df_filtrado['start_date']).dt.year
            df_filtrado['month_year'] = df_filtrado['year'].astype(str) + '-' + df_filtrado['month'].astype(str).str.zfill(2)
            df_filtrado['month_year'] = pd.to_datetime(df_filtrado['month_year'], format='%Y-%m')
            df_filtrado['month_year'] = df_filtrado['month_year'].dt.strftime('%Y-%m')

            col_evo1, col_evo2 = st.columns(2)
            with col_evo1:
                fig_gasto_meses = px.line(
                    df_filtrado.groupby('month_year')['budget'].mean().reset_index(),
                    x='month_year',
                    y='budget',
                    title='Gasto medio mensual',
                    labels={'month_year': 'Mes', 'budget': 'Gasto Medio (€)'},
                    color_discrete_sequence=['#636EFA']
                )
                fig_gasto_meses.update_layout(
                    xaxis_tickangle=45,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=13),
                    title_font=dict(size=16),
                    title_x=0.0
                )
                st.plotly_chart(fig_gasto_meses, use_container_width=True)

                fig_beneficio_meses = px.line(
                    df_filtrado.groupby('month_year')['beneficio_neto'].mean().reset_index(),
                    x='month_year',
                    y='beneficio_neto',
                    title='Beneficio neto medio mensual',
                    labels={'month_year': 'Mes', 'beneficio_neto': 'Beneficio Neto Medio (€)'},
                    color_discrete_sequence=['#00CC96']
                )
                fig_beneficio_meses.update_layout(
                    xaxis_tickangle=45,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=13),
                    title_font=dict(size=16),
                    title_x=0.0
                )
                st.plotly_chart(fig_beneficio_meses, use_container_width=True)

            with col_evo2:
                fig_roi_meses = px.line(
                    df_filtrado.groupby('month_year')['roi'].mean().reset_index(),
                    x='month_year',
                    y='roi',
                    title='ROI medio mensual',
                    labels={'month_year': 'Mes', 'roi': 'ROI Medio'},
                    color_discrete_sequence=['#AB63FA']
                )
                fig_roi_meses.update_layout(
                    xaxis_tickangle=45,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=13),
                    title_font=dict(size=16),
                    title_x=0.0
                )
                st.plotly_chart(fig_roi_meses, use_container_width=True)

                fig_conversion_meses = px.line(
                    df_filtrado.groupby('month_year')['conversion_rate'].mean().reset_index(),
                    x='month_year',
                    y='conversion_rate',
                    title='Tasa de conversión media mensual',
                    labels={'month_year': 'Mes', 'conversion_rate': 'Tasa de Conversión Media'},
                    color_discrete_sequence=['#FFA15A']
                )
                fig_conversion_meses.update_layout(
                    xaxis_tickangle=45,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=13),
                    title_font=dict(size=16),
                    title_x=0.0
                )
                st.plotly_chart(fig_conversion_meses, use_container_width=True)
                
        st.subheader("Campañas destacadas en julio 2023")    
        #encontrar campañas realizadas en julio
        if df is not None:
            # Asegurarse de que 'start_date' es de tipo string antes de usar .str
            if df['start_date'].dtype == 'object':
                campañas_julio = df[df['start_date'].str.contains('2023-07', na=False)]
            else:
                # Si no es string, convertir a string
                campañas_julio = df[df['start_date'].astype(str).str.contains('2023-07', na=False)]
            if not campañas_julio.empty:
                campañas_julio_seleccionado = campañas_julio[['campaign_name', 'start_date', 'budget', 'revenue', 'roi', 'conversion_rate']].copy()
                st.dataframe(campañas_julio_seleccionado)
            else:
                st.write("No hay campañas en julio de 2023 con los filtros actuales.")
        else:
            st.write("El DataFrame 'df' no está disponible.")
        

    #------------------ 2. Tipo de Campaña -----------------
    with pestañas_principales[1]:
        st.markdown("### 🏠 Análisis por tipo de campaña")
        st.markdown("Compara el rendimiento de cada tipo de campaña en los principales KPIs.")
        
        st.subheader("¿Qué tipo de campaña se usa más?")
        #gráfica de barras que muestre el uso de los diferentes tipos de campaña
        fig_uso_campañas = px.bar(
            df['type'].value_counts().reset_index(),
            x='count',
            y='type',
            title='Uso de tipos de campañas',
            labels={'index': 'type', 'type': 'Número de Campañas'},
            color='type',
            color_discrete_map=colores_campañas
        )
        fig_uso_campañas.update_layout(title_text='Uso de Tipos de campaña', title_x=0.0)
        st.plotly_chart(fig_uso_campañas, use_container_width=True)
        

        st.subheader("¿Qué tipo de campaña es más efectiva?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Campaña con más conversión", 
                    df_filtrado.groupby('type')['conversion_rate'].mean().idxmax())
        with col2:
            st.metric("Campaña con más ROI", 
                    df_filtrado.groupby('type')['roi'].mean().idxmax())
        with col3:
            st.metric("Campaña con más beneficio neto", 
                    df_filtrado.groupby('type')['beneficio_neto'].mean().idxmax())

        st.markdown("---")
        camp_tab1, camp_tab2 = st.tabs([
            "Gasto y Ganancia", 
            "KPIs",
        ])
        with camp_tab1:
            col1, col2 = st.columns(2)
            st.subheader("Relación entre gasto y ganancia por tipo de campaña")
            with col1:
                st.subheader("Gasto por campaña")
                # Calcular porcentaje de gasto por tipo de campaña
                porcentaje_gasto_campañas = (
                    df.groupby('type')['budget'].sum().reset_index(name='Gasto')
                )
                porcentaje_gasto_campañas['Porcentaje de Gasto'] = 100 * porcentaje_gasto_campañas['Gasto'] / porcentaje_gasto_campañas['Gasto'].sum()
                porcentaje_gasto_campañas = porcentaje_gasto_campañas.rename(columns={'type': 'Tipo de Campaña'})

                fig_gasto_campaña = px.pie(
                    porcentaje_gasto_campañas,
                    values='Porcentaje de Gasto',
                    names='Tipo de Campaña',
                    title='Distribución de la Gasto por Tipo de campaña',
                    color='Tipo de Campaña',
                    color_discrete_map=colores_campañas
                )
                fig_gasto_campaña.update_traces(textposition='inside', textinfo='percent+label')
                fig_gasto_campaña.update_layout(title_text='Distribución de la Gasto por Tipo campaña', title_x=0.5)
                st.plotly_chart(fig_gasto_campaña, use_container_width=True)

            with col2:
                porcentaje_ganancia_campaña = (
                    df.groupby('type')['revenue'].sum().reset_index(name='Ganancia')
                )
                porcentaje_ganancia_campaña['Porcentaje de Ganancia'] = 100 * porcentaje_ganancia_campaña['Ganancia'] / porcentaje_ganancia_campaña['Ganancia'].sum()

                fig_ganancia_campaña = px.pie(
                    porcentaje_ganancia_campaña,
                    values='Porcentaje de Ganancia',
                    names='type',
                    title='Ganancia por tipo de campaña',
                    color='type',
                    color_discrete_map=colores_campañas
                )
                fig_ganancia_campaña.update_traces(textposition='inside', textinfo='percent+label')
                fig_ganancia_campaña.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_x=0.0,
                    font=dict(size=13)
                )
                st.plotly_chart(fig_ganancia_campaña, use_container_width=True)

            st.markdown("---")
            st.subheader("Gasto y ganancia por tipo de campaña")
            fig_gasto_ganancia_campaña = px.scatter(
                df,
                x="budget",
                y="revenue",
                color="type",
                size="revenue",
                hover_data=['campaign_name'],
                color_discrete_map=colores_campañas,
                title="Gasto y ganancia por tipo de campaña",
                labels={"budget": "Gasto (€)", "revenue": "Ganancia (€)", "type": "Tipo de Campaña"},
                log_x=True
            )
            fig_gasto_ganancia_campaña.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Gasto (escala log)",
                yaxis_title="Ganancia",
                legend_title="Tipo de Campaña",
                font=dict(size=13),
                title_font=dict(size=16),
                title_x=0.0
            )
            st.plotly_chart(fig_gasto_ganancia_campaña, use_container_width=True)

        with camp_tab2:
            st.subheader("Comparativa de KPIs por tipo de campaña")
            col1, col2, col3 = st.columns(3)
            with col1:
                #Roi por campaña
                fig_roi_campaña = px.bar(
                    df.groupby('type')['roi'].mean().reset_index(),
                    x='type',
                    y='roi',
                    title='ROI medio por tipo de campaña',
                    color='type',
                    color_discrete_map=colores_campañas
                )
                fig_roi_campaña.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=13),
                    title_font=dict(size=16),
                    title_x=0.0,
                    showlegend=False
                )
                st.plotly_chart(fig_roi_campaña, use_container_width=True)
        
            with col2:
                #Conversión por campaña
                fig_conversion_campaña = px.box(
                    df,
                    x='type',
                    y='conversion_rate',
                    color='type',
                    title='Tasa de Conversión por Tipo de Campaña',
                    color_discrete_map=colores_campañas
                )
                fig_conversion_campaña.update_layout(title_text='Distribución de la Tasa de Conversión por Tipo de Campaña', title_x=0.0)
                st.plotly_chart(fig_conversion_campaña, use_container_width=True)
            
            with col3:
                #Beneficio neto por campaña
                fig_beneficio_campaña = px.bar(
                    df.groupby('type')['beneficio_neto'].sum().reset_index(),
                    x='type',
                    y='beneficio_neto',
                    title='Beneficio Neto por Tipo de Campaña',
                    color='type',
                    color_discrete_map=colores_campañas
                )
                fig_beneficio_campaña.update_layout(title_text='Beneficio Neto por Tipo de Campaña', title_x=0.0)
                st.plotly_chart(fig_beneficio_campaña, use_container_width=True)
        
        #Relacion gasto con kpis
        col_kpis1, col_kpis2 = st.columns(2)
        with col_kpis1:
            st.subheader("Relación entre gasto y ROI por tipo de campaña")
            fig_gasto_roi_camapaña = px.scatter(
                df,
                x="budget",
                y="roi",
                color="type",
                size="revenue",
                hover_data=['campaign_name'],
                color_discrete_map= colores_campañas,
                title="Gasto y ROI por Tipo de Campaña",
                labels={"budget": "Gasto", "roi": "ROI", "type": "Tipo de Campaña"},
                log_x=True
            )
            fig_gasto_roi_camapaña.update_layout(
                title_x=0.0,
                xaxis_title="Gasto (escala log)",
                yaxis_title="ROI",
                legend_title="Campaña",
                hovermode="closest"
            )
            st.plotly_chart(fig_gasto_roi_camapaña, use_container_width=True)
        
        with col_kpis2:
            st.subheader("Relación entre gasto y Conversión por tipo de campaña")
            fig_gasto_conversion_campaña = px.scatter(
                df,
                x="budget",
                y="conversion_rate",
                color="type",
                size="revenue",
                hover_data=['campaign_name'],
                color_discrete_map= colores_campañas,
                title="Gasto y Conversión por Tipo de Campaña",
                labels={"budget": "Gasto", "conversion_rate": "Tasa de conversión", "type": "Tipo de Campaña"},
                log_x=True
            )
            fig_gasto_conversion_campaña.update_layout(
                title_x=0.0,
                xaxis_title="Gasto (escala log)",
                yaxis_title="Conversión",
                legend_title="Campaña",
                hovermode="closest"
            )
            st.plotly_chart(fig_gasto_conversion_campaña, use_container_width=True)
            
    #------------------ 3. Canales -----------------
    with pestañas_principales[2]:
        st.markdown("### 🪂 Análisis por canal")
        st.markdown("Compara el rendimiento de cada canal en los principales KPIs.")
        
        st.subheader("¿Qué tipo canal se usa más?")
        #gráfica de barras que muestre el uso de los diferentes canales
        fig_uso_canales = px.bar(
            df['channel'].value_counts().reset_index(),
            x='count',
            y='channel',
            title='Uso de los Canales de Marketing',
            labels={'index': 'Canal', 'channel': 'Número de Campañas'},
            color='channel',
            color_discrete_map=colores_canales
        )
        fig_uso_canales.update_layout(title_text='Uso de los Canales de Marketing', title_x=0.0)
        st.plotly_chart(fig_uso_canales, use_container_width=True)
        
        st.subheader("¿Qué tipo canal es más efectivo?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Canal con más conversión", 
                    df_filtrado.groupby('channel')['conversion_rate'].mean().idxmax())
        with col2:
            st.metric("Canal con más ROI", 
                    df_filtrado.groupby('channel')['roi'].mean().idxmax())
        with col3:
            st.metric("Canal con más beneficio neto", 
                    df_filtrado.groupby('channel')['beneficio_neto'].mean().idxmax())

        st.markdown("---")
        can_tab1, can_tab2 = st.tabs([
            "Gasto y Ganancia", 
            "KPIs",
        ])
        with can_tab1:
            st.subheader("Relación entre gasto y ganancia por canal")
            col1, col2 = st.columns(2)
            with col1:
                porcentaje_gasto_canales = (
                    df.groupby('channel')['budget'].sum().reset_index(name='Gasto')
                )
                porcentaje_gasto_canales['Porcentaje de Gasto'] = 100 * porcentaje_gasto_canales['Gasto'] / porcentaje_gasto_canales['Gasto'].sum()
                porcentaje_gasto_canales = porcentaje_gasto_canales.rename(columns={'channel': 'Canal'})

                fig_gasto_canal = px.pie(
                    porcentaje_gasto_canales,
                    values='Porcentaje de Gasto',
                    names='Canal',
                    title='Distribución del gasto por canal',
                    color='Canal',
                    color_discrete_map=colores_canales
                )
                fig_gasto_canal.update_traces(textposition='inside', textinfo='percent+label')
                fig_gasto_canal.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_x=0.0,
                    font=dict(size=13)
                )
                st.plotly_chart(fig_gasto_canal, use_container_width=True, key="fig_gasto_canal")

            with col2:
                porcentaje_ganancia_canal = (
                    df.groupby('channel')['revenue'].sum().reset_index(name='Ganancia')
                )
                porcentaje_ganancia_canal['Porcentaje de Ganancia'] = 100 * porcentaje_ganancia_canal['Ganancia'] / porcentaje_ganancia_canal['Ganancia'].sum()

                fig_ganancia_canal = px.pie(
                    porcentaje_ganancia_canal,
                    values='Porcentaje de Ganancia',
                    names='channel',
                    title='Ganancia por canal',
                    color='channel',
                    color_discrete_map=colores_canales
                )
                fig_ganancia_canal.update_traces(textposition='inside', textinfo='percent+label')
                fig_ganancia_canal.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    title_x=0.0,
                    font=dict(size=13)
                )
                st.plotly_chart(fig_ganancia_canal, use_container_width=True, key="fig_ganancia_canal")

            st.markdown("---")
            st.subheader("Gasto y ganancia por canal")
            fig_gasto_ganancia_canal = px.scatter(
                df,
                x="budget",
                y="revenue",
                color="channel",
                size="revenue",
                hover_data=['campaign_name'],
                color_discrete_map=colores_canales,
                title="Gasto y ganancia por canal",
                labels={"budget": "Gasto (€)", "revenue": "Ganancia (€)", "channel": "Canal"},
                log_x=True
            )
            fig_gasto_ganancia_canal.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Gasto (escala log)",
                yaxis_title="Ganancia",
                legend_title="Canal",
                font=dict(size=13),
                title_font=dict(size=16),
                title_x=0.0
            )
            st.plotly_chart(fig_gasto_ganancia_canal, use_container_width=True, key="fig_gasto_ganancia_canal")

        with can_tab2:
            st.subheader("Comparativa de KPIs por canal")
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_roi_canal = px.bar(
                    df.groupby('channel')['roi'].mean().reset_index(),
                    x='channel',
                    y='roi',
                    title='ROI medio por canal',
                    color='channel',
                    color_discrete_map=colores_canales
                )
                fig_roi_canal.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(size=13),
                    title_font=dict(size=16),
                    title_x=0.0,
                    showlegend=False
                )
                st.plotly_chart(fig_roi_canal, use_container_width=True, key="fig_roi_canal")
        
            with col2:
                fig_conversion_canal = px.box(
                    df,
                    x='channel',
                    y='conversion_rate',
                    color='channel',
                    title='Tasa de Conversión por Canal',
                    color_discrete_map=colores_canales
                )
                fig_conversion_canal.update_layout(title_text='Tasa de Conversión por Canal', title_x=0.0)
                st.plotly_chart(fig_conversion_canal, use_container_width=True, key="fig_conversion_canal")
            
            with col3:
                fig_beneficio_canal = px.bar(
                    df.groupby('channel')['beneficio_neto'].sum().reset_index(),
                    x='channel',
                    y='beneficio_neto',
                    title='Beneficio Neto por Canal',
                    color='channel',
                    color_discrete_map=colores_canales
                )
                fig_beneficio_canal.update_layout(title_text='Beneficio Neto por Canal', title_x=0.0)
                st.plotly_chart(fig_beneficio_canal, use_container_width=True, key="fig_beneficio_canal")
        
        col_kpis1, col_kpis2 = st.columns(2)
        with col_kpis1:
            st.subheader("Relación entre gasto y ROI por canal")
            fig_gasto_roi_canal = px.scatter(
                df,
                x="budget",
                y="roi",
                color="channel",
                size="revenue",
                hover_data=['campaign_name'],
                color_discrete_map= colores_canales,
                title="Relación entre Gasto y ROI por Canal",
                labels={"budget": "Gasto", "roi": "ROI", "channel": "Canal"},
                log_x=True
            )
            fig_gasto_roi_canal.update_layout(
                title_x=0.0,
                xaxis_title="Gasto (escala log)",
                yaxis_title="ROI",
                legend_title="Canal",
                hovermode="closest"
            )
            st.plotly_chart(fig_gasto_roi_canal, use_container_width=True, key="fig_gasto_roi_canal")
        
        with col_kpis2:
            st.subheader("Relación entre gasto y Conversión por canal")
            fig_gasto_conversion = px.scatter(
                df,
                x="budget",
                y="conversion_rate",
                color="channel",
                size="revenue",
                hover_data=['campaign_name'],
                color_discrete_map= colores_canales,
                title="Gasto y Conversión por Canal",
                labels={"budget": "Gasto", "conversion_rate": "Tasa de conversión", "channel": "Canal"},
                log_x=True
            )
            fig_gasto_conversion.update_layout(
                title_x=0.0,
                xaxis_title="Gasto (escala log)",
                yaxis_title="Conversión",
                legend_title="Canal",
                hovermode="closest"
            )
            st.plotly_chart(fig_gasto_conversion, use_container_width=True, key="fig_gasto_conversion")
            
    #------------------ 4. Audiencia -----------------
    with pestañas_principales[3]:
        st.markdown("### 👽 Análisis por audiencia")
        st.markdown("Compara el rendimiento de los KPIs según la audiencia a la que se dirige.")

        col1, col2, col3 = st.columns(3)
        with col1:
            b2b = round(df[df['target_audience'] == 'B2B'].shape[0] / df.shape[0] * 100,2)
            st.metric("Porcentaje campañas B2B", f"{b2b}%")
        with col2:
            b2c = round(df[df['target_audience'] == 'B2C'].shape[0] / df.shape[0] * 100,2)
            st.metric("Porcentaje campañas B2C", f"{b2c}%")
        with col3:
            st.metric("Total campañas", df.shape[0])
        
        st.markdown("---")
            
        col_tar1, col_tar2, col_tar3, col_tar4 = st.columns(4)
        with col_tar1:
            st.metric("Audiencia con más gasto", 
                    df_filtrado.groupby('target_audience')['budget'].mean().idxmax())
        with col_tar2:
            st.metric("Audiencia con más ROI", 
                    df_filtrado.groupby('target_audience')['roi'].mean().idxmax())
        with col_tar3:
            st.metric("Audiencia con más conversión", 
                    df_filtrado.groupby('target_audience')['conversion_rate'].mean().idxmax())
        with col_tar4:
            st.metric("Audiencia con más beneficio neto", 
                    df_filtrado.groupby('target_audience')['beneficio_neto'].mean().idxmax())
                
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distrubusión del Roi por tipo de audiencia")
            fig_roi_target = px.violin(
                df,
                x='target_audience',
                y='roi',
                color='target_audience',
                box=True,
                points='all',
                title='ROI por Audiencia Objetivo',
                labels={'target_audience': 'Audiencia Objetivo', 'roi': 'ROI'},
                color_discrete_map=colores_target
            )
            fig_roi_target.update_layout(title_text='Distribución del ROI por Audiencia Objetivo', title_x=0.0)
            st.plotly_chart(fig_roi_target, use_container_width=True)
        
        with col2:
            st.subheader("Distribución de la conversión por audiencia")
            fig_conversion_target = px.box(
                df,
                x='target_audience',
                y='conversion_rate',
                color='target_audience',
                title='Distribución de la Tasa de Conversión por Audiencia Objetivo',
                labels={'target_audience': 'Audiencia Objetivo', 'conversion_rate': 'Tasa de Conversión'},
                color_discrete_map=colores_target
            )
            fig_conversion_target.update_layout(title_text='Distribución de la Tasa de Conversión por Audiencia Objetivo', title_x=0.0)
            st.plotly_chart(fig_conversion_target, use_container_width=True)
        st.markdown("---")
        
        fig_gasto_ganancia_target = px.scatter(
            df,
            x='budget',
            y='revenue',
            color='target_audience',
            size='revenue',
            size_max=40,
            title='Relación entre Gasto y Ganancia por Audiencia Objetivo (Escala Logarítmica)',
            labels={'budget': 'Gasto', 'revenue': 'Ganancia'},
            color_discrete_map=colores_target,
            log_x=True,
            log_y=True
        )
        fig_gasto_ganancia_target.update_layout(title_text='Relación entre Gasto y Ganancia por Audiencia Objetivo (Escala Logarítmica)', title_x=0.0)
        st.plotly_chart(fig_gasto_ganancia_target, use_container_width=True)
        
        st.subheader("Relación entre Gasto y Tasa de Conversión por Audiencia Objetivo")
        fig_gasto_conversion_target = px.scatter(
            df,
            x='budget',
            y='conversion_rate',
            color='target_audience',
            size='conversion_rate',
            size_max=40,
            title=' Gasto y Tasa de Conversión por Audiencia (Escala Logarítmica)',
            labels={'budget': 'Gasto', 'conversion_rate': 'Tasa de Conversión'},
            color_discrete_map=colores_target,
            log_x=True
        )
        fig_gasto_conversion_target.update_layout(title_text='Relación entre Gasto y Tasa de Conversión por Audiencia Objetivo (Escala Logarítmica)', title_x=0.0)
        st.plotly_chart(fig_gasto_conversion_target, use_container_width=True)
        
    #------------------ 5. Mejores Campañas -----------------
    with pestañas_principales[4]:
        st.markdown("### 💟 Mejores Campañas")
        st.markdown("Conozcamos las características concretas de las campañas con mejores resultados. .")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<p style='font-size: 0.8em;'>Campaña con más gasto: <br><strong>{df_filtrado.loc[df_filtrado['budget'].idxmax()]['campaign_name']}</strong></p>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='font-size: 0.8em;'>Campaña con más ROI: <br><strong>{df_filtrado.loc[df_filtrado['roi'].idxmax()]['campaign_name']}</strong></p>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<p style='font-size: 0.8em;'>Campaña con más conversión: <br><strong>{df_filtrado.loc[df_filtrado['conversion_rate'].idxmax()]['campaign_name']}</strong></p>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<p style='font-size: 0.8em;'>Campaña con más beneficio neto: <br><strong>{df_filtrado.loc[df_filtrado['beneficio_neto'].idxmax()]['campaign_name']}</strong></p>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Características de las mejores campañas")
        st.markdown("A continuación, se presentan las campañas con los mejores resultados en términos de ROI, conversión y beneficio neto.")
        fila = df[df['campaign_name'].isin(['Outlier Budget','Realigned radical hardware','Persevering zero administration interface', 'too manu conversions', 'Advanced systematic complexity'])]
        st.dataframe(fila)

    #------------------ 6. Conclusiones -----------------
    with pestañas_principales[5]:
        st.markdown("""
        <h2 style="color:#000;">📊 CONCLUSIONES EJECUTIVAS Y RECOMENDACIONES ESTRATÉGICAS</h2>
        <h4 style="color:#000;">ANÁLISIS BIANUAL DE CAMPAÑAS DE MARKETING</h4>
        <hr>
        <div style="background-color:#111; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#00CC96;">🔎 RESUMEN EJECUTIVO</h4>
            <ul>
            <li><b>Período analizado:</b> Últimos 2 años de actividad</li>
            <li><b>Métricas clave:</b> Gasto, Ganancia, ROI, Tasa de Conversión, Beneficio Neto</li>
            <li><b>Dimensiones:</b> Tipo de campaña, Canal de distribución, Audiencia objetivo</li>
            <li><b>Enfoque:</b> Dashboard interactivo con filtrado temporal y segmentación</li>
            </ul>
        </div>
        <br>
        <div style="background-color:#222; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#636EFA;">🌟 HALLAZGOS PRINCIPALES</h4>
            <ul>
            <li><b>Diversificación de Estrategias:</b>
            <ul>
            <li>7 tipos de campañas activas: <span style="color:#636EFA;">redes sociales</span>, <span style="color:#EF553B;">webinar</span>, <span style="color:#00CC96;">email</span>, <span style="color:#AB63FA;">podcast</span>, <span style="color:#FFA15A;">B2B</span>, evento, y <i>Desconocido</i></li>
            <li>5 canales principales: Referral, Unknown, orgánico, pagado, promoción</li>
            <li>2 audiencias objetivo: <b>B2B</b> y <b>B2C</b></li>
            </ul>
            </li>
            <li><b>Patrones Temporales:</b>
            <ul>
            <li>Estacionalidad: <b style="color:#EF553B;">Pico en julio 2023</b> (campañas destacadas)</li>
            <li>Evolución mensual de métricas clave con tendencias claras</li>
            <li>Variabilidad temporal en gasto, ROI y conversión</li>
            </ul>
            </li>
            <li><b>Campañas de Alto Rendimiento:</b>
            <ul>
            <li><b>Outlier Budget</b> – Mayor gasto</li>
            <li><b>Realigned radical hardware</b> – Alto ROI</li>
            <li><b>Persevering zero administration interface</b> – Mejor conversión</li>
            <li><b>Advanced systematic complexity</b> – Mayor beneficio neto</li>
            </ul>
            </li>
            </ul>
        </div>
        <br>
        <div style="background-color:#111; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#00CC96;">📈 ANÁLISIS POR DIMENSIONES</h4>
            <ul>
            <li><b>Tipos de campaña:</b>
            <ul>
            <li><b>Email:</b> Consistentemente efectivo en conversión</li>
            <li><b>Redes Sociales:</b> Mayor volumen, ROI variable</li>
            <li><b>Webinars:</b> Alto engagement y ROI</li>
            <li><b>Podcasts:</b> Resultados destacados en B2B</li>
            <li>Concentración del gasto en redes sociales</li>
            <li>Oportunidades de optimización presupuestaria</li>
            </ul>
            </li>
            <li><b>Canales de distribución:</b>
            <ul>
            <li><b>Orgánico:</b> Mejor relación costo-beneficio</li>
            <li><b>Pagado:</b> Mayor volumen, ROI moderado</li>
            <li><b>Referral:</b> Conversiones de alta calidad</li>
            <li><b>Unknown:</b> Requiere mejor tracking y atribución</li>
            <li>Potencial de rebalanceo hacia canales eficientes</li>
            </ul>
            </li>
            <li><b>Segmentación de audiencia:</b>
            <ul>
            <li><b>B2B:</b> Ciclos largos, mayor valor por conversión</li>
            <li><b>B2C:</b> Mayor volumen, conversiones rápidas</li>
            <li>ROI diferenciado por segmento</li>
            <li>Estrategias específicas según audiencia</li>
            </ul>
            </li>
            </ul>
        </div>
        <br>
        <div style="background-color:#222; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#AB63FA;">💡 INSIGHTS CRÍTICOS</h4>
            <ul>
            <li>No linealidad entre inversión y retorno: <b>Punto de saturación</b> en ciertas campañas</li>
            <li><b>Julio 2023</b> destaca por alta actividad y resultados</li>
            <li>Gap en clasificación/tracking para campañas/canales <i>Desconocido/Unknown</i></li>
            <li>Desalineación estratégica entre canales y tipos de campaña</li>
            <li><b>B2B:</b> Mayor ROI, menor volumen | <b>B2C:</b> Escalable, márgenes ajustados</li>
            <li><b>Principio de Pareto:</b> 20% de campañas concentran la mayoría del beneficio neto</li>
            <li>Tasa de conversión inversamente proporcional al tamaño de la audiencia</li>
            <li>Canales saturados muestran ROI decreciente; oportunidades en canales emergentes</li>
            </ul>
        </div>
        <br>
        <div style="background-color:#111; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#FFA15A;">🚀 RECOMENDACIONES ESTRATÉGICAS</h4>
            <ul>
            <li><b>Corto plazo (1-3 meses):</b>
            <ul>
            <li>Reclasificación de campañas <i>Desconocido</i></li>
            <li>Mejor tracking para canal <i>Unknown</i></li>
            <li>Análisis profundo de campañas <b>outlier</b> de julio 2023</li>
            <li>Optimización presupuestaria hacia canales de mayor ROI</li>
            </ul>
            </li>
            <li><b>Mediano plazo (3-6 meses):</b>
            <ul>
            <li>Rebalanceo de portfolio de tipos de campaña</li>
            <li>Estrategias diferenciadas <b>B2B</b> vs <b>B2C</b></li>
            <li>Modelo predictivo para campañas de alto potencial</li>
            <li>Dashboard en tiempo real para decisiones ágiles</li>
            </ul>
            </li>
            <li><b>Largo plazo (6-12 meses):</b>
            <ul>
            <li>Integración de datos de múltiples fuentes</li>
            <li>Automatización de optimización presupuestaria</li>
            <li>Desarrollo de benchmarks industriales</li>
            <li>Implementación de testing A/B sistemático</li>
            </ul>
            </li>
            </ul>
        </div>
        <br>
        <div style="background-color:#222; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#636EFA;">📏 MÉTRICAS DE SEGUIMIENTO</h4>
            <ul>
            <li>ROI promedio por tipo de campaña</li>
            <li>Costo por conversión por canal</li>
            <li>Beneficio neto mensual</li>
            <li>Eficiencia presupuestaria</li>
            <li><b>Indicadores de alerta:</b>
            <ul>
            <li>Campañas con ROI &lt; 1.0</li>
            <li>Canales con tendencia decreciente en conversión</li>
            <li>Desbalance presupuestario &gt; 70% en un solo canal</li>
            <li>Campañas sin clasificación &gt; 5% del total</li>
            </ul>
            </li>
            </ul>
        </div>
        <br>
        <div style="background-color:#111; padding:1em; border-radius:8px; color:#fff;">
            <h4 style="color:#00CC96;">🛠️ PRÓXIMOS PASOS</h4>
            <ul>
            <li>Sesión de <b>deep-dive</b> en campañas de alto rendimiento</li>
            <li>Workshop de optimización presupuestaria</li>
            <li>Mejoras en tracking y atribución</li>
            <li>Desarrollo de casos de uso específicos por vertical</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.info("Dashboard desarrollado con capacidades de filtrado temporal, segmentación multidimensional y visualizaciones interactivas para facilitar la toma de decisiones basada en datos.")
