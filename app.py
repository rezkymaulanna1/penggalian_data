import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Smart Farming Data Mining Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Global Styling
# =========================
st.markdown("""
<style>
    .main {
        background-color: #F7F9FB;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #111827 100%);
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.98rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        padding: 0.75rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .card-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 0.9rem;
        padding: 1rem 1rem 0.5rem 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0F766E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown('<div class="app-title">🌱 Smart Farming Data Mining Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Dashboard akademik untuk eksplorasi data, evaluasi model, dan prediksi yield berbasis fitur pertanian serta sensor.</div>',
    unsafe_allow_html=True
)

# =========================
# Data Loading
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv('Smart_Farming_Crop_Yield_2024.csv')
    df['sowing_date'] = pd.to_datetime(df['sowing_date'])
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def train_models(df):
    target = 'yield_kg_per_hectare'
    drop_cols = ['farm_id', 'sensor_id', 'yield_kg_per_hectare', 'sowing_date', 'harvest_date', 'timestamp']
    X = df.drop(columns=drop_cols)
    y = df[target]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

    models = {
        'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            'pipeline': pipe,
            'mae': mean_absolute_error(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds),
            'actual': y_test,
            'preds': preds,
        }
    return results

df = load_data()
results = train_models(df)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🌿 Smart Farming")
    st.caption("Academic & Professional Dashboard")
    page = st.radio(
        "Navigasi",
        ['Overview', 'Eksplorasi Data', 'Modeling', 'Prediksi', 'Batch Prediksi']
    )
    st.markdown("---")
    st.markdown("**Dataset Summary**")
    st.write(f"Rows: {len(df):,}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Crop types: {df['crop_type'].nunique()}")
    st.write(f"Regions: {df['region'].nunique()}")

# =========================
# Helper
# =========================
def metric_row():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Data", f"{len(df):,}")
    c2.metric("Jumlah Fitur", df.shape[1])
    c3.metric("Crop Type", df['crop_type'].nunique())
    c4.metric("Region", df['region'].nunique())

# =========================
# Pages
# =========================
if page == 'Overview':
    metric_row()

    tab1, tab2, tab3 = st.tabs(["Ringkasan", "Struktur Data", "Preview"])

    with tab1:
        st.markdown("### Ringkasan Dataset")
        st.write("Dataset memuat variabel lingkungan, irigasi, pupuk, NDVI, serta status penyakit tanaman untuk memprediksi yield_kg_per_hectare.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.write("**Informasi Utama**")
            st.write("- Cocok untuk analisis regresi.")
            st.write("- Mendukung eksplorasi hubungan fitur dan hasil panen.")
            st.write("- Dapat digunakan untuk evaluasi model machine learning.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.write("**Komponen Data**")
            st.write("- Fitur numerik: kelembapan tanah, pH, suhu, hujan, dll.")
            st.write("- Fitur kategorikal: region, crop type, irrigation, fertilizer.")
            st.write("- Target: yield_kg_per_hectare.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.dataframe(df.describe(include='all').T, use_container_width=True)

    with tab3:
        st.dataframe(df.head(10), use_container_width=True)

elif page == 'Eksplorasi Data':
    st.subheader('Eksplorasi Data')

    tab1, tab2, tab3 = st.tabs(["Distribusi", "Agregasi", "Korelasi"])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df['yield_kg_per_hectare'], bins=30, kde=True, ax=ax, color='#0F766E')
        ax.set_title('Distribusi Yield')
        ax.set_xlabel('Yield (kg/hectare)')
        ax.set_ylabel('Frekuensi')
        st.pyplot(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            crop_avg = df.groupby('crop_type')['yield_kg_per_hectare'].mean().sort_values()
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.barh(crop_avg.index, crop_avg.values, color='#2563EB')
            ax2.set_title('Rata-rata Yield per Crop Type')
            ax2.set_xlabel('Yield rata-rata')
            st.pyplot(fig2, use_container_width=True)

        with col2:
            region_avg = df.groupby('region')['yield_kg_per_hectare'].mean().sort_values()
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            ax3.barh(region_avg.index, region_avg.values, color='#14B8A6')
            ax3.set_title('Rata-rata Yield per Region')
            ax3.set_xlabel('Yield rata-rata')
            st.pyplot(fig3, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes(include=['int64', 'float64'])
        fig4, ax4 = plt.subplots(figsize=(11, 6))
        sns.heatmap(num_df.corr(), cmap='YlGnBu', ax=ax4, linewidths=0.3)
        ax4.set_title('Korelasi Fitur Numerik')
        st.pyplot(fig4, use_container_width=True)

elif page == 'Modeling':
    st.subheader('Evaluasi Model')

    summary = pd.DataFrame([
        {'Model': k, 'MAE': v['mae'], 'RMSE': v['rmse'], 'R2': v['r2']}
        for k, v in results.items()
    ]).sort_values('R2', ascending=False)

    st.dataframe(summary, use_container_width=True)

    selected = st.selectbox('Pilih model', list(results.keys()))
    res = results[selected]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(res['actual'], res['preds'], alpha=0.75, color='#0F766E', edgecolors='white', linewidth=0.5)
    mn = min(res['actual'].min(), res['preds'].min())
    mx = max(res['actual'].max(), res['preds'].max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    ax.set_xlabel('Actual Yield')
    ax.set_ylabel('Predicted Yield')
    ax.set_title(f'Actual vs Predicted — {selected}')
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Ringkasan performa")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{res['mae']:.2f}")
    c2.metric("RMSE", f"{res['rmse']:.2f}")
    c3.metric("R²", f"{res['r2']:.4f}")

elif page == 'Prediksi':
    st.subheader('Prediksi Yield Baru')
    model_name = st.selectbox('Model', list(results.keys()))

    with st.form('pred_form'):
        col1, col2 = st.columns(2)

        with col1:
            region = st.selectbox('Region', sorted(df['region'].dropna().astype(str).unique()))
            crop_type = st.selectbox('Crop Type', sorted(df['crop_type'].dropna().astype(str).unique()))
            irrigation_type = st.selectbox('Irrigation Type', sorted(df['irrigation_type'].dropna().astype(str).unique()))
            fertilizer_type = st.selectbox('Fertilizer Type', sorted(df['fertilizer_type'].dropna().astype(str).unique()))
            disease = st.selectbox('Crop Disease Status', sorted(df['crop_disease_status'].dropna().astype(str).unique()))
            total_days = st.number_input('Total Days', 1, 365, 120)
            ndvi = st.number_input('NDVI Index', 0.0, 1.0, 0.6)

        with col2:
            soil_moisture = st.number_input('Soil Moisture %', 0.0, 100.0, 30.0)
            soil_ph = st.number_input('Soil pH', 0.0, 14.0, 6.5)
            temperature = st.number_input('Temperature C', -10.0, 60.0, 25.0)
            rainfall = st.number_input('Rainfall mm', 0.0, 500.0, 150.0)
            humidity = st.number_input('Humidity %', 0.0, 100.0, 70.0)
            sunlight = st.number_input('Sunlight Hours', 0.0, 24.0, 6.0)
            pesticide = st.number_input('Pesticide Usage ml', 0.0, 100.0, 20.0)
            latitude = st.number_input('Latitude', -90.0, 90.0, 20.0)
            longitude = st.number_input('Longitude', -180.0, 180.0, 80.0)

        submitted = st.form_submit_button('Prediksi')

    if submitted:
        input_df = pd.DataFrame([{
            'region': region,
            'crop_type': crop_type,
            'soil_moisture_%': soil_moisture,
            'soil_pH': soil_ph,
            'temperature_C': temperature,
            'rainfall_mm': rainfall,
            'humidity_%': humidity,
            'sunlight_hours': sunlight,
            'irrigation_type': irrigation_type,
            'fertilizer_type': fertilizer_type,
            'pesticide_usage_ml': pesticide,
            'total_days': total_days,
            'latitude': latitude,
            'longitude': longitude,
            'NDVI_index': ndvi,
            'crop_disease_status': disease,
        }])
        pred = results[model_name]['pipeline'].predict(input_df)[0]
        st.success(f'Prediksi yield: {pred:,.2f} kg/hectare')

elif page == 'Batch Prediksi':
    st.subheader('Batch Prediksi via CSV')
    st.caption('Upload CSV dengan kolom fitur yang sama seperti input model.')
    model_name = st.selectbox('Model batch', list(results.keys()), key='batch_model')
    upload = st.file_uploader('Upload CSV batch', type=['csv'], key='batch_upload')

    if upload is not None:
        batch_df = pd.read_csv(upload)
        needed = [
            'region', 'crop_type', 'soil_moisture_%', 'soil_pH', 'temperature_C',
            'rainfall_mm', 'humidity_%', 'sunlight_hours', 'irrigation_type',
            'fertilizer_type', 'pesticide_usage_ml', 'total_days', 'latitude',
            'longitude', 'NDVI_index', 'crop_disease_status'
        ]
        missing = [c for c in needed if c not in batch_df.columns]
        if missing:
            st.error('Kolom yang kurang: ' + ', '.join(missing))
        else:
            preds = results[model_name]['pipeline'].predict(batch_df[needed])
            batch_df['predicted_yield_kg_per_hectare'] = preds
            st.dataframe(batch_df.head(50), use_container_width=True)
            st.download_button(
                'Download hasil prediksi',
                batch_df.to_csv(index=False).encode('utf-8'),
                'prediksi_batch.csv',
                'text/csv'
            )
