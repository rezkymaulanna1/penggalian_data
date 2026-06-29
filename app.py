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
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Farming Analytics Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .stApp {
        background: linear-gradient(to bottom, #F8FAFC, #FFFFFF);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .card {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 1.25rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .section-title {
        color: #1E293B;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .small-text {
        color: #64748B;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# DATA LOADING
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv('Smart_Farming_Crop_Yield_2024.csv')
    df['sowing_date'] = pd.to_datetime(df['sowing_date'])
    df['harvest_date'] = pd.to_datetime(df['harvest_date'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# =========================
# MODEL TRAINING
# =========================
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
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

# =========================
# LOAD DATA
# =========================
df = load_data()
results = train_models(df)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🌱 Smart Farming")
st.sidebar.caption("Dashboard analisis hasil panen berbasis data sensor dan lingkungan")

page = st.sidebar.radio(
    "Navigasi",
    ['Overview', 'Eksplorasi Data', 'Evaluasi Model', 'Prediksi Individu', 'Batch Prediksi']
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset")
st.sidebar.write(f"**Total data:** {len(df):,}")
st.sidebar.write(f"**Jumlah fitur:** {df.shape[1]}")
st.sidebar.write(f"**Crop type:** {df['crop_type'].nunique()}")
st.sidebar.write(f"**Region:** {df['region'].nunique()}")

# =========================
# HEADER
# =========================
st.title("🌱 Smart Farming Analytics Dashboard")
st.caption("Dashboard akademik untuk eksplorasi data, evaluasi model, dan prediksi yield pertanian")

# =========================
# OVERVIEW
# =========================
if page == 'Overview':
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Data", f"{len(df):,}")
    c2.metric("Jumlah Fitur", df.shape[1])
    c3.metric("Crop Type", df['crop_type'].nunique())
    c4.metric("Region", df['region'].nunique())

    st.markdown("### Ringkasan Dataset")
    st.markdown("""
    <div class="card">
    Dataset ini digunakan untuk menganalisis hubungan antara kondisi lingkungan, input pertanian, dan hasil panen.
    Variabel yang dianalisis mencakup kelembapan tanah, pH tanah, curah hujan, suhu, NDVI, jenis irigasi,
    jenis pupuk, serta status penyakit tanaman.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Lihat 10 data teratas"):
        st.dataframe(df.head(10), use_container_width=True)

# =========================
# EXPLORATION
# =========================
elif page == 'Eksplorasi Data':
    tab1, tab2, tab3 = st.tabs(["Distribusi Yield", "Agregasi", "Korelasi"])

    with tab1:
        st.markdown("### Distribusi Yield")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['yield_kg_per_hectare'], bins=30, kde=True, ax=ax, color='#2E7D32')
        ax.set_xlabel("Yield (kg/hectare)")
        ax.set_ylabel("Frekuensi")
        ax.set_title("Distribusi Yield Panen")
        st.pyplot(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Rata-rata Yield per Crop Type")
            crop_avg = df.groupby('crop_type')['yield_kg_per_hectare'].mean().sort_values()
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.barh(crop_avg.index, crop_avg.values, color='#1565C0')
            ax2.set_xlabel("Rata-rata Yield")
            ax2.set_ylabel("Crop Type")
            st.pyplot(fig2, use_container_width=True)

        with col2:
            st.markdown("### Rata-rata Yield per Region")
            region_avg = df.groupby('region')['yield_kg_per_hectare'].mean().sort_values()
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.barh(region_avg.index, region_avg.values, color='#2E7D32')
            ax3.set_xlabel("Rata-rata Yield")
            ax3.set_ylabel("Region")
            st.pyplot(fig3, use_container_width=True)

    with tab3:
        st.markdown("### Korelasi Fitur Numerik")
        num_df = df.select_dtypes(include=['int64', 'float64'])
        fig4, ax4 = plt.subplots(figsize=(12, 7))
        sns.heatmap(num_df.corr(), cmap='YlGnBu', annot=False, ax=ax4)
        ax4.set_title("Heatmap Korelasi")
        st.pyplot(fig4, use_container_width=True)

# =========================
# MODEL EVALUATION
# =========================
elif page == 'Evaluasi Model':
    st.markdown("### Perbandingan Performa Model")
    summary = pd.DataFrame([
        {'Model': k, 'MAE': v['mae'], 'RMSE': v['rmse'], 'R2': v['r2']}
        for k, v in results.items()
    ]).sort_values('R2', ascending=False)

    st.dataframe(summary, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        selected 
