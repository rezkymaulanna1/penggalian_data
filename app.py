import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Prediksi Hasil Pertanian", 
    page_icon="🌾", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TAMBAHAN UNTUK TAMPILAN AKADEMIS ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 0px;}
    .sub-header {font-size: 1.2rem; color: #4B5563; text-align: center; margin-bottom: 30px; font-style: italic;}
    .section-header {color: #2563EB; border-bottom: 2px solid #E5E7EB; padding-bottom: 5px; margin-top: 30px;}
    </style>
""", unsafe_allow_html=True)

# --- FUNGSI CACHE DATA & MODEL ---
@st.cache_data
def load_data():
    # Menambahkan error handling jika file tidak ditemukan saat demo
    try:
        df = pd.read_csv('Smart_Farming_Crop_Yield_2024.csv')
        df['sowing_date'] = pd.to_datetime(df['sowing_date'])
        df['harvest_date'] = pd.to_datetime(df['harvest_date'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        # Fallback dummy data untuk testing UI jika file asli tidak ada
        return pd.DataFrame()

@st.cache_resource
def train_models(df):
    if df.empty: return {}
    target = 'yield_kg_per_hectare'
    drop_cols = ['farm_id','sensor_id','yield_kg_per_hectare','sowing_date','harvest_date','timestamp']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

    models = {
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    
    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            'pipeline': pipe,
            'mae': mean_absolute_error(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds),
            'actual': y_test,
            'preds': preds,
            'model_obj': model # Menyimpan model untuk feature importance
        }
    return results

# --- HEADER UTAMA ---
st.markdown('<p class="main-header">🌾 Implementasi Data Mining untuk Prediksi Hasil Pertanian</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analisis Berbasis Sensor Cerdas dan Fitur Lingkungan</p>', unsafe_allow_html=True)

df = load_data()
if df.empty:
    st.error("Dataset 'Smart_Farming_Crop_Yield_2024.csv' tidak ditemukan. Harap pastikan file berada di direktori yang sama.")
    st.stop()

with st.spinner("Melatih model machine learning..."):
    results = train_models(df)

# --- SIDEBAR NAVIGASI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913483.png", width=100) # Ikon akademik/pertanian
    st.title("Navigasi Penelitian")
    page = st.radio('Pilih Modul:', ['1. Tinjauan Dataset', '2. Analisis Eksploratori (EDA)', '3. Evaluasi Model', '4. Simulasi Prediksi (Single)', '5. Prediksi Batch (CSV)'])
    
    st.divider()
    st.markdown("**Informasi Studi**")
    st.caption("Penelitian ini membandingkan algoritma berbasis *Tree* (Decision Tree & Random Forest) dalam memprediksi *Crop Yield* berdasarkan variabel agronomi dan iklim.")

# --- HALAMAN 1: OVERVIEW ---
if page == '1. Tinjauan Dataset':
    st.markdown('<h3 class="section-header">Bab I: Tinjauan Dataset dan Metrik Utama</h3>', unsafe_allow_html=True)
    
    # Abstrak Singkat
    st.info("**Abstrak/Konteks:** Dataset ini memuat rekam jejak pertanian cerdas (*smart farming*), mencakup variabel iklim, praktik irigasi, penggunaan pupuk, hingga status penyakit tanaman. Tujuan pemodelan adalah memprediksi variabel dependen `yield_kg_per_hectare` secara akurat.")
    
    # Metrik Akademis
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Observasi (N)', f"{len(df):,}")
    col2.metric('Dimensi Fitur', df.shape[1])
    col3.metric('Varietas Tanaman', df['crop_type'].nunique())
    col4.metric('Wilayah Studi', df['region'].nunique())

    st.markdown("#### Sampel Observasi Data")
    st.dataframe(df.head(10), use_container_width=True)

# --- HALAMAN 2: EDA ---
elif page == '2. Analisis Eksploratori (EDA)':
    st.markdown('<h3 class="section-header">Bab II: Analisis Eksploratori Data (EDA)</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Target", "Analisis Kategorikal", "Korelasi Variabel"])
    
    with tab1:
        st.markdown("#### Distribusi Variabel Dependen (Yield)")
        fig = px.histogram(df, x="yield_kg_per_hectare", nbins=40, marginal="box", 
                           color_discrete_sequence=['#2E8B57'], title="Histogram & Boxplot Distribusi Hasil Panen")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            crop_avg = df.groupby('crop_type')['yield_kg_per_hectare'].mean().reset_index()
            fig2 = px.bar(crop_avg, x='yield_kg_per_hectare', y='crop_type', orientation='h',
                          title="Rata-rata Yield Berdasarkan Jenis Tanaman", color_discrete_sequence=['#4682B4'])
            fig2.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)
        with c2:
            region_avg = df.groupby('region')['yield_kg_per_hectare'].mean().reset_index()
            fig3 = px.bar(region_avg, x='yield_kg_per_hectare', y='region', orientation='h',
                          title="Rata-rata Yield Berdasarkan Wilayah", color_discrete_sequence=['#8FBC8F'])
            fig3.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown("#### Matriks Korelasi Variabel Numerik")
        num_df = df.select_dtypes(include=['int64','float64']).drop(columns=['farm_id', 'sensor_id'], errors='ignore')
        corr = num_df.corr()
        fig4 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='YlGnBu')
        st.plotly_chart(fig4, use_container_width=True)

# --- HALAMAN 3: MODELING ---
elif page == '3. Evaluasi Model':
    st.markdown('<h3 class="section-header">Bab III: Komparasi dan Evaluasi Model Machine Learning</h3>', unsafe_allow_html=True)
    
    # Tabel Ringkasan Metrik
    st.markdown("#### Ringkasan Performa Pemodelan (Test Set)")
    summary = pd.DataFrame([
        {'Algoritma': k, 'R² Score': v['r2'], 'RMSE': v['rmse'], 'MAE': v['mae']}
        for k, v in results.items()
    ]).sort_values('R² Score', ascending=False).reset_index(drop=True)
    
    # Styling Tabel
    st.dataframe(summary.style.highlight_max(subset=['R² Score'], color='#D4EDDA')
                 .highlight_min(subset=['RMSE', 'MAE'], color='#D4EDDA')
                 .format({'R² Score': '{:.4f}', 'RMSE': '{:.2f}', 'MAE': '{:.2f}'}), 
                 use_container_width=True)

    st.markdown("#### Analisis Visual: Actual vs Predicted")
    selected = st.selectbox('Pilih Algoritma untuk Divisualisasikan:', list(results.keys()))
    res = results[selected]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res['actual'], y=res['preds'], mode='markers', 
                             marker=dict(opacity=0.6, color='#2E8B57'), name='Prediksi vs Aktual'))
    
    # Garis Ideal
    mn = min(res['actual'].min(), res['preds'].min())
    mx = max(res['actual'].max(), res['preds'].max())
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', 
                             line=dict(color='red', dash='dash'), name='Garis Ideal (Perfect Fit)'))
    
    fig.update_layout(xaxis_title='Nilai Aktual (kg/ha)', yaxis_title='Nilai Prediksi (kg/ha)', 
                      title=f'Kesesuaian Model: {selected}', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# --- HALAMAN 4: PREDIKSI SINGLE ---
elif page == '4. Simulasi Prediksi (Single)':
    st.markdown('<h3 class="section-header">Bab IV: Simulasi Prediksi Berbasis Form Input</h3>', unsafe_allow_html=True)
    
    model_name = st.selectbox('Pilih Algoritma Prediksi:', list(results.keys()))
    
    with st.form('pred_form'):
        st.markdown("Masukkan parameter agrikultur untuk memprediksi potensi *yield*.")
        
        # Menggunakan kolom agar form terlihat rapi dan profesional
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**1. Data Lokasi & Tanaman**")
            region = st.selectbox('Wilayah (Region)', sorted(df['region'].dropna().astype(str).unique()))
            crop_type = st.selectbox('Jenis Tanaman', sorted(df['crop_type'].dropna().astype(str).unique()))
            disease = st.selectbox('Status Penyakit', sorted(df['crop_disease_status'].dropna().astype(str).unique()))
            total_days = st.number_input('Total Hari Tanam', 1, 365, 120)
            
        with col2:
            st.markdown("**2. Kondisi Iklim & Lingkungan**")
            temperature = st.number_input('Suhu (°C)', -10.0, 60.0, 25.0)
            humidity = st.number_input('Kelembaban (%)', 0.0, 100.0, 70.0)
            rainfall = st.number_input('Curah Hujan (mm)', 0.0, 500.0, 150.0)
            sunlight = st.number_input('Durasi Sinar Matahari (Jam)', 0.0, 24.0, 6.0)
            
        with col3:
            st.markdown("**3. Praktik Pertanian & Tanah**")
            soil_moisture = st.number_input('Kelembaban Tanah (%)', 0.0, 100.0, 30.0)
            soil_ph = st.number_input('pH Tanah', 0.0, 14.0, 6.5)
            irrigation_type = st.selectbox('Sistem Irigasi', sorted(df['irrigation_type'].dropna().astype(str).unique()))
            fertilizer_type = st.selectbox('Jenis Pupuk', sorted(df['fertilizer_type'].dropna().astype(str).unique()))
            pesticide = st.number_input('Penggunaan Pestisida (ml)', 0.0, 100.0, 20.0)
            
        st.divider()
        c_lat, c_lon, c_ndvi = st.columns(3)
        latitude = c_lat.number_input('Garis Lintang (Latitude)', -90.0, 90.0, 20.0)
        longitude = c_lon.number_input('Garis Bujur (Longitude)', -180.0, 180.0, 80.0)
        ndvi = c_ndvi.number_input('Indeks NDVI', 0.0, 1.0, 0.6)

        submitted = st.form_submit_button('Jalankan Simulasi Prediksi', type="primary", use_container_width=True)

    if submitted:
        input_dict = {
            'region': region, 'crop_type': crop_type, 'soil_moisture_%': soil_moisture,
            'soil_pH': soil_ph, 'temperature_C': temperature, 'rainfall_mm': rainfall,
            'humidity_%': humidity, 'sunlight_hours': sunlight, 'irrigation_type': irrigation_type,
            'fertilizer_type': fertilizer_type, 'pesticide_usage_ml': pesticide, 'total_days': total_days,
            'latitude': latitude, 'longitude': longitude, 'NDVI_index': ndvi, 'crop_disease_status': disease,
        }
        input_df = pd.DataFrame([input_dict])
        
        with st.spinner('Menghitung prediksi...'):
            pred = results[model_name]['pipeline'].predict(input_df)[0]
            
        st.success('Simulasi Berhasil Diselesaikan!')
        st.info(f"**Estimasi Hasil Panen (Yield):** `{pred:,.2f} kg/hectare` menggunakan algoritma **{model_name}**.")

# --- HALAMAN 5: PREDIKSI BATCH ---
elif page == '5. Prediksi Batch (CSV)':
    st.markdown('<h3 class="section-header">Bab V: Prediksi Skor Masal (Batch Processing)</h3>', unsafe_allow_html=True)
    st.markdown("Modul ini digunakan untuk mengevaluasi data observasi dalam jumlah besar menggunakan file CSV.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        model_name = st.selectbox('Pilih Algoritma:', list(results.keys()), key='batch_model')
    with col2:
        upload = st.file_uploader('Unggah Dataset Uji (Format .csv)', type=['csv'], key='batch_upload')

    if upload is not None:
        batch_df = pd.read_csv(upload)
        needed = ['region','crop_type','soil_moisture_%','soil_pH','temperature_C','rainfall_mm','humidity_%','sunlight_hours','irrigation_type','fertilizer_type','pesticide_usage_ml','total_days','latitude','longitude','NDVI_index','crop_disease_status']
        missing = [c for c in needed if c not in batch_df.columns]
        
        if missing:
            st.error('Validasi Gagal: Terdapat kolom yang hilang dalam dataset unggahan.')
            st.warning('Kolom yang dibutuhkan: ' + ', '.join(missing))
        else:
            with st.spinner('Memproses inferensi model...'):
                preds = results[model_name]['pipeline'].predict(batch_df[needed])
                batch_df['predicted_yield_kg_per_hectare'] = preds
                
            st.success("Inferensi Batch Selesai!")
            st.dataframe(batch_df.head(50), use_container_width=True)
            
            # Tombol Download Rapi
            csv_data = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil Prediksi (CSV)",
                data=csv_data,
                file_name='hasil_prediksi_batch.csv',
                mime='text/csv',
                type="primary"
            )
