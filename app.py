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

st.set_page_config(page_title="Smart Farming Data Mining Dashboard", page_icon="🌱", layout="wide")

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
    drop_cols = ['farm_id','sensor_id','yield_kg_per_hectare','sowing_date','harvest_date','timestamp']
    X = df.drop(columns=drop_cols)
    y = df[target]

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
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

st.title('🌱 Smart Farming Data Mining Dashboard')
st.caption('Prediksi hasil panen (yield_kg_per_hectare) berbasis fitur pertanian dan sensor')

df = load_data()
results = train_models(df)

page = st.sidebar.radio('Navigasi', ['Overview', 'Eksplorasi Data', 'Modeling', 'Prediksi', 'Batch Prediksi'])

if page == 'Overview':
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Data', f"{len(df):,}")
    c2.metric('Jumlah Fitur', df.shape[1])
    c3.metric('Crop Type', df['crop_type'].nunique())
    c4.metric('Region', df['region'].nunique())

    st.subheader('Ringkasan Dataset')
    st.write('Dataset memuat variabel lingkungan, irigasi, pupuk, NDVI, dan status penyakit tanaman untuk memprediksi yield_kg_per_hectare.')
    st.dataframe(df.head(10), use_container_width=True)

elif page == 'Eksplorasi Data':
    st.subheader('Distribusi Yield')
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df['yield_kg_per_hectare'], bins=30, kde=True, ax=ax, color='#2e8b57')
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Rata-rata Yield per Crop Type')
        fig2, ax2 = plt.subplots(figsize=(6,4))
        crop_avg = df.groupby('crop_type')['yield_kg_per_hectare'].mean().sort_values()
        ax2.barh(crop_avg.index, crop_avg.values, color='#4682b4')
        st.pyplot(fig2)
    with col2:
        st.subheader('Rata-rata Yield per Region')
        fig3, ax3 = plt.subplots(figsize=(6,4))
        region_avg = df.groupby('region')['yield_kg_per_hectare'].mean().sort_values()
        ax3.barh(region_avg.index, region_avg.values, color='#8fbc8f')
        st.pyplot(fig3)

    st.subheader('Korelasi Fitur Numerik')
    num_df = df.select_dtypes(include=['int64','float64'])
    fig4, ax4 = plt.subplots(figsize=(10,6))
    sns.heatmap(num_df.corr(), cmap='YlGnBu', ax=ax4)
    st.pyplot(fig4)

elif page == 'Modeling':
    st.subheader('Perbandingan Model Regresi')
    summary = pd.DataFrame([
        {'Model': k, 'MAE': v['mae'], 'RMSE': v['rmse'], 'R2': v['r2']}
        for k, v in results.items()
    ]).sort_values('R2', ascending=False)
    st.dataframe(summary, use_container_width=True)

    selected = st.selectbox('Pilih model', list(results.keys()))
    res = results[selected]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(res['actual'], res['preds'], alpha=0.7, color='#2e8b57')
    mn = min(res['actual'].min(), res['preds'].min())
    mx = max(res['actual'].max(), res['preds'].max())
    ax.plot([mn,mx],[mn,mx],'r--')
    ax.set_xlabel('Actual Yield')
    ax.set_ylabel('Predicted Yield')
    ax.set_title(f'Actual vs Predicted — {selected}')
    st.pyplot(fig)

elif page == 'Prediksi':
    st.subheader('Prediksi Yield Baru')
    model_name = st.selectbox('Model', list(results.keys()))

    with st.form('pred_form'):
        region = st.selectbox('Region', sorted(df['region'].dropna().astype(str).unique()))
        crop_type = st.selectbox('Crop Type', sorted(df['crop_type'].dropna().astype(str).unique()))
        soil_moisture = st.number_input('Soil Moisture %', 0.0, 100.0, 30.0)
        soil_ph = st.number_input('Soil pH', 0.0, 14.0, 6.5)
        temperature = st.number_input('Temperature C', -10.0, 60.0, 25.0)
        rainfall = st.number_input('Rainfall mm', 0.0, 500.0, 150.0)
        humidity = st.number_input('Humidity %', 0.0, 100.0, 70.0)
        sunlight = st.number_input('Sunlight Hours', 0.0, 24.0, 6.0)
        irrigation_type = st.selectbox('Irrigation Type', sorted(df['irrigation_type'].dropna().astype(str).unique()))
        fertilizer_type = st.selectbox('Fertilizer Type', sorted(df['fertilizer_type'].dropna().astype(str).unique()))
        pesticide = st.number_input('Pesticide Usage ml', 0.0, 100.0, 20.0)
        total_days = st.number_input('Total Days', 1, 365, 120)
        latitude = st.number_input('Latitude', -90.0, 90.0, 20.0)
        longitude = st.number_input('Longitude', -180.0, 180.0, 80.0)
        ndvi = st.number_input('NDVI Index', 0.0, 1.0, 0.6)
        disease = st.selectbox('Crop Disease Status', sorted(df['crop_disease_status'].dropna().astype(str).unique()))
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
    st.caption('Upload CSV dengan kolom fitur yang sama seperti dataset. Jika kolom yield_kg_per_hectare tersedia, sistem akan menampilkan evaluasi metrik dan visualisasi perbandingan aktual vs prediksi.')
    model_name = st.selectbox('Model batch', list(results.keys()), key='batch_model')
    upload = st.file_uploader('Upload CSV batch', type=['csv'], key='batch_upload')

    if upload is not None:
        batch_df = pd.read_csv(upload)
        needed = ['region','crop_type','soil_moisture_%','soil_pH','temperature_C','rainfall_mm','humidity_%','sunlight_hours','irrigation_type','fertilizer_type','pesticide_usage_ml','total_days','latitude','longitude','NDVI_index','crop_disease_status']
        missing = [c for c in needed if c not in batch_df.columns]
        if missing:
            st.error('Kolom yang kurang: ' + ', '.join(missing))
        else:
            preds = results[model_name]['pipeline'].predict(batch_df[needed])
            batch_df['predicted_yield_kg_per_hectare'] = preds

            # --- Metrics Dashboard ---
            has_actual = 'yield_kg_per_hectare' in batch_df.columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric('Jumlah Data', f"{len(batch_df):,}")
            col2.metric('Rata-rata Prediksi', f"{preds.mean():,.2f}")
            col3.metric('Min Prediksi', f"{preds.min():,.2f}")
            col4.metric('Max Prediksi', f"{preds.max():,.2f}")

            if has_actual:
                mae = mean_absolute_error(batch_df['yield_kg_per_hectare'], preds)
                rmse = np.sqrt(mean_squared_error(batch_df['yield_kg_per_hectare'], preds))
                r2 = r2_score(batch_df['yield_kg_per_hectare'], preds)
                st.markdown('#### Evaluasi Model pada Batch Data')
                c1, c2, c3 = st.columns(3)
                c1.metric('MAE', f"{mae:,.2f}")
                c2.metric('RMSE', f"{rmse:,.2f}")
                c3.metric('R²', f"{r2:.4f}")

            # --- Visualization 1: Distribusi Prediksi ---
            st.markdown('#### Distribusi Hasil Prediksi Yield')
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            sns.histplot(batch_df['predicted_yield_kg_per_hectare'], bins=30, kde=True, color='#2e8b57', ax=ax1, label='Prediksi')
            if has_actual:
                sns.histplot(batch_df['yield_kg_per_hectare'], bins=30, kde=True, color='#cd5c5c', ax=ax1, alpha=0.6, label='Aktual')
                ax1.legend()
            ax1.set_xlabel('Yield (kg/hectare)')
            ax1.set_ylabel('Frekuensi')
            ax1.set_title('Distribusi Yield: Prediksi vs Aktual' if has_actual else 'Distribusi Yield Prediksi')
            st.pyplot(fig1)

            # --- Visualization 2: Scatter Plot (Aktual vs Prediksi) ---
            if has_actual:
                st.markdown('#### Scatter Plot: Aktual vs Prediksi')
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.scatter(batch_df['yield_kg_per_hectare'], batch_df['predicted_yield_kg_per_hectare'], alpha=0.6, color='#4682b4', edgecolors='k', linewidth=0.5)
                mn = min(batch_df['yield_kg_per_hectare'].min(), batch_df['predicted_yield_kg_per_hectare'].min())
                mx = max(batch_df['yield_kg_per_hectare'].max(), batch_df['predicted_yield_kg_per_hectare'].max())
                ax2.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction')
                ax2.set_xlabel('Aktual Yield (kg/hectare)')
                ax2.set_ylabel('Prediksi Yield (kg/hectare)')
                ax2.set_title('Actual vs Predicted Yield')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)

            # --- Visualization 3: Box Plot per Crop Type ---
            st.markdown('#### Distribusi Prediksi Yield per Crop Type')
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='crop_type', y='predicted_yield_kg_per_hectare', data=batch_df, palette='Set2', ax=ax3)
            ax3.set_xlabel('Crop Type')
            ax3.set_ylabel('Prediksi Yield (kg/hectare)')
            ax3.set_title('Box Plot Prediksi Yield berdasarkan Jenis Tanaman')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            st.pyplot(fig3)

            # --- Visualization 4: Bar Plot per Region ---
            st.markdown('#### Rata-rata Prediksi Yield per Region')
            fig4, ax4 = plt.subplots(figsize=(10, 5))
            region_avg = batch_df.groupby('region')['predicted_yield_kg_per_hectare'].mean().sort_values()
            colors = sns.color_palette('YlGn', len(region_avg))
            ax4.barh(region_avg.index, region_avg.values, color=colors)
            ax4.set_xlabel('Rata-rata Prediksi Yield (kg/hectare)')
            ax4.set_title('Rata-rata Prediksi Yield per Region')
            st.pyplot(fig4)

            # --- Visualization 5: Feature Importance (if tree-based) ---
            if model_name in ['Decision Tree', 'Random Forest'] and hasattr(results[model_name]['pipeline'].named_steps['model'], 'feature_importances_'):
                st.markdown('#### Feature Importance')
                model = results[model_name]['pipeline'].named_steps['model']
                preprocessor = results[model_name]['pipeline'].named_steps['preprocessor']

                # Get feature names
                num_features = preprocessor.transformers_[0][2]
                cat_features = preprocessor.transformers_[1][2]
                cat_onehot = preprocessor.named_transformers_['cat']['onehot']
                cat_feature_names = cat_onehot.get_feature_names_out(cat_features)
                all_features = list(num_features) + list(cat_feature_names)

                importances = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=True).tail(15)
                fig5, ax5 = plt.subplots(figsize=(8, 6))
                importances.plot(kind='barh', color='#2e8b57', ax=ax5)
                ax5.set_title('Top 15 Feature Importances')
                ax5.set_xlabel('Importance Score')
                st.pyplot(fig5)

            # --- Data Table ---
            st.markdown('#### Preview Hasil Prediksi')
            st.dataframe(batch_df.head(50), use_container_width=True)

            # --- Download ---
            st.download_button('Download hasil prediksi', batch_df.to_csv(index=False).encode('utf-8'), 'prediksi_batch.csv', 'text/csv')
