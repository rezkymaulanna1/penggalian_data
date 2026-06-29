elif page == 'Batch Prediksi':
    st.subheader('Batch Prediksi via CSV')
    st.caption('Upload CSV untuk menghasilkan prediksi sekaligus analisis visual hasil prediksi.')

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

            tab1, tab2, tab3 = st.tabs(["Ringkasan", "Visualisasi", "Data Hasil"])

            with tab1:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Baris", f"{len(batch_df):,}")
                c2.metric("Rata-rata Prediksi", f"{batch_df['predicted_yield_kg_per_hectare'].mean():,.2f}")
                c3.metric("Prediksi Tertinggi", f"{batch_df['predicted_yield_kg_per_hectare'].max():,.2f}")
                c4.metric("Prediksi Terendah", f"{batch_df['predicted_yield_kg_per_hectare'].min():,.2f}")

                st.markdown("### Statistik Prediksi")
                st.dataframe(
                    batch_df['predicted_yield_kg_per_hectare'].describe().to_frame().T,
                    use_container_width=True
                )

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Distribusi Prediksi")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    sns.histplot(
                        batch_df['predicted_yield_kg_per_hectare'],
                        bins=25,
                        kde=True,
                        color='#0F766E',
                        ax=ax1
                    )
                    ax1.set_title('Distribusi Prediksi Yield')
                    ax1.set_xlabel('Predicted Yield')
                    ax1.set_ylabel('Frekuensi')
                    st.pyplot(fig1, use_container_width=True)

                with col2:
                    st.markdown("#### Rata-rata Prediksi per Region")
                    region_pred = batch_df.groupby('region')['predicted_yield_kg_per_hectare'].mean().sort_values()
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.barh(region_pred.index.astype(str), region_pred.values, color='#2563EB')
                    ax2.set_title('Prediksi Yield per Region')
                    ax2.set_xlabel('Rata-rata Prediksi')
                    ax2.set_ylabel('Region')
                    st.pyplot(fig2, use_container_width=True)

                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("#### Rata-rata Prediksi per Crop Type")
                    crop_pred = batch_df.groupby('crop_type')['predicted_yield_kg_per_hectare'].mean().sort_values()
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.barh(crop_pred.index.astype(str), crop_pred.values, color='#14B8A6')
                    ax3.set_title('Prediksi Yield per Crop Type')
                    ax3.set_xlabel('Rata-rata Prediksi')
                    ax3.set_ylabel('Crop Type')
                    st.pyplot(fig3, use_container_width=True)

                with col4:
                    st.markdown("#### NDVI vs Prediksi")
                    fig4, ax4 = plt.subplots(figsize=(8, 4))
                    ax4.scatter(
                        batch_df['NDVI_index'],
                        batch_df['predicted_yield_kg_per_hectare'],
                        alpha=0.75,
                        color='#0F766E',
                        edgecolors='white'
                    )
                    ax4.set_title('NDVI vs Predicted Yield')
                    ax4.set_xlabel('NDVI Index')
                    ax4.set_ylabel('Predicted Yield')
                    ax4.grid(alpha=0.2)
                    st.pyplot(fig4, use_container_width=True)

            with tab3:
                st.markdown("### Hasil Prediksi Lengkap")
                display_cols = needed + ['predicted_yield_kg_per_hectare']
                st.dataframe(
                    batch_df[display_cols].style.format({
                        'predicted_yield_kg_per_hectare': '{:,.2f}'
                    }),
                    use_container_width=True
                )

                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download hasil prediksi',
                    csv,
                    'prediksi_batch.csv',
                    'text/csv'
                )elif page == 'Batch Prediksi':
    st.subheader('Batch Prediksi via CSV')
    st.caption('Upload CSV untuk menghasilkan prediksi sekaligus analisis visual hasil prediksi.')

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

            tab1, tab2, tab3 = st.tabs(["Ringkasan", "Visualisasi", "Data Hasil"])

            with tab1:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Baris", f"{len(batch_df):,}")
                c2.metric("Rata-rata Prediksi", f"{batch_df['predicted_yield_kg_per_hectare'].mean():,.2f}")
                c3.metric("Prediksi Tertinggi", f"{batch_df['predicted_yield_kg_per_hectare'].max():,.2f}")
                c4.metric("Prediksi Terendah", f"{batch_df['predicted_yield_kg_per_hectare'].min():,.2f}")

                st.markdown("### Statistik Prediksi")
                st.dataframe(
                    batch_df['predicted_yield_kg_per_hectare'].describe().to_frame().T,
                    use_container_width=True
                )

            with tab2:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Distribusi Prediksi")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    sns.histplot(
                        batch_df['predicted_yield_kg_per_hectare'],
                        bins=25,
                        kde=True,
                        color='#0F766E',
                        ax=ax1
                    )
                    ax1.set_title('Distribusi Prediksi Yield')
                    ax1.set_xlabel('Predicted Yield')
                    ax1.set_ylabel('Frekuensi')
                    st.pyplot(fig1, use_container_width=True)

                with col2:
                    st.markdown("#### Rata-rata Prediksi per Region")
                    region_pred = batch_df.groupby('region')['predicted_yield_kg_per_hectare'].mean().sort_values()
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.barh(region_pred.index.astype(str), region_pred.values, color='#2563EB')
                    ax2.set_title('Prediksi Yield per Region')
                    ax2.set_xlabel('Rata-rata Prediksi')
                    ax2.set_ylabel('Region')
                    st.pyplot(fig2, use_container_width=True)

                col3, col4 = st.columns(2)

                with col3:
                    st.markdown("#### Rata-rata Prediksi per Crop Type")
                    crop_pred = batch_df.groupby('crop_type')['predicted_yield_kg_per_hectare'].mean().sort_values()
                    fig3, ax3 = plt.subplots(figsize=(8, 4))
                    ax3.barh(crop_pred.index.astype(str), crop_pred.values, color='#14B8A6')
                    ax3.set_title('Prediksi Yield per Crop Type')
                    ax3.set_xlabel('Rata-rata Prediksi')
                    ax3.set_ylabel('Crop Type')
                    st.pyplot(fig3, use_container_width=True)

                with col4:
                    st.markdown("#### NDVI vs Prediksi")
                    fig4, ax4 = plt.subplots(figsize=(8, 4))
                    ax4.scatter(
                        batch_df['NDVI_index'],
                        batch_df['predicted_yield_kg_per_hectare'],
                        alpha=0.75,
                        color='#0F766E',
                        edgecolors='white'
                    )
                    ax4.set_title('NDVI vs Predicted Yield')
                    ax4.set_xlabel('NDVI Index')
                    ax4.set_ylabel('Predicted Yield')
                    ax4.grid(alpha=0.2)
                    st.pyplot(fig4, use_container_width=True)

            with tab3:
                st.markdown("### Hasil Prediksi Lengkap")
                display_cols = needed + ['predicted_yield_kg_per_hectare']
                st.dataframe(
                    batch_df[display_cols].style.format({
                        'predicted_yield_kg_per_hectare': '{:,.2f}'
                    }),
                    use_container_width=True
                )

                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    'Download hasil prediksi',
                    csv,
                    'prediksi_batch.csv',
                    'text/csv'
                )
