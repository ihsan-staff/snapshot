
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
#from google.colab import auth
import google.auth
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(layout="wide")

# Helper function to format DPW list for hover data
def format_dpw_list(dpw_string, max_per_line=4):
    if not isinstance(dpw_string, str) or not dpw_string:
        return ""
    dpws = [d.strip() for d in dpw_string.split(',')] 
    formatted_lines = []
    for i in range(0, len(dpws), max_per_line):
        formatted_lines.append(', '.join(dpws[i:i+max_per_line]))
    return '<br>'.join(formatted_lines)

@st.cache_data(ttl=3600) # Cache data for 1 hour
def load_data():
    # In Streamlit Cloud, you typically authenticate with st.secrets or a service account.
    # For local Colab execution, you might have used auth.authenticate_user() previously.
    # For simplicity and broad compatibility with external deployment, let's use a public CSV export URL if available
    # or prompt the user for credentials if full gspread features are strictly needed.
    
    # Using the gviz/tq endpoint for public access to CSV data
    gsheet_url = "https://docs.google.com/spreadsheets/d/1hg0PuNymzzMp1CslB11XDulzAZR5BPOIYkxFkWHDCDU/gviz/tq?tqx=out:csv"

    try:
        df = pd.read_csv(gsheet_url)
        df = df.rename(columns={
            'Email Address': 'Email',
            'Asal DPW': 'DPW',
            'Jenjang keanggotaan yang dikelola BKAP DPW:': 'Jenjang',
            'Biro yang sudah dibentuk di BKAP DPW:': 'Biro',
            'Jumlah personil kepengurusan BKAP DPW:': 'Personil',
            'Apakah BKAP DPW telah menyusun program Kaderisasi Tahun 2026?': 'Program',
            'Dalam kurun waktu November 2025 - Januari 2026, berapa kali rapat rutin internal BKAP DPW terlaksana?': 'Internal',
            'Dalam kurun waktu November 2025 - Januari 2026, berapa kali rapat rutin koordinasi BKAP DPW dengan BKAP DPD terlaksana?': 'Rakor',
            'Dalam kurun waktu November 2025 - Januari 2026, berapa kali KunKer ke BKAP DPD terlaksana?': 'KunKer',
            'Program rekrutmen yang sudah dicanangkan BKAP DPW:': 'Rekrutmen',
            'Rata-rata terlaksananya UPA Utama': 'UPAU',
            'Rata-rata kehadiran pembimbing dalam pelaksanaan UPA Utama': 'PembimbingAU',
            'Rata-rata terlaksananya UPA Dewasa': 'UPAD',
            'Rata-rata kehadiran pembimbing dalam pelaksanaan UPA Dewasa': 'PembimbingAD',
            'Rata-rata terlaksananya UPA Madya': 'UPAM',
            'Rata-rata kehadiran pembimbing dalam pelaksanaan UPA Madya': 'PembimbingAM',
            'Apakah data Anggota dan data UPA sudah  diperbaharui pada aplikasi Sapulidi?': 'Sapulidi',
            'Beri tanda centang pada fitur Sapulidi yang sudah anda gunakan': 'Fitur',
            'Rata-rata kejelasan Anggota UPA terhadap konsep, kerangka, arah kebijakan dan strategi Kurikulum 1447 H': 'Kejelasan_Konsep_UPA',
            'Rata-rata suasana ukhuwah di UPA': 'Suasana_Ukhuwah_UPA',
            'Rata-rata suasana dakwah harakiyah di UPA': 'Suasana_Dakwah_Harakiyah_UPA',
            'Rata-rata suasana ilmiah tsaqafiyah di UPA': 'Suasana_Ilmiah_Tsaqafiyah_UPA',
            'Rata-rata tingkat kemampuan ekonomi Anggota UPA': 'Tingkat_Ekonomi_UPA',
            'Rata-rata terselesaikannya qadhayah Anggota UPA': 'Qadhayah_Anggota_UPA',
            'Rata-rata jumlah Anggota UPA yang sedang bertugas menjadi Pembina UPA Penggerak': 'Jml_Pembina_UPA_Penggerak',
            'Rata-rata program rekrutmen Anggota Muda di UPA': 'Program_Rekrutmen_Anggota_Muda',
            'Rata-rata implementasi Kurikulum 1447 H di UPA': 'Implementasi_Kurikulum_UPA'
        })

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Comprehensive Clustering Logic (must be performed here to reuse df) ---
@st.cache_data(ttl=3600)
def perform_comprehensive_clustering(df_input):
    df_preprocessed = df_input.copy()
    df_preprocessed = df_preprocessed.set_index('DPW')

    numerical_cols = [
        'Personil', 'Internal', 'Rakor', 'KunKer', 'Kejelasan_Konsep_UPA'
    ]

    for col in numerical_cols:
        df_preprocessed[col] = pd.to_numeric(df_preprocessed[col], errors='coerce')
        df_preprocessed[col] = df_preprocessed[col].fillna(df_preprocessed[col].mean())

    multi_choice_dfs = []
    multi_choice_cols = ['Jenjang', 'Biro', 'Rekrutmen', 'Fitur']
    for col in multi_choice_cols:
        df_preprocessed[col] = df_preprocessed[col].fillna('')
        exploded_df = df_preprocessed.assign(temp_col=df_preprocessed[col].str.split(', ')).explode('temp_col')
        exploded_df['temp_col'] = exploded_df['temp_col'].str.strip()
        exploded_df = exploded_df[exploded_df['temp_col'] != '']
        dummies = pd.get_dummies(exploded_df['temp_col'], prefix=col)
        processed_multi_choice = exploded_df.reset_index()[['DPW']].join(dummies).groupby('DPW').sum()
        multi_choice_dfs.append(processed_multi_choice)

    single_choice_cols = [
        'Program', 'UPAU', 'PembimbingAU', 'UPAD', 'PembimbingAD', 'UPAM',
        'PembimbingAM', 'Sapulidi', 'Suasana_Ukhuwah_UPA', 'Suasana_Dakwah_Harakiyah_UPA',
        'Suasana_Ilmiah_Tsaqafiyah_UPA', 'Tingkat_Ekonomi_UPA', 'Qadhayah_Anggota_UPA',
        'Jml_Pembina_UPA_Penggerak', 'Program_Rekrutmen_Anggota_Muda',
        'Implementasi_Kurikulum_UPA'
    ]

    single_choice_encoded_df = pd.DataFrame(index=df_preprocessed.index)
    for col in single_choice_cols:
        df_preprocessed[col] = df_preprocessed[col].fillna('Missing')
        dummies = pd.get_dummies(df_preprocessed[col], prefix=col)
        single_choice_encoded_df = single_choice_encoded_df.join(dummies)

    final_df_for_clustering = df_preprocessed[numerical_cols].copy()
    for processed_df in multi_choice_dfs:
        final_df_for_clustering = final_df_for_clustering.merge(processed_df, left_index=True, right_index=True, how='left')
    final_df_for_clustering = final_df_for_clustering.merge(single_choice_encoded_df, left_index=True, right_index=True, how='left')
    final_df_for_clustering = final_df_for_clustering.fillna(0)

    scaler = StandardScaler()
    scaled_numerical_features = scaler.fit_transform(final_df_for_clustering[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=[f"Scaled_{col}" for col in numerical_cols], index=final_df_for_clustering.index)
    final_df_for_clustering_scaled = scaled_numerical_df.join(final_df_for_clustering.drop(columns=numerical_cols))

    k_optimal_comprehensive = 4 # Based on previous Elbow/Silhouette analysis
    kmeans_comprehensive = KMeans(n_clusters=k_optimal_comprehensive, random_state=42, n_init='auto')
    kmeans_comprehensive.fit(final_df_for_clustering_scaled)
    final_df_for_clustering_scaled['Comprehensive_Cluster_Label'] = kmeans_comprehensive.labels_

    cluster_labels_df = final_df_for_clustering_scaled.reset_index()[['DPW', 'Comprehensive_Cluster_Label']]
    df_with_comprehensive_clusters = df_input.merge(cluster_labels_df, on='DPW', how='left')

    cluster_characterization_df = final_df_for_clustering.reset_index().merge(
        final_df_for_clustering_scaled[['Comprehensive_Cluster_Label']].reset_index(),
        on='DPW',
        how='left'
    )

    cluster_summary_numerical = cluster_characterization_df.groupby('Comprehensive_Cluster_Label')[numerical_cols].mean().reset_index()
    cluster_summary_numerical = cluster_summary_numerical.sort_values(by='Personil').reset_index(drop=True)

    cluster_names_comprehensive = {
        cluster_summary_numerical.loc[0, 'Comprehensive_Cluster_Label']: 'DPW Kecil & Kurang Aktif',
        cluster_summary_numerical.loc[1, 'Comprehensive_Cluster_Label']: 'DPW Sedang & Cukup Aktif',
        cluster_summary_numerical.loc[2, 'Comprehensive_Cluster_Label']: 'DPW Besar & Aktif',
        cluster_summary_numerical.loc[3, 'Comprehensive_Cluster_Label']: 'DPW Sangat Besar & Sangat Aktif'
    }

    df_with_comprehensive_clusters['Comprehensive_Cluster_Name'] = df_with_comprehensive_clusters['Comprehensive_Cluster_Label'].map(cluster_names_comprehensive)
    cluster_characterization_df['Comprehensive_Cluster_Name'] = cluster_characterization_df['Comprehensive_Cluster_Label'].map(cluster_names_comprehensive)
    final_cluster_summary = cluster_characterization_df.groupby('Comprehensive_Cluster_Name')[numerical_cols].agg(['mean', 'count', 'min', 'max'])
    
    return df_with_comprehensive_clusters, final_cluster_summary


# --- Load Data ---
df = load_data()

if not df.empty:
    df_with_comprehensive_clusters, final_cluster_summary = perform_comprehensive_clustering(df.copy())

    st.title('Dashboard Snapshot BKAP DPW')
    total_responden = df.shape[0]  

    # --- Bagian 1: Kepengurusan ---
    st.header('Bagian 1: Kepengurusan')
    #st.header(f'Total responden: {total_responden}')
    st.header('Tes')

    st.subheader('Jenjang Keanggotaan yang dikelola BKAP DPW')
    df['Jenjang'] = df['Jenjang'].fillna('')
    jenjang_exploded = df.assign(Jenjang=df['Jenjang'].str.split(', ')).explode('Jenjang')
    jenjang_exploded['Jenjang'] = jenjang_exploded['Jenjang'].str.strip()
    jenjang_exploded = jenjang_exploded[jenjang_exploded['Jenjang'] != '']
    jenjang_counts_plotly = jenjang_exploded.groupby(['DPW', 'Jenjang']).size().reset_index(name='Jumlah')

    color_map = {
        'Utama': '#FE5000',
        'Dewasa': 'blue',
        'Madya': 'green',
        'Muda': '#0080ff',
        'Pratama': 'red'
    }

    fig_jenjang_dpw = px.bar(
        jenjang_counts_plotly,
        x='DPW',
        y='Jumlah',
        color='Jenjang',
#        title='Jenjang Keanggotaan yang dikelola BKAP DPW',
        labels={'DPW': 'DPW', 'Jenjang': 'Jenjang'},
        color_discrete_map=color_map,
        height=600
    )
    st.plotly_chart(fig_jenjang_dpw, use_container_width=True)

    st.subheader('Persentase DPW yang Mengelola Setiap Jenjang Keanggotaan')
    dpw_per_jenjang = jenjang_exploded.groupby('Jenjang')['DPW'].nunique().reset_index(name='Jumlah_DPW')
    total_dpw_unik = jenjang_exploded['DPW'].nunique()
    dpw_per_jenjang['Persentase_DPW'] = (dpw_per_jenjang['Jumlah_DPW'] / total_dpw_unik) * 100
    dpws_in_each_jenjang = jenjang_exploded.groupby('Jenjang')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    dpw_per_jenjang = pd.merge(dpw_per_jenjang, dpws_in_each_jenjang, on='Jenjang')
    dpw_per_jenjang['Formatted_DPW_List'] = dpw_per_jenjang['DPW_List'].apply(format_dpw_list)

    fig_jenjang_percent = px.bar(
        dpw_per_jenjang,
        x='Jenjang',
        y='Persentase_DPW',
#        title='Persentase DPW yang Mengelola Setiap Jenjang Keanggotaan',
        color='Jenjang',
        color_discrete_map=color_map,
        hover_data={'Jumlah_DPW': True, 'Formatted_DPW_List': True},
        labels={'Jenjang': 'Jenjang Keanggotaan', 'Persentase_DPW': 'Persentase DPW'}
    )
    fig_jenjang_percent.update_traces(
        texttemplate='%{y:.2f}%',
        textposition='outside',
        hovertemplate='<b>Jenjang</b>: %{x}<br>' +
                      'Jumlah DPW: %{customdata[0]}<br>' +
                      'DPW: %{customdata[1]}'
    )
    st.plotly_chart(fig_jenjang_percent, use_container_width=True)

    st.subheader('Persentase DPW berdasarkan Jenjang Keanggotaan yang Dikelola')
    df['Jenjang'] = df['Jenjang'].fillna('')
    jenjang_distribution = df['Jenjang'].value_counts(normalize=True).reset_index()
    jenjang_distribution.columns = ['Jenjang_Kombinasi', 'Persentase']
    jenjang_distribution['Persentase'] = jenjang_distribution['Persentase'] * 100
    dpws_in_each_combination = df.groupby('Jenjang')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    jenjang_distribution = pd.merge(jenjang_distribution, dpws_in_each_combination, left_on='Jenjang_Kombinasi', right_on='Jenjang', how='left')
    jenjang_distribution = jenjang_distribution.drop(columns=['Jenjang'], errors='ignore')
    jenjang_distribution['Jumlah_DPW'] = jenjang_distribution['DPW_List'].apply(lambda x: len(x.split(', ')) if x else 0)
    jenjang_distribution['CustomHoverData'] = jenjang_distribution.apply(
        lambda row: (row['Jumlah_DPW'], row['DPW_List']), axis=1
    )
    jenjang_distribution['Formatted_DPW_List'] = jenjang_distribution['DPW_List'].apply(format_dpw_list)

    fig_jenjang_pie = px.pie(
        jenjang_distribution,
        values='Persentase',
        names='Jenjang_Kombinasi',
        # title='Distribusi Persentase Kombinasi Jenjang Keanggotaan',
        hole=0.3,
        hover_data=['Jumlah_DPW', 'Formatted_DPW_List']
    )
    fig_jenjang_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>Kombinasi Jenjang</b>: %{label}<br>' +
                      'Jumlah DPW: %{customdata[0][0]}<br>' +
                      'DPW: %{customdata[0][1]}'
    )
    st.plotly_chart(fig_jenjang_pie, use_container_width=True)

    st.subheader('Klaster BKAP DPW Berdasarkan Jumlah Personil')
    personil_per_dpw = df.groupby('DPW')['Personil'].sum().reset_index()
    X = personil_per_dpw[['Personil']].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(X)
    personil_per_dpw['Cluster'] = kmeans.labels_
    cluster_summary = personil_per_dpw.groupby('Cluster')['Personil'].agg(['mean', 'min', 'max', 'count']).reset_index()
    cluster_summary = cluster_summary.sort_values(by='mean').reset_index(drop=True)
    cluster_labels = {
        cluster_summary.loc[0, 'Cluster']: 'Personil Ramping',
        cluster_summary.loc[1, 'Cluster']: 'Personil Sedang',
        cluster_summary.loc[2, 'Cluster']: 'Personil Gemuk'
    }
    cluster_color_map = {
        'Personil Ramping': '#636efa',
        'Personil Sedang': '#00cc96',
        'Personil Gemuk': '#EF553B'
    }
    personil_per_dpw['Cluster_Label'] = personil_per_dpw['Cluster'].map(cluster_labels)
    total_dpw_count = personil_per_dpw['DPW'].nunique()
    cluster_counts = personil_per_dpw.groupby('Cluster_Label').agg(
        Jumlah_DPW=('DPW', 'nunique')
    ).reset_index()
    cluster_counts['Persentase'] = (cluster_counts['Jumlah_DPW'] / total_dpw_count) * 100
    dpw_list_per_cluster = personil_per_dpw.groupby('Cluster_Label')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    cluster_counts = pd.merge(cluster_counts, dpw_list_per_cluster, on='Cluster_Label')
    cluster_counts['Formatted_DPW_List'] = cluster_counts['DPW_List'].apply(format_dpw_list)

    fig_personil_cluster = make_subplots(rows=2, cols=1, specs=[[{'type':'bar'}], [{'type':'domain'}]],
                        subplot_titles=('Jumlah Personil BKAP DPW per Klaster', 'Distribusi DPW per Klaster'))

    personil_per_dpw_sorted = personil_per_dpw.sort_values(by=['Cluster_Label', 'Personil'], ascending=[True, True])
    bar_chart_personil = go.Bar(
        x=personil_per_dpw_sorted['DPW'],
        y=personil_per_dpw_sorted['Personil'],
        marker_color=personil_per_dpw_sorted['Cluster_Label'].map(cluster_color_map),
        name='Jumlah Personil per DPW',
        showlegend=False,
        hovertemplate=
            "<b>DPW</b>: %{x}<br>" +
            "<b>Jumlah Personil</b>: %{y}<br>" +
            "<b>Cluster</b>: %{customdata}",
        customdata=personil_per_dpw_sorted['Cluster_Label']
    )
    fig_personil_cluster.add_trace(bar_chart_personil, row=1, col=1)

    pie_chart_personil = go.Pie(
        labels=cluster_counts['Cluster_Label'],
        values=cluster_counts['Persentase'],
        marker=dict(colors=cluster_counts['Cluster_Label'].map(cluster_color_map)),
        name='Distribusi DPW per Cluster',
        pull=[0.05 if label == 'Personil Gemuk' else 0 for label in cluster_counts['Cluster_Label']],
        hovertemplate=
            "<b>Cluster</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=cluster_counts[['Jumlah_DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    )
    fig_personil_cluster.add_trace(pie_chart_personil, row=2, col=1)

    fig_personil_cluster.update_layout(
#        title_text='Analisis Clustering Jumlah Personil BKAP DPW',
        height=1200,
        showlegend=True
    )
    fig_personil_cluster.update_xaxes(title_text="DPW", row=1, col=1)
    fig_personil_cluster.update_yaxes(title_text="Jumlah Personil", row=1, col=1)
    st.plotly_chart(fig_personil_cluster, use_container_width=True)

    # --- Biro Clustering ---
    st.subheader('Klaster BKAP DPW Berdasarkan Jumlah Biro yang Dibentuk')
    df['Biro'] = df['Biro'].fillna('')
    biro_exploded = df.assign(Biro=df['Biro'].str.split(', ')).explode('Biro')
    biro_exploded['Biro'] = biro_exploded['Biro'].str.strip()
    biro_exploded = biro_exploded[biro_exploded['Biro'] != '']
    biro_counts_per_dpw = biro_exploded.groupby('DPW')['Biro'].nunique().reset_index(name='Jumlah_Biro')

    # Perform KMeans clustering for Biro
    X_biro = biro_counts_per_dpw[['Jumlah_Biro']].values.reshape(-1, 1)
    kmeans_biro = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans_biro.fit(X_biro)
    biro_counts_per_dpw['Biro_Cluster'] = kmeans_biro.labels_

    biro_cluster_summary = biro_counts_per_dpw.groupby('Biro_Cluster')['Jumlah_Biro'].agg(['mean', 'min', 'max', 'count']).reset_index()
    biro_cluster_summary = biro_cluster_summary.sort_values(by='mean').reset_index(drop=True)
    biro_cluster_labels = {
        biro_cluster_summary.loc[0, 'Biro_Cluster']: 'Biro Ramping',
        biro_cluster_summary.loc[1, 'Biro_Cluster']: 'Biro Sedang',
        biro_cluster_summary.loc[2, 'Biro_Cluster']: 'Biro Gemuk'
    }
    biro_cluster_color_map = {
        'Biro Ramping': '#636efa',
        'Biro Sedang': '#00cc96',
        'Biro Gemuk': '#EF553B'
    }
    biro_counts_per_dpw['Biro_Cluster_Label'] = biro_counts_per_dpw['Biro_Cluster'].map(biro_cluster_labels)

    total_dpw_count_biro = biro_counts_per_dpw['DPW'].nunique()
    biro_cluster_counts = biro_counts_per_dpw.groupby('Biro_Cluster_Label').agg(
        Jumlah_DPW=('DPW', 'nunique')
    ).reset_index()
    biro_cluster_counts['Persentase'] = (biro_cluster_counts['Jumlah_DPW'] / total_dpw_count_biro) * 100
    biro_dpw_list_per_cluster = biro_counts_per_dpw.groupby('Biro_Cluster_Label')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    biro_cluster_counts = pd.merge(biro_cluster_counts, biro_dpw_list_per_cluster, on='Biro_Cluster_Label')
    biro_cluster_counts['Formatted_DPW_List'] = biro_cluster_counts['DPW_List'].apply(format_dpw_list)

    fig_biro_cluster = make_subplots(rows=2, cols=1, specs=[[{'type':'bar'}], [{'type':'domain'}]],
                        subplot_titles=('Jumlah Biro BKAP DPW per Klaster', 'Distribusi DPW per Klaster'))

    biro_counts_per_dpw_sorted = biro_counts_per_dpw.sort_values(by=['Biro_Cluster_Label', 'Jumlah_Biro'], ascending=[True, True])
    bar_chart_biro = go.Bar(
        x=biro_counts_per_dpw_sorted['DPW'],
        y=biro_counts_per_dpw_sorted['Jumlah_Biro'],
        marker_color=biro_counts_per_dpw_sorted['Biro_Cluster_Label'].map(biro_cluster_color_map),
        name='Jumlah Biro per DPW',
        showlegend=False,
        hovertemplate=
            "<b>DPW</b>: %{x}<br>" +
            "<b>Jumlah Biro</b>: %{y}<br>" +
            "<b>Cluster</b>: %{customdata}",
        customdata=biro_counts_per_dpw_sorted['Biro_Cluster_Label']
    )
    fig_biro_cluster.add_trace(bar_chart_biro, row=1, col=1)

    pie_chart_biro = go.Pie(
        labels=biro_cluster_counts['Biro_Cluster_Label'],
        values=biro_cluster_counts['Persentase'],
        marker=dict(colors=biro_cluster_counts['Biro_Cluster_Label'].map(biro_cluster_color_map)),
        name='Distribusi DPW per Cluster',
        pull=[0.05 if label == 'Biro Gemuk' else 0 for label in biro_cluster_counts['Biro_Cluster_Label']],
        hovertemplate=
            "<b>Cluster</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=biro_cluster_counts[['Jumlah_DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    )
    fig_biro_cluster.add_trace(pie_chart_biro, row=2, col=1)

    fig_biro_cluster.update_layout(
#        title_text='Analisis Clustering Jumlah Biro yang Dibentuk BKAP DPW',
        height=1200,
        showlegend=True
    )
    fig_biro_cluster.update_xaxes(title_text="DPW", row=1, col=1)
    fig_biro_cluster.update_yaxes(title_text="Jumlah Biro", row=1, col=1)
    st.plotly_chart(fig_biro_cluster, use_container_width=True)

    # --- Rekapitulasi Program Rekrutmen ---
    st.subheader('Rekapitulasi Program Rekrutmen BKAP DPW')
    df['Rekrutmen'] = df['Rekrutmen'].fillna('')
    rekrutmen_exploded = df.assign(Rekrutmen=df['Rekrutmen'].str.split(', ')).explode('Rekrutmen')
    rekrutmen_exploded['Rekrutmen'] = rekrutmen_exploded['Rekrutmen'].str.strip()
    rekrutmen_exploded = rekrutmen_exploded[rekrutmen_exploded['Rekrutmen'] != '']

    rekrutmen_counts = rekrutmen_exploded['Rekrutmen'].value_counts().reset_index()
    rekrutmen_counts.columns = ['Program Rekrutmen', 'Jumlah DPW']

    dpw_list_by_rekrutmen = rekrutmen_exploded.groupby('Rekrutmen')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    rekrutmen_counts = pd.merge(rekrutmen_counts, dpw_list_by_rekrutmen, left_on='Program Rekrutmen', right_on='Rekrutmen', how='left')
    rekrutmen_counts = rekrutmen_counts.drop(columns=['Rekrutmen'], errors='ignore')
    
    # Add formatted DPW list for display
    rekrutmen_counts['Formatted_DPW_List'] = rekrutmen_counts['DPW_List'].apply(format_dpw_list)
    st.dataframe(rekrutmen_counts.drop(columns=['DPW_List']), use_container_width=True)


    # --- Bagian 2: UPA Pelopor ---
    st.header('Bagian 2: UPA Pelopor')

    upa_color_map = {
        'Lengkap 100%': '#EF5000',
        'Lebih dari 80%': '#2ca02c',
        'Lebih dari 60%': '#1f77b4',
        'Lebih dari 40%': '#9467bd',
        'Kurang dari 40%': '#d62728',
        'Data tidak tersedia': '#7f7f7f'
    }

    # UPA Utama
    st.subheader('Terlaksananya UPA Utama dan Kehadiran Pembimbing')
    upa_utama_counts = df['UPAU'].value_counts().reset_index()
    upa_utama_counts.columns = ['Status UPA Utama', 'Jumlah DPW']
    upa_utama_counts['Persentase'] = (upa_utama_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_upau = df.groupby('UPAU')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    upa_utama_counts = pd.merge(upa_utama_counts, dpw_list_by_upau, left_on='Status UPA Utama', right_on='UPAU', how='left')
    upa_utama_counts = upa_utama_counts.drop(columns=['UPAU'], errors='ignore')
    upa_utama_counts['Formatted_DPW_List'] = upa_utama_counts['DPW_List'].apply(format_dpw_list)

    pembimbing_au_counts = df['PembimbingAU'].value_counts().reset_index()
    pembimbing_au_counts.columns = ['Kehadiran Pembimbing UPA Utama', 'Jumlah DPW']
    pembimbing_au_counts['Persentase'] = (pembimbing_au_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_pembimbingau = df.groupby('PembimbingAU')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    pembimbing_au_counts = pd.merge(pembimbing_au_counts, dpw_list_by_pembimbingau, left_on='Kehadiran Pembimbing UPA Utama', right_on='PembimbingAU', how='left')
    pembimbing_au_counts = pembimbing_au_counts.drop(columns=['PembimbingAU'], errors='ignore')
    pembimbing_au_counts['Formatted_DPW_List'] = pembimbing_au_counts['DPW_List'].apply(format_dpw_list)

    upa_utama_colors = [upa_color_map.get(label, '#9467bd') for label in upa_utama_counts['Status UPA Utama']]
    pembimbing_au_colors = [upa_color_map.get(label, '#9467bd') for label in pembimbing_au_counts['Kehadiran Pembimbing UPA Utama']]

    fig_upa_utama = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                            subplot_titles=('Persentase DPW Berdasarkan Terlaksananya UPA Utama',
                                            'Persentase DPW Berdasarkan Kehadiran Pembimbing UPA Utama'))

    fig_upa_utama.add_trace(go.Pie(
        labels=upa_utama_counts['Status UPA Utama'],
        values=upa_utama_counts['Persentase'],
        name='UPA Utama',
        hole=0.3,
        marker=dict(colors=upa_utama_colors),
        hovertemplate=
            "<b>Status UPA Utama</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=upa_utama_counts[['Jumlah DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    ), 1, 1)

    fig_upa_utama.add_trace(go.Pie(
        labels=pembimbing_au_counts['Kehadiran Pembimbing UPA Utama'],
        values=pembimbing_au_counts['Persentase'],
        name='Pembimbing UPA Utama',
        hole=0.3,
        marker=dict(colors=pembimbing_au_colors),
        hovertemplate=
            "<b>Kehadiran Pembimbing UPA Utama</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=pembimbing_au_counts[['Jumlah DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    ), 1, 2)

    fig_upa_utama.update_layout(
    #title_text='Analisis Terlaksananya UPA Utama dan Kehadiran Pembimbing', 
    height=600)
    st.plotly_chart(fig_upa_utama, use_container_width=True)

    # UPA Dewasa
    st.subheader('Terlaksananya UPA Dewasa dan Kehadiran Pembimbing')
    upa_dewasa_counts = df['UPAD'].value_counts().reset_index()
    upa_dewasa_counts.columns = ['Status UPA Dewasa', 'Jumlah DPW']
    upa_dewasa_counts['Persentase'] = (upa_dewasa_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_upad = df.groupby('UPAD')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    upa_dewasa_counts = pd.merge(upa_dewasa_counts, dpw_list_by_upad, left_on='Status UPA Dewasa', right_on='UPAD', how='left')
    upa_dewasa_counts = upa_dewasa_counts.drop(columns=['UPAD'], errors='ignore')
    upa_dewasa_counts['Formatted_DPW_List'] = upa_dewasa_counts['DPW_List'].apply(format_dpw_list)

    pembimbing_ad_counts = df['PembimbingAD'].value_counts().reset_index()
    pembimbing_ad_counts.columns = ['Kehadiran Pembimbing UPA Dewasa', 'Jumlah DPW']
    pembimbing_ad_counts['Persentase'] = (pembimbing_ad_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_pembimbingad = df.groupby('PembimbingAD')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    pembimbing_ad_counts = pd.merge(pembimbing_ad_counts, dpw_list_by_pembimbingad, left_on='Kehadiran Pembimbing UPA Dewasa', right_on='PembimbingAD', how='left')
    pembimbing_ad_counts = pembimbing_ad_counts.drop(columns=['PembimbingAD'], errors='ignore')
    pembimbing_ad_counts['Formatted_DPW_List'] = pembimbing_ad_counts['DPW_List'].apply(format_dpw_list)

    upa_dewasa_colors = [upa_color_map.get(label, '#9467bd') for label in upa_dewasa_counts['Status UPA Dewasa']]
    pembimbing_ad_colors = [upa_color_map.get(label, '#9467bd') for label in pembimbing_ad_counts['Kehadiran Pembimbing UPA Dewasa']]

    fig_upa_dewasa = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                            subplot_titles=('Persentase DPW Berdasarkan Terlaksananya UPA Dewasa',
                                            'Persentase DPW Berdasarkan Kehadiran Pembimbing UPA Dewasa'))

    fig_upa_dewasa.add_trace(go.Pie(
        labels=upa_dewasa_counts['Status UPA Dewasa'],
        values=upa_dewasa_counts['Persentase'],
        name='UPA Dewasa',
        hole=0.3,
        marker=dict(colors=upa_dewasa_colors),
        hovertemplate=
            "<b>Status UPA Dewasa</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=upa_dewasa_counts[['Jumlah DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    ), 1, 1)

    fig_upa_dewasa.add_trace(go.Pie(
        labels=pembimbing_ad_counts['Kehadiran Pembimbing UPA Dewasa'],
        values=pembimbing_ad_counts['Persentase'],
        name='Pembimbing UPA Dewasa',
        hole=0.3,
        marker=dict(colors=pembimbing_ad_colors),
        hovertemplate=
            "<b>Kehadiran Pembimbing UPA Dewasa</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=pembimbing_ad_counts[['Jumlah DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    ), 1, 2)

    fig_upa_dewasa.update_layout(
    #title_text='Analisis Terlaksananya UPA Dewasa dan Kehadiran Pembimbing', 
      height=600)
    st.plotly_chart(fig_upa_dewasa, use_container_width=True)

    # UPA Madya
    st.subheader('Terlaksananya UPA Madya dan Kehadiran Pembimbing')
    upa_madya_counts = df['UPAM'].value_counts().reset_index()
    upa_madya_counts.columns = ['Status UPA Madya', 'Jumlah DPW']
    upa_madya_counts['Persentase'] = (upa_madya_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_upam = df.groupby('UPAM')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    upa_madya_counts = pd.merge(upa_madya_counts, dpw_list_by_upam, left_on='Status UPA Madya', right_on='UPAM', how='left')
    upa_madya_counts = upa_madya_counts.drop(columns=['UPAM'], errors='ignore')
    upa_madya_counts['Formatted_DPW_List'] = upa_madya_counts['DPW_List'].apply(format_dpw_list)

    pembimbing_am_counts = df['PembimbingAM'].value_counts().reset_index()
    pembimbing_am_counts.columns = ['Kehadiran Pembimbing UPA Madya', 'Jumlah DPW']
    pembimbing_am_counts['Persentase'] = (pembimbing_am_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_pembimbingam = df.groupby('PembimbingAM')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    pembimbing_am_counts = pd.merge(pembimbing_am_counts, dpw_list_by_pembimbingam, left_on='Kehadiran Pembimbing UPA Madya', right_on='PembimbingAM', how='left')
    pembimbing_am_counts = pembimbing_am_counts.drop(columns=['PembimbingAM'], errors='ignore')
    pembimbing_am_counts['Formatted_DPW_List'] = pembimbing_am_counts['DPW_List'].apply(format_dpw_list)

    upa_madya_colors = [upa_color_map.get(label, '#9467bd') for label in upa_madya_counts['Status UPA Madya']]
    pembimbing_am_colors = [upa_color_map.get(label, '#9467bd') for label in pembimbing_am_counts['Kehadiran Pembimbing UPA Madya']]

    fig_upa_madya = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                            subplot_titles=('Persentase DPW Berdasarkan Terlaksananya UPA Madya',
                                            'Persentase DPW Berdasarkan Kehadiran Pembimbing UPA Madya'))

    fig_upa_madya.add_trace(go.Pie(
        labels=upa_madya_counts['Status UPA Madya'],
        values=upa_madya_counts['Persentase'],
        name='UPA Madya',
        hole=0.3,
        marker=dict(colors=upa_madya_colors),
        hovertemplate=
            "<b>Status UPA Madya</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=upa_madya_counts[['Jumlah DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    ), 1, 1)

    fig_upa_madya.add_trace(go.Pie(
        labels=pembimbing_am_counts['Kehadiran Pembimbing UPA Madya'],
        values=pembimbing_am_counts['Persentase'],
        name='Pembimbing UPA Madya',
        hole=0.3,
        marker=dict(colors=pembimbing_am_colors),
        hovertemplate=
            "<b>Kehadiran Pembimbing UPA Madya</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}",
        customdata=pembimbing_am_counts[['Jumlah DPW', 'Formatted_DPW_List']].values,
        textinfo='percent+label'
    ), 1, 2)

    fig_upa_madya.update_layout(
    #title_text='Analisis Terlaksananya UPA Madya dan Kehadiran Pembimbing', 
       height=600)
    st.plotly_chart(fig_upa_madya, use_container_width=True)


    # --- Bagian 3: Sapulidi ---
    st.header('Bagian 3: Sapulidi')

    st.subheader('Progress Pembaharuan Data Anggota dan UPA di Aplikasi Sapulidi')
    sapulidi_counts = df['Sapulidi'].value_counts().reset_index()
    sapulidi_counts.columns = ['Status Sapulidi', 'Jumlah DPW']
    sapulidi_counts['Persentase'] = (sapulidi_counts['Jumlah DPW'] / len(df)) * 100
    dpw_list_by_sapulidi = df.groupby('Sapulidi')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    sapulidi_counts = pd.merge(sapulidi_counts, dpw_list_by_sapulidi, left_on='Status Sapulidi', right_on='Sapulidi', how='left')
    sapulidi_counts = sapulidi_counts.drop(columns=['Sapulidi'], errors='ignore')
    sapulidi_counts['Formatted_DPW_List'] = sapulidi_counts['DPW_List'].apply(format_dpw_list)

    sapulidi_color_map = {
        'Sudah 100%': '#EF5000',
        'Lebih dari 75%': 'green',
        'Lebih dari 50%': 'blue',
        'Kurang dari 50%': 'red'
    }

    fig_sapulidi_pie = px.pie(
        sapulidi_counts,
        values='Persentase',
        names='Status Sapulidi',
#        title='Progress Pembaharuan Data Anggota dan UPA di Aplikasi Sapulidi',
        color='Status Sapulidi',
        color_discrete_map=sapulidi_color_map,
        hole=0.3,
        custom_data=['Jumlah DPW', 'Formatted_DPW_List']
    )

    fig_sapulidi_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate=
            "<b>Status Sapulidi</b>: %{label}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>Jumlah DPW</b>: %{customdata[0][0]}<br>" +
            "<b>DPW</b>: %{customdata[0][1]}"
    )
    st.plotly_chart(fig_sapulidi_pie, use_container_width=True)

    st.subheader('Penggunaan Fitur Sapulidi')
    df['Fitur'] = df['Fitur'].fillna('')
    fitur_exploded = df.assign(Fitur=df['Fitur'].str.split(', ')).explode('Fitur')
    fitur_exploded['Fitur'] = fitur_exploded['Fitur'].str.strip()
    fitur_exploded = fitur_exploded[fitur_exploded['Fitur'] != '']

    fitur_counts = fitur_exploded['Fitur'].value_counts().reset_index()
    fitur_counts.columns = ['Fitur', 'Jumlah DPW']

    dpw_list_by_feature = fitur_exploded.groupby('Fitur')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    fitur_counts = pd.merge(fitur_counts, dpw_list_by_feature, on='Fitur', how='left')
    fitur_counts['Formatted_DPW_List'] = fitur_counts['DPW_List'].apply(format_dpw_list)

    total_unique_dpw = df['DPW'].nunique()
    fitur_counts['Persentase DPW'] = (fitur_counts['Jumlah DPW'] / total_unique_dpw) * 100
    fitur_counts = fitur_counts.sort_values(by='Jumlah DPW', ascending=True)

    fig_bar_fitur = px.bar(
        fitur_counts,
        x='Jumlah DPW',
        y='Fitur',
        orientation='h',
 #       title='Penggunaan Fitur Sapulidi',
        labels={
            'Jumlah DPW': 'Jumlah DPW yang Menggunakan',
            'Fitur': 'Fitur Sapulidi'
        },
        height=600,
        text_auto=True,
        custom_data=[fitur_counts['Formatted_DPW_List'], fitur_counts['Persentase DPW']]
    )

    fig_bar_fitur.update_traces(
        hovertemplate=
            "<b>Fitur</b>: %{y}<br>" +
            "<b>Jumlah DPW</b>: %{x}<br>" +
            "<b>Persentase</b>: %{customdata[1]:.2f}%<br>" +
            "<b>DPW</b>: %{customdata[0]}<extra></extra>"
    )
    st.plotly_chart(fig_bar_fitur, use_container_width=True)

    # --- Comprehensive Clustering Visualization ---
    st.header('Analisis Klaster Komprehensif BKAP DPW')
    st.subheader('Sebaran Klaster BKAP DPW Berdasarkan Seluruh Aspek Data')

    # Display summary table
    st.write("### Ringkasan Karakteristik Klaster Komprehensif")
    st.dataframe(final_cluster_summary.round(2))
    
    # Bar chart for distribution of DPWs per comprehensive cluster
    comprehensive_cluster_counts = df_with_comprehensive_clusters['Comprehensive_Cluster_Name'].value_counts().reset_index()
    comprehensive_cluster_counts.columns = ['Comprehensive_Cluster_Name', 'Jumlah_DPW']
    comprehensive_cluster_counts['Persentase'] = (comprehensive_cluster_counts['Jumlah_DPW'] / len(df)) * 100
    dpw_list_per_comprehensive_cluster = df_with_comprehensive_clusters.groupby('Comprehensive_Cluster_Name')['DPW'].apply(lambda x: ', '.join(x.unique())).reset_index(name='DPW_List')
    comprehensive_cluster_counts = pd.merge(comprehensive_cluster_counts, dpw_list_per_comprehensive_cluster, on='Comprehensive_Cluster_Name')
    comprehensive_cluster_counts['Formatted_DPW_List'] = comprehensive_cluster_counts['DPW_List'].apply(format_dpw_list)

    comprehensive_color_map = {
        'DPW Sangat Besar & Sangat Aktif': '#EF5000',
        'DPW Besar & Aktif': 'green',
        'DPW Sedang & Cukup Aktif': 'blue',
        'DPW Kecil & Kurang Aktif': 'red'
    }

    fig_bar_comprehensive_clusters = px.bar(
        comprehensive_cluster_counts,
        x='Comprehensive_Cluster_Name',
        y='Jumlah_DPW',
        color='Comprehensive_Cluster_Name',
        color_discrete_map = comprehensive_color_map,
        title='Distribusi DPW per Klaster Komprehensif',
        labels={
            'Comprehensive_Cluster_Name': 'Nama Klaster',
            'Jumlah_DPW': 'Jumlah DPW'
        },
        height=600,
        text_auto=True,
        hover_data={'Formatted_DPW_List': True}
    )
    fig_bar_comprehensive_clusters.update_traces(
        hovertemplate=
            "<b>Klaster</b>: %{x}<br>" +
            "<b>Persentase DPW</b>: %{value:.2f}%<br>" +
            "<b>DPW</b>: %{customdata[0]}"
    )
    fig_bar_comprehensive_clusters.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar_comprehensive_clusters, use_container_width=True)

    
