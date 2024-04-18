import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import altair as alt

st.write("""
SILVA JEN RETNO - 2024
""")

# Sidebar untuk navigasi
with st.sidebar:
    selected = option_menu('Menu',
                           ['Home',
                            'Data Visualization',
                            'Clustering'],

                            icons = ['house-fill', 
                                     'database-fill',
                                     'stars'],
                            default_index = 0)


### Home

if selected == 'Home':

    #page tittle
    # Menampilkan gambar dari file lokal
    from PIL import Image
    image = Image.open('dash.jpg')
    st.image(image, caption='')

    st.subheader('MENGANALISIS KEBERHASILAN AKADEMIK: STUDI KOMPREHENSIF FAKTOR KINERJA SISWA')
    
    st.markdown("""
            Dataset yang berjudul "Menganalisis Keberhasilan Akademik Studi Komprehensif: Faktor Kinerja Siswa" merupakan
            data yang diambil dari platform Kaggle. Dataset ini mencakup berbagai aspek yang terkait dengan kinerja siswa di sekolah. 
            Kumpulan data ini berfungsi sebagai sumber daya untuk mengeksplorasi berbagai dinamika yang memengaruhi hasil akademik siswa di sekolah. 
            Informasi ini, dapat digunakan untuk memahami lebih lanjut tentang atribut-atribut yang tersedia dalam dataset 
            dan potensi hubungannya dengan keberhasilan akademik siswa.
            Berikut link dari dataset : https://www.kaggle.com/datasets/jacksondivakarr/student-classification-dataset/data
            """)
    
    st.write('##### DATASET AWAL')
    # Read data
    data = pd.read_csv('student.csv')
    st.write (data)
    st.markdown('Dataset ini akan melewati serangkaian langkah untuk membersihkan, menyiapkan, dan memvalidasi data agar akurat, relevan, dan representatif. Tujuannya adalah untuk mengurangi kesalahan dan bias dalam proses analisis.')

    st.write('##### DATASET AKHIR')
    df = pd.read_csv('data_cleaned.csv')
    st.write (df)
    st.markdown('Dataset ini telah dimodifikasi untuk disesuaikan dengan keperluan analisis. Proses ini melibatkan transformasi variabel dalam dataset ke format yang dapat diolah oleh algoritma yang akan digunakan, contohnya seperti mengubah variabel kategorikal menjadi numerik.')

    ### Visualisasi 1 : Hubungan Nilai Dengan Usia Siswa
    st.write('##### Hubungan Antara Nilai Dengan Usia Siswa')
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Student_Age', y='Grade', marker='o', color='blue')
    plt.xlabel('Usia Siswa')
    plt.ylabel('Nilai')
    plt.grid(True)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.markdown('''
                *Interpretasi:*
                Berdasarkan grafik, terlihat bahwa nilai siswa mengalami tren penurunan seiring dengan bertambahnya usia. 
                Hal ini terlihat dari garis yang menurun dari kanan ke kiri. Penurunan ini cukup signifikan pada usia 20 tahun ke atas.

                *Insight:*
                Siswa yang lebih tua mengalami penurunan nilai yang signifikan daripada siswa yang lebih muda. Grafik garis usia yang memiliki nilai tertinggi adalah usia 18 tahun menunjukkan bahwa sebagian besar siswa mencapai tingkat kesuksesan akademik yang relatif tinggi di usia muda.

                *Actionable Insight:*
                Memberikan dukungan dan motivasi untuk belajar dapat diarahkan kepada siswa dengan rentang usia 20 tahun ke atas untuk membantu meningkatkan fokus dan motivasi belajar.
                ''')


    ### Visualisasi 2 : Perbandingan Nilai Siswa Yang Memiliki dan Tidak Memiliki Pekerjaan Tambahan 
    st.write('##### Perbandingan Nilai Siswa Yang Memiliki dan Tidak Memiliki Pekerjaan Tambahan')
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Additional_Work', y='Grade', palette='Set3')
    plt.xlabel('Pekerjaan Tambahan')
    plt.ylabel('Nilai Siswa')
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.markdown('''
                *Interpretasi:*
                Grafik diatas menunjukkan rata-rata nilai siswa berdasarkan status pekerjaan mereka. Dapat dilihat bahwa siswa yang tidak memiliki pekerjaan tambahan memiliki nilai rata-rata yang lebih tinggi daripada yang memiliki pekerjaan tambahan

                *Insight:*
                Perbedaan antara nilai rata-rata antara siswa yang memiliki pekerjaan tambahan dan yang tidak memiliki pekerjaan tambahan terlihat kecil, tetapi penting untuk memahami apa saja faktor-faktor yang bisa mempengaruhi kinerja akademik siswa.

                *Actionable Insight:*
                Perlunya kesadaran akan pentingnya pendidikan dalam jangka panjang, siswa jjuga perlu mengatur waktu dengan baik antara bekerja dan belajar agar tidak saling mengganggu.
                ''')


### DATA VISUALIZATION

if selected == 'Data Visualization':

    ### DATA DISTRIBUTION

    st.write('##### Data Distribution')
    st.markdown('''
                Visualisasi ini menampilkan distribusi umur siswa dalam bentuk histogram. 
                Histogram adalah grafik yang menunjukkan bagaimana data terdistribusi di sepanjang sumbu horizontal, 
                dengan frekuensi kemunculan nilai-nilai umur ditampilkan di sumbu vertikal. 
                Dengan melihat histogram, kita dapat memahami pola umum dari distribusi umur siswa, 
                seperti apakah distribusinya condong ke kelompok umur tertentu atau merata di sepanjang rentang umur. 
                Histogram ini membantu kita untuk mendapatkan gambaran visual yang jelas tentang karakteristik umur siswa dalam dataset yang kita analisis.
                Siswa yang memiliki usia 18 tahun memiliki frekuensi yang paling tinggi.
                ''')
    
    df = pd.read_csv('data_cleaned.csv')
    df1 = pd.DataFrame(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df1['Student_Age'], bins=10, color='blue', alpha=0.7)
    ax.set_title('Histogram Student Age')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)


    ### DATA RELATIONSHIP

    st.write('##### Data Relationship')
    st.markdown('''
                 Visualisasi ini menunjukkan hubungan antar variabel angka dalam data. 
                Hubungan ini digambarkan dengan matriks korelasi yang membantu memahami seberapa kuat hubungan antar variabel. 
                Nilai pada matriks berkisar antara -1 dan 1. Nilai 1 menunjukkan hubungan yang sangat kuat dan positif, 
                -1 menunjukkan hubungan yang sangat kuat dan negatif, dan 0 menunjukkan tidak ada hubungan. 
                Dari visualisasi korelasi ini, terlihat bahwa setiap variabel memiliki nilai yang menunjukkan hubungannya dengan variabel lain.
                ''')
    df = pd.read_csv('data_cleaned.csv')
    df2 = pd.DataFrame(df)

    fig, ax = plt.subplots()
    sns.heatmap(df2.corr().round(2), annot=True, cmap='Blues', ax=ax)
    plt.show()

    st.pyplot(fig)


    ### DATA COMPOSITION

    st.write('##### Data Composition')
    st.markdown('''
                Visualisasi ini menampilkan rata-rata nilai fitur numerik berdasarkan kategori Nilai Siswa (Grade). 
                Setiap kotak sel menunjukkan rata-rata nilai fitur tersebut untuk setiap kategori Nilai Siswa (Grade). 
                Intensitas warna menunjukkan nilai, di mana warna yang lebih gelap menunjukkan nilai yang lebih tinggi dan warna yang lebih terang menunjukkan nilai yang lebih rendah. 
                Visualisasi ini berguna untuk melihat pola hubungan antara kategori Nilai Siswa (Grade) dengan berbagai fitur numerik dalam dataset.
                ''')
    df = pd.read_csv('data_cleaned.csv')
    df3 = pd.DataFrame(df)

    numeric_columns = df3.select_dtypes(include=['int64', 'float64']).columns
    class_composition = df3.groupby('Grade')[numeric_columns].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu')
    plt.title('Komposisi berdasarkan Nilai Siswa (Grade)')
    plt.xlabel('Nilai Siswa')
    plt.ylabel('Fitur')
    st.pyplot(plt)


    ### DATA COMPARISON

    st.write('##### DATA Comparison')
    st.markdown('''
                Grafik ini menunjukkan hubungan antara Nilai Siswa (Grade) dengan Jam Belajar Siswa. 
                Sumbu X menunjukkan jam belajar siswa, sedangkan sumbu Y menunjukkan nilai. 
                Garis pada grafik menunjukkan tren nilai siswa seiring dengan bertambahnya jam belajar.
                Berdasarkan grafik, terlihat bahwa nilai siswa mengalami tren peningkatan seiring dengan bertambahnya jam belajar. 
                Hal ini terlihat dari garis yang naik dari kiri ke kanan. 
                Peningkatan ini cukup signifikan pada jam belajar 4 jam ke atas.
                ''')
    df = pd.read_csv('data_cleaned.csv')

    fig, ax = plt.subplots()
    sns.lineplot(x="Weekly_Study_Hours",
                y="Grade",
                hue="Reading",
                palette="Set1",
                data=df,
                ax=ax)

    st.pyplot(fig)



### CLUSTERING

if selected == 'Clustering':
    st.subheader("Data Clustering")
    
    num_clusters = st.slider("Number of Clusters", 2, 10, 3)

    data = pd.read_csv('data_cleaned.csv')

    X = data[['Grade', 'Student_Age']]
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(X)

    data['Cluster'] = clusters

    fig = px.scatter(data, x='Grade', y='Student_Age', color='Cluster', 
                     title='Data Clustering: Usia Siswa dan Nilai Siswa')
    st.plotly_chart(fig)

    st.write("""Visualisasi ini menunjukkan hasil dari analisis data clustering berdasarkan variabel Nilai Siswa (Grade) dan Usia Siswa (Student_Age). 
             Tampaknya  ada tren penurunan nilai siswa seiring dengan bertambahnya usia.  
             Hal ini terlihat dari kumpulan titik-titik data yang umumnya menurun dari kanan ke kiri pada grafik.""")
