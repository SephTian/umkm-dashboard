import pandas as pd  # pip install pandas openpyxl
import plotly.express as px  # pip install plotly-express
import streamlit as st  # pip install streamlit
from sklearn.neighbors import KNeighborsClassifier # pip install scikit-learn
from sklearn.model_selection import GridSearchCV


st.set_page_config(page_title="Dashboard UMKM", page_icon=":bar_chart:", layout="wide")

# ==== CLASSIFICATION ====
def klasifikasi (dataframe, provinsi, sektor, produksi, tahun):
    #Memasukkan data dan menampilkan data
    initial_df = dataframe

    #mengambil data yang diperlukan
    new_df = initial_df[["Provinsi","Sektor","Kemampuan_Produksi","Tahun_Mulai", "Pembiayaan_Bank"]]

    #Mengubah string menjadi angka
    prov_uniq = {'Jawa Barat':1, 'DKI Jakarta':2}
    sek_uniq = {'Barang Kayu dan Hasil Hutan Lainnya':1,
    'Tekstil Barang Kulit dan Alas Kaki':2, 'Makanan dan Minuman':3,
    'Barang Lainnya':4, 'Alat Angkutan, Mesin dan Lainnya':5,
    'Semen dan Barang Galian':6, 'Pupuk, Kimia dan Barang dari Karet':7,
    'Logam Dasar Besi dan Baja':8}

    datmin_df = new_df.copy()
    datmin_df["Provinsi"] = datmin_df["Provinsi"].replace(prov_uniq)
    datmin_df["Sektor"] = datmin_df["Sektor"].replace(sek_uniq)

    #Set data for train
    Features = ["Provinsi","Sektor","Kemampuan_Produksi","Tahun_Mulai"]
    x = datmin_df[Features].copy()
    y = datmin_df['Pembiayaan_Bank']
    parameter = {
       "n_neighbors": [2,3,4,5,6,7,8,9,10],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    #Optimisasi Parameter
    model = GridSearchCV(KNeighborsClassifier(),parameter, cv=3, n_jobs=-1, verbose=1)
    model.fit(x,y)
    n_neighbors = model.best_params_["n_neighbors"]
    weights = model.best_params_["weights"]
    p = model.best_params_["p"]

    #Pelatihan data
    knn = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,p=p)
    knn.fit(x,y)

    #Replace inputan provinsi dan sektor agar valuenya menjadi angka
    for key, value in prov_uniq.items():
        provinsi = provinsi.replace(key, str(value))
    for key, value in sek_uniq.items():
        sektor = sektor.replace(key, str(value))

    #Mencoba buat DF baru
    data_input = {"Provinsi": provinsi,
        "Sektor": sektor, 
        "Kemampuan_Produksi":produksi,
        "Tahun_Mulai":tahun}
    
    predict_data = pd.DataFrame(data_input, index=[0])

    #Memprediksi Data
    predict_data["Prediction"] = knn.predict(predict_data)
    if predict_data["Prediction"].values == "Tidak Ada":
        return "Kemungkinan untuk mendapatkan pembiayaan bank adalah ' TIDAK ADA '"    
    elif predict_data["Prediction"].values == "Ada":
        return "Kemungkinan untuk mendapatkan pembiayaan bank adalah ' ADA '"

# ==== TAKE EXCEL DATA ====
@st.cache_data
def get_data_from_excel():
    df = pd.read_excel(
        io="Data_After_Prepro_rev.xlsx",
        engine="openpyxl",
    )
    return df
umkm_df = get_data_from_excel()

# ==== MAINPAGE ====
st.title(":bar_chart: Dashboard UMKM")
st.write("Pada Bagian ini anda dapat melihat grafik berdasarkan data dalam database UMKM")

#Inisiasi row pertama
left_column, middle_column, right_column = st.columns(3)

# -- total perusahaan --
total_perusahaan = int(umkm_df["Nama_Perusahaan"].count())
with left_column:
    st.subheader("Total Perusahaan :")
    st.subheader(f"{total_perusahaan} UMKM")

# -- total perusahaan --
df_sektor = umkm_df.groupby(['Sektor']).count()
total_sektor = int(df_sektor["Nama_Perusahaan"].count())
with middle_column:
    st.subheader("Total Sektor :")
    st.subheader(f"{total_sektor} sektor")

df_provinsi = umkm_df.groupby(['Provinsi']).count()
total_provinsi = int(df_provinsi["Nama_Perusahaan"].count())    
with right_column:
    st.subheader("Total Provinsi :")
    st.subheader(f"{total_provinsi} provinsi")


left_column, middle_column, right_column = st.columns(3)
# -- Bar Chart jml umk per provinsi --
jml_perusahaan_provinsi = umkm_df.groupby(['Provinsi']).count()
fig_perusahaan_provinsi = px.bar(
    jml_perusahaan_provinsi,
    x=jml_perusahaan_provinsi.index,
    y="Nama_Perusahaan",
    orientation="v",
    title="<b>Jumlah UMKM per Provinsi</b>",
    template="plotly_white",
)
left_column.plotly_chart(fig_perusahaan_provinsi, use_container_width=True)

# -- Bar Chart jml umkm per sektor --
jml_perusahaan_sektor = umkm_df.groupby(['Sektor']).count().sort_values(by=["Nama_Perusahaan"])
fig_perusahaan_sektor = px.bar(
    jml_perusahaan_sektor,
    x="Nama_Perusahaan",
    y=jml_perusahaan_sektor.index,
    orientation="h",
    title="<b>Jumlah UMKM per Sektor</b>",
    template="plotly_white",
)
middle_column.plotly_chart(fig_perusahaan_sektor, use_container_width=True)

# -- Line Chart kenaikan umkm per tahun --
kenaikan_perusahaan = umkm_df.groupby(['Tahun_Mulai']).count().cumsum()
fig_kenaikan = px.line(
    kenaikan_perusahaan["Nama_Perusahaan"],
    x=kenaikan_perusahaan.index,
    y="Nama_Perusahaan",
    orientation="h",
    title="<b>Jumlah UMKM per tahun</b>",
)
right_column.plotly_chart(fig_kenaikan, use_container_width=True)


st.markdown("##")

# -- Bagian Klasifikasi --
st.title(":bar_chart: Prediksi Pembiayaan")
st.write("Pada bagian ini anda dapat memprediksi pembiayaan bank sesuai data UMKM anda")
st.write("============================================================================")

# -> memasukkan data yang diperlukan <-
provinsi_data = st.selectbox("Pilih Provinsi :", 
                             umkm_df["Provinsi"].unique(), 
                             label_visibility="visible")
sektor_data = st.selectbox("Pilih Sektor :", 
                             umkm_df["Sektor"].unique(), 
                             label_visibility="visible")
produksi_data = st.number_input("Masukkan Jumlah Kemampuan Produksi :", 
                min_value=int(0), 
                max_value=int(10000), 
                value=int(0),
                label_visibility="visible")
tahun_data = st.number_input("Masukkan Tahun Mulai :", 
                min_value=int(1980), 
                max_value=int(2023), 
                value=int(1980),
                label_visibility="visible")

hasil_prediksi = st.button("Mulai Prediksi")

# -> menampilkan data <-
if hasil_prediksi:
    prediksi = klasifikasi(umkm_df, provinsi_data,sektor_data,produksi_data,tahun_data)
    st.write(prediksi)

st.markdown("##")
st.markdown("##")

# -- Bagian List Data --
st.title(":bar_chart: List Data")
st.write("List dari database UMKM")
umkm_df["Tahun_Mulai"] = umkm_df["Tahun_Mulai"].astype(str)
st.dataframe(umkm_df)
