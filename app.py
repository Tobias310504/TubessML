from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Konfigurasi dasar aplikasi
st.set_page_config(
    page_title="Prediksi Harga Mobil Bekas",
    page_icon="₹",
    layout="wide",
)


# BASE_DIR dipakai agar path file tetap aman walaupun app dijalankan dari folder lain.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model_harga_mobil_ridge_log.pkl"
DATA_PATH = BASE_DIR / "data" / "Dataset_carDeckho.csv"

# Dataset yang digunakan adalah dataset mobil bekas India/CarDekho, sehingga
# nilai selling_price paling tepat ditampilkan sebagai Indian Rupee (INR).
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"
CURRENCY_NAME = "Indian Rupee"


# function untuk styling UI
def inject_custom_css():
    """Menambahkan CSS agar tampilan Streamlit lebih rapi.

    Streamlit secara default sudah bisa dipakai, tetapi tampilannya cukup polos.
    CSS ini membuat app terasa seperti dashboard: spacing lebih lega, card lebih
    jelas, dan hasil prediksi lebih mudah dilihat.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, #eef6f3 42%, #f8fafc 100%);
            color: #17202a;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
        }

        [data-testid="stSidebar"] {
            background: #12312d;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        .app-header {
            background: linear-gradient(135deg, #12312d 0%, #176b61 52%, #d97706 100%);
            border-radius: 8px;
            padding: 24px 28px;
            color: #ffffff;
            margin-bottom: 22px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
        }

        .app-header h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 8px 0;
            letter-spacing: 0;
        }

        .app-header p {
            margin: 0;
            color: #e8fff8;
            max-width: 780px;
            font-size: 1rem;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(255,255,255,0.32);
            background: rgba(255,255,255,0.12);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 0.82rem;
            color: #ffffff;
        }

        .quick-nav-label {
            color: #e8fff8;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 18px;
            margin-bottom: 6px;
        }

        .soft-panel {
            border: 1px solid #d8e7df;
            background: rgba(255,255,255,0.86);
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .model-summary-panel {
            border: 1px solid #c5ddd5;
            background: linear-gradient(180deg, #ffffff 0%, #f3fbf8 100%);
            border-radius: 8px;
            padding: 26px 28px;
            margin-bottom: 22px;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
        }

        .model-summary-panel h2 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 12px 0;
            color: #12312d;
            letter-spacing: 0;
        }

        .model-summary-panel p {
            font-size: 1.05rem;
            line-height: 1.65;
            color: #334155;
            max-width: 1050px;
            margin: 0;
        }

        .result-panel {
            border: 1px solid #b9dfd3;
            background: #ffffff;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
        }

        .result-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 8px;
        }

        .result-price {
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 800;
            color: #0f766e;
            margin: 0 0 8px 0;
        }

        .helper-text {
            color: #64748b;
            font-size: 0.92rem;
            margin: 0;
        }

        .section-kicker {
            color: #0f766e;
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe7e2;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] p {
            color: #64748b;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #12312d;
        }

        .model-metrics [data-testid="stMetric"] {
            padding: 22px 24px;
            min-height: 136px;
            border-color: #c5ddd5;
        }

        .model-metrics [data-testid="stMetricValue"] {
            font-size: 2.35rem;
        }

        .stButton > button {
            border-radius: 8px;
            background: #0f766e;
            color: white;
            border: 0;
            font-weight: 700;
        }

        .stButton > button:hover {
            background: #115e59;
            color: white;
            border: 0;
        }

        .quick-nav-row .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.34);
            background: rgba(255,255,255,0.13);
            color: #ffffff;
            box-shadow: none;
            min-height: 42px;
        }

        .quick-nav-row .stButton > button:hover {
            background: rgba(255,255,255,0.24);
            color: #ffffff;
            border: 1px solid rgba(255,255,255,0.52);
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title, subtitle, badges):
    """Membuat header halaman yang konsisten di semua page.

    Parameter badges tetap ada agar pemanggilan fungsi mudah dibaca, tetapi
    navigasi yang benar-benar bisa diklik dibuat oleh render_quick_nav().
    """
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def change_page(page_name):
    """Mengubah halaman aktif.

    Streamlit menjalankan ulang script setiap ada interaksi. Karena itu pilihan
    halaman disimpan di st.session_state agar klik tombol navigasi tetap diingat.
    """
    st.session_state["page"] = page_name


def render_quick_nav():
    """Membuat tombol navigasi cepat di bawah header.

    Tombol ini menjawab kebutuhan UI: badge/area seperti button di bawah banner
    sekarang benar-benar punya fungsi untuk pindah halaman.
    """
    st.markdown("<div class='quick-nav-label'>Quick navigation</div>", unsafe_allow_html=True)
    st.markdown("<div class='quick-nav-row'>", unsafe_allow_html=True)
    nav_col_1, nav_col_2, nav_col_3, nav_col_4 = st.columns(4)

    with nav_col_1:
        st.button("Prediksi", width="stretch", on_click=change_page, args=("Prediksi Harga",))
    with nav_col_2:
        st.button("EDA", width="stretch", on_click=change_page, args=("Dataset & EDA",))
    with nav_col_3:
        st.button("Model", width="stretch", on_click=change_page, args=("Penjelasan Model",))
    with nav_col_4:
        st.button("Data", width="stretch", on_click=change_page, args=("Tentang Dataset",))

    st.markdown("</div>", unsafe_allow_html=True)


inject_custom_css()



def format_price(value):
    """Mengubah angka harga menjadi string mata uang yang mudah dibaca.

    Contoh:
    350000 -> ₹350,000

    Kenapa dibuat fungsi?
    Supaya semua angka harga di aplikasi punya format yang konsisten.
    Jika nanti dataset diganti ke Rupiah, cukup ubah CURRENCY_SYMBOL menjadi "Rp".
    """
    return f"{CURRENCY_SYMBOL}{value:,.0f}"


# Load model dan dataset
@st.cache_resource
def load_model_package():
    """Load model dari file pickle.

    Kenapa pakai cache_resource?
    Streamlit menjalankan ulang script setiap ada interaksi user. Tanpa cache,
    file model akan dibaca berulang-ulang. Dengan cache, model cukup diload sekali.

    Catatan keamanan:
    File pickle hanya aman diload jika kita percaya sumber file-nya.
    Di sini file model berasal dari training kita sendiri.
    """
    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan: {MODEL_PATH}")
        st.stop()

    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_dataset():
    """Load dataset mentah untuk kebutuhan EDA dan pilihan input.

    Dataset tidak dipakai untuk melatih ulang model di web ini. Model sudah
    dilatih sebelumnya di notebook. Dataset dipakai agar app bisa:
    - menampilkan ringkasan data
    - membuat grafik EDA
    - mengambil daftar pilihan nama mobil dan kategori input
    """
    if not DATA_PATH.exists():
        st.error(f"File dataset tidak ditemukan: {DATA_PATH}")
        st.stop()

    return pd.read_csv(DATA_PATH)


model_package = load_model_package()
raw_df = load_dataset()

model = model_package["model"]
feature_columns = model_package["feature_columns"]
max_year = model_package["max_year"]


# Feature engineering
def add_features(data, reference_year=max_year):
    """Membuat fitur turunan agar input web sama dengan input model.

    Model di notebook tidak menerima langsung kolom year dan name.
    Dari kolom tersebut, kita membuat:
    - brand: kata pertama dari nama mobil
    - model_family: dua kata pertama dari nama mobil
    - car_age: umur mobil berdasarkan max_year dari dataset training
    - km_per_year: rata-rata kilometer per tahun

    Fungsi ini harus konsisten dengan fungsi feature engineering di notebook.
    Jika tidak konsisten, prediksi web bisa berbeda dengan prediksi notebook.
    """
    data = data.copy()
    data["brand"] = data["name"].astype(str).str.split().str[0]
    data["model_family"] = data["name"].astype(str).str.split().str[:2].str.join(" ")
    data["car_age"] = reference_year - data["year"]
    data["km_per_year"] = data["km_driven"] / (data["car_age"] + 1)
    return data


def clean_dataset_for_eda(data):
    """Membersihkan dataset untuk tampilan EDA.

    Proses ini dibuat mirip dengan notebook:
    - hapus duplikat
    - hapus Test Drive Car
    - pastikan harga positif dan kilometer tidak negatif
    - tambahkan fitur turunan
    """
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in ["year", "selling_price", "km_driven"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned[cleaned["owner"] != "Test Drive Car"].copy()
    cleaned = cleaned[cleaned["selling_price"] > 0].copy()
    cleaned = cleaned[cleaned["km_driven"] >= 0].copy()
    cleaned = cleaned.dropna().reset_index(drop=True)
    cleaned = add_features(cleaned)

    return cleaned


clean_df = clean_dataset_for_eda(raw_df)

PAGES = ["Prediksi Harga", "Dataset & EDA", "Penjelasan Model", "Tentang Dataset"]

if "page" not in st.session_state:
    st.session_state["page"] = "Prediksi Harga"


# untuk membuat sidebar sebagai navigasi
st.sidebar.title("Navigasi")
sidebar_page = st.sidebar.radio(
    "Pilih halaman",
    PAGES,
    index=PAGES.index(st.session_state["page"]),
)

if sidebar_page != st.session_state["page"]:
    st.session_state["page"] = sidebar_page

page = st.session_state["page"]

st.sidebar.divider()
st.sidebar.caption("Model: Ridge Regression + log target")
st.sidebar.caption("Preprocessing: OneHotEncoder + StandardScaler")
st.sidebar.caption(f"Mata uang: {CURRENCY_NAME} ({CURRENCY_CODE})")



# page 1: untuk memprediksi harga
if page == "Prediksi Harga":
    render_header(
        "Prediksi Harga Mobil Bekas",
        (
            "Masukkan spesifikasi mobil, lalu aplikasi akan memprediksi harga "
            "menggunakan model terbaik dari notebook."
        ),
        [
            "Ridge Regression",
            "Log target",
            f"{CURRENCY_CODE} / {CURRENCY_SYMBOL}",
            "CarDekho India",
        ],
    )
    render_quick_nav()

    top_metric_1, top_metric_2, top_metric_3, top_metric_4 = st.columns(4)
    top_metric_1.metric("R2 test", "0.798")
    top_metric_2.metric("MAE test", format_price(91_211))
    top_metric_3.metric("Data bersih", f"{len(clean_df):,} baris")
    top_metric_4.metric("Median harga", format_price(clean_df["selling_price"].median()))

    # Ambil pilihan input dari dataset agar user tidak perlu menebak kategori valid.
    car_names = sorted(clean_df["name"].dropna().unique())
    fuel_options = sorted(clean_df["fuel"].dropna().unique())
    seller_options = sorted(clean_df["seller_type"].dropna().unique())
    transmission_options = sorted(clean_df["transmission"].dropna().unique())
    owner_options = sorted(clean_df["owner"].dropna().unique())

    left_col, right_col = st.columns([1.2, 0.8], gap="large")

    with left_col:
        st.markdown("<div class='section-kicker'>Input</div>", unsafe_allow_html=True)
        st.subheader("Spesifikasi Mobil")

        # st.form membuat input tidak langsung memicu prediksi.
        # Prediksi baru dijalankan saat tombol submit ditekan.
        with st.form("prediction_form"):
            use_custom_name = st.checkbox(
                "Isi nama mobil manual",
                value=False,
                help=(
                    "Aktifkan jika nama mobil tidak ada di daftar. "
                    "Kategori baru tetap bisa diproses karena encoder memakai handle_unknown='ignore'."
                ),
            )

            if use_custom_name:
                car_name = st.text_input(
                    "Nama mobil",
                    value="Hyundai Creta 1.6 CRDi SX",
                    help="Minimal isi merek dan model agar fitur brand/model_family bisa dibuat.",
                )
            else:
                car_name = st.selectbox(
                    "Nama mobil",
                    car_names,
                    index=car_names.index("Hyundai Creta 1.6 CRDi SX")
                    if "Hyundai Creta 1.6 CRDi SX" in car_names
                    else 0,
                )

            year = st.number_input(
                "Tahun mobil",
                min_value=1990,
                max_value=2030,
                value=2018,
                step=1,
                help="Tahun akan diubah menjadi car_age berdasarkan tahun maksimum data training.",
            )

            km_driven = st.number_input(
                "Kilometer ditempuh",
                min_value=0,
                max_value=1_000_000,
                value=45_000,
                step=1_000,
            )

            input_col_1, input_col_2 = st.columns(2)

            with input_col_1:
                fuel = st.selectbox("Bahan bakar", fuel_options)
                transmission = st.selectbox("Transmisi", transmission_options)

            with input_col_2:
                seller_type = st.selectbox("Tipe penjual", seller_options)
                owner = st.selectbox("Kepemilikan", owner_options)

            submitted = st.form_submit_button("Prediksi Harga", width="stretch")

    with right_col:
        st.markdown("<div class='section-kicker'>Output</div>", unsafe_allow_html=True)
        st.subheader("Estimasi Harga")

        if submitted:
            # Bentuk DataFrame mentah sesuai input user.
            input_df = pd.DataFrame(
                {
                    "name": [car_name],
                    "year": [year],
                    "km_driven": [km_driven],
                    "fuel": [fuel],
                    "seller_type": [seller_type],
                    "transmission": [transmission],
                    "owner": [owner],
                }
            )

            # terapkan feature engineering yang sama seperti notebook
            processed_input = add_features(input_df)

            # ambil hanya kolom yang benar-benar digunakan saat training model
            model_input = processed_input[feature_columns]

            # prediksi harga
            prediction = model.predict(model_input)[0]
            prediction = max(prediction, 0)

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-label">Prediksi harga</div>
                    <div class="result-price">{format_price(prediction)}</div>
                    <p class="helper-text">
                        Harga ditampilkan dalam {CURRENCY_NAME} ({CURRENCY_CODE}),
                        mengikuti satuan harga pada dataset.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.write("Fitur yang dikirim ke model:")
            st.dataframe(model_input, width="stretch", hide_index=True)
        else:
            st.markdown(
                """
                <div class="soft-panel">
                    <div class="result-label">Menunggu input</div>
                    <p class="helper-text">
                        Isi form di sebelah kiri, lalu klik tombol Prediksi Harga.
                        Hasil estimasi akan tampil di area ini.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("<div class='section-kicker'>Pipeline</div>", unsafe_allow_html=True)
    st.subheader("Cara Kerja Singkat")
    st.markdown(
        """
        <div class="soft-panel">
            Input user tidak langsung dipakai mentah. Aplikasi membuat fitur
            <code>brand</code>, <code>model_family</code>, <code>car_age</code>,
            dan <code>km_per_year</code>, lalu model melakukan preprocessing
            kategori dan numerik secara otomatis melalui pipeline yang sudah disimpan.
        </div>
        """,
        unsafe_allow_html=True,
    )



# page 2: menampilkan dataset & EDA
elif page == "Dataset & EDA":
    render_header(
        "Dataset & EDA",
        (
            "Ringkasan data, distribusi harga, outlier, dan kategori utama yang "
            "dipakai untuk memahami karakter dataset sebelum modeling."
        ),
        ["Data understanding", "EDA", "Outlier check", "INR price"],
    )
    render_quick_nav()

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Baris awal", f"{len(raw_df):,}")
    metric_2.metric("Baris setelah cleaning", f"{len(clean_df):,}")
    metric_3.metric("Jumlah brand", f"{clean_df['brand'].nunique():,}")
    metric_4.metric("Median harga", format_price(clean_df["selling_price"].median()))

    st.subheader("Preview Dataset")
    st.dataframe(clean_df.head(30), width="stretch")

    st.markdown("<div class='section-kicker'>Target distribution</div>", unsafe_allow_html=True)
    st.subheader("Distribusi Harga")
    plot_col_1, plot_col_2 = st.columns(2, gap="large")

    with plot_col_1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(clean_df["selling_price"], bins=40, color="#2563eb", edgecolor="white")
        ax.set_title("Selling Price")
        ax.set_xlabel("Harga")
        ax.set_ylabel("Jumlah data")
        st.pyplot(fig)

    with plot_col_2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.log1p(clean_df["selling_price"]), bins=40, color="#16a34a", edgecolor="white")
        ax.set_title("log1p(Selling Price)")
        ax.set_xlabel("Log harga")
        ax.set_ylabel("Jumlah data")
        st.pyplot(fig)

    st.markdown(
        """
        <div class="soft-panel">
            Histogram kiri terlihat right-skewed: mayoritas mobil berada pada harga
            rendah-menengah, sedangkan sedikit mobil punya harga sangat tinggi.
            Karena itu notebook memakai transformasi <code>log1p</code> agar model lebih stabil.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-kicker'>Outlier</div>", unsafe_allow_html=True)
    st.subheader("Ringkasan Outlier Harga")
    price_p95 = clean_df["selling_price"].quantile(0.95)
    price_p99 = clean_df["selling_price"].quantile(0.99)
    price_max = clean_df["selling_price"].max()

    outlier_col_1, outlier_col_2, outlier_col_3 = st.columns(3)
    outlier_col_1.metric("Harga p95", format_price(price_p95))
    outlier_col_2.metric("Harga p99", format_price(price_p99))
    outlier_col_3.metric("Harga maksimum", format_price(price_max))

    st.markdown(
        """
        <div class="soft-panel">
            p95 berarti sekitar 95% data memiliki harga di bawah angka tersebut.
            Jika harga maksimum jauh lebih besar dari p95/p99, berarti terdapat harga ekstrem.
            Pada konteks harga mobil, nilai ekstrem belum tentu salah karena bisa merepresentasikan mobil mahal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-kicker'>Categories</div>", unsafe_allow_html=True)
    st.subheader("Distribusi Kategori")
    category_column = st.selectbox(
        "Pilih kolom kategori",
        ["fuel", "seller_type", "transmission", "owner", "brand"],
    )

    category_counts = clean_df[category_column].value_counts().head(15)
    st.bar_chart(category_counts)

    st.subheader("Statistik Harga")
    st.dataframe(
        clean_df["selling_price"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        .to_frame("selling_price"),
        width="stretch",
    )



# page 3: berisi penjelasan Model
elif page == "Penjelasan Model":
    render_header(
        "Penjelasan Model",
        (
            "Alasan pemilihan model, preprocessing, transformasi target, dan cara "
            "membaca hasil prediksi."
        ),
        ["Ridge Regression", "OneHotEncoder", "StandardScaler", "log1p target"],
    )
    render_quick_nav()

    st.markdown("<div class='section-kicker'>Model summary</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="model-summary-panel">
            <h2>Model yang Digunakan</h2>
            <p>
                Model utama adalah <strong>Ridge Regression</strong> dengan target harga
                yang ditransformasi menggunakan <code>log1p</code>. Model ini dipilih
                karena performanya stabil, cocok untuk data tabular, dan masih mudah
                dijelaskan melalui koefisien fitur.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='model-metrics'>", unsafe_allow_html=True)
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("R2 test", "0.798")
    metric_col_2.metric("MAE test", format_price(91_211))
    metric_col_3.metric("RMSE test", format_price(247_725))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-kicker'>Preprocessing choice</div>", unsafe_allow_html=True)
    st.subheader("Kenapa Tidak Memakai Mapping Angka Manual?")
    st.write(
        "Kolom seperti `fuel`, `transmission`, `brand`, dan `model_family` adalah kategori "
        "nominal. Jika kategori diberi angka manual, model linear bisa menganggap ada urutan "
        "matematis antar kategori. Karena itu app memakai pipeline model dari notebook yang "
        "menggunakan OneHotEncoder."
    )

    st.markdown("<div class='section-kicker'>Target transform</div>", unsafe_allow_html=True)
    st.subheader("Kenapa Target Harga Pakai Log?")
    st.write(
        "Distribusi harga mobil bekas right-skewed. Ada banyak mobil murah-menengah dan sedikit "
        "mobil yang sangat mahal. Transformasi `log1p` mengurangi pengaruh outlier harga sehingga "
        "model lebih stabil."
    )

    st.markdown("<div class='section-kicker'>Outlier policy</div>", unsafe_allow_html=True)
    st.subheader("Apakah Outlier Harga Sebaiknya Dihapus?")
    st.write(
        "Untuk kasus harga mobil, outlier tidak otomatis berarti data salah. Mobil yang sangat mahal "
        "bisa saja valid karena merek, tipe, tahun, atau kelas mobilnya memang berbeda. Karena itu "
        "model utama tidak menghapus outlier harga valid, tetapi memakai transformasi log agar "
        "pengaruh harga ekstrem lebih terkendali."
    )

    st.markdown(
        """
        <div class="soft-panel">
            Menghapus outlier boleh dilakukan jika tujuan model hanya untuk mobil harga umum
            atau jika nilai ekstrem terbukti salah input. Namun jika outlier adalah mobil mahal yang valid,
            menghapusnya membuat model kurang mampu memprediksi mobil mahal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Di notebook, eksperimen capping p99 sudah dicoba. Hasilnya sedikit membantu area harga umum, "
        "tetapi performa keseluruhan menurun. Karena itu rekomendasi akhir tetap: gunakan data valid "
        "apa adanya, pakai log target, dan gunakan plot log/fokus 95% untuk visualisasi."
    )

    st.markdown("<div class='section-kicker'>Features</div>", unsafe_allow_html=True)
    st.subheader("Fitur yang Dipakai Model")
    feature_df = pd.DataFrame({"Fitur": feature_columns})
    st.dataframe(feature_df, width="stretch", hide_index=True)

    st.subheader("Koefisien Fitur Terbesar")

    try:
        # ambil komponen pipeline dari model yang sudah dilatih
        fitted_pipeline = model.regressor_
        fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
        fitted_ridge = fitted_pipeline.named_steps["model"]

        feature_names = fitted_preprocessor.get_feature_names_out()

        coef_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": fitted_ridge.coef_,
            }
        )
        coef_df["Abs Coefficient"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Abs Coefficient", ascending=False).head(20)

        st.dataframe(coef_df, width="stretch", hide_index=True)
    except Exception as error:
        st.warning(f"Koefisien tidak bisa ditampilkan: {error}")

    st.subheader("Kesimpulan")
    st.write(
        "Web app ini menggunakan model yang sudah dilatih di notebook. Tujuan utamanya adalah "
        "memberikan demo sederhana: user memasukkan data mobil, aplikasi membuat fitur turunan, "
        "lalu model memprediksi harga."
    )



# page 4: informasi mengenai Dataset
else:
    render_header(
        "Tentang Dataset",
        (
            "Informasi ringkas tentang sumber konteks dataset, mata uang, kolom, "
            "dan cara data dipakai oleh aplikasi."
        ),
        ["CarDekho India", "INR / ₹", "Used cars", "Regression dataset"],
    )
    render_quick_nav()

    st.markdown("<div class='section-kicker'>Dataset context</div>", unsafe_allow_html=True)
    st.subheader("Konteks Dataset")
    st.markdown(
        f"""
        <div class="soft-panel">
            Dataset ini mengikuti struktur dataset mobil bekas CarDekho India.
            Karena konteksnya India, kolom <code>selling_price</code> ditampilkan dalam
            <strong>{CURRENCY_NAME} ({CURRENCY_CODE})</strong> dengan simbol
            <strong>{CURRENCY_SYMBOL}</strong>. Label mata uang ini tidak melakukan konversi kurs,
            hanya mengikuti satuan asli dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Kolom Dataset")
    column_description = pd.DataFrame(
        [
            ["name", "Nama mobil"],
            ["year", "Tahun produksi mobil"],
            ["selling_price", f"Harga jual mobil dalam {CURRENCY_CODE}"],
            ["km_driven", "Jarak tempuh mobil"],
            ["fuel", "Jenis bahan bakar"],
            ["seller_type", "Tipe penjual"],
            ["transmission", "Jenis transmisi"],
            ["owner", "Status kepemilikan"],
        ],
        columns=["Kolom", "Keterangan"],
    )
    st.dataframe(column_description, width="stretch", hide_index=True)

    st.subheader("Cara Dataset Dipakai di Web")
    usage_col_1, usage_col_2 = st.columns(2, gap="large")

    with usage_col_1:
        st.markdown(
            """
            <div class="soft-panel">
                <strong>Untuk prediksi</strong><br>
                Dataset dipakai untuk menyediakan pilihan input seperti nama mobil,
                jenis bahan bakar, transmisi, tipe penjual, dan status kepemilikan.
                Model tidak dilatih ulang saat web dijalankan.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with usage_col_2:
        st.markdown(
            """
            <div class="soft-panel">
                <strong>Untuk EDA</strong><br>
                Dataset dipakai untuk menampilkan preview data, ringkasan statistik,
                distribusi harga, distribusi kategori, dan ringkasan outlier.
            </div>
            """,
            unsafe_allow_html=True,
        )
