# Panduan Deploy ke Streamlit Community Cloud

Folder ini adalah folder siap deploy untuk web app prediksi harga mobil bekas.

## Isi Folder

- `app.py`: file utama Streamlit
- `requirements.txt`: daftar library yang akan di-install oleh Streamlit Cloud
- `.streamlit/config.toml`: konfigurasi tema tampilan
- `data/Dataset_Raw.csv`: dataset untuk EDA dan pilihan input
- `model/model_harga_mobil_ridge_log.pkl`: model hasil training dari notebook
- `.gitignore`: file yang tidak perlu masuk GitHub

## Langkah Deploy

1. Buat repository baru di GitHub.
2. Upload semua isi folder ini ke repository tersebut.
3. Buka Streamlit Community Cloud.
4. Pilih repository GitHub tadi.
5. Set main file path ke:

```text
app.py
```

6. Klik Deploy.

## Catatan Penting

- Jangan ubah struktur folder `data/` dan `model/`, karena `app.py` sudah membaca file dari lokasi tersebut.
- Harga ditampilkan dalam Indian Rupee (`INR / ₹`) karena dataset berasal dari konteks CarDekho India.
- File `.pkl` hanya aman digunakan jika sumber model dipercaya. Pada project ini model berasal dari notebook training sendiri.

## Cara Jalankan Lokal

Masuk ke folder ini, lalu jalankan:

```bash
streamlit run app.py
```

Jika berhasil, browser akan membuka:

```text
http://localhost:8501
```
