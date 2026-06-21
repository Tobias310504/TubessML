# Web App Prediksi Harga Mobil Bekas

Web app ini dibuat dengan Streamlit untuk memprediksi harga mobil bekas dari model Machine Learning yang sudah dilatih di notebook.

Harga pada dataset ditampilkan sebagai **Indian Rupee (INR / ₹)** karena dataset yang digunakan adalah dataset mobil bekas India/CarDekho. Simbol mata uang ini hanya label satuan dataset, bukan hasil konversi kurs.

## Isi Folder

- `app.py`: kode utama web app
- `requirements.txt`: daftar library yang dibutuhkan
- `model/model_harga_mobil_ridge_log.pkl`: model hasil training
- `data/Dataset_carDeckho.csv`: dataset untuk EDA dan pilihan input

## Cara Menjalankan Lokal

```bash
streamlit run app.py
```
## Catatan

File `.pkl` hanya aman digunakan jika sumber model dipercaya. Pada project ini model berasal dari training notebook sendiri.
