import pandas as pd
from sklearn.impute import SimpleImputer

# Memuat dataset dari file CSV
file_path = r'D:\SEMESTER 5\Data Mining\tugas 3\Cars Dataset.csv'
data = pd.read_csv(file_path)

# Handle missing values
# Untuk kolom numerik (Engine, Power, Seats), gunakan rata-rata (mean) untuk mengisi nilai yang hilang
imputer = SimpleImputer(missing_values=pd.NA, strategy='mean')

# Bersihkan kolom 'Engine' dan 'Power' dengan menghapus karakter non-numerik dan mengubahnya menjadi float
data['Engine'] = data['Engine'].replace({' CC': ''}, regex=True).astype(float)
data['Power'] = pd.to_numeric(data['Power'].replace({' bhp': '', 'null': pd.NA}, regex=True), errors='coerce')

# Isi nilai yang hilang di kolom numerik (Engine, Power, Seats)
data['Engine'] = imputer.fit_transform(data[['Engine']])
data['Power'] = imputer.fit_transform(data[['Power']])
data['Seats'] = imputer.fit_transform(data[['Seats']])

# Untuk kolom kategorikal (Fuel_Type, Transmission, Owner_Type), gunakan nilai yang paling sering (modus)
imputer_categorical = SimpleImputer(strategy='most_frequent')

# Ubah hasil fit_transform menjadi array 1D dengan .ravel()
data['Fuel_Type'] = imputer_categorical.fit_transform(data[['Fuel_Type']]).ravel()
data['Transmission'] = imputer_categorical.fit_transform(data[['Transmission']]).ravel()
data['Owner_Type'] = imputer_categorical.fit_transform(data[['Owner_Type']]).ravel()

# Simpan hasilnya ke file Excel dan CSV
output_excel_path = r'D:\SEMESTER 5\Data Mining\tugas 3\Cars_Dataset_Cleaned.xlsx'
output_csv_path = r'D:\SEMESTER 5\Data Mining\tugas 3\Cars_Dataset_Cleaned.csv'

data.to_excel(output_excel_path, index=False)
data.to_csv(output_csv_path, index=False)

print("Data preprocessing selesai dan disimpan ke Excel dan CSV.")
