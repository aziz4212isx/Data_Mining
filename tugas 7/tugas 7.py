import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Dataset laptop dalam bentuk array
data_laptop = [
    ["Tinggi", "Mahal", "Core i7", "16GB", "SSD", "Ya"],
    ["Rendah", "Murah", "Core i3", "4GB", "HDD", "Tidak"],
    ["Sedang", "Sedang", "Core i5", "8GB", "SSD", "Ya"],
    ["Tinggi", "Mahal", "Core i7", "8GB", "HDD", "Tidak"],
    ["Rendah", "Murah", "Core i3", "8GB", "SSD", "Ya"],
    ["Sedang", "Mahal", "Core i5", "16GB", "SSD", "Ya"],
    ["Tinggi", "Sedang", "Core i7", "8GB", "SSD", "Ya"],
    ["Rendah", "Sedang", "Core i3", "4GB", "HDD", "Tidak"],
    ["Sedang", "Murah", "Core i5", "8GB", "HDD", "Tidak"],
    ["Tinggi", "Mahal", "Core i7", "16GB", "HDD", "Ya"]
]

# Memisahkan fitur (X) dan label (y)
X = np.array([row[:-1] for row in data_laptop])
y = np.array([row[-1] for row in data_laptop])

# Inisialisasi LabelEncoder untuk setiap kolom
feature_names = ['Tinggi', 'Harga', 'Prosesor', 'RAM', 'Penyimpanan']
encoders = {}
X_encoded = np.empty(X.shape)

# Mengubah setiap kolom fitur menjadi numerik
for i in range(X.shape[1]):
    encoders[feature_names[i]] = LabelEncoder()
    X_encoded[:, i] = encoders[feature_names[i]].fit_transform(X[:, i])

# Mengubah label menjadi numerik
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Membuat dan melatih model Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_encoded, y_encoded)

# Membuat prediksi
y_pred = clf.predict(X_encoded)

# Visualisasi pohon keputusan
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=feature_names, 
          class_names=label_encoder.classes_, 
          filled=True, rounded=True, fontsize=12)
plt.show()

# Menghitung akurasi
accuracy = np.mean(y_pred == y_encoded)

# Membuat list untuk menyimpan perbandingan hasil
comparison = []
for i in range(len(y)):
    actual = y[i]
    predicted = label_encoder.inverse_transform([y_pred[i]])[0]
    is_correct = actual == predicted
    comparison.append({
        'Data ke': i+1,
        'Sebenarnya': actual,
        'Prediksi': predicted,
        'Hasil': 'Benar' if is_correct else 'Salah'
    })

# Menampilkan hasil detail
print("\nDetail Hasil Prediksi:")
print("-" * 60)
print("| {:^8} | {:^12} | {:^12} | {:^10} |".format('Data ke', 'Sebenarnya', 'Prediksi', 'Hasil'))
print("-" * 60)
for item in comparison:
    print("| {:^8} | {:^12} | {:^12} | {:^10} |".format(
        item['Data ke'],
        item['Sebenarnya'],
        item['Prediksi'],
        item['Hasil']
    ))
print("-" * 60)

# Menampilkan ringkasan hasil
print("\nRingkasan Hasil Evaluasi:")
print("-" * 40)
print(f"Label sebenarnya: {y.tolist()}")
print(f"Hasil prediksi  : {label_encoder.inverse_transform(y_pred).tolist()}")
print(f"Prediksi benar  : {sum(y_pred == y_encoded)}")
print(f"Prediksi salah  : {sum(y_pred != y_encoded)}")
print(f"Akurasi        : {accuracy:.2%}")
print("-" * 40)

# Menampilkan importance feature
print("\nTingkat Kepentingan Fitur:")
print("-" * 40)
for feature, importance in zip(feature_names, clf.feature_importances_):
    print(f"{feature:12}: {importance:.4f}")
print("-" * 40)