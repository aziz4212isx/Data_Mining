# Import library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Mengimpor dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # Menggunakan kolom fitur
y = dataset.iloc[:, 4].values       # Menggunakan kolom target

# Membagi dataset menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Normalisasi fitur
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Membangun model Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Memprediksi hasil test set
y_pred = classifier.predict(X_test)

# Membuat confusion matrix dan menghitung akurasi
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan hasil
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualisasi hasil
plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], color='red', label='Actual Class 0')
plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], color='green', label='Actual Class 1')
plt.title('Naive Bayes Classification (Test set)')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.legend()
plt.show()
