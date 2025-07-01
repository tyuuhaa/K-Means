import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("penjualan_topup.csv")

# Menampilkan data awal
print("Data Awal:")
print(df)

# Fitur yang digunakan untuk clustering
X = df[["TotalTransaksi", "TotalPengeluaran", "RataKunjunganPerBulan"]]

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menentukan jumlah cluster dengan metode Elbow
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Buat model K-Means (misal 3 cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Tampilkan hasil cluster
print("\nHasil Cluster:")
print(df[["CustomerID", "Cluster"]])

# Visualisasi cluster (menggunakan dua fitur pertama)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap='viridis')
plt.title("Visualisasi Cluster Pelanggan")
plt.xlabel("Total Transaksi (scaled)")
plt.ylabel("Total Pengeluaran (scaled)")
plt.grid()
plt.show()
