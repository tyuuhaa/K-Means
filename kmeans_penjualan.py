# 1. Import Library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
df = pd.read_csv("penjualan_topup.csv")
print("Data Awal:")
print(df)

# 3. Persiapan Data
X = df[["TotalTransaksi", "TotalPengeluaran", "RataKunjunganPerBulan"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Jalankan K-Means Clustering (misal: 3 cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 5. Tampilkan Hasil Clustering
print("\nHasil Cluster:")
print(df[["CustomerID", "TotalTransaksi", "TotalPengeluaran", "RataKunjunganPerBulan", "Cluster"]])

# 6. Visualisasi Clustering
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="Set1", s=100)
plt.title("Visualisasi Cluster Pelanggan")
plt.xlabel("Total Transaksi (scaled)")
plt.ylabel("Total Pengeluaran (scaled)")
plt.grid(True)
plt.show()
