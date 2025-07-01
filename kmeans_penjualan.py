# K-Means Clustering untuk Segmentasi Pelanggan Website Top-Up

# 1. Import Library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
df = pd.read_csv("penjualan_topup.csv")
print("Data Awal:")
print(df.head())

# 3. Pilih Fitur yang Akan Digunakan
fitur = ["TotalTransaksi", "TotalPengeluaran", "RataKunjunganPerBulan"]
X = df[fitur]

# 4. Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Jalankan K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 6. Visualisasi Cluster (2D: TotalTransaksi vs TotalPengeluaran)
plt.figure(figsize=(10, 6))

# Warna & Label Cluster
warna = ["red", "blue", "green", "purple"]
label_cluster = ["Pembeli Baru", "Pembeli Setia", "Pembeli Musiman", "Pembeli Besar"]

for i in range(4):
    plt.scatter(
        X_scaled[df["Cluster"] == i, 0],
        X_scaled[df["Cluster"] == i, 1],
        s=100,
        c=warna[i],
        label=f"Cluster {i} - {label_cluster[i]}"
    )

plt.title("Visualisasi Cluster Pelanggan Top-Up")
plt.xlabel("Total Transaksi (scaled)")
plt.ylabel("Total Pengeluaran (scaled)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Tampilkan Hasil Cluster
print("\nHasil Cluster:")
print(df[["CustomerID"] + fitur + ["Cluster"]])

# 8. Ringkasan Setiap Cluster
print("\nRata-Rata Setiap Cluster:")
print(df.groupby("Cluster")[fitur].mean())

# 9. Simpan ke File Baru (opsional)
df.to_csv("hasil_cluster_penjualan.csv", index=False)
