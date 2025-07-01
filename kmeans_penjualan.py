# K-Means Clustering untuk Segmentasi Pelanggan Website Top-Up

# 1. Import Library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
df = pd.read_csv("penjualan_topup.csv")

# 3. Pilih Fitur untuk Clustering
fitur = ["TotalTransaksi", "TotalPengeluaran", "RataKunjunganPerBulan"]
X = df[fitur]

# 4. Normalisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Jalankan K-Means Clustering (4 cluster)
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 6. Hitung Rata-Rata Tiap Cluster
cluster_summary = df.groupby("Cluster")[fitur].mean().round(0).astype(int)

# 7. Skoring dan Penentuan Label Final
cluster_summary["Skor"] = cluster_summary["TotalTransaksi"] + \
                          cluster_summary["TotalPengeluaran"] / 100000 + \
                          cluster_summary["RataKunjunganPerBulan"]

# Urutkan cluster berdasarkan skor total
cluster_sorted = cluster_summary.sort_values("Skor")

# Label final berurutan
label_urut = ["Pembeli Baru", "Pembeli Musiman", "Pembeli Setia", "Pembeli Besar"]
cluster_sorted["LabelFinal"] = label_urut

# Buat mapping dari Cluster ke LabelFinal
label_mapping = cluster_sorted["LabelFinal"].to_dict()
df["LabelCluster"] = df["Cluster"].map(label_mapping)

# 8. Visualisasi
plt.figure(figsize=(10, 6))

# Warna sesuai urutan label
label_to_color = {
    "Pembeli Baru": "blue",
    "Pembeli Musiman": "purple",
    "Pembeli Setia": "green",
    "Pembeli Besar": "red"
}

# Tampilkan scatter plot berdasarkan label yang sudah ditetapkan
for label in label_urut:  # urutan: Baru, Musiman, Setia, Besar
    subset = df[df["LabelCluster"] == label]
    plt.scatter(
        subset["TotalTransaksi"],
        subset["TotalPengeluaran"],
        s=100,
        c=label_to_color[label],
        label=label
    )

plt.title("Visualisasi Cluster Pelanggan Top-Up")
plt.xlabel("Total Transaksi")
plt.ylabel("Total Pengeluaran")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Simpan hasil lengkap
df.to_csv("hasil_cluster_penjualan.csv", index=False)
