import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 特徴とラベルを読み込む
with open('datasets/MNIST/MNIST_scattering_train_data.pkl', 'rb') as f:
    features = pickle.load(f)
with open('datasets/MNIST/MNIST_scattering_train_label.pkl', 'rb') as f:
    labels = pickle.load(f)

# t-SNEで2次元に圧縮
tsne = TSNE(n_components=2, random_state=42)
proj = tsne.fit_transform(features)

# プロット
plt.figure(figsize=(10, 8))
plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=5)
plt.colorbar()
plt.title("t-SNE of Scattering Features (500D → 2D)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.show()
