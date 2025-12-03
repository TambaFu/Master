# filepath: c:\Users\tamba\OneDrive - 国立大学法人岩手大学\ドキュメント\Self-Expressive-Network\src\tes.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N = 200 
phi = 2 * np.pi * np.random.rand(N)
#np.random.rand(N)を自己表現係数にする
theta = (np.pi / 132.0) * np.sin(4 * phi)# theta = (pi/12)*sin(4*phi)緯度

# X は MATLAB と同様に shape=(3, N)
X = np.vstack([
    np.cos(theta) * np.cos(phi),
    np.cos(theta) * np.sin(phi),
    np.sin(theta)
])

# 必要なら (N,3) に変換
X_rows = X.T

# 簡単な確認 / 可視化
print("X shape:", X.shape)         # (3, N)
print("X_rows shape:", X_rows.shape)  # (N, 3)

# Plotting the random points on the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_rows[:,0], X_rows[:,1], X_rows[:,2], s=10)

# X, Y, Z軸の比率を1:1:1に設定して、球が歪まないようにする
ax.set_box_aspect([1, 1, 1]) # 3Dプロットでアスペクト比を固定する推奨される方法

# Plotting the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))


# 透明度 (alpha) を非常に小さく設定して透明に近づける
ax.plot_surface(x, y, z, color='w', alpha=0.2) # 非常に薄い白
# 球面がわかりやすいようにワイヤーフレームを追加
ax.plot_wireframe(x, y, z, color='gray', alpha=0.6, linewidth=0.5)

plt.show()