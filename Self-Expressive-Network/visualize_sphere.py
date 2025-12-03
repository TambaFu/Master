"""高次元データを球面上に可視化するスクリプト。

再構成ベクトル x̂ のような高次元データを、3次元に次元圧縮（PCA または t-SNE）
したうえで、単位ベクトルへ正規化してユニットスフィア上に描画します。

使用例:
 # 最も基本的な使い方（NumPy配列）
 python visualize_sphere.py --data reconstructed_data.npy --labels true_labels.npy

 # 次元圧縮に t-SNE を使用
 python visualize_sphere.py --data data.npy --labels labels.npy --reduction tsne

 # 複数データをグリッドで並べて表示
 python visualize_sphere.py --data data1.npy data2.npy --labels labels.npy --grid 1 2
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


def normalize_to_sphere(X):
    """ベクトルを単位長に正規化し、球面上へ射影します。"""
    norms = np.linalg.norm(X, axis=1)
    # 0 除算を避ける
    norms[norms == 0] = 1
    return X / norms[:, np.newaxis]


def reduce_dimensions(X, method='pca', n_components=3, random_state=42):
    """PCA または t-SNE を用いて高次元データを 3 次元へ圧縮します。"""
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("method は 'pca' または 'tsne' を指定してください")
    
    return reducer.fit_transform(X)


def plot_sphere_surface(ax, alpha=0.1, color='gray'):
    """半透明の単位球を描画します。"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def plot_one_sphere(X, labels=None, ax=None, title=None):
    """1 つのサブプロット上にデータ点を球面として描画します。"""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # 球面（ユニットスフィア）を描画
    plot_sphere_surface(ax)
    
    # データ点を描画
    if labels is not None:
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                           c=labels, cmap='tab10',
                           s=50, alpha=0.8)
        # 単独プロットの場合のみカラーバーを追加
        if plt.gcf().get_axes()[-1] is ax:
            plt.colorbar(scatter)
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=50, alpha=0.8)
    
    if title:
        ax.set_title(title)
    
    # アスペクト比を等しく保つ
    ax.set_box_aspect([1,1,1])
    return ax


def visualize_sphere(Xs, labels=None, reduction='pca', grid=None,
                    titles=None, figsize=None, save_path=None, show=True):
    """1 つまたは複数のデータ配列を球面上に可視化します。
    
    引数:
        Xs: (n_samples, n_features) の ndarray または そのリスト
        labels: (n_samples,) の配列 または None（クラス色分けに利用）
        reduction: 'pca' もしくは 'tsne'（3 次元への圧縮手法）
        grid: (行, 列) のタプル。None の場合は自動レイアウト
        titles: 各サブプロットのタイトルのリスト
        figsize: 図のサイズ（インチ）
        save_path: 図の保存パス
        show: 画面表示を行うかどうか
    """
    # Xs をリスト化して扱う
    if isinstance(Xs, np.ndarray):
        Xs = [Xs]
    
    # グリッドレイアウトを決定
    n_plots = len(Xs)
    if grid is None:
        if n_plots <= 3:
            grid = (1, n_plots)
        else:
            grid = (2, (n_plots + 1) // 2)
    
    if figsize is None:
        figsize = (6 * grid[1], 6 * grid[0])
    
    # 図を作成
    fig = plt.figure(figsize=figsize)
    
    # 各データセットを処理
    for i, X in enumerate(Xs):
        # 必要に応じて次元圧縮
        if X.shape[1] > 3:
            X = reduce_dimensions(X, method=reduction)
        
        # 単位球へ正規化（射影）
        X = normalize_to_sphere(X)
        
        # サブプロットを作成
        ax = fig.add_subplot(grid[0], grid[1], i+1, projection='3d')
        
        # 描画
        title = titles[i] if titles and i < len(titles) else None
        plot_one_sphere(X, labels, ax=ax, title=title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"図を保存しました: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def load_data(path):
    """.npy または .pkl からデータを読み込みます。"""
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("ファイルは .npy または .pkl を指定してください")


def main():
    parser = argparse.ArgumentParser(description='データを球面（ユニットスフィア）上に可視化します。')
    parser.add_argument('--data', required=True, nargs='+',
                       help='データファイルのパス（複数可）.npy または .pkl')
    parser.add_argument('--labels', help='ラベルファイルのパス .npy または .pkl（任意）')
    parser.add_argument('--reduction', choices=['pca', 'tsne'],
                       default='pca', help='3 次元への次元圧縮手法')
    parser.add_argument('--grid', type=int, nargs=2,
                       help='グリッドレイアウト（行 列）')
    parser.add_argument('--titles', nargs='+',
                       help='各サブプロットのタイトル')
    parser.add_argument('--save', '-s', help='図の保存先パス')
    parser.add_argument('--no-show', action='store_true',
                       help='描画ウィンドウを表示しない（保存のみ）')
    args = parser.parse_args()
    
    # Load all data files
    Xs = [load_data(path) for path in args.data]
    
    # Load labels if provided
    labels = None
    if args.labels:
        labels = load_data(args.labels)
    
    # Visualize
    visualize_sphere(Xs, labels=labels, reduction=args.reduction,
                    grid=args.grid, titles=args.titles,
                    save_path=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()