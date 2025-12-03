"""各ステージで再構成された x̂ を球面上に描画するスクリプト（tes.py 風）。

multi_main3.py が保存した再構成データ（xhat_*.npy）を結果フォルダから自動で見つけ、
ステージごとに読み込んで球面上に並べて可視化します。データが3次元より高次元の場合は、
PCA（またはオプションで t-SNE）により3次元へ次元圧縮し、単位球へ正規化してから描画します。

想定する x̂ のファイル名（multi_main3.py の保存形式）:
    {folder}/xhat_{dataset}_N{N}_stage{stage}_epoch{epoch_tag}.npy
ここで:
    - folder: "{dataset}_result"
    - epoch_tag: xhat_every=0 の場合は "final"、それ以外は整数エポック

ラベル（任意）があれば自動で読み込み、クラス別に色分けします:
    {folder}/{dataset}_labels_{N}.pkl

使用例:
    # 各ステージの最新の x̂ を自動検出して描画し、PNG 保存
    python plot_xhat_stages.py --dataset CIFAR10 --N 5000 --reduction pca --save CIFAR10_result/xhat_sphere.png

    # ステージとエポックを明示指定
    python plot_xhat_stages.py --dataset MNIST --N 2000 --stages 0 1 --epoch final

    # t-SNE を利用（低速だが場合によっては分離が見やすい）
    python plot_xhat_stages.py --dataset MNIST --N 200 --reduction tsne
"""

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import numpy as np

# 既存の可視化ユーティリティ（visualize_sphere.py）を再利用
try:
    from visualize_sphere import visualize_sphere  # type: ignore
except Exception as e:
    print("Failed to import visualize_sphere.visualize_sphere. Make sure 'visualize_sphere.py' is in the same folder.")
    raise


def find_xhat_files(folder: str, dataset: str, N: int, stages: Optional[List[int]] = None,
                    epoch: Optional[str] = None) -> List[Tuple[int, str]]:
    """ステージごとの x̂ の .npy ファイルを検索して取得します。

    引数:
        folder: 結果フォルダ（例: 'CIFAR10_result'）
        dataset: データセット名（例: 'CIFAR10'）
        N: 学習時のサンプル数（ファイル名照合に使用）
        stages: 対象とするステージ番号のリスト。None の場合は自動検出
        epoch: 'final' または 文字列の整数（例: '12'）。None は最新を自動選択

    戻り値:
        (stage, filepath) のリスト。stage 昇順でソート済み。
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Results folder not found: {folder}")

    pattern = re.compile(rf"^xhat_{re.escape(dataset)}_N{N}_stage(\d+)_epoch(.+)\.npy$")
    candidates = {}

    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if not m:
            continue
        stg = int(m.group(1))
        tag = m.group(2)
    # 指定されたステージのみ残す（指定がある場合）
        if stages is not None and stg not in stages:
            continue
    # 指定されたエポックのみ残す（指定がある場合）
        if epoch is not None and tag != str(epoch):
            continue
    # ステージごとの候補を収集
        candidates.setdefault(stg, []).append((tag, os.path.join(folder, fname)))

    selected = []
    for stg, files in candidates.items():
    # 'final' があれば最優先で選択
        finals = [fp for tag, fp in files if tag == 'final']
        if finals and (epoch is None or epoch == 'final'):
            selected.append((stg, finals[0]))
            continue
    # それ以外は数値エポックがあれば最大（最新）を選択
        numeric = []
        for tag, fp in files:
            try:
                numeric.append((int(tag), fp))
            except ValueError:
                # Non-numeric tag; if epoch explicitly matches, we'd have picked above
                pass
        if numeric:
            numeric.sort(key=lambda x: x[0])
            selected.append((stg, numeric[-1][1]))
        else:
            # 数値に変換できない場合は文字列順で最後を選択
            files.sort(key=lambda x: x[0])
            selected.append((stg, files[-1][1]))

    if not selected:
        raise FileNotFoundError(
            f"No x̂ files found in '{folder}' for dataset={dataset}, N={N}"
            + (f", stages={stages}" if stages else "")
            + (f", epoch={epoch}" if epoch else ".")
        )

    selected.sort(key=lambda x: x[0])
    return selected


def maybe_load_labels(folder: str, dataset: str, N: int) -> Optional[np.ndarray]:
    """multi_main3.py が保存したラベルの読み込みを試みます。なければ None を返します。"""
    for ext in ('.pkl', '.npy'):
        path = os.path.join(folder, f"{dataset}_labels_{N}{ext}")
        if os.path.isfile(path):
            try:
                if ext == '.pkl':
                    import pickle
                    with open(path, 'rb') as f:
                        labels = pickle.load(f)
                else:
                    labels = np.load(path)
                labels = np.asarray(labels)
                return labels
            except Exception:
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description='各ステージで再構成された x̂ を球面上に可視化します。')
    parser.add_argument('--dataset', required=True, help='データセット名（例: MNIST, CIFAR10）')
    parser.add_argument('--N', type=int, required=True, help='学習時のサンプル数（ファイル名の照合に使用）')
    parser.add_argument('--stages', type=int, nargs='+', help='描画するステージ番号（省略時は自動検出）')
    parser.add_argument('--epoch', help="読み込むエポックタグ（'final' または 整数文字列）。省略時は自動選択")
    parser.add_argument('--folder', help='結果フォルダ（既定: {dataset}_result）')
    parser.add_argument('--reduction', choices=['pca', 'tsne'], default='pca', help='描画前の次元圧縮手法')
    parser.add_argument('--save', '-s', help='結合図の保存先パス')
    parser.add_argument('--no-show', action='store_true', help='インタラクティブな表示を行わない')
    parser.add_argument('--max-points', type=int, default=5000, help='高速化のため各ステージで間引く最大点数')
    args = parser.parse_args()

    folder = args.folder or f"{args.dataset}_result"

    try:
        stage_files = find_xhat_files(folder, args.dataset, args.N, stages=args.stages, epoch=args.epoch)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    # ステージごとにデータを読み込み
    Xs = []
    titles = []
    for stg, fpath in stage_files:
        X = np.load(fpath)
        if not isinstance(X, np.ndarray):
            print(f"Warning: {fpath} did not load as numpy array. Skipping.")
            continue
    # 必要に応じて間引き（速度と視認性の向上）
        if args.max_points and X.shape[0] > args.max_points:
            idx = np.random.RandomState(0).choice(X.shape[0], args.max_points, replace=False)
            X = X[idx]
        Xs.append(X)
        titles.append(f"Stage {stg}\n{os.path.basename(fpath)}")

    if not Xs:
        print("No valid x̂ arrays loaded. Nothing to plot.")
        sys.exit(1)

    # ラベルを読み込み（任意）
    labels = maybe_load_labels(folder, args.dataset, args.N)

    # 共有ヘルパー（visualize_sphere）で可視化
    visualize_sphere(Xs, labels=labels, reduction=args.reduction, titles=titles,
                     save_path=args.save, show=not args.no_show)


if __name__ == '__main__':
    main()
