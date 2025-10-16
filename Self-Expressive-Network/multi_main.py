import numpy as np  # 数値計算ライブラリNumPyをインポート
import torch  # PyTorch（深層学習ライブラリ）をインポート
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn  # ニューラルネットワーク構築用モジュール
import torch.nn.functional as F  # PyTorchの関数型API
import torch.nn.init as init  # 重み初期化用モジュール
import torch.optim as optim  # 最適化アルゴリズム（例：Adam）
import utils  # 補助関数などを定義した独自モジュール
from sklearn import cluster  # クラスタリング手法（例：KMeans）
import pickle  # Pythonオブジェクトをファイルに保存・読み込みするためのモジュール
import scipy.sparse as sparse  # 疎行列を扱うためのモジュール
import time  # 時間計測用
from sklearn.preprocessing import normalize  # 正規化処理
from sklearn.neighbors import kneighbors_graph  # k近傍グラフの構築
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score  # クラスタリング評価指標
from metrics.cluster.accuracy import clustering_accuracy  # 精度計算のための独自実装関数
import argparse  # コマンドライン引数の解析用モジュール
import random  # 乱数生成用
from tqdm import tqdm  # プログレスバー表示
import os  # OS依存の処理（例：フォルダ作成）
import csv  # CSVファイル操作
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用するGPUを指定（GPU 0）
# 新しく追加するライブラリ
import matplotlib.pyplot as plt
import pandas as pd

class MLP(nn.Module):  # 多層パーセプトロン（MLP）を定義するクラス
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims  # 入力次元
        self.hid_dims = hid_dims  # 中間層の次元リスト
        self.output_dims = out_dims  # 出力次元
        self.layers = nn.ModuleList()  # 層をリスト形式で保持

        # 最初の全結合層とReLU活性化
        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())

        # 残りの中間層を順に追加
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        # 出力層
        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)

        if kaiming_init:  # Kaiming初期化を使う場合
            self.reset_parameters()

    def reset_parameters(self):  # 重み初期化関数
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)  # Kaiming初期化
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)  # Xavier初期化
        init.zeros_(self.out_layer.bias)

    def forward(self, x):  # 順伝播の処理
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)  # 線形→ReLU
        h = self.out_layer(h)  # 最終出力層
        h = torch.tanh_(h)  # 出力にtanhを適用[-1から1に正規化]
        return h


class AdaptiveSoftThreshold(nn.Module):  # 適応型ソフトしきい値処理の定義
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        # 学習可能なバイアスパラメータを登録（初期値ゼロ）
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):  # 順伝播処理
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)  # ソフトしきい値処理


class SENet(nn.Module):  # Self-Expressive Networkの本体
    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims  # スケーリング係数（分母）

        # クエリとキー用のMLPを別々に定義
        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)  # ソフトしきい値を1次元で適用

    def query_embedding(self, queries):  # クエリ埋め込み生成
        q_emb = self.net_q(queries)
        return q_emb

    def key_embedding(self, keys):  # キー埋め込み生成
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):  # 自己表現係数の計算
        c = self.thres(q_emb.mm(k_emb.t()))  # 内積 → ソフトしきい値
        return self.shrink * c  # スケーリングして返す


    def forward(self, queries, keys):  # モデルの順伝播全体（クエリ・キー→係数）
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out

class MultiSENet(nn.Module):  # Self-Expressive Networkの本体
    def __init__(self, input_dims, hid_dims, out_dims, num_stages, kaiming_init=True):
        super(MultiSENet, self).__init__()#親クラスのインスタンス継承
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims  # スケーリング係数（分母）
        self.numstages=num_stages
        self.net_q_list = nn.ModuleList()
        self.net_k_list = nn.ModuleList()
        self.thres_list = nn.ModuleList()  # ソフトしきい値を1次元で適用
        # クエリとキー用のMLPを別々に定義
        for _ in range(self.numstages):
            self.net_q_list.append(MLP(input_dims=self.input_dims,
                            hid_dims=self.hid_dims,
                            out_dims=self.out_dims,
                            kaiming_init=self.kaiming_init))

            self.net_k_list.append(MLP(input_dims=self.input_dims,
                            hid_dims=self.hid_dims,
                            out_dims=self.out_dims,
                            kaiming_init=self.kaiming_init))
            self.thres_list.append(AdaptiveSoftThreshold(1))

    def query_embedding(self, stage, queries):  # クエリ埋め込み生成
        q_emb = self.net_q_list[stage](queries)
        return q_emb

    def key_embedding(self, stage, keys):  # キー埋め込み生成
        k_emb = self.net_k_list[stage](keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb, stage):  # 自己表現係数の計算
        c = self.thres_list[stage](q_emb.mm(k_emb.t()))  # 内積 → ソフトしきい値
        return self.shrink * c  # スケーリングして返す


    def forward(self, queries, keys):
        rec_batch = torch.zeros_like(queries).cuda()
        reg = torch.zeros([1]).cuda()

        # 初期値として入力を保持
        x_rec = keys

        for stage in range(self.numstages):
            # 現在の入力でクエリとキーを計算
            if stage == 0:
                # 第1段階：元の入力を使用
                q = self.query_embedding(stage,queries)
                k = self.key_embedding(stage,keys)
            else:
                # 第2段階以降：前段階の再構成結果を使用
                q = self.query_embedding(stage,x_rec)
                k = self.key_embedding(stage,x_rec)  # 図では x̂_j も再構成結果
            c = self.get_coeff(q, k, stage)

            # 再構成（図のx̂_i = Σ c_ij * x_j）
            if stage == 0:
                x_reconstructed = c.mm(keys)  # 第1段階は元のkeysを使用
            else:
                x_reconstructed = c.mm(x_rec)  # 第2段階以降は前段階の結果を使用

            # 対角補正
            #.sum(dim=1, keepdim=True)各行の和
            #p*k対応する要素同士掛け算
            diag_c = self.thres_list[stage]((q * k).sum(dim=1, keepdim=True)) * self.shrink
            rec_batch = x_reconstructed - diag_c * x_rec

            # 正則化
            reg = regularizer(c)
            reg = reg - regularizer(diag_c)

            # 次段の入力に渡す（detachで勾配遮断）テンソルだけを渡す
            x_rec = rec_batch.detach()

        return c, rec_batch, reg


def regularizer(c, lmbd=1.0):  # 正則化項（L1とL2の混合）を計算
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


def get_sparse_rep(multisenet, data, stage, batch_size=10, chunk_size=100, non_zeros=1000):  # 疎な自己表現係数行列を取得
    N, D = data.shape  # データのサンプル数と次元数を取得
    non_zeros = min(N, non_zeros)  # 非ゼロ要素数を制限
    C = torch.empty([batch_size, N])  # 結果格納用のテンソル

    # バッチサイズとチャンクサイズはNの約数である必要がある
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    if (N % chunk_size != 0):
        raise Exception("chunk_size should be a factor of dataset size.")

    val = []  # スパース行列の値
    indicies = []  # スパース行列のインデックス
    with torch.no_grad():
        multisenet.eval()  # モデルを推論モードに切り替え
        for i in range(data.shape[0] // batch_size):
            chunk = data[i * batch_size:(i + 1) * batch_size].cuda()
            q = multisenet.query_embedding(stage-1,chunk)  # クエリを埋め込みに変換
            for j in range(data.shape[0] // chunk_size):
                chunk_samples = data[j * chunk_size: (j + 1) * chunk_size].cuda()
                k = multisenet.key_embedding(stage-1,chunk_samples)  # キーの埋め込み
                temp = multisenet.get_coeff(q, k, stage-1)  # 自己表現係数
                C[:, j * chunk_size:(j + 1) * chunk_size] = temp.cpu()

            rows = list(range(batch_size))
            cols = [j + i * batch_size for j in rows]
            C[rows, cols] = 0.0  # 対角要素（自己ループ）を0に設定

            _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)  # 大きい順に非ゼロを抽出
            val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
            index = index.reshape([-1]).cpu().data.numpy()
            indicies.append(index)

    val = np.concatenate(val, axis=0)  # 値を連結
    indicies = np.concatenate(indicies, axis=0)  # インデックスを連結
    indptr = [non_zeros * i for i in range(N + 1)]  # 各行のポインタ

    # CSR形式の疎行列に変換
    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse



def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):  # k近傍グラフによる類似度行列作成
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)  # 対称グラフに変換
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)  # 相互近傍のみを残す
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


def evaluate(multisenet, data, num_stage, labels, num_subspaces, spectral_dim, non_zeros=1000, n_neighbors=3,
             batch_size=10000, chunk_size=10000, affinity='nearest_neighbor', knn_mode='symmetric'):
    # 自己表現係数の疎行列を生成
    C_sparse = get_sparse_rep(multisenet=multisenet, data=data, stage=num_stage, batch_size=batch_size,
                              chunk_size=chunk_size, non_zeros=non_zeros)
    C_sparse_normalized = normalize(C_sparse).astype(np.float32)  # 正規化

    # 類似度行列の作成
    if affinity == 'symmetric':
        Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    elif affinity == 'nearest_neighbor':
        Aff = get_knn_Aff(C_sparse_normalized, k=n_neighbors, mode=knn_mode)
    else:
        raise Exception("affinity should be 'symmetric' or 'nearest_neighbor'")

    # スペクトラルクラスタリング実行
    preds = utils.spectral_clustering(Aff, num_subspaces, spectral_dim)

    # 評価指標の計算
    acc = clustering_accuracy(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds, average_method='geometric')
    ari = adjusted_rand_score(labels, preds)
    return acc, nmi, ari


def same_seeds(seed):  # 乱数シードを固定して再現性を確保
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def plot_learning_curve(data, save_path):
    """
    学習曲線（損失と精度の推移）をプロットする関数
    Args:
        data (pd.DataFrame): 損失と精度のデータを含むDataFrame
        save_path (str): 画像の保存先パス
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(data['Iteration'], data['Loss'], color=color, label='Total Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # 共通のx軸を持つ第二のy軸を作成
    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(data['Iteration'], data['Accuracy'], color=color, linestyle='--', marker='o', label='Clustering Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title('Learning Curve')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(save_path)
    print(f"Learning curve saved to {save_path}")
    plt.close()

if __name__ == "__main__":  # スクリプトを直接実行したときに実行されるメイン処理
    parser = argparse.ArgumentParser()  # 引数解析器を作成
    parser.add_argument('--dataset', type=str, default="MNIST")  # 使用するデータセット名
    parser.add_argument('--num_subspaces', type=int, default=10)  # クラスタ数
    parser.add_argument('--gamma', type=float, default=200.0)  # 再構成誤差の重み
    parser.add_argument('--lmbd', type=float, default=0.9)  # L1とL2のバランス係数
    parser.add_argument('--hid_dims', type=int, default=[1024,1024,1024])  # 隠れ層の構成
    parser.add_argument('--out_dims', type=int, default=1024)  # 埋め込みの出力次元
    parser.add_argument('--total_iters', type=int, default=100000)  # 学習ステップ数
    parser.add_argument('--save_iters', type=int, default=200000)  # モデル保存の間隔
    parser.add_argument('--eval_iters', type=int, default=200000)  # 評価の間隔
    parser.add_argument('--lr', type=float, default=1e-3)  # 学習率
    parser.add_argument('--lr_min', type=float, default=0.0)  # 最小学習率
    parser.add_argument('--batch_size', type=int, default=100)  # 学習時のバッチサイズ
    parser.add_argument('--chunk_size', type=int, default=10000)  # 分割処理のチャンクサイズ
    parser.add_argument('--non_zeros', type=int, default=1000)  # 自己表現係数の非ゼロ数
    parser.add_argument('--n_neighbors', type=int, default=3)  # k-NNの近傍数
    parser.add_argument('--spectral_dim', type=int, default=15)  # スペクトラルクラスタリングの次元
    parser.add_argument('--num_stages', type=int, default=2)  # 多段数
    parser.add_argument('--affinity', type=str, default="nearest_neighbor")  # 類似度行列のタイプ
    parser.add_argument('--mean_subtract', dest='mean_subtraction', action='store_true')  # 平均減算の有無
    parser.set_defaults(mean_subtraction=False)
    parser.add_argument('--seed', type=int, default=0)  # 乱数シード
    args = parser.parse_args()  # 引数をパース

    # データセットに応じてパラメータを調整
    if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST':
        args.__setattr__('gamma', 200.0)
        args.__setattr__('spectral_dim', 15)
        args.__setattr__('mean_subtract', False)
        args.__setattr__('lr_min', 0.0)
    elif args.dataset == 'EMNIST':
        args.__setattr__('gamma', 150.0)
        args.__setattr__('num_subspaces', 26)
        args.__setattr__('spectral_dim', 26)
        args.__setattr__('mean_subtract', True)
        args.__setattr__('chunk_size', 10611)
        args.__setattr__('lr_min', 1e-3)
    elif args.dataset == 'CIFAR10':
        args.__setattr__('gamma', 200)
        args.__setattr__('num_subspaces', 10)
        args.__setattr__('chunk_size', 10000)
        args.__setattr__('total_iters', 50000)
        args.__setattr__('eval_iters', 1000)
        args.__setattr__('lr_min', 0.0)
        args.__setattr__('spectral_dim', 10)
        args.__setattr__('mean_subtract', False)
        args.__setattr__('affinity', 'symmetric')
    else:
        raise Exception("Only MNIST, FashionMNIST, EMNIST and CIFAR10 are currently supported.")

    # 実験設定の表示
    fit_msg = "Experiments on {}, numpy_seed=0, total_iters=100000, lambda=0.9, gamma=200.0".format(args.dataset, args.seed)
    print(fit_msg)

    folder = "{}_result".format(args.dataset)  # 結果保存フォルダ名
    if not os.path.exists(folder):
        os.mkdir(folder)  # なければ作成

    same_seeds(args.seed)  # シードを固定して再現性確保
    tic = time.time()  # 開始時間の記録

    # データセット読み込み（前処理済み特徴量）
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
        with open('datasets/{}/{}_scattering_train_data.pkl'.format(args.dataset, args.dataset), 'rb') as f:
            train_samples = pickle.load(f)
        with open('datasets/{}/{}_scattering_train_label.pkl'.format(args.dataset, args.dataset), 'rb') as f:
            train_labels = pickle.load(f)
        with open('datasets/{}/{}_scattering_test_data.pkl'.format(args.dataset, args.dataset), 'rb') as f:
            test_samples = pickle.load(f)
        with open('datasets/{}/{}_scattering_test_label.pkl'.format(args.dataset, args.dataset), 'rb') as f:
            test_labels = pickle.load(f)
        full_samples = np.concatenate([train_samples, test_samples], axis=0)
        full_labels = np.concatenate([train_labels, test_labels], axis=0)
    elif args.dataset in ["CIFAR10"]:
        with open('datasets/CIFAR10-MCR2/cifar10-features.npy', 'rb') as f:
            full_samples = np.load(f)
        with open('datasets/CIFAR10-MCR2/cifar10-labels.npy', 'rb') as f:
            full_labels = np.load(f)
    else:
        raise Exception("Only MNIST, FashionMNIST and EMNIST are currently supported.")

    # 平均減算（必要な場合）
    if args.mean_subtract:
        print("Mean Subtraction")
        full_samples = full_samples - np.mean(full_samples, axis=0, keepdims=True)

    # ラベルを0から始まるように変換
    full_labels = full_labels - np.min(full_labels)

    # 評価結果を保存するCSVファイルを準備
    result = open('{}/results.csv'.format(folder), 'w')
    writer = csv.writer(result)
    writer.writerow(["N", "ACC", "NMI", "ARI"])
    # 追加: 学習過程のデータを保存するリスト
    learning_data = {
        'Iteration': [],
        'Loss': [],
        'Accuracy': []
    }
    global_steps = 0  # 学習全体のステップ数カウント
    
    # 複数のサンプルサイズで実験N=20000の時
    for N in [200]:
        #sampleをNこランダムで選ぶ
        sampled_idx = np.random.choice(full_samples.shape[0], N, replace=False)
        samples, labels = full_samples[sampled_idx], full_labels[sampled_idx]
        block_size = min(N, 10000)#=10000

        # サンプルを保存（後で検証や可視化に使用可能）
        with open('{}/{}_samples_{}.pkl'.format(folder, args.dataset, N), 'wb') as f:
            pickle.dump(samples, f)
        with open('{}/{}_labels_{}.pkl'.format(folder, args.dataset, N), 'wb') as f:
            pickle.dump(labels, f)

        all_samples, ambient_dim = samples.shape[0], samples.shape[1]  # サンプル数と特徴次元

        data = torch.from_numpy(samples).float()  # NumPy→Tensor
        data = utils.p_normalize(data)  # データを正規化

        n_iter_per_epoch = samples.shape[0] // args.batch_size  # N/100 N=20000の時200 1エポックあたりのイテレーション数
        #block_size min(N,10000)=N
        n_step_per_iter = round(all_samples // block_size)  #N/N=1 再構成に必要な分割数N=20000の時2
        n_epochs = args.total_iters // n_iter_per_epoch  # 100000/(N/100)総エポック数N=20000の時500

        # モデル・最適化器・スケジューラの初期化
        multisenet = MultiSENet(ambient_dim, args.hid_dims, args.out_dims, args.num_stages, kaiming_init=True).cuda()
        optimizer = optim.Adam(multisenet.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=args.lr_min)

        n_iters = 0  # モデル学習ステップカウント
        pbar = tqdm(range(n_epochs), ncols=120)  # プログレスバー

        for epoch in pbar:  # エポックごとのループN=20000の時500回
            pbar.set_description(f"Epoch {epoch}")
            randidx = torch.randperm(data.shape[0])  # データをシャッフル
            reconstructed_data = torch.zeros_like(data).cuda()  # [20000, d] の空配列
            for stage in range(args.num_stages):
                if stage != 0:
                    randidx = torch.randperm(reconstructed_data.shape[0])  # データをシャッフル
                for i in range(n_iter_per_epoch):  # ミニバッチ単位のループN=20000の時200回
                    multisenet.train()  # 訓練モードに切り替え
                    if stage == 0:
                        batch_idx = randidx[i * args.batch_size : (i + 1) * args.batch_size]#100このデータのインデックス
                        #インデックスに対応したデータを取得
                        batch = data[batch_idx].cuda()  # バッチを取得しGPUへ
                        # 第1段階：元の入力を使用
                        q_batch = multisenet.query_embedding(stage,batch)  # クエリ埋め込み
                        k_batch = multisenet.key_embedding(stage,batch)  # キー埋め込み
                    else:
                        batch_idx = randidx[i * args.batch_size : (i + 1) * args.batch_size]#100このデータのインデックス
                        #インデックスに対応したデータを取得
                        batch = reconstructed_data[batch_idx].cuda()  # バッチを取得しGPUへ
                        # 第2段階以降：前段階の再構成結果を使用
                        q_batch = multisenet.query_embedding(stage,batch)
                        k_batch = multisenet.key_embedding(stage,batch)  # 図では x̂_j も再構成結果

                    rec_batch = torch.zeros_like(batch).cuda()  # 再構成バッチ初期化
                    reg = torch.zeros([1]).cuda()  # 正則化項初期化
                    
                    for j in range(n_step_per_iter):  # 再構成のために全体を分割N=20000の時2回
                        if stage == 0:
                            #block_size=10000
                            block = data[j * block_size: (j + 1) * block_size].cuda()#メモリを節約しながらすべてのサンプルに対して
                            block_cloned = block.clone()
                            k_block = multisenet.key_embedding(stage,block)  # 各ブロックのキー
                            c = multisenet.get_coeff(q_batch, k_block, stage)  # 自己表現係数
                            # 再構成（図のx̂_i = Σ c_ij * x_j）
                            x_reconstructed = c.mm(block_cloned)  # 第1段階は元のkeysを使用
                        else:
                            #block_size=10000
                            block = reconstructed_data[j * block_size: (j + 1) * block_size].cuda()#メモリを節約しながらすべてのサンプルに対して
                            block_cloned = block.clone()
                            k_block = multisenet.key_embedding(stage,block)  # 各ブロックのキー
                            c = multisenet.get_coeff(q_batch, k_block, stage)  # 自己表現係数
                            # 再構成（図のx̂_i = Σ c_ij * x_j）
                            x_reconstructed = c.mm(block_cloned)  # 第2段階以降は前段階の結果を使用
                        rec_batch = rec_batch + x_reconstructed  # 再構成
                        reg = reg + regularizer(c, args.lmbd)  # 正則化項を加算

                    # 対角成分（自己ループ）を補正
                    diag_c = multisenet.thres_list[stage]((q_batch * k_batch).sum(dim=1, keepdim=True)) * multisenet.shrink
                    rec_batch = rec_batch - diag_c * batch
                    reg = reg - regularizer(diag_c, args.lmbd)
                    # 損失計算（再構成誤差＋正則化）
                    rec_loss = torch.sum(torch.pow(batch - rec_batch, 2))
                    loss = (0.5 * args.gamma * rec_loss + reg) / args.batch_size
                    # 追加: 学習過程のデータを記録
                    if n_iters % 1000 == 0:  # 100イテレーションごとに記録
                        learning_data['Iteration'].append(n_iters)
                        learning_data['Loss'].append(loss.item())
                    # 勾配更新処理
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(multisenet.parameters(), 0.001)  # 勾配のノルムを制限
                    optimizer.step()

                    global_steps += 1
                    n_iters += 1
                    temp_reconstructed_data = reconstructed_data.clone() # まず現在の全体をクローン
                    temp_reconstructed_data[batch_idx] = rec_batch.detach() # クローンした部分を更新
                    reconstructed_data = temp_reconstructed_data
                    # 一定間隔でモデル保存
                    if n_iters % args.save_iters == 0:
                        with open('{}/SENet_{}_N{:d}_iter{:d}.pth.tar'.format(folder, args.dataset, N, n_iters), 'wb') as f:
                            torch.save(multisenet.state_dict(), f)
                        print("Model Saved.")

                    # 一定間隔で評価を実行
                    if n_iters % args.eval_iters == 0:
                        print("Evaluating on sampled data...")
                        acc, nmi, ari = evaluate(multisenet, data=data,num_stage=args.num_stages, labels=labels, num_subspaces=args.num_subspaces, affinity=args.affinity,
                                                spectral_dim=args.spectral_dim, non_zeros=args.non_zeros, n_neighbors=args.n_neighbors,
                                                batch_size=block_size, chunk_size=block_size,
                                                knn_mode='symmetric')
                        print("ACC-{:.6f}, NMI-{:.6f}, ARI-{:.6f}".format(acc, nmi, ari))
                        # 追加: 評価結果を記録
                        learning_data['Accuracy'].append(acc)
            # プログレスバーに損失を表示
            pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
                             rec_loss="{:3.4f}".format(rec_loss.item() / args.batch_size),
                             reg="{:3.4f}".format(reg.item() / args.batch_size))
            scheduler.step()  # 学習率スケジューラ更新

        # 全体データに対して最終評価
        print("Evaluating on {}-full...".format(args.dataset))
        full_data = torch.from_numpy(full_samples).float()
        full_data = utils.p_normalize(full_data)
        acc, nmi, ari = evaluate(multisenet, data=full_data,num_stage=args.num_stages, labels=full_labels, num_subspaces=args.num_subspaces, affinity=args.affinity,
                                spectral_dim=args.spectral_dim, non_zeros=args.non_zeros, n_neighbors=args.n_neighbors, batch_size=args.chunk_size,
                                chunk_size=args.chunk_size, knn_mode='symmetric')
        print("N-{:d}: ACC-{:.6f}, NMI-{:.6f}, ARI-{:.6f}".format(N, acc, nmi, ari))
        writer.writerow([N, acc, nmi, ari])  # CSVに記録
        result.flush()
        # 学習曲線データのDataFrameを作成して保存
        df = pd.DataFrame(learning_data)
        df.to_csv(f'{folder}/learning_data_{args.dataset}_{N}.csv', index=False)
        # 学習曲線をプロット
        plot_learning_curve(df, f'{folder}/learning_curve_{args.dataset}_{N}.png')

        # モデル保存（最終）
        with open('{}/SENet_{}_N{:d}.pth.tar'.format(folder, args.dataset, N), 'wb') as f:
            torch.save(multisenet.state_dict(), f)

        torch.cuda.empty_cache()  # GPUメモリを解放
    result.close()  # 結果ファイルを閉じる
