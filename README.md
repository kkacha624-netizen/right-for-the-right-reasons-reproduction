# Right for the Right Reasons 再現実験

## 概要

このリポジトリは、以下の論文に含まれる実験を完全再現することを目的とする。

**Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations**
Andrew Slavin Ross, Michael C. Hughes, Finale Doshi-Velez
IJCAI 2017

本研究では、分類精度だけでなく、モデルが「正しい理由」に基づいて予測しているかを考慮する。
具体的には、入力特徴量に対する予測出力の勾配を説明として用い、人間が指定した重要でない特徴に対する勾配を抑制することで、モデルの判断根拠を制約する。

本リポジトリでは、論文内で報告されている実験設定・評価指標・比較手法を可能な限り忠実に再現する。

---

## 再現対象

本リポジトリでは、論文中の以下の実験を再現対象とする。

* Synthetic dataset における実験
* Decoy MNIST における実験
* 20 Newsgroups における実験
* Diabetes dataset における実験
* 入力勾配に基づく説明の可視化
* Explanation regularization の有無による性能比較
* 論文中の表・図に対応する結果の再現

再現対象の詳細は `docs/reproduction_plan.md` に記録する。

---

## 手法の概要

通常の教師あり学習では、モデルは以下のような分類損失を最小化する。

```text
classification loss
```

Right for the Right Reasons では、これに加えて、入力勾配に対する正則化項を導入する。

```text
total loss = classification loss + explanation regularization
```

ここで、入力勾配は「各入力特徴量が予測にどの程度影響しているか」を表す。
人間が「この特徴は判断根拠にしてほしくない」と指定した特徴に対して、勾配が大きくならないように制約する。

---

## リポジトリ構成

```text
right-for-the-right-reasons-reproduction/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ uv.lock
│
├─ docs/
│  ├─ paper_summary.md
│  ├─ reproduction_plan.md
│  ├─ experiment_checklist.md
│  └─ implementation_notes.md
│
├─ src/
│  ├─ rrr/
│  │  ├─ __init__.py
│  │  ├─ losses.py
│  │  ├─ gradients.py
│  │  ├─ models.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ visualize.py
│  │
│  └─ datasets/
│     ├─ __init__.py
│     ├─ synthetic.py
│     ├─ decoy_mnist.py
│     ├─ newsgroups.py
│     └─ diabetes.py
│
├─ configs/
│  ├─ synthetic/
│  ├─ decoy_mnist/
│  ├─ newsgroups/
│  └─ diabetes/
│
├─ scripts/
│  ├─ prepare_data.py
│  ├─ run_synthetic.py
│  ├─ run_decoy_mnist.py
│  ├─ run_newsgroups.py
│  ├─ run_diabetes.py
│  └─ reproduce_all.py
│
├─ experiments/
│  ├─ synthetic/
│  ├─ decoy_mnist/
│  ├─ newsgroups/
│  └─ diabetes/
│
├─ results/
│  ├─ figures/
│  ├─ tables/
│  └─ logs/
│
├─ notebooks/
│  └─ analysis.ipynb
│
└─ tests/
   ├─ test_losses.py
   ├─ test_gradients.py
   └─ test_datasets.py
```

---

## 環境

使用環境は以下の通り。

```text
OS: Windows 11 / Ubuntu
Python: 3.12
Package manager: uv
```

主要ライブラリは以下を使用する予定である。

```text
numpy
pandas
scikit-learn
matplotlib
torch
torchvision
tqdm
pyyaml
```

正確な依存関係は `pyproject.toml` および `uv.lock` に記録する。

---

## 環境構築

### 1. リポジトリをクローン

```bash
git clone https://github.com/<your-username>/right-for-the-right-reasons-reproduction.git
cd right-for-the-right-reasons-reproduction
```

### 2. 仮想環境を作成

```bash
uv venv
```

### 3. 仮想環境を有効化

Windows PowerShell の場合：

```powershell
.venv\Scripts\Activate.ps1
```

コマンドプロンプトの場合：

```cmd
.venv\Scripts\activate.bat
```

Linux / macOS の場合：

```bash
source .venv/bin/activate
```

### 4. 依存関係をインストール

```bash
uv sync
```

---

## データセット準備

データセットは `data/` ディレクトリに配置する。

```text
data/
├─ raw/
├─ processed/
└─ external/
```

`data/` ディレクトリはファイルサイズが大きくなる可能性があるため、Git 管理には含めない。

データセットの準備は以下のコマンドで行う。

```bash
python scripts/prepare_data.py
```

各データセットの作成・前処理手順は `docs/reproduction_plan.md` に記録する。

---

## 実験一覧

### Experiment 1: Synthetic dataset

合成データセットを用いて、通常のモデルと Explanation Regularization を加えたモデルを比較する。

```bash
python scripts/run_synthetic.py
```

主な確認項目：

* 通常学習モデルの精度
* RRR 正則化ありモデルの精度
* 入力勾配に基づく説明の違い
* 論文中の結果との比較

---

### Experiment 2: Decoy MNIST

MNIST 画像に人工的な Decoy feature を加えたデータセットを用いる。
モデルが本来の数字領域ではなく、Decoy feature に依存して分類してしまうかを確認する。

```bash
python scripts/run_decoy_mnist.py
```

主な確認項目：

* Decoy feature に依存した通常モデルの挙動
* RRR による Decoy feature への依存抑制
* テスト環境が変化した場合の汎化性能
* 入力勾配の可視化

---

### Experiment 3: 20 Newsgroups

テキスト分類データセットを用いて、特定の単語や特徴に対する依存を抑制する実験を行う。

```bash
python scripts/run_newsgroups.py
```

主な確認項目：

* 通常モデルと RRR モデルの分類性能
* 説明制約の有無による汎化性能の違い
* モデルが重視した単語特徴の比較

---

### Experiment 4: Diabetes dataset

医療系データセットを用いて、説明制約がモデルの判断根拠に与える影響を確認する。

```bash
python scripts/run_diabetes.py
```

主な確認項目：

* 通常モデルと RRR モデルの性能比較
* 人間が指定した特徴制約の影響
* 入力勾配に基づく説明の比較

---

## 全実験の実行

すべての実験を一括で実行する場合は、以下を実行する。

```bash
python scripts/reproduce_all.py
```

実験結果は以下に保存する。

```text
results/
├─ figures/
├─ tables/
└─ logs/
```

---

## 結果の保存形式

各実験の結果は、以下の形式で保存する。

```text
experiments/
└─ <dataset_name>/
   └─ <experiment_id>/
      ├─ config.yaml
      ├─ metrics.json
      ├─ notes.md
      └─ figures/
```

例：

```text
experiments/
└─ decoy_mnist/
   └─ exp001_rrr_lambda_1.0/
      ├─ config.yaml
      ├─ metrics.json
      ├─ notes.md
      └─ figures/
```

`metrics.json` には以下のような情報を保存する。

```json
{
  "dataset": "decoy_mnist",
  "method": "rrr",
  "accuracy": 0.0,
  "loss": 0.0,
  "regularization_strength": 0.0,
  "seed": 0
}
```

---

## 再現結果の比較

論文中の結果と本リポジトリで得られた結果を比較する。

比較表は以下に保存する。

```text
results/tables/reproduction_summary.csv
```

比較する項目：

* Dataset
* Model
* Regularization
* Paper result
* Reproduced result
* Difference
* Notes

---

## 実験管理方針

本リポジトリでは、実験ごとに以下を記録する。

* 実験 ID
* 使用データセット
* モデル構造
* 損失関数
* 正則化係数
* 学習率
* バッチサイズ
* エポック数
* 乱数シード
* 評価指標
* 論文結果との差分
* 実装上の注意点

実験メモは各実験ディレクトリ内の `notes.md` に記録する。

---

## Git 管理しないファイル

以下のファイル・ディレクトリは Git 管理に含めない。

```text
data/
checkpoints/
outputs/
runs/
wandb/
mlruns/
*.pth
*.pt
*.pkl
*.ckpt
*.zip
*.tar.gz
```

ただし、以下は Git 管理に含める。

```text
configs/
docs/
src/
scripts/
tests/
results/tables/
results/figures/
experiments/**/config.yaml
experiments/**/metrics.json
experiments/**/notes.md
```

---

## 再現チェックリスト

* [ ] 論文を読み、実験条件を整理する
* [ ] 再現対象の表・図を一覧化する
* [ ] Synthetic dataset を実装する
* [ ] Decoy MNIST を実装する
* [ ] 20 Newsgroups の前処理を実装する
* [ ] Diabetes dataset の前処理を実装する
* [ ] 通常モデルを実装する
* [ ] 入力勾配の計算処理を実装する
* [ ] Explanation regularization を実装する
* [ ] 各データセットで通常学習を実行する
* [ ] 各データセットで RRR 学習を実行する
* [ ] 論文中の結果と比較する
* [ ] 図表を再作成する
* [ ] 再現できた点・できなかった点を記録する

---

## 注意事項

本リポジトリは、論文内の実験を完全再現することを目標とする。
ただし、以下の要因により、論文中の数値と完全には一致しない可能性がある。

* ライブラリのバージョン差
* 乱数シードの違い
* データセット取得元の違い
* 前処理の細部の違い
* 論文中に明記されていないハイパーパラメータ
* GPU / CPU による計算差

差分が生じた場合は、`docs/implementation_notes.md` および各実験の `notes.md` に記録する。

---

## ライセンス

このリポジトリのコードは MIT License の下で公開する予定である。
ただし、使用するデータセットおよび元論文のライセンス・利用条件には従うこと。

---

## 参考文献

Andrew Slavin Ross, Michael C. Hughes, Finale Doshi-Velez.
Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations.
IJCAI 2017.
