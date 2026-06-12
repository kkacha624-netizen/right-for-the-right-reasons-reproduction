# Right for the Right Reasons 再現実験計画書

## 1. 目的

本ドキュメントは、以下の論文に含まれる実験を完全再現するための計画を整理するものである。

**Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations**
Andrew Slavin Ross, Michael C. Hughes, Finale Doshi-Velez
IJCAI 2017

本リポジトリでは、論文で提案された **Right for the Right Reasons, RRR** の手法を実装し、論文内の主要実験を可能な限り忠実に再現する。

---

## 2. 論文の概要

通常の教師あり学習では、モデルの予測が正解しているかどうかを中心に学習を行う。
しかし、モデルが正しい予測をしていても、その判断根拠が不適切である場合がある。

本論文では、モデルが「正しい理由」に基づいて予測するように、入力特徴量に対する予測出力の勾配を制約する。

具体的には、以下のような目的関数を用いる。

```text
total loss = classification loss + explanation regularization
```

ここで、`explanation regularization` は、モデルが使ってほしくない特徴量に対して大きな勾配を持たないようにするための正則化項である。

---

## 3. 再現の方針

本再現実験では、以下を目標とする。

* 論文中の実験をすべて再現する
* 論文中の表・図に対応する結果を再作成する
* 通常学習モデルと RRR モデルを比較する
* 入力勾配に基づく説明を可視化する
* 論文結果との差分を記録する
* 再現できた点・再現できなかった点を明確にする

完全再現を目指すが、論文中に明記されていない設定がある場合は、以下のように扱う。

* 論文本文・補足資料・著者実装を確認する
* それでも不明な場合は、再現時に採用した設定を明記する
* 論文結果との差分が出た場合は、その原因候補を記録する

---

## 4. 再現対象の実験

本リポジトリでは、以下の実験を再現対象とする。

| 実験ID    | 実験名                                     | 目的                          | 状態  |
| ------- | --------------------------------------- | --------------------------- | --- |
| EXP-001 | Synthetic dataset                       | 人工データで RRR の効果を確認する         | 未着手 |
| EXP-002 | Decoy MNIST                             | 不適切な特徴への依存を RRR で抑制できるか確認する | 未着手 |
| EXP-003 | 20 Newsgroups                           | テキスト分類における説明制約の効果を確認する      | 未着手 |
| EXP-004 | Diabetes dataset                        | 医療系データにおける説明制約の効果を確認する      | 未着手 |
| EXP-005 | Unsupervised explanation regularization | 教師なし的な説明制約による多様なモデル学習を確認する  | 未着手 |
| EXP-006 | Explanation visualization               | 入力勾配に基づく説明を可視化する            | 未着手 |

---

## 5. 共通実装方針

### 5.1 モデル

論文で使用されているモデル構造を確認し、可能な限り同じ構造を実装する。

実装予定ファイル：

```text
src/rrr/models.py
```

実装予定クラス：

```text
LinearClassifier
MLPClassifier
TextClassifier
```

確認項目：

* 各実験で使用されたモデル構造
* 隠れ層の数
* 隠れユニット数
* 活性化関数
* Dropout の有無
* 正則化の有無
* 最適化手法
* 学習率
* エポック数
* バッチサイズ

---

### 5.2 損失関数

RRR の損失関数を実装する。

実装予定ファイル：

```text
src/rrr/losses.py
```

実装する損失：

```text
classification loss
gradient regularization loss
total RRR loss
```

確認項目：

* 分類損失の種類
* 勾配正則化項の定義
* 正則化係数 lambda の値
* マスク行列 A の扱い
* L1 / L2 などのノルム設定
* バッチ処理時の正則化項の平均方法

---

### 5.3 入力勾配の計算

モデル出力の入力特徴量に対する勾配を計算する。

実装予定ファイル：

```text
src/rrr/gradients.py
```

実装する処理：

```text
compute_input_gradients
compute_explanation_penalty
apply_feature_mask
```

確認項目：

* どの出力に対して勾配を取るか
* 正解クラスの logit に対する勾配か
* 予測クラスの logit に対する勾配か
* 損失に対する勾配か
* softmax 前の値を使うか
* softmax 後の値を使うか

---

### 5.4 評価指標

各実験で以下の指標を記録する。

```text
accuracy
loss
classification loss
explanation regularization loss
test accuracy
validation accuracy
```

必要に応じて以下も記録する。

```text
precision
recall
F1 score
AUC
gradient norm
masked gradient norm
```

---

### 5.5 乱数シード

再現性確保のため、すべての実験で乱数シードを固定する。

記録対象：

```text
Python random seed
NumPy random seed
PyTorch random seed
CUDA random seed
```

複数 seed で実験する場合は、論文の設定に従う。

---

## 6. EXP-001: Synthetic dataset

### 6.1 目的

人工データセットを用いて、通常の学習ではモデルが不適切な特徴に依存する可能性があることを確認する。
その上で、RRR によって不適切な特徴への依存を抑制できるかを検証する。

---

### 6.2 実装予定ファイル

```text
src/datasets/synthetic.py
scripts/run_synthetic.py
configs/synthetic/baseline.yaml
configs/synthetic/rrr.yaml
```

---

### 6.3 実験条件

確認・実装する項目：

* データ生成方法
* 入力次元数
* サンプル数
* 学習データ数
* 検証データ数
* テストデータ数
* ラベル生成規則
* 不適切な特徴の作り方
* explanation annotation の作り方
* train / validation / test の分割方法

---

### 6.4 比較手法

以下を比較する。

```text
Baseline model
RRR model
```

必要に応じて以下も比較する。

```text
モデル構造違い
正則化係数違い
アノテーション量違い
```

---

### 6.5 保存する結果

```text
experiments/synthetic/<experiment_id>/config.yaml
experiments/synthetic/<experiment_id>/metrics.json
experiments/synthetic/<experiment_id>/notes.md
results/figures/synthetic/
results/tables/synthetic_results.csv
```

---

### 6.6 完了条件

* Synthetic dataset を再現できている
* Baseline と RRR の両方を実行できる
* 論文中の対応する結果と比較できる
* 入力勾配の違いを可視化できる
* 結果の差分を notes.md に記録している

---

## 7. EXP-002: Decoy MNIST

### 7.1 目的

MNIST に人工的な decoy feature を加え、通常のモデルが本来の数字領域ではなく decoy feature に依存するかを確認する。
RRR を用いることで、decoy feature への依存を抑制し、より適切な特徴に基づいて分類できるかを検証する。

---

### 7.2 実装予定ファイル

```text
src/datasets/decoy_mnist.py
scripts/run_decoy_mnist.py
configs/decoy_mnist/baseline.yaml
configs/decoy_mnist/rrr.yaml
```

---

### 7.3 実験条件

確認・実装する項目：

* MNIST の取得方法
* decoy feature の付与方法
* decoy feature の位置
* decoy feature とラベルの対応関係
* train / validation / test の分割方法
* 学習時とテスト時で decoy feature の分布を変えるか
* explanation annotation の作り方
* 画像中のどの領域をマスクするか

---

### 7.4 比較手法

以下を比較する。

```text
Baseline model
RRR model
```

追加で確認する可能性があるもの：

```text
decoy feature あり学習・ありテスト
decoy feature あり学習・なしテスト
decoy feature あり学習・decoy 変化テスト
```

---

### 7.5 可視化

以下を可視化する。

```text
入力画像
decoy feature
Baseline の入力勾配
RRR の入力勾配
マスク領域
```

保存先：

```text
results/figures/decoy_mnist/
```

---

### 7.6 保存する結果

```text
experiments/decoy_mnist/<experiment_id>/config.yaml
experiments/decoy_mnist/<experiment_id>/metrics.json
experiments/decoy_mnist/<experiment_id>/notes.md
results/figures/decoy_mnist/
results/tables/decoy_mnist_results.csv
```

---

### 7.7 完了条件

* Decoy MNIST を生成できる
* Baseline と RRR の両方を学習できる
* decoy feature への依存の違いを確認できる
* 論文中の結果と比較できる
* 入力勾配の可視化結果を保存できる

---

## 8. EXP-003: 20 Newsgroups

### 8.1 目的

20 Newsgroups を用いたテキスト分類において、モデルが不適切な単語特徴に依存するかを確認する。
RRR により、指定された単語特徴への依存を抑制できるかを検証する。

---

### 8.2 実装予定ファイル

```text
src/datasets/newsgroups.py
scripts/run_newsgroups.py
configs/newsgroups/baseline.yaml
configs/newsgroups/rrr.yaml
```

---

### 8.3 実験条件

確認・実装する項目：

* 使用するカテゴリ
* 文書の前処理方法
* tokenization
* stop words の扱い
* TF-IDF / bag-of-words のどちらを使うか
* 語彙数
* 特徴量の正規化
* train / validation / test の分割方法
* explanation annotation の作り方
* 制約対象となる単語特徴

---

### 8.4 比較手法

以下を比較する。

```text
Baseline model
RRR model
```

必要に応じて以下も比較する。

```text
異なる正則化係数
異なるアノテーション量
異なる語彙サイズ
```

---

### 8.5 保存する結果

```text
experiments/newsgroups/<experiment_id>/config.yaml
experiments/newsgroups/<experiment_id>/metrics.json
experiments/newsgroups/<experiment_id>/notes.md
results/tables/newsgroups_results.csv
results/figures/newsgroups/
```

---

### 8.6 完了条件

* 20 Newsgroups の前処理を再現できる
* Baseline と RRR の両方を学習できる
* 指定単語への依存度の違いを確認できる
* 論文中の結果と比較できる
* 結果を表として保存できる

---

## 9. EXP-004: Diabetes dataset

### 9.1 目的

Diabetes dataset を用いて、医療系データにおける説明制約の効果を検証する。
モデルが指定された特徴量に不適切に依存しないよう、RRR による勾配制約を適用する。

---

### 9.2 実装予定ファイル

```text
src/datasets/diabetes.py
scripts/run_diabetes.py
configs/diabetes/baseline.yaml
configs/diabetes/rrr.yaml
```

---

### 9.3 実験条件

確認・実装する項目：

* 使用する Diabetes dataset の取得元
* タスク設定
* 入力特徴量
* 目的変数
* 前処理方法
* 欠損値処理
* 標準化の有無
* train / validation / test の分割方法
* explanation annotation の作り方
* 制約対象の特徴量

---

### 9.4 比較手法

以下を比較する。

```text
Baseline model
RRR model
```

必要に応じて以下も比較する。

```text
正則化係数違い
特徴制約の有無
一部特徴のみ制約
```

---

### 9.5 保存する結果

```text
experiments/diabetes/<experiment_id>/config.yaml
experiments/diabetes/<experiment_id>/metrics.json
experiments/diabetes/<experiment_id>/notes.md
results/tables/diabetes_results.csv
results/figures/diabetes/
```

---

### 9.6 完了条件

* Diabetes dataset を取得・前処理できる
* Baseline と RRR の両方を学習できる
* 指定特徴量への依存度を比較できる
* 論文中の結果と比較できる
* 再現差分を記録できる

---

## 10. EXP-005: Unsupervised explanation regularization

### 10.1 目的

教師ありの explanation annotation がない場合でも、異なる説明を持つ複数のモデルを学習できるかを確認する。

---

### 10.2 実装予定ファイル

```text
src/rrr/unsupervised.py
scripts/run_unsupervised.py
configs/unsupervised/
```

---

### 10.3 実験条件

確認・実装する項目：

* 複数モデルの学習方法
* モデル間で異なる決定境界を促す正則化項
* 比較対象のモデル数
* 使用するデータセット
* 評価指標
* 可視化方法

---

### 10.4 保存する結果

```text
experiments/unsupervised/<experiment_id>/config.yaml
experiments/unsupervised/<experiment_id>/metrics.json
experiments/unsupervised/<experiment_id>/notes.md
results/figures/unsupervised/
results/tables/unsupervised_results.csv
```

---

### 10.5 完了条件

* 教師なし explanation regularization を実装できる
* 複数モデルの説明の違いを確認できる
* 論文中の結果と比較できる
* 可視化結果を保存できる

---

## 11. EXP-006: Explanation visualization

### 11.1 目的

入力勾配に基づく説明を可視化し、Baseline と RRR の判断根拠の違いを確認する。

---

### 11.2 実装予定ファイル

```text
src/rrr/visualize.py
scripts/visualize_explanations.py
```

---

### 11.3 可視化対象

以下を可視化する。

```text
Synthetic dataset の入力勾配
Decoy MNIST の入力勾配
20 Newsgroups の重要単語
Diabetes dataset の特徴量重要度
```

---

### 11.4 保存する結果

```text
results/figures/explanations/
```

---

### 11.5 完了条件

* 入力勾配を計算できる
* 勾配を図として保存できる
* Baseline と RRR の違いを比較できる
* 論文中の可視化結果と比較できる

---

## 12. 実験IDの命名規則

各実験は以下の形式で管理する。

```text
exp<番号>_<dataset>_<method>_<補足>
```

例：

```text
exp001_synthetic_baseline
exp002_synthetic_rrr_lambda_1e-2
exp003_decoy_mnist_baseline
exp004_decoy_mnist_rrr_lambda_1e-2
```

各実験ディレクトリには以下を保存する。

```text
config.yaml
metrics.json
notes.md
```

必要に応じて以下も保存する。

```text
figures/
logs/
```

---

## 13. 設定ファイルの形式

各実験では `configs/` 以下の YAML ファイルで設定を管理する。

例：

```yaml
experiment:
  name: exp001_decoy_mnist_rrr
  seed: 0

dataset:
  name: decoy_mnist
  data_dir: data/processed/decoy_mnist
  batch_size: 64

model:
  name: mlp
  hidden_dim: 256
  activation: relu

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 0.0

rrr:
  enabled: true
  lambda: 1.0
  mask_type: annotation

output:
  experiment_dir: experiments/decoy_mnist/exp001_decoy_mnist_rrr
```

---

## 14. 結果ファイルの形式

各実験の `metrics.json` は以下の形式にする。

```json
{
  "experiment_id": "exp001_decoy_mnist_rrr",
  "dataset": "decoy_mnist",
  "method": "rrr",
  "seed": 0,
  "accuracy": null,
  "loss": null,
  "classification_loss": null,
  "explanation_regularization_loss": null,
  "paper_result": null,
  "reproduced_result": null,
  "difference": null,
  "notes": ""
}
```

---

## 15. 論文結果との比較表

最終的に以下の比較表を作成する。

保存先：

```text
results/tables/reproduction_summary.csv
```

列：

```text
experiment_id
dataset
method
metric
paper_result
reproduced_result
difference
status
notes
```

`status` は以下のいずれかにする。

```text
reproduced
partially_reproduced
not_reproduced
unknown
```

---

## 16. 実装タスク一覧

### 16.1 ドキュメント

* [ ] `docs/paper_summary.md` を作成する
* [ ] `docs/reproduction_plan.md` を作成する
* [ ] `docs/experiment_checklist.md` を作成する
* [ ] `docs/implementation_notes.md` を作成する

---

### 16.2 データセット

* [ ] Synthetic dataset を実装する
* [ ] Decoy MNIST を実装する
* [ ] 20 Newsgroups の前処理を実装する
* [ ] Diabetes dataset の前処理を実装する
* [ ] explanation annotation / mask を作成する

---

### 16.3 モデル

* [ ] Baseline model を実装する
* [ ] RRR model を実装する
* [ ] テキスト分類用モデルを実装する
* [ ] 医療データ用モデルを実装する

---

### 16.4 損失関数

* [ ] classification loss を実装する
* [ ] 入力勾配計算を実装する
* [ ] explanation regularization loss を実装する
* [ ] total loss を実装する
* [ ] 教師なし explanation regularization を実装する

---

### 16.5 学習・評価

* [ ] 学習ループを実装する
* [ ] 評価ループを実装する
* [ ] metrics 保存処理を実装する
* [ ] 実験ログ保存処理を実装する
* [ ] 全実験一括実行スクリプトを実装する

---

### 16.6 可視化

* [ ] 入力勾配の可視化を実装する
* [ ] Decoy MNIST の可視化を実装する
* [ ] テキスト特徴量の可視化を実装する
* [ ] 結果比較グラフを作成する

---

## 17. 再現完了条件

本再現実験は、以下を満たした場合に完了とする。

* [ ] 論文中の主要実験がすべて実装されている
* [ ] 各実験で Baseline と RRR の比較ができる
* [ ] 論文中の表・図に対応する結果を作成できる
* [ ] 各実験の設定ファイルが保存されている
* [ ] 各実験の結果が `metrics.json` として保存されている
* [ ] 論文結果との比較表が作成されている
* [ ] 再現できた点とできなかった点が記録されている
* [ ] 実行手順が README に記載されている

---

## 18. 不明点・確認事項

以下は実装前に確認する。

* [ ] 各実験で使用された正確なモデル構造
* [ ] 各実験のハイパーパラメータ
* [ ] 各データセットの正確な前処理
* [ ] explanation annotation の具体的な作成方法
* [ ] Decoy MNIST の decoy feature の作り方
* [ ] 20 Newsgroups で制約対象とした単語
* [ ] Diabetes dataset の取得元
* [ ] 教師なし explanation regularization の正確な実装式
* [ ] 論文中の表・図と実験IDの対応関係
* [ ] 著者実装の有無

---

## 19. 想定されるリポジトリ内ファイル

```text
docs/
├─ paper_summary.md
├─ reproduction_plan.md
├─ experiment_checklist.md
└─ implementation_notes.md

src/
├─ rrr/
│  ├─ losses.py
│  ├─ gradients.py
│  ├─ models.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ visualize.py
└─ datasets/
   ├─ synthetic.py
   ├─ decoy_mnist.py
   ├─ newsgroups.py
   └─ diabetes.py

configs/
├─ synthetic/
├─ decoy_mnist/
├─ newsgroups/
├─ diabetes/
└─ unsupervised/

scripts/
├─ prepare_data.py
├─ run_synthetic.py
├─ run_decoy_mnist.py
├─ run_newsgroups.py
├─ run_diabetes.py
├─ run_unsupervised.py
├─ visualize_explanations.py
└─ reproduce_all.py

experiments/
├─ synthetic/
├─ decoy_mnist/
├─ newsgroups/
├─ diabetes/
└─ unsupervised/

results/
├─ figures/
├─ tables/
└─ logs/
```

---

## 20. 備考

本ドキュメントは、実装開始前の計画書である。
実装を進める中で、論文本文・補足資料・著者実装を確認し、必要に応じて更新する。

特に、論文中に明記されていないハイパーパラメータや前処理については、採用した設定とその理由を必ず記録する。
