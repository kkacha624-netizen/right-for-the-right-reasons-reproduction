## 論文本文に基づく確認メモ

`docs/reproduction_plan.md` の不明点を確認する際は、以下の論文箇所を参照する。

### 共通設定

- [ ] モデル構造を確認する
  - 根拠: Section 3 Empirical Evaluation, p.3 / IJCAI紙面 p.2664
  - 2 hidden layers
  - hidden size: 50 and 30
  - ReLU nonlinearities
  - softmax output

- [ ] 学習設定を確認する
  - 根拠: Section 3 Empirical Evaluation, p.3 / IJCAI紙面 p.2664
  - optimizer: Adam
  - batch size: 256
  - `λ2 = 0.0001`
  - 多くの実験で `λ1 = 1000`

- [ ] `λ1` の調整方針を確認する
  - 根拠: Appendix A Cross-Validation, p.8 / IJCAI紙面 p.2668
  - cross entropy と “right reasons” loss が同程度の大きさになるように調整する。

### RRR loss / annotation mask

- [ ] annotation matrix `A` の定義を確認する
  - 根拠: Section 2.1 Loss Functions that Constrain Explanations, p.2 / IJCAI紙面 p.2663
  - `A ∈ {0,1}^{N×D}` は、入力次元 `d` がサンプル `n` の予測に無関係であるべきかを表す binary mask。

- [ ] RRR loss の式を確認する
  - 根拠: Section 2.1 Loss Functions that Constrain Explanations, p.2 / IJCAI紙面 p.2663
  - loss は以下の3項からなる。
    - cross entropy
    - explanation regularization
    - weight regularization

- [ ] 正則化対象の勾配を確認する
  - 根拠: Section 2.1, p.3 / IJCAI紙面 p.2664
  - 正則化では log probability の入力勾配を penalize する。
  - 可視化では predicted probability の勾配も使われる。

### Toy Color dataset

- [ ] `Synthetic dataset` という表記を `Toy Color dataset` に修正するか検討する
  - 根拠: Section 3.1 Toy Color Dataset, p.3 / IJCAI紙面 p.2664
  - `5×5×3` RGB images
  - 4 possible colors
  - corner pixels がすべて同じ色かどうか
  - top-middle 3 pixels がすべて異なる色かどうか
  - class 1 は両方を満たし、class 2 はどちらも満たさない。

### 20 Newsgroups

- [ ] 使用カテゴリを確認する
  - 根拠: Section 3.2 Real-world Datasets, p.4 / IJCAI紙面 p.2665
  - `alt.atheism`
  - `soc.religion.christian`

- [ ] 特徴量化方法を確認する
  - 根拠: Section 3.2, p.4 / IJCAI紙面 p.2665
  - TF-IDF vectorizer
  - 5000 components
  - A=0 で 94% accuracy

### Iris-Cancer
- [ ] データ作成方法を確認する
  - 根拠: Section 3.2, p.4 / IJCAI紙面 p.2665
  - Iris dataset の class 1/2 を使用
  - Breast Cancer Wisconsin dataset の各 class 先頭50件を使用
  - `X ∈ R^{100×34}`
  - `y ∈ {0,1}`

- [ ] 評価設定を確認する
  - 根拠: Section 3.2, p.4 / IJCAI紙面 p.2665
  - 350 random train-test splits
  - 通常 test accuracy: 平均 92%
  - test set から 4 Iris components を除くと平均 81%

### Decoy MNIST

- [ ] decoy feature の作成方法を確認する
  - 根拠: Section 3.2 Real-world Datasets, p.4 / IJCAI紙面 p.2665
  - `4×4` gray swatches
  - randomly chosen corners
  - training では shade が digit label `y` の関数
  - 具体的には `255 - 25y`
  - test では shade は random

- [ ] baseline performance を確認する
  - 根拠: Section 3.2, p.4 / IJCAI紙面 p.2665
  - standard MNIST: train 98%, test 96%
  - Decoy MNIST: train 99.6%, test 55%

### explanation constraints の実験結果

- [ ] Iris-Cancer の説明制約結果を確認する
  - 根拠: Figure 8, p.6 / IJCAI紙面 p.2666
  - `A=0` では Iris dimensions への gradient が大きく、Iris を除いた test set で精度が落ちる。
  - Iris dimensions を `A=1` とすると、Iris あり/なしの test accuracy がほぼ一致する。

- [ ] Decoy MNIST の説明制約結果を確認する
  - 根拠: Figure 9, p.6 / IJCAI紙面 p.2666
  - swatch-randomized test set では `A=0` の accuracy が低い。
  - swatch 領域を `A=1` として学習すると、standard MNIST baseline に近い性能になる。

### find-another-explanation

- [ ] 反復的に `A` を更新する方法を確認する
  - 根拠: Section 2.2, p.3 / IJCAI紙面 p.2664
  - `A0 = 0` から始め、前モデルの大きな gradient components を次の `A` に加える。

- [ ] real-world dataset での結果を確認する
  - 根拠: Figure 10, p.6–7 / IJCAI紙面 p.2667
  - Iris-Cancer
  - 20 Newsgroups
  - Decoy MNIST
