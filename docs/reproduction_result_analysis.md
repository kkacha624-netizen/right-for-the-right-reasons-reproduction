# 再現実験結果の分析

このメモは、Colab で実行したフル実験結果 `results (1).zip` をもとに、Ross, Hughes, Doshi-Velez の論文 **Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations** の各実験と比較したものである。

今回の実行では `device: cuda` が使われており、`results/tables/reproduction_summary.csv` と各実験ディレクトリ内の `metrics.json` の値は一致している。

## 全体評価

| 実験 | 対応する成果物 | 状態 | 短評 |
| --- | --- | --- | --- |
| Toy Color / Synthetic | `experiments/toy_color/metrics.json` | 部分的に再現 | 実験パイプラインは動作し、RRR も高精度を維持している。ただし、論文中の可視化やハイパーパラメータ探索までは完全には再現していない。 |
| Iris-Cancer | `experiments/iris_cancer/metrics.json` | 部分的に再現 | 精度は高いが、論文で見られる「Iris 特徴を除くと baseline が劣化する」挙動は出ていない。現在の分割では cancer 特徴だけで分類できている可能性がある。 |
| 20 Newsgroups | `experiments/newsgroups/metrics.json` | 部分的に再現 | 実験は正常に動作しているが、論文メモにある baseline 約 94% より低い。前処理や特徴量化の違いが大きい可能性がある。 |
| Decoy MNIST | `experiments/decoy_mnist/metrics.json` | 再現成功 | 通常モデルは decoy に依存し、RRR は decoy がランダム化されても高精度を維持している。論文の中心的な主張とよく一致している。 |
| Find Another Explanation | `experiments/find_another_explanation/metrics.json` | 探索的再現 | 反復的に特徴をマスクする処理は動いているが、最終 round で精度が大きく低下している。論文レベルの再現には追加調整が必要。 |
| Explanation Visualization | `results/figures/explanations/toy_color_gradients.png` | 診断用可視化として再現 | 入力、baseline 勾配、RRR 勾配、annotation mask を確認できる。全論文図に対応する可視化までは未実装。 |

## 結果一覧

| 実験 | 手法 | 評価内容 | 結果 |
| --- | --- | --- | ---: |
| Toy Color | Baseline | Test accuracy | 0.9945 |
| Toy Color | RRR | Test accuracy | 0.9755 |
| Iris-Cancer | Baseline | Test accuracy | 0.9667 |
| Iris-Cancer | RRR | Test accuracy | 0.9333 |
| Iris-Cancer | Baseline | Iris 特徴を除いた test accuracy | 0.9667 |
| Iris-Cancer | RRR | Iris 特徴を除いた test accuracy | 0.9333 |
| 20 Newsgroups | Baseline | Test accuracy | 0.8061 |
| 20 Newsgroups | RRR | Test accuracy | 0.7936 |
| Decoy MNIST | Baseline | swatch が相関した test accuracy | 0.9977 |
| Decoy MNIST | RRR | swatch が相関した test accuracy | 0.9900 |
| Decoy MNIST | Baseline | swatch をランダム化した test accuracy | 0.5809 |
| Decoy MNIST | RRR | swatch をランダム化した test accuracy | 0.9900 |
| Find Another Explanation | Iterative RRR | Round 0 accuracy | 0.9915 |
| Find Another Explanation | Iterative RRR | Round 1 accuracy | 0.9335 |
| Find Another Explanation | Iterative RRR | Round 2 accuracy | 0.4890 |
| Explanation Visualization | Baseline | Test accuracy | 0.9930 |
| Explanation Visualization | RRR | Test accuracy | 0.9695 |

## 実験別の分析

### Toy Color / Synthetic

論文側の実験設定:

- Toy Color dataset は `5 x 5 x 3` の RGB 入力を使う。
- ラベルに関係する特徴は、画像中の特定位置に集中している。
- RRR により、不要な特徴に対する入力勾配が抑制されるかを確認する。

今回の結果:

- Baseline accuracy: `0.9945`
- RRR accuracy: `0.9755`
- RRR は baseline より少し精度が下がるが、高精度を維持している。
- RRR の explanation regularization loss は `3.7657e-06` で、勾配制約が適用されている。

分析:

- Toy Color の基本的な RRR 実験パイプラインは動作している。
- 勾配制約を加えても分類性能が大きく崩れていない点は、論文の主張と整合している。
- ただし、論文中の図や複数条件の比較を完全に再現したわけではないため、現段階では部分的な再現と判断する。

### Iris-Cancer

論文側の実験設定:

- Iris dataset と Breast Cancer Wisconsin dataset を結合したデータセットを使う。
- 論文では、baseline model が Iris 特徴に依存し、test 時に Iris 特徴を除くと精度が低下することを示している。
- 実装メモでは、通常 test accuracy は平均約 `92%`、Iris の 4 次元を除くと平均約 `81%` と記録されている。
- RRR では Iris 次元への依存を抑制し、Iris 特徴の有無による精度差が小さくなることが期待される。

今回の結果:

- Baseline accuracy: `0.9667`
- Baseline without Iris: `0.9667`
- RRR accuracy: `0.9333`
- RRR without Iris: `0.9333`

分析:

- 各モデルで、Iris 特徴あり・なしの精度が一致している。
- これは「Iris 特徴に依存していない」という意味では望ましいが、論文で報告されている baseline の劣化は再現できていない。
- 現在の train/test split では、cancer 特徴だけで十分に分類できている可能性がある。
- 論文により近づけるには、報告されている `350` 個のランダム train-test split を実行し、平均と標準偏差を出す必要がある。
- また、Iris と cancer のサンプル対応、クラスの対応、行の選び方が論文と一致しているかを再確認する必要がある。

### 20 Newsgroups

論文側の実験設定:

- 使用カテゴリは `alt.atheism` と `soc.religion.christian`。
- 実装メモでは、TF-IDF vectorizer、`5000` components、`A=0` で約 `94%` accuracy と記録されている。
- RRR では、指定された単語特徴への依存を抑制し、分類性能と特徴依存の変化を比較する。

今回の結果:

- Baseline accuracy: `0.8061`
- RRR accuracy: `0.7936`
- RRR は baseline よりわずかに低いが、近い精度を維持している。

分析:

- 実験自体は正常に実行できている。
- しかし、論文メモにある baseline 約 `94%` とは差が大きい。
- 原因として、headers / footers / quotes の除去、stop words の扱い、TF-IDF の設定、語彙構築、正規化、train/test split の違いが考えられる。
- 現段階では「実装確認としては成功」だが、「論文数値の再現」としては不十分である。
- 次の改善として、論文または著者実装に合わせて前処理を詰める必要がある。

### Decoy MNIST

論文側の実験設定:

- MNIST 画像のランダムな角に `4 x 4` の gray swatch を付与する。
- training では swatch の濃さが digit label の関数になっている。
- 具体的には `255 - 25y`。
- test では swatch の濃さをランダム化する。
- 通常モデルは swatch に依存してしまい、ランダム化 test で性能が落ちる。
- RRR は swatch 領域への勾配を抑制し、ランダム化 test でも高精度を維持することが期待される。

今回の結果:

- Baseline correlated-swatch accuracy: `0.9977`
- Baseline randomized-swatch accuracy: `0.5809`
- RRR correlated-swatch accuracy: `0.9900`
- RRR randomized-swatch accuracy: `0.9900`

分析:

- 今回の最も強い再現結果である。
- baseline は swatch がラベルと相関している test ではほぼ完全に分類できている。
- しかし、swatch をランダム化すると accuracy が `0.5809` まで落ちる。
- これは baseline が本来の数字ではなく、decoy swatch に強く依存していたことを示している。
- 一方 RRR は、swatch をランダム化しても `0.9900` を維持している。
- これは「正しい理由で分類する」ように学習できていることを示しており、論文の中心的な主張とよく一致している。

### Find Another Explanation

論文側の実験設定:

- `A = 0` から開始する。
- 各 round で学習したモデルの大きな勾配成分を次の `A` に追加する。
- これにより、前回とは異なる説明を持つモデルを学習する。

今回の結果:

| Round | Accuracy | 新たに mask された特徴 index |
| --- | ---: | --- |
| 0 | 0.9915 | `[2, 62, 14, 74, 12, 60, 72, 0]` |
| 1 | 0.9335 | `[61, 1, 13, 73, 8, 53, 11, 5]` |
| 2 | 0.4890 | `[58, 17, 19, 15, 20, 31, 47, 66]` |

分析:

- 反復的に mask を更新する処理は実装され、実行できている。
- ただし round 2 で accuracy が `0.4890` まで低下している。
- これは現在の top-k masking が強すぎるか、論文の thresholding / feature selection と一致していない可能性が高い。
- 現段階では探索的再現であり、論文レベルの再現とは言いにくい。
- 改善するには、mask 追加ルール、threshold、対象データセット、評価指標を論文に合わせて調整する必要がある。

### Explanation Visualization

論文側の実験設定:

- 入力勾配に基づく説明を可視化し、モデルがどの特徴に依存しているかを確認する。
- baseline と RRR の説明の違いを見ることが目的である。

今回の結果:

- Figure: `results/figures/explanations/toy_color_gradients.png`
- Baseline accuracy: `0.9930`
- RRR accuracy: `0.9695`

分析:

- 可視化画像は正常に生成されている。
- 画像には、入力、baseline gradient、RRR gradient、annotation mask が含まれている。
- 現段階では Toy Color に対する診断用可視化として有用である。
- ただし、Decoy MNIST の swatch 領域や、20 Newsgroups の重要単語、Iris-Cancer の特徴量勾配まで可視化できると、論文との対応がより明確になる。

## 結果アーカイブ内の注意点

以下の 2 つのファイルは空である。

```text
experiments/exp001_baseline/metrics.json
experiments/exp002_gradcam/metrics.json
```

これは以前の雛形ファイルが zip に混ざっているだけであり、今回の分析では使用していない。

有効な結果は以下に保存されている。

```text
experiments/toy_color/
experiments/iris_cancer/
experiments/newsgroups/
experiments/decoy_mnist/
experiments/find_another_explanation/
experiments/explanation_visualization/
```

## 今後の改善点

1. Decoy MNIST は現在の最も強い再現結果として採用する。
2. Iris-Cancer は論文通り `350` 個の random split を実行し、平均と標準偏差を報告する。
3. 20 Newsgroups は前処理、TF-IDF、語彙、annotation words を論文または著者実装に合わせる。
4. Decoy MNIST の入力勾配可視化を追加し、swatch 領域への依存が RRR で抑制されていることを図で示す。
5. `reproduce_all.py` または zip 作成前処理で、古い空の実験ディレクトリを除外する。
