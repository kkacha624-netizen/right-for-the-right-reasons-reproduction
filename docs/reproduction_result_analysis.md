# 再現実験結果の分析

このメモは、Colab で再実行した `results (2).zip` の内容をもとに、Ross, Hughes, Doshi-Velez の論文 **Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations** の各実験と比較したものである。

今回の実行では、全ての主要実験が `device: cuda` で実行され、`results/tables/reproduction_summary.csv` 上の全項目が `reproduced` 判定になっている。前回まで残っていた古い雛形ファイル `exp001_baseline` と `exp002_gradcam` は zip から除外されており、成果物の構成も整理された。

## 成果物の確認

今回の zip には以下が含まれている。

```text
results/tables/reproduction_summary.csv
results/tables/newsgroups_baseline_top_gradient_words.csv
results/tables/newsgroups_rrr_top_gradient_words.csv
results/figures/explanations/toy_color_gradients.png
results/figures/explanations/decoy_mnist/sample_00.png ... sample_04.png
results/figures/samples/toy_color/sample_00.png ... sample_04.png
results/figures/samples/decoy_mnist/sample_00.png ... sample_04.png
results/figures/samples/newsgroups/sample_00.png ... sample_04.png
results/figures/samples/iris_cancer/sample_00.png ... sample_04.png
experiments/*/metrics.json
```

古い空の雛形ファイルは含まれていない。

## 全体評価

| 実験 | 状態 | 短評 |
| --- | --- | --- |
| Toy Color / Synthetic | 再現成功 | `lambda_rrr=100.0` に調整したことで、RRR accuracy が `0.973` まで回復し、corner 勾配抑制と top-middle 勾配移行も確認できた。 |
| Iris-Cancer | 再現成功 | 350 split 集計が行われ、baseline の Iris 除去時劣化と RRR によるギャップ縮小が確認できた。 |
| 20 Newsgroups | 再現成功 | TF-IDF 5000 components、metadata を残す設定で baseline accuracy `0.9442` となり、論文メモの約 `94%` に一致した。 |
| Decoy MNIST | 再現成功 | baseline は random swatch で崩れ、RRR は高精度を維持した。swatch 勾配も大きく抑制されている。 |
| Find Another Explanation | 再現成功 | cutoff `0.67` を使い、round 0/1 は高精度、round 2 で性能低下という論文の挙動に近い結果が出た。 |
| Explanation Visualization | 再現成功 | Toy Color の勾配可視化に加え、Decoy MNIST の baseline/RRR 勾配図も保存された。 |

## 結果一覧

| 実験 | 手法 | 評価内容 | 結果 | 判定 |
| --- | --- | --- | ---: | --- |
| Toy Color | Baseline | Test accuracy | 0.9860 | reproduced |
| Toy Color | RRR | Test accuracy | 0.9730 | reproduced |
| Toy Color | Baseline | corner gradient fraction | 0.5319 | 参考 |
| Toy Color | RRR | corner gradient fraction | 0.0009 | reproduced |
| Toy Color | Baseline | top-middle gradient fraction | 0.1739 | 参考 |
| Toy Color | RRR | top-middle gradient fraction | 0.6819 | reproduced |
| Iris-Cancer | Baseline | Test accuracy | 0.9667 | reproduced |
| Iris-Cancer | RRR | Test accuracy | 0.9333 | reproduced |
| Iris-Cancer | Baseline | 350 split mean accuracy | 0.9671 ± 0.0299 | reproduced |
| Iris-Cancer | Baseline | Iris removed mean accuracy | 0.9334 ± 0.0441 | reproduced |
| Iris-Cancer | RRR | 350 split mean accuracy | 0.9328 ± 0.0475 | reproduced |
| Iris-Cancer | RRR | Iris removed mean accuracy | 0.9330 ± 0.0474 | reproduced |
| 20 Newsgroups | Baseline | Test accuracy | 0.9442 | reproduced |
| 20 Newsgroups | RRR | Test accuracy | 0.9372 | reproduced |
| Decoy MNIST | Baseline | correlated swatch accuracy | 0.9977 | reproduced |
| Decoy MNIST | RRR | correlated swatch accuracy | 0.9821 | reproduced |
| Decoy MNIST | Baseline | random swatch accuracy | 0.5809 | reproduced |
| Decoy MNIST | RRR | random swatch accuracy | 0.9823 | reproduced |
| Decoy MNIST | Baseline | swatch gradient fraction | 0.2401 | 参考 |
| Decoy MNIST | RRR | swatch gradient fraction | 0.000037 | reproduced |
| Find Another Explanation | Round 0 | Accuracy | 0.9885 | reproduced |
| Find Another Explanation | Round 1 | Accuracy | 0.9015 | reproduced |
| Find Another Explanation | Round 2 | Accuracy | 0.5125 | reproduced |
| Explanation Visualization | Baseline | Test accuracy | 0.9875 | reproduced |
| Explanation Visualization | RRR | Test accuracy | 0.9800 | reproduced |

## 実験別の分析

### Toy Color / Synthetic

論文側の狙い:

- `5 x 5 x 3` の RGB toy image を使う。
- 複数の rule で分類できる状況を作り、RRR が annotation で禁止された説明を避け、別の妥当な rule を使うかを見る。
- accuracy だけでなく、入力勾配が望ましい領域へ移ることが重要である。

今回の結果:

- Baseline accuracy: `0.9860`
- RRR accuracy: `0.9730`
- Baseline corner gradient fraction: `0.5319`
- RRR corner gradient fraction: `0.0009`
- Baseline top-middle gradient fraction: `0.1739`
- RRR top-middle gradient fraction: `0.6819`
- `lambda_rrr`: `100.0`

分析:

- 前回は `lambda_rrr=1000.0` で RRR accuracy が `0.8080` まで低下していたが、今回は `lambda_rrr=100.0` に下げたことで `0.9730` まで回復した。
- RRR は corner への勾配を `0.5319` から `0.0009` まで抑制している。
- 同時に top-middle への勾配比率は `0.1739` から `0.6819` へ増えている。
- これは「禁止された説明を避け、別の正しい説明を使う」という論文の toy experiment の主張とよく一致している。

結論:

- 再現成功。

### Iris-Cancer

論文側の狙い:

- Iris と Breast Cancer Wisconsin を結合したデータセットを使う。
- Baseline は Iris 特徴に依存しやすく、Iris 特徴を test 時に除去すると性能が落ちる。
- RRR は Iris 特徴への依存を抑え、Iris 特徴を除いても性能が変わりにくくなる。

今回の結果:

- 350 split が実行されている。
- Baseline mean accuracy: `0.9671 ± 0.0299`
- Baseline without Iris mean accuracy: `0.9334 ± 0.0441`
- RRR mean accuracy: `0.9328 ± 0.0475`
- RRR without Iris mean accuracy: `0.9330 ± 0.0474`

分析:

- Baseline は Iris 除去により `0.9671` から `0.9334` に低下している。
- RRR は Iris あり `0.9328`、Iris なし `0.9330` でほぼ一致している。
- 論文メモにある baseline の劣化幅よりは小さいが、方向性は一致している。
- RRR によって Iris 特徴への依存が抑制されるという挙動は再現できている。

結論:

- 再現成功。

### 20 Newsgroups

論文側の狙い:

- `alt.atheism` と `soc.religion.christian` の 2 クラス分類。
- TF-IDF `5000` components。
- baseline、つまり `A=0` で約 `94%` accuracy。
- RRR により、指定された単語特徴への依存を抑える。

今回の結果:

- Categories: `alt.atheism`, `soc.religion.christian`
- TF-IDF max features: `5000`
- remove_metadata: `false`
- Baseline accuracy: `0.9442`
- RRR accuracy: `0.9372`
- baseline / RRR の高勾配単語 top 50 CSV が保存されている。

baseline の高勾配語上位:

```text
clh, host, rutgers, nntp, athos, mail, atheism, christ, posting, christians, psuvm
```

RRR の高勾配語上位:

```text
clh, rutgers, christ, host, mail, geneva, athos, nntp, friend, posting, arrogance
```

分析:

- Baseline accuracy `0.9442` は論文メモの約 `94%` と非常によく一致している。
- RRR accuracy も `0.9372` で、性能低下は小さい。
- 高勾配語 CSV により、どの語が説明に強く寄与しているかを後から確認できるようになった。
- ただし、上位語には `host`, `nntp`, `posting` など metadata 由来の語も含まれている。論文メモの 94% に合わせるには metadata を残す必要があるが、説明の意味解釈ではこの点に注意が必要である。

結論:

- 分類性能の再現は成功。
- 説明語の分析も成果物として保存された。

### Decoy MNIST

論文側の狙い:

- MNIST 画像に `4 x 4` gray swatch を付ける。
- train では swatch shade が digit label の関数になっている。
- test では swatch shade をランダム化する。
- Baseline は swatch に依存して random swatch test で崩れる。
- RRR は swatch 領域への勾配を抑制し、random swatch test でも高精度を維持する。

今回の結果:

- Baseline correlated swatch accuracy: `0.9977`
- Baseline random swatch accuracy: `0.5809`
- RRR correlated swatch accuracy: `0.9821`
- RRR random swatch accuracy: `0.9823`
- Baseline swatch gradient fraction: `0.2401`
- RRR swatch gradient fraction: `0.000037`
- Decoy MNIST の baseline/RRR 勾配画像が 5 枚保存されている。

分析:

- Baseline は swatch が相関している test ではほぼ完全に分類できる。
- swatch をランダム化すると `0.5809` まで低下しており、decoy に依存していたことが明確である。
- RRR は random swatch でも `0.9823` を維持している。
- swatch gradient fraction は `0.2401` から `0.000037` まで低下しており、説明の面でも decoy 依存が抑制されている。
- 追加された勾配画像により、入力、swatch mask、baseline gradient、RRR gradient を視覚的に比較できる。

結論:

- 再現成功。
- 今回の中でも最も強い再現結果である。

### Find Another Explanation

論文側の狙い:

- `A = 0` から開始する。
- あるモデルで大きな勾配を持った特徴を、次 round の annotation mask に追加する。
- これにより、前回とは異なる説明を持つモデルを学習する。
- 論文では cutoff `c = 0.67` が使われる。

今回の結果:

| Round | Accuracy | Masked fraction | Cutoff |
| --- | ---: | ---: | ---: |
| 0 | 0.9885 | 0.0474 | 0.67 |
| 1 | 0.9015 | 0.0961 | 0.67 |
| 2 | 0.5125 | 0.1889 | 0.67 |

分析:

- `cutoff=0.67` が使われている。
- Round 0 は高精度。
- Round 1 も `0.9015` を維持しており、別の説明でも分類できている。
- Round 2 では `0.5125` まで低下している。
- Toy Color の主要 rule を順に禁止していくと、最終的に有効な説明が尽きて性能が落ちる、という論文の挙動に近い。

結論:

- 再現成功。

### Explanation Visualization

論文側の狙い:

- 入力勾配を可視化し、baseline と RRR の判断根拠の違いを見る。

今回の結果:

- Toy Color figure: `results/figures/explanations/toy_color_gradients.png`
- Decoy MNIST figures: `results/figures/explanations/decoy_mnist/sample_00.png ... sample_04.png`
- Baseline accuracy: `0.9875`
- RRR accuracy: `0.9800`

分析:

- Toy Color の可視化は正常に生成されている。
- Decoy MNIST についても、入力、swatch mask、baseline gradient、RRR gradient の比較図が 5 枚生成された。
- これにより、分類性能だけでなく説明制約の効果を画像として確認できる。

結論:

- 再現成功。

## サンプル画像と追加分析ファイル

今回の zip では、ユーザー要望に対応して各データセットのサンプル画像が保存されている。

| データセット | 保存先 | 枚数 | 内容 |
| --- | --- | ---: | --- |
| Toy Color | `results/figures/samples/toy_color/` | 5 | `5 x 5` RGB toy image |
| Decoy MNIST | `results/figures/samples/decoy_mnist/` | 5 | swatch 付き MNIST image |
| 20 Newsgroups | `results/figures/samples/newsgroups/` | 5 | 文書テキストを画像化したもの |
| Iris-Cancer | `results/figures/samples/iris_cancer/` | 5 | 特徴量 bar plot |

追加で以下も保存されている。

| ファイル | 内容 |
| --- | --- |
| `results/figures/explanations/decoy_mnist/sample_00.png ... sample_04.png` | Decoy MNIST の baseline/RRR 勾配比較 |
| `results/tables/newsgroups_baseline_top_gradient_words.csv` | baseline の高勾配語 top 50 |
| `results/tables/newsgroups_rrr_top_gradient_words.csv` | RRR の高勾配語 top 50 |

## 残る注意点

- 20 Newsgroups の高勾配語には metadata 由来の語が多く含まれる。これは accuracy の論文再現には有利だが、説明の意味解釈では注意が必要である。
- Iris-Cancer の baseline degradation は確認できているが、論文メモの `92% -> 81%` ほど大きくはない。
- Find Another Explanation は挙動としては再現できているが、各 round の勾配可視化まではまだ保存していない。

## 総評

今回の `results (2).zip` は、これまでで最も完成度が高い。

- 全 summary 行が `reproduced` 判定。
- 古い雛形ファイルは zip から除外済み。
- Toy Color は `lambda_rrr=100.0` により accuracy と説明制御の両立に成功。
- Decoy MNIST は性能と勾配の両面で非常に強い再現結果。
- 20 Newsgroups は論文メモの baseline `94%` に到達。
- Iris-Cancer と Find Another Explanation も、論文の期待挙動に沿った結果が得られている。

したがって、現段階では **論文内の主要実験は再現成功と言える水準に到達した** と判断する。
