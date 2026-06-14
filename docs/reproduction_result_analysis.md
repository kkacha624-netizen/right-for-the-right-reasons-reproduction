# 再現実験結果の分析

このメモは、Colab で再実行した `results.zip` の内容をもとに、Ross, Hughes, Doshi-Velez の論文 **Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations** の各実験と比較したものである。

今回の実行では、主要実験が `device: cuda` で実行されている。`results/tables/reproduction_summary.csv` と各 `experiments/<experiment>/metrics.json` の値は一致しており、前回問題だった summary と個別 metrics の不一致は解消されている。

また、各データセットのサンプル画像が 5 枚ずつ保存されている。

```text
results/figures/samples/toy_color/sample_00.png ... sample_04.png
results/figures/samples/decoy_mnist/sample_00.png ... sample_04.png
results/figures/samples/newsgroups/sample_00.png ... sample_04.png
results/figures/samples/iris_cancer/sample_00.png ... sample_04.png
```

## 全体評価

| 実験 | 状態 | 短評 |
| --- | --- | --- |
| Toy Color / Synthetic | 部分的に再現 | RRR は corner への勾配を強く抑制し、top-middle rule へ説明を移している。ただし RRR accuracy が `0.808` で、成功基準の `0.90` を下回った。 |
| Iris-Cancer | 再現成功 | 350 split 集計が行われ、baseline の Iris 除去時劣化と RRR によるギャップ縮小が確認できた。 |
| 20 Newsgroups | 再現成功 | TF-IDF 5000 components、metadata を残す設定で baseline accuracy `0.9442` となり、論文メモの約 `94%` に一致した。 |
| Decoy MNIST | 再現成功 | baseline は random swatch で崩れ、RRR は高精度を維持した。swatch 勾配も大きく抑制されている。 |
| Find Another Explanation | 再現成功 | `cutoff=0.67` を使い、1・2 round は高精度、後半 round で性能低下という論文の挙動に近い結果が出た。 |
| Explanation Visualization | 再現成功 | Toy Color の入力、baseline gradient、RRR gradient、annotation mask の図が正常に生成された。 |

## 結果一覧

| 実験 | 手法 | 評価内容 | 結果 | 判定 |
| --- | --- | --- | ---: | --- |
| Toy Color | Baseline | Test accuracy | 0.9860 | partial |
| Toy Color | RRR | Test accuracy | 0.8080 | partial |
| Toy Color | Baseline | corner gradient fraction | 0.5319 | 参考 |
| Toy Color | RRR | corner gradient fraction | 0.0006 | 良好 |
| Toy Color | Baseline | top-middle gradient fraction | 0.1739 | 参考 |
| Toy Color | RRR | top-middle gradient fraction | 0.7060 | 良好 |
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
- 単に accuracy が高いだけでなく、入力勾配が望ましい領域へ移ることが重要である。

今回の結果:

- Baseline accuracy: `0.9860`
- RRR accuracy: `0.8080`
- Baseline corner gradient fraction: `0.5319`
- RRR corner gradient fraction: `0.0006`
- Baseline top-middle gradient fraction: `0.1739`
- RRR top-middle gradient fraction: `0.7060`

分析:

- 説明の制御という観点ではかなり良い。RRR は corner への勾配をほぼゼロに抑え、top-middle へ勾配を移している。
- これは「禁止された説明を避けて別の説明を使う」という論文の定性的主張と一致している。
- 一方で、RRR accuracy が `0.8080` に落ちており、現在の成功条件 `rrr_high_accuracy >= 0.90` を満たしていない。
- 原因として、`lambda_rrr=1000` が Toy Color には強すぎる、または corner を完全に抑制する annotation と現在のデータ生成が分類を難しくしすぎている可能性がある。

結論:

- 説明制御は成功。
- 分類性能込みの完全再現としてはまだ partial。
- 次は `lambda_rrr` を下げる、epoch 数を増やす、または paper に近い Toy Color の annotation 設定を再確認するのがよい。

### Iris-Cancer

論文側の狙い:

- Iris と Breast Cancer Wisconsin を結合したデータセットを使う。
- Baseline は Iris 特徴に依存しやすく、Iris 特徴を test 時に除去すると性能が落ちる。
- RRR は Iris 特徴への依存を抑え、Iris 特徴を除いても性能が変わりにくくなる。
- 論文メモでは baseline accuracy 約 `92%`、Iris 除去時約 `81%` が目安である。

今回の結果:

- 350 split が実行されている。
- Baseline mean accuracy: `0.9671 ± 0.0299`
- Baseline without Iris mean accuracy: `0.9334 ± 0.0441`
- RRR mean accuracy: `0.9328 ± 0.0475`
- RRR without Iris mean accuracy: `0.9330 ± 0.0474`

分析:

- Baseline は Iris 除去で `0.9671` から `0.9334` に低下しており、Iris 特徴への依存が確認できる。
- RRR は Iris あり `0.9328`、Iris なし `0.9330` でほぼ一致している。
- これは「RRR により、禁止特徴を使わないモデルになる」という論文の主張と整合している。
- 論文メモの `92% -> 81%` より baseline の低下幅は小さいが、方向性と RRR の効果は明確である。

結論:

- 再現成功と判断できる。
- 数値差の大きさまで論文に揃えるには、Iris-Cancer の行対応や split 条件をさらに厳密に確認するとよい。

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

分析:

- Baseline accuracy が `0.9442` で、論文メモの約 `94%` と非常によく一致している。
- RRR も `0.9372` で、baseline から大きく崩れていない。
- 前回の `0.80` 台から大きく改善しており、metadata を残す前処理が論文設定に近かったと考えられる。

結論:

- 分類性能の観点では再現成功。
- 今後は、RRR がどの単語への依存を抑制したかを、重みまたは入力勾配上位語として表に出すとさらに論文対応が明確になる。

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

分析:

- Baseline は swatch が相関している場合にはほぼ完全に分類できる。
- しかし swatch をランダム化すると `0.5809` まで低下しており、decoy に依存していたことが明確である。
- RRR は random swatch でも `0.9823` を維持している。
- swatch gradient fraction も `0.2401` から `0.000037` へ大きく低下している。
- これは、分類性能と説明の両面で RRR の効果が出ていることを示している。

結論:

- 再現成功。
- 今回の全実験の中で最も強い再現結果である。

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
- Round 1 も `0.9015` を維持しており、別の説明でもある程度分類できている。
- Round 2 では `0.5125` まで低下している。
- Toy Color の主要 rule を順に禁止していくと、最終的に有効な説明が尽きて性能が落ちる、という論文の挙動に近い。

結論:

- 再現成功。
- 今後は、各 round の入力勾配可視化も保存するとさらに説得力が増す。

### Explanation Visualization

論文側の狙い:

- 入力勾配を可視化し、baseline と RRR の判断根拠の違いを見る。

今回の結果:

- Figure: `results/figures/explanations/toy_color_gradients.png`
- Baseline accuracy: `0.9875`
- RRR accuracy: `0.9800`
- `figure_created: true`

分析:

- 図は正常に生成されている。
- baseline と RRR の accuracy も十分高い。
- Toy Color の説明可視化としては成功している。

結論:

- 再現成功。
- 次は Decoy MNIST の swatch 領域の勾配可視化を追加すると、最も重要な実験の説明面も図で示せる。

## サンプル画像の出力確認

今回の zip には、各データセット 5 枚ずつのサンプル画像が含まれている。

| データセット | 保存先 | 枚数 | 内容 |
| --- | --- | ---: | --- |
| Toy Color | `results/figures/samples/toy_color/` | 5 | `5 x 5` RGB toy image |
| Decoy MNIST | `results/figures/samples/decoy_mnist/` | 5 | swatch 付き MNIST image |
| 20 Newsgroups | `results/figures/samples/newsgroups/` | 5 | 文書テキストを画像化したもの |
| Iris-Cancer | `results/figures/samples/iris_cancer/` | 5 | 特徴量 bar plot |

この点はユーザー要望を満たしている。

## 古い雛形ファイルについて

以前の zip には以下の空ファイルが含まれていた。

```text
experiments/exp001_baseline/metrics.json
experiments/exp002_gradcam/metrics.json
```

これらは古い雛形ファイルであり、現在はリポジトリから削除済みである。
また、`scripts/package_results.py` を使って zip を作成すると、これらの古い実験ディレクトリが残っていても成果物には含めない。

## 次の改善案

1. Toy Color の RRR accuracy を `0.90` 以上に戻すため、`run_synthetic.py` のデフォルト `lambda_rrr` を `100.0` に下げた。次回 Colab 実行で確認する。
2. Toy Color について、論文の図に近い round ごとの gradient visualization を追加する。
3. Decoy MNIST の baseline / RRR の入力勾配画像を 5 枚保存する処理を追加した。
4. 20 Newsgroups で、baseline と RRR の高勾配単語を CSV に出す処理を追加した。
5. zip 作成前に、古い空の `exp001_baseline` と `exp002_gradcam` を除外する `scripts/package_results.py` を追加した。

## 総評

今回の再実行では、前回より明確に再現度が上がった。

- Iris-Cancer は 350 split 集計により再現成功と判断できる。
- 20 Newsgroups は論文メモの約 `94%` baseline に到達した。
- Decoy MNIST は性能と勾配の両面で非常に強い再現結果になっている。
- Find Another Explanation も cutoff `0.67` を使った挙動として整っている。
- Toy Color は説明制御は成功しているが、RRR accuracy が低いため partial とする。

したがって、現時点では **Toy Color 以外はおおむね再現成功、Toy Color は正則化強度の調整が必要**という評価である。
