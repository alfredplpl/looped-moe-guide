# Looped MoEを図とコードで読む

[English version](README.en.md)

小さな言語モデルを作る実験で使った構成です。**中央のブロックを繰り返す Looped Transformer** と、**トークンごとに処理先を選ぶ Mixture of Experts（MoE）** を組み合わせています。

## 1. 同じ4層を2周する

先頭2層を通り、中央4層を2周し、末尾2層から次トークンの予測を出します。中央では、1周目の出力を2周目の入力に渡します。

```mermaid
flowchart LR
    T["入力トークンと位置"] --> A["先頭2層"]
    A --> B["中央4層 · 1周目"]
    B --> C["中央4層 · 2周目"]
    C --> D["末尾2層"]
    D --> E["次トークンの予測"]
    B -. "Attention・ルーター・expertの重みを共有" .- C
```

コードでの実行順は `0, 1, (2, 3, 4, 5), (2, 3, 4, 5), 6, 7` です。中央4層は同じモジュールをもう一度呼ぶので、更新される重みは**8層分**。順伝播は**12層分**です。逆伝播では両方の周回の勾配が共有パラメータに足し合わされます。

## 2. 各トークンを2個のexpertへ送る

各ブロックでは、まず因果的セルフアテンションで前にあるトークンの情報を集めます。その次に、ルーターが各トークンの隠れ状態を見て、42個のexpertからスコア上位2個を選びます。各expertはSwiGLUのフィードフォワードネットワークです。

```mermaid
flowchart LR
    H["Attention後の隠れ状態"] --> R["ルーター · 42個を採点"]
    R --> A["1位のexpert"]
    R --> B["2位のexpert"]
    A --> S["2つの出力を重み付きで足す"]
    B --> S
    S --> O["残差に加える"]
```

重みは上位2個のスコアをsoftmaxで正規化して決めます。次の周回では隠れ状態が更新されているため、同じルーターでも異なるexpertを選べます。学習時は、割り当てが一部のexpertに偏りすぎないよう補助損失を加えます。

## 3. 実際に学習した構成

| 項目 | 設定 |
|---|---|
| 層 | 独立8層、実行12層 |
| 隠れ幅・Attention | 512、8ヘッド |
| Expert | 各層42個、上位2個を使用、SwiGLUの中間幅2,944 |
| 位置表現・文脈長 | 学習可能な絶対位置埋め込み、512トークン |
| 語彙 | 16,384、入力埋め込みと出力の重みを共有 |
| 保存するパラメータ | **1,536,606,720** |
| 1トークンの順伝播で使う重みの要素数 | **130,032,128**。周回ごとに同じ重みの使用も数える |

同じ寸法で12層すべてに独立した重みを持たせた構成は約2.30Bです。中央を共有した構成では、保存するパラメータが約33%減ります。12層分を通るため、その削減率をそのまま順伝播時間には当てはめられません。

この実験では日英1:8の約30億トークンで事前学習し、その後に日本語のMagpieで回答部分だけを学習しました。事前学習の検証損失は2.690、SFTの検証損失は2.004です。それぞれの評価は別の検証データを使っています。 [Magpie SFT v1.0](https://huggingface.co/datasets/llm-jp/magpie-sft-v1.0)

## CPUで試す

```bash
python -m pip install -r requirements.txt
python walkthrough.py
```

乱数で初期化した小さなモデルを使います。ブロックが呼ばれた順番、中央の同じブロックが1周目と2周目で選んだexpert、1回の学習ステップで得た共有重みの勾配を表示します。

- [moe.py](moe.py) で `MoEConfig.block_order` → `MoELanguageModel.forward` → `ExpertLayer.forward` の順に読むと、実行の流れを追えます。
- [walkthrough.py](walkthrough.py) は上の実装をCPUで観察する短いスクリプトです。

## 参考文献

- [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://proceedings.iclr.cc/paper_files/paper/2025/file/2676109d49d1eb26d6bc584a8f556305-Paper-Conference.pdf)
- [OLMoE: Open Mixture-of-Experts Language Models](https://arxiv.org/abs/2409.02060)
