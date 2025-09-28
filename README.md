# article
論文とその解析、Pythonでのコーディング等をまとめたリポジトリです。

## ディレクトリ構成
- `resources/2311.01985/2311.01985.pdf` : 論文 *Maximizing Portfolio Predictability with Machine Learning* 本体
- `resources/2311.01985/2311.01985_explained.md` : 論文の概要、手法、実験結果の要約ノート
- `resources/2311.01985/mpp_pipeline.py` : 正規化線形化アルゴリズム (NLA) を用いたMPP最適化ユーティリティ
- `resources/2311.01985/mpp_snp_demo.py` : S&P500銘柄を用いたシンプルな予測・MPP検証スクリプト
- `resources/2311.01985/2311.01985.txt` : PDFから抽出したテキスト
- `streamlit_app.py` : Streamlitによる論文解説とMPPシミュレーションUI

## セットアップ
パッケージ管理には [uv](https://docs.astral.sh/uv/) を使用します。初回は以下の手順で仮想環境と依存関係を準備してください。

```bash
# uv が未インストールの場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# .venv を作成しつつ依存パッケージを同期
uv sync
```

> `uv sync` により `.venv` が自動生成され、`cvxpy`・`yfinance`・`pandas`・`numpy`・`streamlit` など必要なライブラリがインストールされます。

## MPPパイプラインの使い方
1. `resources/2311.01985/mpp_pipeline.py` の `NormalizedLinearizationMPP` でMPPウェイトを計算できます。`solve(verbose=True)` を指定すると反復時のスケーリング値が表示され、収束状況を確認できます。
2. `resources/2311.01985/mpp_snp_demo.py` を実行すると、主要S&P500銘柄の月次リターンをダウンロードし、3ヶ月移動平均による単純予測をもとにMPP最適化を行います。

```bash
uv run python resources/2311.01985/mpp_snp_demo.py
```

出力例:
- 反復ごとのスケーリング値 (収束判定の指標)
- MPPの推定 `R^2`
- 上位銘柄のウェイト

※デモの予測モデルは単純な移動平均のため `R^2` は負になる場合があります。Elastic Net や Random Forest 等で予測精度を高めると論文に近い結果を再現しやすくなります。

## Streamlitアプリ
`streamlit_app.py` では、論文解説の閲覧とMPPシミュレーションをWeb UIで実行できます。

```bash
uv run streamlit run streamlit_app.py
```

- サイドバーで対象銘柄・期間・移動平均期間・制約などを設定できます。
- 「シミュレーションを実行」を押すとデータ取得と最適化を行い、MPPウェイト、スケーリング履歴、累積リターン、直近のリターンと予測を表示します。
- `yfinance` を利用するためネットワーク接続が必要です。API制限に注意してください。

## 次のステップの提案
- 論文で使用されている機械学習モデル (Elastic Net, Random Forest, SVR) を実装し、検証期間を設けてハイパーパラメータを選定する
- 得られたMPPウェイトでアウトオブサンプルのパフォーマンス指標 (シャープレシオ、ドローダウン、回転率など) を算出して効果を評価する
- 取引コストやリスク制約を追加し、実運用を想定したシミュレーションを行う
