from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODULE_DIR = BASE_DIR / "resources" / "2311.01985"
if MODULE_DIR.exists():
    sys.path.append(str(MODULE_DIR))
else:
    raise FileNotFoundError("resources/2311.01985 ディレクトリが見つかりません。")

from mpp_snp_demo import MPPDemoResult, TICKERS, run_demo

st.set_page_config(
    page_title="Maximizing Portfolio Predictability",
    layout="wide",
)

st.title("Maximizing Portfolio Predictability with Machine Learning")

explanation_path = Path("resources/2311.01985/2311.01985_explained.md")
with st.expander("論文の解説を表示", expanded=False):
    if explanation_path.exists():
        st.markdown(explanation_path.read_text(), unsafe_allow_html=False)
    else:
        st.warning("解説ファイルが見つかりませんでした。")

st.sidebar.header("シミュレーション設定")
start_date = st.sidebar.date_input("開始日", value=date(2015, 1, 1))
end_date_enabled = st.sidebar.checkbox("終了日を指定", value=False)
end_date = None
if end_date_enabled:
    end_date_value = st.sidebar.date_input("終了日", value=date.today())
    end_date = end_date_value.isoformat()

lookback = st.sidebar.slider("移動平均の期間 (ヶ月)", min_value=1, max_value=12, value=3)
weight_cap = st.sidebar.slider("ウェイト上限", min_value=0.05, max_value=0.5, value=0.2, step=0.01)
rho = st.sidebar.slider("期待リターン制約 ρ", min_value=-0.2, max_value=0.2, value=-0.05, step=0.01)
selected = st.sidebar.multiselect(
    "対象銘柄",
    options=TICKERS,
    default=TICKERS,
)
extra_tickers = st.sidebar.text_input("追加ティッカー (カンマ区切り)")
if extra_tickers.strip():
    selected.extend([ticker.strip().upper() for ticker in extra_tickers.split(",") if ticker.strip()])

start_iso = start_date.isoformat()

st.subheader("MPPシミュレーション")
run_clicked = st.button("シミュレーションを実行")

if run_clicked:
    if len(selected) < 2:
        st.error("銘柄は最低2つ選択してください。")
    else:
        with st.spinner("データ取得と最適化を実行中..."):
            try:
                result: MPPDemoResult = run_demo(
                    tickers=selected,
                    start=start_iso,
                    end=end_date,
                    lookback=lookback,
                    weight_cap=weight_cap,
                    rho=rho,
                    verbose=False,
                )
            except Exception as exc:
                st.error(f"シミュレーションに失敗しました: {exc}")
            else:
                st.success(
                    f"計算完了: R² = {result.r2:.4f}, 反復回数 = {result.iterations}"
                )
                st.markdown("### MPPウェイト")
                st.dataframe(result.weights.to_frame())

                st.markdown("### スケーリング履歴")
                scaling_df = pd.DataFrame(
                    {"Scaling": result.scaling_history},
                    index=pd.Index(range(1, len(result.scaling_history) + 1), name="Iteration"),
                )
                st.line_chart(scaling_df)

                ordered_returns = result.returns[result.weights.index]
                portfolio_returns = ordered_returns.mul(result.weights.values, axis=1).sum(axis=1)
                eq_weight = ordered_returns.mean(axis=1)
                cumulative = pd.DataFrame(
                    {
                        "MPP": (1 + portfolio_returns).cumprod(),
                        "Equal Weight": (1 + eq_weight).cumprod(),
                    }
                )
                cumulative.index.name = "Date"

                st.markdown("### 累積リターン (インサンプル)")
                st.line_chart(cumulative)

                st.markdown("### 直近のリターンと予測")
                preview = pd.concat(
                    {
                        "Realized": ordered_returns.tail(5),
                        "Forecast": result.forecasts[result.weights.index].tail(5),
                    },
                    axis=1,
                )
                st.dataframe(preview)

                st.info(
                    "R² が負の場合は予測精度が十分でないことを示します。\n"
                    "Elastic Net や Random Forest などのモデルに置き換えると改善が期待できます。"
                )
else:
    st.write("サイドバーでパラメータを設定し、シミュレーションを実行してください。")
