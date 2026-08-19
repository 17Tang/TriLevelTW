import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="台股三關價即時分析", layout="wide")


def format_ticker(stock_id: str) -> str:
    stock_id = stock_id.strip().upper()
    if stock_id in ["^TWII", "0000", "TWA00", "加權", "大盤"]:
        return "^TWII"
    elif stock_id in ["^TWOII", "OTC", "櫃買"]:
        return "^TWOII"
    if stock_id.endswith(".TW") or stock_id.endswith(".TWO"):
        return stock_id
    return f"{stock_id}.TW"


@st.cache_data(ttl=300)  # 快取 5 分鐘，避免重複請求被 Yahoo 擋掉
def get_stock_three_passes(stock_id: str, days: int = 30):
    ticker_symbol = format_ticker(stock_id)
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="6mo")

    # 若上市查無資料，嘗試上櫃
    if df.empty and ticker_symbol.endswith(".TW"):
        ticker_symbol = ticker_symbol.replace(".TW", ".TWO")
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="6mo")

    if df.empty:
        return None, "查無此代號或近期無交易數據，請確認輸入是否正確。"

    # 取基礎欄位
    df = df[["Open", "High", "Low", "Close"]].copy()

    # 1. 昨收、漲跌、漲跌幅、振幅
    df["昨收"] = df["Close"].shift(1)
    df["漲跌"] = df["Close"] - df["昨收"]
    df["漲跌幅(%)"] = (df["漲跌"] / df["昨收"]) * 100
    df["振幅(%)"] = ((df["High"] - df["Low"]) / df["昨收"]) * 100

    # 2. 三關價計算
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)
    diff = prev_high - prev_low

    df["上關(空方防守)"] = prev_high + (diff * 0.382)
    df["中關(日關)"] = (prev_high + prev_low) / 2
    df["下關(多方防守)"] = prev_low - (diff * 0.382)

    # 3. 說明區間判斷
    def judge_position(row):
        close = row["Close"]
        up = row["上關(空方防守)"]
        mid = row["中關(日關)"]
        down = row["下關(多方防守)"]

        if pd.isna(up) or pd.isna(mid) or pd.isna(down):
            return "資料計算中"
        if close > up:
            return "強勢：漲破空方防守"
        elif close >= mid:
            return "偏多：介於中關與上關之間"
        elif close >= down:
            return "偏空：小於中關但未破多方防守"
        else:
            return "弱勢：跌破多方防守"

    df["說明"] = df.apply(judge_position, axis=1)

    # 欄位重新命名與排序
    df.rename(
        columns={
            "Open": "開盤",
            "High": "最高",
            "Low": "最低",
            "Close": "收盤",
        },
        inplace=True,
    )
    cols = [
        "開盤",
        "最高",
        "最低",
        "收盤",
        "昨收",
        "漲跌",
        "漲跌幅(%)",
        "振幅(%)",
        "上關(空方防守)",
        "中關(日關)",
        "下關(多方防守)",
        "說明",
    ]
    result = df[cols].tail(days).sort_index(ascending=False)
    result.index = result.index.strftime("%Y-%m-%d")
    result.index.name = "日期"

    return result, None


# --- Streamlit 網頁介面 ---
st.title("📈 台股近 30 日三關價分析看板")
st.caption("支援輸入代號（如：2330、8069）或指數名稱（大盤、櫃買）")

# 側邊欄或頂部輸入
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("輸入股號 / 指數名稱", value="2330")
with col2:
    days_to_show = st.number_input(
        "顯示天數", min_value=5, max_value=90, value=30
    )

if user_input:
    with st.spinner("載入資料中..."):
        df_result, err = get_stock_three_passes(user_input, days=days_to_show)

    if err:
        st.error(err)
    else:
        # 最新一日資訊卡片
        latest = df_result.iloc[0]
        latest_date = df_result.index[0]

        st.subheader(f"最新交易日 ({latest_date}) 概況")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            "收盤價",
            f"{latest['收盤']:.2f}",
            f"{latest['漲跌']:+.2f} ({latest['漲跌幅(%)']:+.2f}%)",
        )
        kpi2.metric("上關 (空防)", f"{latest['上關(空方防守)']:.2f}")
        kpi3.metric("中關 (日關)", f"{latest['中關(日關)']:.2f}")
        kpi4.metric("下關 (多防)", f"{latest['下關(多方防守)']:.2f}")

        # 狀態標籤
        st.info(f"當日型態判斷：**{latest['說明']}**")

        # 格式化表格輸出
        st.write("### 詳細交易歷史清單")

        # 透過 Styler 上色：上漲紅字/底色，下跌綠字/底色 (符合台股習慣)
        def color_change(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #ff4b4b;"
                elif val < 0:
                    return "color: #09ab3b;"
            return ""

        styled_df = (
            df_result.style.format(
                {
                    "開盤": "{:.2f}",
                    "最高": "{:.2f}",
                    "最低": "{:.2f}",
                    "收盤": "{:.2f}",
                    "昨收": "{:.2f}",
                    "漲跌": "{:+.2f}",
                    "漲跌幅(%)": "{:+.2f}%",
                    "振幅(%)": "{:.2f}%",
                    "上關(空方防守)": "{:.2f}",
                    "中關(日關)": "{:.2f}",
                    "下關(多方防守)": "{:.2f}",
                }
            )
            .map(color_change, subset=["漲跌", "漲跌幅(%)"])
            .highlight_max(
                axis=0, subset=["收盤"], color="#ffd2d2"
            )  # 近期高點高亮
        )

        st.dataframe(styled_df, use_container_width=True, height=600)
