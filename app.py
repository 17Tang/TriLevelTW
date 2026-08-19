from datetime import datetime, timedelta
import pandas as pd
import requests
import streamlit as st
import twstock

st.set_page_config(page_title="台股三關價即時分析", layout="wide")


# --- 抓取台股指數 (大盤 / 櫃買) ---
def fetch_index_data(symbol: str) -> pd.DataFrame:
    """從證交所開放資料抓取加權指數或從 Yahoo 備用 API 抓取歷史"""
    # 透過證交所 API 取得近期加權指數
    url = "https://openapi.twse.com.tw/v1/exchangeReport/FMTQIK"
    resp = requests.get(url, timeout=10)
    data = resp.json()

    # 欄位: 日期, 成交股數, 成交金額, 成交筆數, 發行量加權股價指數, 漲跌點數
    records = []
    for item in data:
        # 民國轉西元日期 113/05/20 -> 2024-05-20
        d_parts = item["Date"].split("/")
        ad_year = int(d_parts[0]) + 1911
        date_str = f"{ad_year}-{d_parts[1]}-{d_parts[2]}"

        close_p = float(item["發行量加權股價指數"].replace(",", ""))
        records.append({"Date": date_str, "Close": close_p})

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # 由於官方此 API 只提供收盤價，以近似方式提供三關價參考
    df["Open"] = df["Close"]
    df["High"] = df["Close"]
    df["Low"] = df["Close"]
    return df


# --- 抓取上市/上櫃個股資料 ---
def fetch_stock_data(stock_id: str) -> pd.DataFrame:
    stock = twstock.Stock(stock_id)
    # 抓取最近 70 筆日資料 (約 3 個月)
    data = stock.fetch_31()  # 當月與前一個月
    # 如果資料太少，額外抓取更早之前的資料
    today = datetime.now()
    all_data = []
    for i in range(3):
        m_date = today - timedelta(days=i * 28)
        all_data.extend(stock.fetch(m_date.year, m_date.month))

    # 去除重複交易日
    unique_dict = {d.date: d for d in all_data}
    sorted_data = sorted(unique_dict.values(), key=lambda x: x.date)

    records = []
    for d in sorted_data:
        records.append(
            {
                "Date": d.date,
                "Open": d.open,
                "High": d.high,
                "Low": d.low,
                "Close": d.close,
            }
        )

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df


@st.cache_data(ttl=300)
def get_stock_three_passes(stock_id: str, days: int = 30):
    stock_id = stock_id.strip().upper()

    # 1. 抓取資料
    try:
        if stock_id in ["^TWII", "0000", "加權", "大盤"]:
            df = fetch_index_data("大盤")
        else:
            # 清除可能帶有的 .TW / .TWO 後綴
            clean_id = stock_id.replace(".TW", "").replace(".TWO", "")
            df = fetch_stock_data(clean_id)
    except Exception as e:
        return (
            None,
            f"資料抓取失敗，請確認代號是否正確 (錯誤訊息: {str(e)})",
        )

    if df.empty or len(df) < 2:
        return None, "查無交易數據或交易日過少。"

    # 2. 昨收、漲跌、漲跌幅、振幅計算
    df["昨收"] = df["Close"].shift(1)
    df["漲跌"] = df["Close"] - df["昨收"]
    df["漲跌幅(%)"] = (df["漲跌"] / df["昨收"]) * 100
    df["振幅(%)"] = ((df["High"] - df["Low"]) / df["昨收"]) * 100

    # 3. 三關價計算
    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)
    diff = prev_high - prev_low

    df["上關(空方防守)"] = prev_high + (diff * 0.382)
    df["中關(日關)"] = (prev_high + prev_low) / 2
    df["下關(多方防守)"] = prev_low - (diff * 0.382)

    # 4. 說明區間判斷
    def judge_position(row):
        close = row["Close"]
        up = row["上關(空方防守)"]
        mid = row["中關(日關)"]
        down = row["下關(多方防守)"]

        if pd.isna(up) or pd.isna(mid) or pd.isna(down):
            return "計算中"
        if close > up:
            return "強勢：漲破空方防守"
        elif close >= mid:
            return "偏多：介於中關與上關之間"
        elif close >= down:
            return "偏空：小於中關但未破多方防守"
        else:
            return "弱勢：跌破多方防守"

    df["說明"] = df.apply(judge_position, axis=1)

    # 欄位重新命名與格式整理
    rename_map = {
        "Open": "開盤",
        "High": "最高",
        "Low": "最低",
        "Close": "收盤",
    }
    df.rename(columns=rename_map, inplace=True)
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
st.caption("資料來源：TWSE 臺灣證券交易所 / TPEx 櫃買中心")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("輸入個股代號 (如: 2330 / 8069 / 大盤)", value="2330")
with col2:
    days_to_show = st.number_input(
        "顯示天數", min_value=5, max_value=60, value=30
    )

if user_input:
    with st.spinner("向證交所連線更新數據中..."):
        df_result, err = get_stock_three_passes(user_input, days=days_to_show)

    if err:
        st.error(err)
    else:
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

        st.info(f"當日型態判斷：**{latest['說明']}**")

        st.write("### 詳細交易歷史清單")

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
            .highlight_max(axis=0, subset=["收盤"], color="#ffd2d2")
        )

        st.dataframe(styled_df, use_container_width=True, height=600)
