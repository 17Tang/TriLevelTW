from datetime import datetime, timedelta
import pandas as pd
import requests
import streamlit as st
import twstock

# 設定頁面資訊
st.set_page_config(page_title="三關價分析看板", layout="wide")


# --- 抓取台股大盤指數 (證交所官方 API) ---
def fetch_index_data() -> pd.DataFrame:
    url = "https://openapi.twse.com.tw/v1/exchangeReport/FMTQIK"
    resp = requests.get(url, timeout=10)
    data = resp.json()

    records = []
    for item in data:
        # 民國年轉西元年 (如 113/05/20 -> 2024-05-20)
        d_parts = item["Date"].split("/")
        ad_year = int(d_parts[0]) + 1911
        date_str = f"{ad_year}-{d_parts[1]}-{d_parts[2]}"

        close_p = float(item["發行量加權股價指數"].replace(",", ""))
        records.append({"Date": date_str, "Close": close_p})

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df["Open"] = df["Close"]
    df["High"] = df["Close"]
    df["Low"] = df["Close"]
    return df


# --- 抓取上市/上櫃個股日 K 線資料 (twstock) ---
def fetch_stock_data(stock_id: str) -> pd.DataFrame:
    stock = twstock.Stock(stock_id)

    today = datetime.now()
    all_data = []
    for i in range(3):
        m_date = today - timedelta(days=i * 28)
        all_data.extend(stock.fetch(m_date.year, m_date.month))

    if not all_data:
        raise ValueError("查無此股票代號或無交易數據")

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

    try:
        if stock_id in ["^TWII", "0000", "加權", "大盤", "TWA00"]:
            df = fetch_index_data()
        else:
            clean_id = stock_id.replace(".TW", "").replace(".TWO", "")
            df = fetch_stock_data(clean_id)
    except Exception as e:
        return (
            None,
            None,
            f"資料抓取失敗，請確認代號是否正確 (錯誤資訊: {str(e)})",
        )

    if df.empty or len(df) < 2:
        return None, None, "查無交易數據或交易日不足。"

    # 1. 昨收、漲跌、漲跌幅、振幅
    df["昨收"] = df["Close"].shift(1)
    df["漲跌"] = df["Close"] - df["昨收"]
    df["漲跌幅(%)"] = (df["漲跌"] / df["昨收"]) * 100
    df["振幅(%)"] = ((df["High"] - df["Low"]) / df["昨收"]) * 100

    # 2. 當日三關價計算 (以昨日高低價推算今日)
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

    # 4. 推算「次日（明日）三關價」 (以最新一日的 High/Low 為基準)
    latest_row = df.iloc[-1]
    curr_high = latest_row["High"]
    curr_low = latest_row["Low"]
    curr_diff = curr_high - curr_low

    next_day_passes = {
        "latest_date": df.index[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_row["Close"],
        "latest_change": latest_row["漲跌"],
        "latest_pct": latest_row["漲跌幅(%)"],
        "next_up": curr_high + (curr_diff * 0.382),
        "next_mid": (curr_high + curr_low) / 2,
        "next_down": curr_low - (curr_diff * 0.382),
        "today_note": latest_row["說明"],
    }

    # 5. 整理表格
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

    return result, next_day_passes, None


# --- 介面呈現 ---
st.title("三關價分析看板")
st.caption("資料來源：TWSE 臺灣證券交易所 / TPEx 櫃買中心")

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("輸入個股代號 (如: 2330 / 8069 / 大盤)", value="2330")
with col2:
    days_to_show = st.number_input(
        "顯示天數", min_value=5, max_value=60, value=30
    )

if user_input:
    with st.spinner("向證券交易所連線更新數據中..."):
        df_result, next_info, err = get_stock_three_passes(
            user_input, days=days_to_show
        )

    if err:
        st.error(err)
    else:
        # 上方展示「最新收盤」與「預計明日三關價」
        st.subheader(
            f"📊 最新收盤 ({next_info['latest_date']}) 與 預計明日三關價"
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(
            f"最新收盤價 ({next_info['latest_date']})",
            f"{next_info['latest_close']:.2f}",
            f"{next_info['latest_change']:+.2f} ({next_info['latest_pct']:+.2f}%)",
        )
        kpi2.metric(
            "預計明日 上關 (空防)",
            f"{next_info['next_up']:.2f}",
            help="明日空方防守價位",
        )
        kpi3.metric(
            "預計明日 中關 (日關)",
            f"{next_info['next_mid']:.2f}",
            help="明日多空強弱分界線",
        )
        kpi4.metric(
            "預計明日 下關 (多防)",
            f"{next_info['next_down']:.2f}",
            help="明日多方防守價位",
        )

        st.markdown(f"**最新交易日型態結算：** {next_info['today_note']}")

        st.write("### 詳細交易歷史清單")

        # 數值顏色樣式函式
        def color_change(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #ff4b4b;"
                elif val < 0:
                    return "color: #09ab3b;"
            return ""

        # 說明欄位顏色樣式函式
        def color_description(val):
            if not isinstance(val, str):
                return ""
            if "漲破" in val or "強勢" in val:
                return "color: #d32f2f; font-weight: bold;"  # 深紅粗體
            elif "偏多" in val:
                return "color: #e65100; font-weight: 500;"  # 橘紅
            elif "偏空" in val:
                return "color: #2e7d32; font-weight: 500;"  # 淺綠
            elif "跌破" in val or "弱勢" in val:
                return "color: #1b5e20; font-weight: bold;"  # 深綠粗體
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
            .map(color_description, subset=["說明"])
        )

        st.dataframe(styled_df, use_container_width=True, height=600)
