from datetime import datetime, timedelta
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="三關價分析看板", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# --- 1. 抓取股票或指數 (大盤 / 櫃買 / 個股) ---
def fetch_stock_or_index(data_id: str, days: int = 45) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime(
        "%Y-%m-%d"
    )
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": data_id,
        "start_date": start_date,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=10)
    data = resp.json()

    if data.get("msg") != "success" or not data.get("data"):
        raise ValueError(f"查無標的【{data_id}】之歷史行情數據。")

    df = pd.DataFrame(data["data"])
    df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df[["Open", "High", "Low", "Close"]]


# --- 2. 抓取台指期貨近月日K ---
def fetch_tx_futures(days: int = 45) -> pd.DataFrame:
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime(
        "%Y-%m-%d"
    )
    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "start_date": start_date,
    }
    resp = requests.get(FINMIND_URL, params=params, timeout=10)
    data = resp.json()

    if data.get("msg") != "success" or not data.get("data"):
        raise ValueError("台指期貨資料取得失敗。")

    raw_df = pd.DataFrame(data["data"])
    # 篩選一般日間交易時段 (position) 並排除價差合約
    if "trading_session" in raw_df.columns:
        raw_df = raw_df[raw_df["trading_session"] == "position"]
    raw_df = raw_df[~raw_df["contract_date"].astype(str).str.contains("/")]

    # 依日期與月份排序，每個交易日取近月第一筆
    raw_df.sort_values(by=["date", "contract_date"], inplace=True)
    near_df = raw_df.groupby("date").first().reset_index()

    near_df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "max": "High",
            "min": "Low",
            "close": "Close",
        },
        inplace=True,
    )
    near_df["Date"] = pd.to_datetime(near_df["Date"])
    near_df.set_index("Date", inplace=True)
    near_df.sort_index(inplace=True)
    return near_df[["Open", "High", "Low", "Close"]]


# --- 三關價核心運算 ---
@st.cache_data(ttl=300)
def get_stock_three_passes(user_input: str, days: int = 30):
    query = user_input.strip().upper()

    try:
        # 分流判斷標的
        if query in ["大盤", "加權", "加權指數", "^TWII", "0000", "TAIEX"]:
            df = fetch_stock_or_index("TAIEX", days=days)
            target_name = "加權指數 (大盤)"
        elif query in [
            "櫃買",
            "上櫃",
            "OTC",
            "^TWOII",
            "櫃買指數",
            "店頭",
            "TPEX",
        ]:
            df = fetch_stock_or_index("TPEx", days=days)
            target_name = "櫃買指數 (TPEx)"
        elif query in ["台指期", "台指", "期貨", "TX", "TX00", "台指近月"]:
            df = fetch_tx_futures(days=days)
            target_name = "台指期貨 (近月連續)"
        else:
            clean_id = query.replace(".TW", "").replace(".TWO", "")
            df = fetch_stock_or_index(clean_id, days=days)
            target_name = f"個股 {clean_id}"
    except Exception as e:
        return None, None, None, f"查詢失敗：{str(e)}"

    if df.empty or len(df) < 2:
        return None, None, None, "查無交易數據或交易日不足。"

    # 1. 昨收、漲跌、漲跌幅、振幅
    df["昨收"] = df["Close"].shift(1)
    df["漲跌"] = df["Close"] - df["昨收"]
    df["漲跌幅(%)"] = (df["漲跌"] / df["昨收"]) * 100
    df["振幅(%)"] = ((df["High"] - df["Low"]) / df["昨收"]) * 100

    # 2. 當日三關價計算
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

    # 4. 推算明日三關價
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

    # 5. 格式整理與倒序排列
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

    return result, next_day_passes, target_name, None


# --- 網頁畫面呈現 ---
st.title("三關價分析看板")
st.caption(
    "支援輸入：大盤 (加權)、櫃買 (上櫃)、台指期 (近月)、個股代號 (如: 2330 / 8069)"
)

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input(
        "請輸入股號或標的名稱",
        value="大盤",
        help="可輸入：大盤、櫃買、台指期、2330、8069...",
    )
with col2:
    days_to_show = st.number_input(
        "顯示天數", min_value=5, max_value=60, value=30
    )

if user_input:
    with st.spinner("資料載入中..."):
        df_result, next_info, target_name, err = get_stock_three_passes(
            user_input, days=days_to_show
        )

    if err:
        st.error(err)
    else:
        st.subheader(
            f"📊 【{target_name}】最新收盤 ({next_info['latest_date']}) 與 預計明日三關價"
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
            help="明日多空強弱分水嶺",
        )
        kpi4.metric(
            "預計明日 下關 (多防)",
            f"{next_info['next_down']:.2f}",
            help="明日多方防守價位",
        )

        st.markdown(f"**最新交易日型態結算：** {next_info['today_note']}")

        st.write("### 詳細交易歷史清單")

        def color_change(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "color: #ff4b4b;"
                elif val < 0:
                    return "color: #09ab3b;"
            return ""

        def color_description(val):
            if not isinstance(val, str):
                return ""
            if "漲破" in val or "強勢" in val:
                return "color: #d32f2f; font-weight: bold;"
            elif "偏多" in val:
                return "color: #e65100; font-weight: 500;"
            elif "偏空" in val:
                return "color: #2e7d32; font-weight: 500;"
            elif "跌破" in val or "弱勢" in val:
                return "color: #1b5e20; font-weight: bold;"
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
