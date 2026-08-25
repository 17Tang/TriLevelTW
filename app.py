from datetime import datetime, timedelta
import io
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="三關價分析看板", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# --- 1. 股票代號與名稱對照表 (快取 1 天) ---
@st.cache_data(ttl=86400)
def get_stock_info_map() -> dict:
    try:
        params = {"dataset": "TaiwanStockInfo"}
        resp = requests.get(FINMIND_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("msg") == "success" and data.get("data"):
            return {
                item["stock_id"]: item["stock_name"] for item in data["data"]
            }
    except Exception:
        pass
    return {}


# --- 2. 抓取台股 / 大盤 / 櫃買 ---
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


# --- 3. 抓取台指期近月日K ---
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
    if "trading_session" in raw_df.columns:
        raw_df = raw_df[raw_df["trading_session"] == "position"]
    raw_df = raw_df[~raw_df["contract_date"].astype(str).str.contains("/")]

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


# --- 4. 抓取美股費城半導體指數 (SOX) ---
def fetch_sox_index(days: int = 45) -> pd.DataFrame:
    url = "https://stooq.com/q/d/l/?s=^sox&i=d"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200 or not resp.text.strip():
        raise ValueError("費城半導體指數連線失敗。")

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "Close" not in df.columns:
        raise ValueError("費城半導體指數解析為空。")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df[["Open", "High", "Low", "Close"]]


# --- 三關價核心運算與計算模型 ---
def compute_three_passes_df(df: pd.DataFrame, days: int = 30):
    if df.empty or len(df) < 2:
        return None, None

    calc_df = df.copy()
    calc_df["昨收"] = calc_df["Close"].shift(1)
    calc_df["漲跌"] = calc_df["Close"] - calc_df["昨收"]
    calc_df["漲跌幅(%)"] = (calc_df["漲跌"] / calc_df["昨收"]) * 100
    calc_df["振幅(%)"] = (
        (calc_df["High"] - calc_df["Low"]) / calc_df["昨收"]
    ) * 100

    prev_high = calc_df["High"].shift(1)
    prev_low = calc_df["Low"].shift(1)
    diff = prev_high - prev_low

    calc_df["上關(空方防守)"] = prev_high + (diff * 0.382)
    calc_df["中關(日關)"] = (prev_high + prev_low) / 2
    calc_df["下關(多方防守)"] = prev_low - (diff * 0.382)

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

    calc_df["說明"] = calc_df.apply(judge_position, axis=1)

    latest_row = calc_df.iloc[-1]
    curr_high = latest_row["High"]
    curr_low = latest_row["Low"]
    curr_diff = curr_high - curr_low

    next_day_passes = {
        "latest_date": calc_df.index[-1].strftime("%Y-%m-%d"),
        "latest_close": latest_row["Close"],
        "latest_change": latest_row["漲跌"],
        "latest_pct": latest_row["漲跌幅(%)"],
        "next_up": curr_high + (curr_diff * 0.382),
        "next_mid": (curr_high + curr_low) / 2,
        "next_down": curr_low - (curr_diff * 0.382),
        "today_note": latest_row["說明"],
    }

    calc_df.rename(
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

    result = calc_df[cols].tail(days).sort_index(ascending=False)
    result.index = result.index.strftime("%Y-%m-%d")
    result.index.name = "日期"

    return result, next_day_passes


# --- 取得查詢標的資料 ---
@st.cache_data(ttl=300)
def get_target_data(user_input: str, days: int = 30):
    query = user_input.strip().upper()
    stock_map = get_stock_info_map()

    try:
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
        elif query in [
            "費半",
            "費城半導體",
            "SOX",
            "^SOX",
            "費城半導體指數",
        ]:
            df = fetch_sox_index(days=days)
            target_name = "費城半導體指數 (SOX)"
        else:
            clean_id = query.replace(".TW", "").replace(".TWO", "")
            df = fetch_stock_or_index(clean_id, days=days)
            stock_name = stock_map.get(clean_id, "")
            target_name = (
                f"{clean_id} {stock_name}".strip()
                if stock_name
                else f"個股 {clean_id}"
            )

        result, next_day_passes = compute_three_passes_df(df, days=days)
        return result, next_day_passes, target_name, None
    except Exception as e:
        return None, None, None, f"查詢失敗：{str(e)}"


# --- 取得大盤、櫃買、台指期近5日摘要 ---
@st.cache_data(ttl=300)
def get_market_overview():
    items = [
        ("加權指數 (大盤)", lambda: fetch_stock_or_index("TAIEX", days=15)),
        ("櫃買指數 (TPEx)", lambda: fetch_stock_or_index("TPEx", days=15)),
        ("台指期 (近月)", lambda: fetch_tx_futures(days=15)),
    ]
    overview_data = {}
    for name, fetch_func in items:
        try:
            df = fetch_func()
            res, _ = compute_three_passes_df(df, days=5)
            overview_data[name] = res
        except Exception:
            overview_data[name] = None
    return overview_data


# --- 樣式設定輔助函式 ---
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


# ================= 網頁畫面呈現 =================
st.title("三關價分析看板")

# --- 頂部常駐：大盤、櫃買、台指期近5日位階總覽 ---
st.markdown("### 🌐 市場核心指數・近 5 日關鍵位階總覽")
overview = get_market_overview()

tabs = st.tabs(["🏛️ 加權指數 (大盤)", "🏢 櫃買指數 (TPEx)", "⚡ 台指期貨 (近月)"])
for idx, (market_name, market_df) in enumerate(overview.items()):
    with tabs[idx]:
        if market_df is not None:
            # 簡要欄位呈現
            display_cols = [
                "收盤",
                "漲跌",
                "漲跌幅(%)",
                "上關(空方防守)",
                "中關(日關)",
                "下關(多方防守)",
                "說明",
            ]
            styled_market = (
                market_df[display_cols]
                .style.format(
                    {
                        "收盤": "{:.2f}",
                        "漲跌": "{:+.2f}",
                        "漲跌幅(%)": "{:+.2f}%",
                        "上關(空方防守)": "{:.2f}",
                        "中關(日關)": "{:.2f}",
                        "下關(多方防守)": "{:.2f}",
                    }
                )
                .map(color_change, subset=["漲跌", "漲跌幅(%)"])
                .map(color_description, subset=["說明"])
            )
            st.dataframe(styled_market, use_container_width=True)
        else:
            st.warning(f"暫無法取得 {market_name} 最新 5 日數據。")

st.markdown("---")

# --- 下方搜尋與詳細歷史分析 ---
st.subheader("🔍 個股與指數個別深度查詢")
st.caption(
    "支援輸入：個股代號 (如: 2330)、大盤、櫃買、台指期、費半 (SOX)"
)

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input(
        "請輸入股號或標的名稱",
        value="2330",
        help="可輸入：2330、大盤、櫃買、台指期、費半...",
    )
with col2:
    days_to_show = st.number_input(
        "顯示天數", min_value=5, max_value=60, value=30
    )

if user_input:
    with st.spinner("資料載入中..."):
        df_result, next_info, target_name, err = get_target_data(
            user_input, days=days_to_show
        )

    if err:
        st.error(err)
    else:
        st.markdown(
            f"### 📊 【{target_name}】最新收盤 ({next_info['latest_date']}) 與 預計明日三關價"
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

        # 狀態卡片醒目標註
        note_text = next_info["today_note"]
        if "強勢" in note_text or "漲破" in note_text:
            st.error(f"🔥 **最新交易日型態結算：{note_text}**")
        elif "偏多" in note_text:
            st.warning(f"📈 **最新交易日型態結算：{note_text}**")
        elif "偏空" in note_text:
            st.info(f"📉 **最新交易日型態結算：{note_text}**")
        else:
            st.success(f"⚠️ **最新交易日型態結算：{note_text}**")

        st.write("#### 詳細交易歷史清單")

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

        st.dataframe(styled_df, use_container_width=True, height=500)
