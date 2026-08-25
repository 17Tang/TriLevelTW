from datetime import datetime, timedelta
import io
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="三關價分析看板", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
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
def fetch_stock_or_index(data_id: str, days: int = 60) -> pd.DataFrame:
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
def fetch_tx_futures(days: int = 60) -> pd.DataFrame:
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


# --- 4. 抓取美股費城半導體指數 (SOX / SOXX) ---
def fetch_sox_index(days: int = 60) -> pd.DataFrame:
    # 透過 Yahoo Finance 公開 Chart API 抓取
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ESOX?range=6mo&interval=1d"
    resp = requests.get(url, headers=HEADERS, timeout=10)

    if resp.status_code == 200:
        res = resp.json()
        result = res.get("chart", {}).get("result")
        if result:
            quote = result[0]
            timestamps = quote.get("timestamp", [])
            indicators = quote.get("indicators", {}).get("quote", [{}])[0]

            dates = [datetime.fromtimestamp(ts).date() for ts in timestamps]
            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(dates),
                    "Open": indicators.get("open"),
                    "High": indicators.get("high"),
                    "Low": indicators.get("low"),
                    "Close": indicators.get("close"),
                }
            )
            df.dropna(inplace=True)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            if not df.empty:
                return df[["Open", "High", "Low", "Close"]]

    # 備援：若 API 被擋則透過 Stooq 下載 SOXX.US (費半 ETF)
    alt_url = "https://stooq.com/q/d/l/?s=soxx.us&i=d"
    alt_resp = requests.get(alt_url, headers=HEADERS, timeout=10)
    if alt_resp.status_code == 200 and alt_resp.text.strip():
        df = pd.read_csv(io.StringIO(alt_resp.text))
        if not df.empty and "Close" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            return df[["Open", "High", "Low", "Close"]]

    raise ValueError("費城半導體指數資料取得失敗，請稍候重試。")


# --- 核心計算：日三關價、周關鍵價、月關鍵價 ---
def compute_three_passes_df(df: pd.DataFrame, days: int = 30):
    if df.empty or len(df) < 2:
        return None, None

    calc_df = df.copy()

    # 1. 日線漲跌與三關價
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

    # 2. 周關鍵價 (當周累計最高與最低之中心價)
    calc_df["Year_Week"] = calc_df.index.to_period("W")
    week_high = calc_df.groupby("Year_Week")["High"].cummax()
    week_low = calc_df.groupby("Year_Week")["Low"].cummin()
    calc_df["周關鍵價"] = (week_high + week_low) / 2

    # 3. 月關鍵價 (當月累計最高與最低之中心價)
    calc_df["Year_Month"] = calc_df.index.to_period("M")
    month_high = calc_df.groupby("Year_Month")["High"].cummax()
    month_low = calc_df.groupby("Year_Month")["Low"].cummin()
    calc_df["月關鍵價"] = (month_high + month_low) / 2

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

    calc_df["說明"] = calc_df.apply(judge_position, axis=1)

    # 推算明日
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
        "curr_week_key": latest_row["周關鍵價"],
        "curr_month_key": latest_row["月關鍵價"],
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
        "周關鍵價",
        "月關鍵價",
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
            "SOXX",
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


# --- 取得市場四大核心概況看板 ---
@st.cache_data(ttl=300)
def get_market_overview():
    items = [
        ("加權指數 (大盤)", lambda: fetch_stock_or_index("TAIEX", days=20)),
        ("櫃買指數 (TPEx)", lambda: fetch_stock_or_index("TPEx", days=20)),
        ("台指期 (近月)", lambda: fetch_tx_futures(days=20)),
        ("費城半導體 (SOX)", lambda: fetch_sox_index(days=20)),
    ]
    overview_list = []
    for name, fetch_func in items:
        try:
            df = fetch_func()
            _, next_info = compute_three_passes_df(df, days=5)
            overview_list.append((name, next_info, None))
        except Exception as e:
            overview_list.append((name, None, str(e)))
    return overview_list


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

# --- 1. 頂部常駐：市場核心指數直觀看板 ---
st.markdown("### 🌐 市場核心指數・即時位階與關鍵價看板")
overview_data = get_market_overview()

cols_top = st.columns(4)
for idx, (market_name, info, err) in enumerate(overview_data):
    with cols_top[idx]:
        if info:
            st.markdown(
                f"<div style='border: 1px solid #333; border-radius: 8px; padding: 12px; background-color: rgba(255,255,255,0.03);'>"
                f"<h4 style='margin:0; font-size:16px;'>{market_name}</h4>"
                f"<p style='color: gray; font-size: 12px; margin: 2px 0 8px 0;'>收盤日期：{info['latest_date']}</p>"
                f"<div style='font-size: 20px; font-weight: bold;'>{info['latest_close']:.2f} "
                f"<span style='font-size: 13px; color: {'#ff4b4b' if info['latest_change'] > 0 else '#09ab3b'};'>"
                f"{info['latest_change']:+.2f} ({info['latest_pct']:+.2f}%)</span></div>"
                f"<hr style='margin: 8px 0; border: none; border-top: 1px solid #444;'/>"
                f"<div style='font-size: 13px; line-height: 1.6;'>"
                f"<b>明日上關(空防)：</b> {info['next_up']:.2f}<br/>"
                f"<b>明日中關(日關)：</b> {info['next_mid']:.2f}<br/>"
                f"<b>明日下關(多防)：</b> {info['next_down']:.2f}<br/>"
                f"<b>本周關鍵價：</b> {info['curr_week_key']:.2f}<br/>"
                f"<b>本月關鍵價：</b> {info['curr_month_key']:.2f}<br/>"
                f"</div>"
                f"<div style='margin-top: 8px; padding: 4px; border-radius: 4px; text-align: center; font-size: 12px; font-weight: bold; "
                f"background-color: {'rgba(211,47,47,0.2)' if '漲破' in info['today_note'] or '強勢' in info['today_note'] else 'rgba(230,81,0,0.2)' if '偏多' in info['today_note'] else 'rgba(46,125,50,0.2)' if '偏空' in info['today_note'] else 'rgba(27,94,32,0.2)'}; "
                f"color: {'#ff5252' if '漲破' in info['today_note'] or '強勢' in info['today_note'] else '#ffa726' if '偏多' in info['today_note'] else '#66bb6a' if '偏空' in info['today_note'] else '#81c784'};'>"
                f"{info['today_note']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(f"{market_name} 資料載入失敗")

st.markdown("---")

# --- 2. 下方個別深度查詢 ---
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
            f"### 📊 【{target_name}】最新收盤 ({next_info['latest_date']}) 與 關鍵位階"
        )

        # 6 格 KPI 指標：收盤、明日上關、中關、下關、周關鍵價、月關鍵價
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric(
            f"最新收盤 ({next_info['latest_date']})",
            f"{next_info['latest_close']:.2f}",
            f"{next_info['latest_change']:+.2f} ({next_info['latest_pct']:+.2f}%)",
        )
        k2.metric(
            "預計明日 上關",
            f"{next_info['next_up']:.2f}",
            help="明日空方防守價位",
        )
        k3.metric(
            "預計明日 中關",
            f"{next_info['next_mid']:.2f}",
            help="明日多空強弱分水嶺",
        )
        k4.metric(
            "預計明日 下關",
            f"{next_info['next_down']:.2f}",
            help="明日多方防守價位",
        )
        k5.metric(
            "本周關鍵價",
            f"{next_info['curr_week_key']:.2f}",
            help="當周高低點中心價",
        )
        k6.metric(
            "本月關鍵價",
            f"{next_info['curr_month_key']:.2f}",
            help="當月高低點中心價",
        )

        # 狀態醒目標註卡片
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
                    "周關鍵價": "{:.2f}",
                    "月關鍵價": "{:.2f}",
                }
            )
            .map(color_change, subset=["漲跌", "漲跌幅(%)"])
            .map(color_description, subset=["說明"])
        )

        st.dataframe(styled_df, use_container_width=True, height=500)
