from datetime import datetime, timedelta
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="三關價波段交易決策系統", layout="wide")

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


# --- 1. 股票名稱對照表 (快取 1 天) ---
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
def fetch_stock_or_index(data_id: str, days: int = 120) -> pd.DataFrame:
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
            "Trading_Volume": "Volume",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open", "High", "Low", "Close", "Volume"]]


# --- 3. 抓取台指期貨近月日K ---
def fetch_tx_futures(days: int = 120) -> pd.DataFrame:
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
            "volume": "Volume",
        },
        inplace=True,
    )
    near_df["Date"] = pd.to_datetime(near_df["Date"])
    near_df.set_index("Date", inplace=True)
    near_df.sort_index(inplace=True)
    if "Volume" not in near_df.columns:
        near_df["Volume"] = 0
    return near_df[["Open", "High", "Low", "Close", "Volume"]]


# --- 4. 抓取費城半導體指數 (SOX) ---
def fetch_sox_index(days: int = 120) -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ESOX?range=1y&interval=1d"
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
                    "Volume": indicators.get("volume", [0] * len(dates)),
                }
            )
            df.dropna(subset=["Close"], inplace=True)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            if not df.empty:
                return df[["Open", "High", "Low", "Close", "Volume"]]

    alt_url = "https://stooq.com/q/d/l/?s=soxx.us&i=d"
    alt_resp = requests.get(alt_url, headers=HEADERS, timeout=10)
    if alt_resp.status_code == 200 and alt_resp.text.strip():
        df = pd.read_csv(io.StringIO(alt_resp.text))
        if not df.empty and "Close" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            if "Volume" not in df.columns:
                df["Volume"] = 0
            return df[["Open", "High", "Low", "Close", "Volume"]]

    raise ValueError("費城半導體指數資料取得失敗。")


# --- 5. 波段策略、損益追蹤與三關價核心引擎 ---
def compute_three_passes_strategy(df: pd.DataFrame, days: int = 30):
    if df.empty or len(df) < 25:
        return None, None

    calc_df = df.copy()

    # 1. 基礎數據
    calc_df["昨收"] = calc_df["Close"].shift(1)
    calc_df["漲跌"] = calc_df["Close"] - calc_df["昨收"]
    calc_df["漲跌幅(%)"] = (calc_df["漲跌"] / calc_df["昨收"]) * 100
    calc_df["振幅(%)"] = (
        (calc_df["High"] - calc_df["Low"]) / calc_df["昨收"]
    ) * 100

    # 2. 三關價 (空防 / AC / 多防)
    prev_high = calc_df["High"].shift(1)
    prev_low = calc_df["Low"].shift(1)
    diff = prev_high - prev_low

    calc_df["空防"] = prev_high + (diff * 0.382)
    calc_df["AC"] = (prev_high + prev_low) / 2
    calc_df["多防"] = prev_low - (diff * 0.382)

    # 3. 周關 / 月關
    calc_df["Year_Week"] = calc_df.index.to_period("W")
    week_high = calc_df.groupby("Year_Week")["High"].cummax()
    week_low = calc_df.groupby("Year_Week")["Low"].cummin()
    calc_df["周關"] = (week_high + week_low) / 2

    calc_df["Year_Month"] = calc_df.index.to_period("M")
    month_high = calc_df.groupby("Year_Month")["High"].cummax()
    month_low = calc_df.groupby("Year_Month")["Low"].cummin()
    calc_df["月關"] = (month_high + month_low) / 2

    # 4. 趨勢大環境判斷 (台股慣例：多頭紅色 🔴、空頭綠色 🟢)
    def get_trend_env(row):
        c, w, m = row["Close"], row["周關"], row["月關"]
        if pd.isna(w) or pd.isna(m):
            return "分析中"
        if c >= w and c >= m:
            return "多頭 🔴"
        elif c <= w and c <= m:
            return "空頭 🟢"
        else:
            return "震盪 🟡"

    calc_df["趨勢環境"] = calc_df.apply(get_trend_env, axis=1)

    # 5. 型態說明
    def judge_position(row):
        close = row["Close"]
        up = row["空防"]
        mid = row["AC"]
        down = row["多防"]
        if pd.isna(up) or pd.isna(mid) or pd.isna(down):
            return "計算中"
        if close > up:
            return "強勢：漲破空防"
        elif close >= mid:
            return "偏多：介於AC與空防"
        elif close >= down:
            return "偏空：介於多防與AC"
        else:
            return "弱勢：跌破多防"

    calc_df["型態說明"] = calc_df.apply(judge_position, axis=1)

    # 6. 狀態機：計算訊號、進場價、出場價、獲利% 與 防守價
    signals = []
    positions = []
    holding_days = []
    entry_prices = []
    exit_prices = []
    pnl_percents = []
    defense_prices = []

    curr_pos = "NONE"  # NONE, LONG_100, LONG_50, SHORT_100, SHORT_50
    days_in_trade = 0
    active_entry_price = np.nan

    for i in range(len(calc_df)):
        row = calc_df.iloc[i]
        c = row["Close"]
        up = row["空防"]
        mid = row["AC"]
        dn = row["多防"]
        trend = row["趨勢環境"]

        if pd.isna(up) or pd.isna(dn) or pd.isna(row["月關"]):
            signals.append("計算中")
            positions.append("無部位")
            holding_days.append(0)
            entry_prices.append(np.nan)
            exit_prices.append(np.nan)
            pnl_percents.append(np.nan)
            defense_prices.append(np.nan)
            continue

        sig = "觀望等待"
        recorded_entry = np.nan
        recorded_exit = np.nan
        calculated_pnl = np.nan

        # --- 狀態切換邏輯 ---
        if curr_pos == "NONE":
            if c > up and ("多頭" in trend or c > row["月關"]):
                curr_pos = "LONG_100"
                days_in_trade = 1
                active_entry_price = c
                recorded_entry = active_entry_price
                calculated_pnl = 0.0
                sig = "🔥 多單進場"
            elif c < dn and ("空頭" in trend or c < row["月關"]):
                curr_pos = "SHORT_100"
                days_in_trade = 1
                active_entry_price = c
                recorded_entry = active_entry_price
                calculated_pnl = 0.0
                sig = "❄️ 空單進場"
            else:
                days_in_trade = 0
                active_entry_price = np.nan
                sig = "空倉觀望"

        elif "LONG" in curr_pos:
            days_in_trade += 1
            recorded_entry = active_entry_price
            if c < dn:  # 跌破多方防守 -> 清倉出場
                recorded_exit = c
                calculated_pnl = ((recorded_exit - active_entry_price) / active_entry_price) * 100
                curr_pos = "NONE"
                sig = "🚨 多單清倉"
                days_in_trade = 0
                active_entry_price = np.nan
            elif c < mid and curr_pos == "LONG_100":  # 破 AC 減碼
                curr_pos = "LONG_50"
                calculated_pnl = ((c - active_entry_price) / active_entry_price) * 100
                sig = "⚠️ 多單減碼50%"
            elif c > up and curr_pos == "LONG_50":  # 重返強勢加碼
                curr_pos = "LONG_100"
                calculated_pnl = ((c - active_entry_price) / active_entry_price) * 100
                sig = "➕ 多單加碼"
            else:
                calculated_pnl = ((c - active_entry_price) / active_entry_price) * 100
                sig = "多單續抱"

        elif "SHORT" in curr_pos:
            days_in_trade += 1
            recorded_entry = active_entry_price
            if c > up:  # 漲破空方防守 -> 清倉出場
                recorded_exit = c
                calculated_pnl = ((active_entry_price - recorded_exit) / active_entry_price) * 100
                curr_pos = "NONE"
                sig = "🚨 空單清倉"
                days_in_trade = 0
                active_entry_price = np.nan
            elif c > mid and curr_pos == "SHORT_100":  # 突破 AC 減碼
                curr_pos = "SHORT_50"
                calculated_pnl = ((active_entry_price - c) / active_entry_price) * 100
                sig = "⚠️ 空單減碼50%"
            elif c < dn and curr_pos == "SHORT_50":
                curr_pos = "SHORT_100"
                calculated_pnl = ((active_entry_price - c) / active_entry_price) * 100
                sig = "➕ 空單加碼"
            else:
                calculated_pnl = ((active_entry_price - c) / active_entry_price) * 100
                sig = "空單續抱"

        pos_display = {
            "NONE": "無部位",
            "LONG_100": "多單 100%",
            "LONG_50": "多單 50%",
            "SHORT_100": "空單 100%",
            "SHORT_50": "空單 50%",
        }.get(curr_pos, "無部位")

        defense_val = (
            dn if "LONG" in curr_pos else (up if "SHORT" in curr_pos else np.nan)
        )

        signals.append(sig)
        positions.append(pos_display)
        holding_days.append(days_in_trade)
        entry_prices.append(recorded_entry)
        exit_prices.append(recorded_exit)
        pnl_percents.append(calculated_pnl)
        defense_prices.append(defense_val)

    calc_df["波段訊號"] = signals
    calc_df["持倉狀態"] = positions
    calc_df["持倉天數"] = holding_days
    calc_df["進場價"] = entry_prices
    calc_df["出場價"] = exit_prices
    calc_df["獲利(%)"] = pnl_percents
    calc_df["防守價"] = defense_prices

    # 預估明日指標
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
        "curr_week_key": latest_row["周關"],
        "curr_month_key": latest_row["月關"],
        "today_note": latest_row["型態說明"],
        "today_signal": latest_row["波段訊號"],
        "today_position": latest_row["持倉狀態"],
        "today_hold_days": latest_row["持倉天數"],
        "today_entry": latest_row["進場價"],
        "today_pnl": latest_row["獲利(%)"],
        "today_defense": latest_row["防守價"],
        "trend_env": latest_row["趨勢環境"],
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

    display_cols = [
        "開盤",
        "最高",
        "最低",
        "收盤",
        "漲跌幅(%)",
        "型態說明",
        "波段訊號",
        "持倉狀態",
        "持倉天數",
        "進場價",
        "出場價",
        "獲利(%)",
        "防守價",
        "空防",
        "AC",
        "多防",
        "周關",
        "月關",
    ]

    result = calc_df[display_cols].tail(days).sort_index(ascending=False)
    result.index = result.index.strftime("%Y-%m-%d")
    result.index.name = "日期"

    return result, next_day_passes


# --- 取得標的資料 ---
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

        result, next_day_passes = compute_three_passes_strategy(df, days=days)
        return result, next_day_passes, target_name, None
    except Exception as e:
        return None, None, None, f"查詢失敗：{str(e)}"


# --- 取得市場四大核心概況 ---
@st.cache_data(ttl=300)
def get_market_overview():
    items = [
        ("加權指數 (大盤)", lambda: fetch_stock_or_index("TAIEX", days=30)),
        ("櫃買指數 (TPEx)", lambda: fetch_stock_or_index("TPEx", days=30)),
        ("台指期 (近月)", lambda: fetch_tx_futures(days=30)),
        ("費城半導體 (SOX)", lambda: fetch_sox_index(days=30)),
    ]
    overview_list = []
    for name, fetch_func in items:
        try:
            df = fetch_func()
            _, next_info = compute_three_passes_strategy(df, days=5)
            overview_list.append((name, next_info, None))
        except Exception as e:
            overview_list.append((name, None, str(e)))
    return overview_list


# --- 樣式設定輔助函式 ---
def color_change(val):
    if isinstance(val, (int, float)) and pd.notna(val):
        if val > 0:
            return "color: #ff4b4b;"
        elif val < 0:
            return "color: #09ab3b;"
    return ""


def color_pnl(val):
    if isinstance(val, (int, float)) and pd.notna(val):
        if val > 0:
            return "color: #ff4b4b; font-weight: bold;"
        elif val < 0:
            return "color: #09ab3b; font-weight: bold;"
    return ""


def color_signal(val):
    if not isinstance(val, str):
        return ""
    if "多單進場" in val or "多單加碼" in val:
        return "background-color: rgba(255, 75, 75, 0.25); color: #ff5252; font-weight: bold;"
    elif "空單進場" in val or "空單加碼" in val:
        return "background-color: rgba(9, 171, 59, 0.25); color: #69f0ae; font-weight: bold;"
    elif "清倉" in val:
        return "background-color: rgba(255, 235, 59, 0.2); color: #ffee58; font-weight: bold;"
    elif "減碼" in val:
        return "color: #ffa726; font-weight: bold;"
    return ""


def color_note(val):
    if not isinstance(val, str):
        return ""
    if "漲破" in val or "強勢" in val:
        return "color: #ff5252; font-weight: bold;"
    elif "偏多" in val:
        return "color: #ffa726;"
    elif "偏空" in val:
        return "color: #66bb6a;"
    elif "跌破" in val or "弱勢" in val:
        return "color: #2e7d32; font-weight: bold;"
    return ""


# ================= 網頁畫面呈現 =================
st.title("三關價波段交易決策系統")

# --- 1. 頂部常駐：市場四大核心看板 ---
st.markdown("### 🌐 市場核心指數・即時位階與波段訊號看板")
overview_data = get_market_overview()

cols_top = st.columns(4)
for idx, (market_name, info, err) in enumerate(overview_data):
    with cols_top[idx]:
        if info:
            st.markdown(
                f"<div style='border: 1px solid #333; border-radius: 8px; padding: 12px; background-color: rgba(255,255,255,0.03);'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<h4 style='margin:0; font-size:15px;'>{market_name}</h4>"
                f"<span style='font-size:11px; padding:2px 6px; border-radius:4px; background-color:#222;'>{info['trend_env']}</span>"
                f"</div>"
                f"<p style='color: gray; font-size: 11px; margin: 2px 0 6px 0;'>日期：{info['latest_date']}</p>"
                f"<div style='font-size: 18px; font-weight: bold;'>{info['latest_close']:.2f} "
                f"<span style='font-size: 12px; color: {'#ff4b4b' if info['latest_change'] > 0 else '#09ab3b'};'>"
                f"{info['latest_change']:+.2f} ({info['latest_pct']:+.2f}%)</span></div>"
                f"<hr style='margin: 6px 0; border: none; border-top: 1px solid #444;'/>"
                f"<div style='font-size: 12px; line-height: 1.5;'>"
                f"<b>明日空防：</b> {info['next_up']:.2f} | <b>AC：</b> {info['next_mid']:.2f}<br/>"
                f"<b>明日多防：</b> {info['next_down']:.2f} | <b>月關：</b> {info['curr_month_key']:.2f}<br/>"
                f"<b>當前持倉：</b> {info['today_position']} ({info['today_hold_days']}天)<br/>"
                f"</div>"
                f"<div style='margin-top: 6px; padding: 3px; border-radius: 4px; text-align: center; font-size: 11px; font-weight: bold; "
                f"background-color: {'rgba(211,47,47,0.2)' if '多單' in info['today_signal'] else 'rgba(46,125,50,0.2)' if '空單' in info['today_signal'] else 'rgba(100,100,100,0.2)'}; "
                f"color: {'#ff5252' if '多單' in info['today_signal'] else '#81c784' if '空單' in info['today_signal'] else '#bbb'};'>"
                f"訊號：{info['today_signal']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(f"{market_name} 載入失敗")

st.markdown("---")

# --- 2. 下方個別深度查詢與決策面板 ---
st.subheader("🔍 個股/指數波段決策與防守分析")
st.caption(
    "支援輸入：個股代號 (如: 2330)、大盤、櫃買、台指期、費半 (SOX)"
)

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input(
        "請輸入股號或標的名稱",
        value="2301",
        help="可輸入：2301、2330、大盤、櫃買、台指期、費半...",
    )
with col2:
    days_to_show = st.number_input(
        "顯示歷史天數", min_value=10, max_value=90, value=30
    )

if user_input:
    with st.spinner("策略模組計算中..."):
        df_result, next_info, target_name, err = get_target_data(
            user_input, days=days_to_show
        )

    if err:
        st.error(err)
    else:
        st.markdown(
            f"### 🎯 【{target_name}】波段狀態與明日戰略部署"
        )

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric(
            f"最新收盤 ({next_info['latest_date']})",
            f"{next_info['latest_close']:.2f}",
            f"{next_info['latest_change']:+.2f} ({next_info['latest_pct']:+.2f}%)",
        )
        k2.metric("趨勢大環境", next_info["trend_env"])
        
        # 持倉與損益顯示
        pnl_val = next_info["today_pnl"]
        pnl_display = f"{pnl_val:+.2f}%" if pd.notna(pnl_val) else "0.00%"
        k3.metric(
            "目前持倉水位",
            f"{next_info['today_position']}",
            f"損益: {pnl_display} ({next_info['today_hold_days']}天)",
        )
        
        defense_str = (
            f"{next_info['today_defense']:.2f}"
            if pd.notna(next_info["today_defense"])
            else "無 (空倉)"
        )
        k4.metric("動態防守點", defense_str, help="跌破多單清倉 / 漲破空單清倉")
        k5.metric(
            "預計明日 空防 (上關)",
            f"{next_info['next_up']:.2f}",
            help="空方主要防守防線",
        )
        k6.metric(
            "預計明日 多防 (下關)",
            f"{next_info['next_down']:.2f}",
            help="多方主要防守防線",
        )

        # 訊號狀態提示卡
        sig_color = (
            "error"
            if "多單" in next_info["today_signal"]
            else (
                "success"
                if "空單" in next_info["today_signal"]
                else "info"
            )
        )
        getattr(st, sig_color)(
            f"🔔 **波段訊號：【{next_info['today_signal']}】** ｜ 型態：{next_info['today_note']} ｜ AC：{next_info['next_mid']:.2f} ｜ 周關：{next_info['curr_week_key']:.2f} ｜ 月關：{next_info['curr_month_key']:.2f}"
        )

        # 詳細歷史決策與三關價報表
        st.write("#### 📋 交易訊號與關鍵價歷史記錄清單")

        format_dict = {
            "開盤": "{:.2f}",
            "最高": "{:.2f}",
            "最低": "{:.2f}",
            "收盤": "{:.2f}",
            "漲跌幅(%)": "{:+.2f}%",
            "進場價": lambda x: f"{x:.2f}" if pd.notna(x) else "-",
            "出場價": lambda x: f"{x:.2f}" if pd.notna(x) else "-",
            "獲利(%)": lambda x: f"{x:+.2f}%" if pd.notna(x) else "-",
            "防守價": lambda x: f"{x:.2f}" if pd.notna(x) else "-",
            "空防": "{:.2f}",
            "AC": "{:.2f}",
            "多防": "{:.2f}",
            "周關": "{:.2f}",
            "月關": "{:.2f}",
        }

        styled_df = (
            df_result.style.format(format_dict)
            .map(color_change, subset=["漲跌幅(%)"])
            .map(color_pnl, subset=["獲利(%)"])
            .map(color_signal, subset=["波段訊號"])
            .map(color_note, subset=["型態說明"])
        )

        st.dataframe(styled_df, use_container_width=True, height=530)
