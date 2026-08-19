from datetime import datetime, timedelta
import io
import pandas as pd
import requests
import streamlit as st
import twstock

st.set_page_config(page_title="三關價分析看板", layout="wide")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# --- 1. 抓取加權指數 (大盤) 完整開高低收 ---
def fetch_twse_index(months: int = 3) -> pd.DataFrame:
    records = []
    today = datetime.now()

    for i in range(months):
        target_date = today - timedelta(days=i * 30)
        date_str = target_date.strftime("%Y%m01")
        url = f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={date_str}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            res_json = resp.json()
            if "data" in res_json:
                for row in res_json["data"]:
                    # 日期格式: 民國年/月/日 (如 113/05/20)
                    parts = row[0].split("/")
                    ad_year = int(parts[0]) + 1911
                    d_str = f"{ad_year}-{parts[1]}-{parts[2]}"

                    def clean_val(v):
                        return float(v.replace(",", ""))

                    records.append(
                        {
                            "Date": d_str,
                            "Open": clean_val(row[1]),
                            "High": clean_val(row[2]),
                            "Low": clean_val(row[3]),
                            "Close": clean_val(row[4]),
                        }
                    )
        except Exception:
            continue

    if not records:
        raise ValueError("加權指數連線失敗，請稍候重試。")

    df = pd.DataFrame(records).drop_duplicates(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df


# --- 2. 抓取櫃買指數 (TPEx 官方 OpenAPI) ---
def fetch_tpex_index() -> pd.DataFrame:
    url = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_index"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        raise ValueError(
            f"櫃買 OpenAPI 連線失敗 (HTTP {resp.status_code})"
        )

    data = resp.json()
    records = []
    for row in data:
        d_raw = str(row.get("Date", ""))
        # 處理日期格式 (8碼西元或民國格式)
        if len(d_raw) == 8 and d_raw.isdigit():
            d_str = f"{d_raw[:4]}-{d_raw[4:6]}-{d_raw[6:]}"
        elif "/" in d_raw:
            parts = d_raw.split("/")
            ad_year = int(parts[0]) + 1911 if int(parts[0]) < 1900 else int(parts[0])
            d_str = f"{ad_year}-{int(parts[1]):02d}-{int(parts[2]):02d}"
        else:
            continue

        try:
            records.append(
                {
                    "Date": d_str,
                    "Open": float(str(row.get("OpeningIndex", 0)).replace(",", "")),
                    "High": float(str(row.get("HighestIndex", 0)).replace(",", "")),
                    "Low": float(str(row.get("LowestIndex", 0)).replace(",", "")),
                    "Close": float(str(row.get("ClosingIndex", 0)).replace(",", "")),
                }
            )
        except Exception:
            continue

    if not records:
        raise ValueError("櫃買指數資料解析為空。")

    df = pd.DataFrame(records).drop_duplicates(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df


# --- 3. 抓取台指期 (近月連續月) 日K資料 ---
def fetch_tx_futures() -> pd.DataFrame:
    # 透過期交所歷史日報下載 CSV (包含連續 30 個交易日)
    today = datetime.now()
    start_date = today - timedelta(days=60)
    url = "https://www.taifex.com.tw/cht/3/futDataDown"

    payload = {
        "down_type": "1",
        "commodity_id": "TX",
        "queryStartDate": start_date.strftime("%Y/%m/%d"),
        "queryEndDate": today.strftime("%Y/%m/%d"),
    }

    resp = requests.post(url, data=payload, headers=HEADERS, timeout=10)
    if resp.status_code != 200 or not resp.text.strip():
        # 備援：若 POST 失敗，嘗試呼叫期交所 OpenAPI
        api_url = "https://openapi.taifex.com.tw/v1/DailyMarketReportFut"
        api_resp = requests.get(api_url, headers=HEADERS, timeout=10)
        data = api_resp.json()
        tx_records = []
        for item in data:
            if item.get("ContractCode") == "TX" and item.get("TradingSession") in ["", "0", "Regular"]:
                d_raw = str(item.get("Date", ""))
                d_str = f"{d_raw[:4]}-{d_raw[4:6]}-{d_raw[6:]}" if len(d_raw) == 8 else d_raw
                tx_records.append(
                    {
                        "Date": d_str,
                        "Month": str(item.get("DeliveryMonth", "")),
                        "Open": float(str(item.get("OpenPrice", 0)).replace(",", "")),
                        "High": float(str(item.get("HighPrice", 0)).replace(",", "")),
                        "Low": float(str(item.get("LowPrice", 0)).replace(",", "")),
                        "Close": float(str(item.get("SettlementPrice") or item.get("ClosePrice", 0)).replace(",", "")),
                    }
                )
        raw_df = pd.DataFrame(tx_records)
    else:
        # 解析 CSV (big5 編碼)
        content = resp.content.decode("big5", errors="ignore")
        csv_df = pd.read_csv(io.StringIO(content))
        # 整理欄位
        csv_df.columns = [c.strip() for c in csv_df.columns]
        # 篩選一般交易時段
        if "交易時段" in csv_df.columns:
            csv_df = csv_df[csv_df["交易時段"].astype(str).str.contains("一般|0")]

        # 篩選欄位：交易日期, 到期月份, 開盤價, 最高價, 最低價, 收盤價/結算價
        records = []
        for _, row in csv_df.iterrows():
            d_raw = str(row.get("交易日期", "")).strip().replace("/", "-")
            month = str(row.get("到期月份(週別)", "")).strip()
            if "/" in month:  # 排除跨月價差合約
                continue
            try:
                records.append(
                    {
                        "Date": d_raw,
                        "Month": month,
                        "Open": float(str(row.get("開盤價", 0)).replace(",", "").replace("-", "0")),
                        "High": float(str(row.get("最高價", 0)).replace(",", "").replace("-", "0")),
                        "Low": float(str(row.get("最低價", 0)).replace(",", "").replace("-", "0")),
                        "Close": float(str(row.get("收盤價") or row.get("結算價", 0)).replace(",", "").replace("-", "0")),
                    }
                )
            except Exception:
                continue
        raw_df = pd.DataFrame(records)

    if raw_df.empty:
        raise ValueError("台指期資料取得為空，請確認是否為交易時段後。")

    raw_df = raw_df[raw_df["Open"] > 0]
    raw_df.sort_values(by=["Date", "Month"], inplace=True)
    # 各交易日只取近月合約 (第一筆)
    near_df = raw_df.groupby("Date").first().reset_index()

    near_df["Date"] = pd.to_datetime(near_df["Date"])
    near_df.set_index("Date", inplace=True)
    near_df.sort_index(inplace=True)
    return near_df[["Open", "High", "Low", "Close"]]


# --- 4. 抓取個股日K (twstock) ---
def fetch_stock_data(stock_id: str) -> pd.DataFrame:
    stock = twstock.Stock(stock_id)
    today = datetime.now()
    all_data = []

    for i in range(3):
        m_date = today - timedelta(days=i * 28)
        all_data.extend(stock.fetch(m_date.year, m_date.month))

    if not all_data:
        raise ValueError(f"查無股票代號【{stock_id}】的交易數據。")

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


# --- 三關價核心運算 ---
@st.cache_data(ttl=300)
def get_stock_three_passes(user_input: str, days: int = 30):
    query = user_input.strip().upper()

    try:
        if query in ["大盤", "加權", "加權指數", "^TWII", "0000", "TWA00"]:
            df = fetch_twse_index()
            target_name = "加權指數 (大盤)"
        elif query in ["櫃買", "上櫃", "OTC", "^TWOII", "櫃買指數", "店頭", "OTC0"]:
            df = fetch_tpex_index()
            target_name = "櫃買指數 (TPEx)"
        elif query in ["台指期", "台指", "期貨", "TX", "TX00", "台指近月"]:
            df = fetch_tx_futures()
            target_name = "台指期貨 (近月連續)"
        else:
            clean_id = query.replace(".TW", "").replace(".TWO", "")
            df = fetch_stock_data(clean_id)
            target_name = f"個股 {clean_id}"
    except Exception as e:
        return None, None, None, f"資料抓取失敗：{str(e)}"

    if df.empty or len(df) < 2:
        return None, None, None, "查無交易數據或交易日不足。"

    # 1. 昨收、漲跌、漲跌幅、振幅
    df["昨收"] = df["Close"].shift(1)
    df["漲跌"] = df["Close"] - df["昨收"]
    df["漲跌幅(%)"] = (df["漲跌"] / df["昨收"]) * 100
    df["振幅(%)"] = ((df["High"] - df["Low"]) / df["昨收"]) * 100

    # 2. 當日三關價
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

    # 5. 格式整理
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
    "支援輸入：個股代號 (如: 2330 / 8069)、大盤 (加權)、櫃買 (上櫃)、台指期 (TX)"
)

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("請輸入股號或標的名稱", value="大盤")
with col2:
    days_to_show = st.number_input(
        "顯示天數", min_value=5, max_value=60, value=30
    )

if user_input:
    with st.spinner("向官方交易所取得即時數據中..."):
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
