from datetime import datetime, timedelta
import io
import re

import pandas as pd
import requests
import streamlit as st
import twstock


# ============================================================
# Streamlit 設定
# ============================================================

st.set_page_config(
    page_title="三關價分析看板",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 基本設定
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

REQUEST_TIMEOUT = 15

# 三關價 Fibonacci
FIBO = 0.382


# ============================================================
# 共用：數字清理
# ============================================================

def clean_number(value):
    """
    將 API / CSV 中的數值轉成 float。
    支援：
    1,234.56
    1234.56
    --
    -
    None
    空字串
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text in ("", "-", "--", "---", "None", "nan", "NaN"):
        return None

    # 去掉千分位
    text = text.replace(",", "")

    # 去掉可能的空白
    text = text.replace(" ", "")

    try:
        return float(text)
    except Exception:
        return None


# ============================================================
# 共用：民國日期轉西元
# ============================================================

def parse_tw_date(value):
    """
    支援：

    115/08/19
    115-08-19
    2026/08/19
    2026-08-19
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = text.replace(".", "/")
    text = text.replace("-", "/")

    parts = text.split("/")

    if len(parts) != 3:
        return None

    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])

        if year < 1900:
            year += 1911

        return datetime(year, month, day)

    except Exception:
        return None


# ============================================================
# 1. 加權指數
# ============================================================

@st.cache_data(ttl=300)
def fetch_twse_index(months: int = 3) -> pd.DataFrame:
    """
    抓取 TWSE 發行量加權股價指數。

    使用 TWSE 官方：
    MI_5MINS_HIST

    每月抓一次。
    """

    records = []

    today = datetime.now()

    for i in range(months):

        target_date = today - timedelta(days=i * 30)

        date_str = target_date.strftime("%Y%m01")

        url = (
            "https://www.twse.com.tw/indicesReport/"
            f"MI_5MINS_HIST?response=json&date={date_str}"
        )

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            rows = data.get("data", [])

            for row in rows:

                if len(row) < 5:
                    continue

                date_value = parse_tw_date(row[0])

                if date_value is None:
                    continue

                open_price = clean_number(row[1])
                high_price = clean_number(row[2])
                low_price = clean_number(row[3])
                close_price = clean_number(row[4])

                if any(
                    value is None
                    for value in [
                        open_price,
                        high_price,
                        low_price,
                        close_price
                    ]
                ):
                    continue

                records.append(
                    {
                        "Date": date_value,
                        "Open": open_price,
                        "High": high_price,
                        "Low": low_price,
                        "Close": close_price,
                    }
                )

        except Exception as e:

            # 不直接中斷，繼續抓其他月份
            continue

    if not records:

        raise ValueError(
            "加權指數資料取得失敗。"
            "請確認 TWSE API 是否正常。"
        )

    df = pd.DataFrame(records)

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.set_index(
        "Date",
        inplace=True
    )

    return df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ]


# ============================================================
# 2. 櫃買指數
# ============================================================

@st.cache_data(ttl=300)
def fetch_tpex_index() -> pd.DataFrame:
    """
    使用 TPEx 官方 OpenAPI：

    https://www.tpex.org.tw/openapi/v1/tpex_index

    官方 OpenAPI 的 tpex_index 為：
    「櫃買指數歷史資料」

    注意：
    TPEx OpenAPI 目前提供的是歷史資料快照，
    因此這裡一次取得後再整理日期。
    """

    url = (
        "https://www.tpex.org.tw/openapi/v1/tpex_index"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        raise ValueError(
            f"櫃買指數 API 連線失敗：{e}"
        )

    # ========================================================
    # OpenAPI 通常直接回傳 list
    # ========================================================

    if isinstance(data, dict):

        # 保留幾種可能格式
        rows = (
            data.get("data")
            or data.get("aaData")
            or data.get("result")
            or []
        )

    elif isinstance(data, list):

        rows = data

    else:

        rows = []

    if not rows:

        raise ValueError(
            "櫃買指數 API 有回應，但沒有取得資料。"
        )

    records = []

    # ========================================================
    # 情況 A：
    # API 回傳 list[dict]
    # ========================================================

    if isinstance(rows[0], dict):

        for row in rows:

            # ------------------------------------------------
            # 找日期欄位
            # ------------------------------------------------

            date_value = None

            date_keys = [
                "Date",
                "date",
                "資料日期",
                "日期",
                "日期(民國年)",
                "交易日期",
            ]

            for key in date_keys:

                if key in row:

                    date_value = parse_tw_date(
                        row[key]
                    )

                    if date_value is not None:
                        break

            if date_value is None:
                continue

            # ------------------------------------------------
            # 找 OHLC
            # ------------------------------------------------

            def find_value(possible_keys):

                for key in possible_keys:

                    if key in row:

                        value = clean_number(
                            row[key]
                        )

                        if value is not None:
                            return value

                return None

            open_price = find_value(
                [
                    "Open",
                    "open",
                    "開市",
                    "開盤",
                    "開盤價",
                ]
            )

            high_price = find_value(
                [
                    "High",
                    "high",
                    "最高",
                    "最高價",
                ]
            )

            low_price = find_value(
                [
                    "Low",
                    "low",
                    "最低",
                    "最低價",
                ]
            )

            close_price = find_value(
                [
                    "Close",
                    "close",
                    "收市",
                    "收盤",
                    "收盤價",
                ]
            )

            if any(
                value is None
                for value in [
                    open_price,
                    high_price,
                    low_price,
                    close_price
                ]
            ):
                continue

            records.append(
                {
                    "Date": date_value,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                }
            )

    # ========================================================
    # 情況 B：
    # API 回傳 list[list]
    # ========================================================

    else:

        for row in rows:

            if len(row) < 5:
                continue

            date_value = parse_tw_date(row[0])

            if date_value is None:
                continue

            open_price = clean_number(row[1])
            high_price = clean_number(row[2])
            low_price = clean_number(row[3])
            close_price = clean_number(row[4])

            if any(
                value is None
                for value in [
                    open_price,
                    high_price,
                    low_price,
                    close_price
                ]
            ):
                continue

            records.append(
                {
                    "Date": date_value,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                }
            )

    if not records:

        raise ValueError(
            "櫃買指數 API 已取得資料，但無法解析 OHLC。"
        )

    df = pd.DataFrame(records)

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.set_index(
        "Date",
        inplace=True
    )

    return df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ]


# ============================================================
# 3. 台指期
# ============================================================

def get_taifex_daily(date_value):
    """
    從 TAIFEX 官方「期貨每日交易行情」取得某一天
    的 TX 一般交易時段資料。

    官方頁面：
    futDailyMarketExcel

    回傳：
    [
        {
            Month,
            Open,
            High,
            Low,
            Close
        },
        ...
    ]
    """

    date_str = date_value.strftime("%Y/%m/%d")

    url = (
        "https://www.taifex.com.tw/cht/3/"
        "futDailyMarketExcel"
    )

    params = {
        "marketCode": "0",
        "commodity_id": "TX",
        "queryStartDate": date_str,
        "queryEndDate": date_str,
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

    except Exception as e:

        raise ValueError(
            f"TAIFEX {date_str} 連線失敗：{e}"
        )

    # ========================================================
    # 期交所網頁可能直接是 HTML table
    # ========================================================

    try:

        tables = pd.read_html(
            io.StringIO(response.text)
        )

    except Exception:

        tables = []

    if not tables:

        return []

    records = []

    # ========================================================
    # 找出包含「契約」的 TX 表格
    # ========================================================

    for table in tables:

        if table.empty:
            continue

        # 多層欄位先攤平
        if isinstance(table.columns, pd.MultiIndex):

            table.columns = [
                "_".join(
                    [
                        str(x)
                        for x in col
                        if str(x) != "nan"
                    ]
                )
                for col in table.columns
            ]

        else:

            table.columns = [
                str(c).strip()
                for c in table.columns
            ]

        # ----------------------------------------------------
        # 找欄位
        # ----------------------------------------------------

        contract_col = None
        month_col = None
        open_col = None
        high_col = None
        low_col = None
        close_col = None

        for col in table.columns:

            col_text = str(col)

            if (
                contract_col is None
                and "契約" in col_text
            ):
                contract_col = col

            if (
                month_col is None
                and "到期" in col_text
            ):
                month_col = col

            if (
                open_col is None
                and "開盤價" in col_text
            ):
                open_col = col

            if (
                high_col is None
                and "最高價" in col_text
            ):
                high_col = col

            if (
                low_col is None
                and "最低價" in col_text
            ):
                low_col = col

            if (
                close_col is None
                and (
                    "最後成交價" in col_text
                    or "收盤價" in col_text
                )
            ):
                close_col = col

        # ====================================================
        # 如果沒有找到必要欄位
        # ====================================================

        if (
            month_col is None
            or open_col is None
            or high_col is None
            or low_col is None
            or close_col is None
        ):
            continue

        # ====================================================
        # 解析每個 TX 合約
        # ====================================================

        for _, row in table.iterrows():

            if contract_col is not None:

                contract = str(
                    row.get(contract_col, "")
                ).strip()

                if contract != "TX":

                    continue

            month = str(
                row.get(month_col, "")
            ).strip()

            # ------------------------------------------------
            # 只接受純月份，例如：
            # 202608
            # 202609
            #
            # 排除：
            # 202608/202609
            # 價差
            # ------------------------------------------------

            if not re.fullmatch(
                r"\d{6}",
                month
            ):
                continue

            open_price = clean_number(
                row.get(open_col)
            )

            high_price = clean_number(
                row.get(high_col)
            )

            low_price = clean_number(
                row.get(low_col)
            )

            close_price = clean_number(
                row.get(close_col)
            )

            if any(
                value is None
                for value in [
                    open_price,
                    high_price,
                    low_price,
                    close_price
                ]
            ):
                continue

            # 有些無成交合約會是 -
            if (
                open_price <= 0
                or high_price <= 0
                or low_price <= 0
                or close_price <= 0
            ):
                continue

            records.append(
                {
                    "Month": month,
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                }
            )

    return records


# ============================================================
# 台指期：抓最近交易日
# ============================================================

@st.cache_data(ttl=300)
def fetch_tx_futures(days: int = 60) -> pd.DataFrame:
    """
    抓取近月台指期日線。

    注意：
    台指期不像股票可以直接把不同月份串起來，
    因此這裡每天先取得 TX 各月份合約，
    再選當日最近的有效月份。

    例如：
    2026/08/19

    TX 202608
    TX 202609
    TX 202610

    若 202608 已到期，則會選 202609。
    """

    records = []

    today = datetime.now()

    # 從最近日期往前找
    # 60 個日曆日通常可以得到足夠的交易日
    for offset in range(days):

        target_date = today - timedelta(
            days=offset
        )

        # 未來日期不抓
        if target_date.date() > today.date():
            continue

        try:

            daily_records = get_taifex_daily(
                target_date
            )

        except Exception:
            continue

        if not daily_records:
            continue

        # ====================================================
        # 選擇真正最近月合約
        # ====================================================

        valid_records = []

        for record in daily_records:

            month = record["Month"]

            try:

                contract_date = datetime.strptime(
                    month,
                    "%Y%m"
                )

            except Exception:

                continue

            valid_records.append(
                (
                    contract_date,
                    record
                )
            )

        if not valid_records:
            continue

        # ----------------------------------------------------
        # 選到期月份最早者
        # ----------------------------------------------------

        valid_records.sort(
            key=lambda x: x[0]
        )

        _, near_contract = valid_records[0]

        records.append(
            {
                "Date": target_date,
                "Month": near_contract["Month"],
                "Open": near_contract["Open"],
                "High": near_contract["High"],
                "Low": near_contract["Low"],
                "Close": near_contract["Close"],
            }
        )

        # ====================================================
        # 已經取得足夠交易日就停止
        # ====================================================

        if len(records) >= 35:
            break

    if not records:

        raise ValueError(
            "台指期資料取得失敗。"
            "TAIFEX 官方每日行情可能暫時無法連線，"
            "或近期沒有可用的 TX 資料。"
        )

    df = pd.DataFrame(records)

    df.drop_duplicates(
        subset=["Date"],
        keep="last",
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.set_index(
        "Date",
        inplace=True
    )

    return df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ]


# ============================================================
# 4. 個股日 K
# ============================================================

@st.cache_data(ttl=300)
def fetch_stock_data(
    stock_id: str
) -> pd.DataFrame:

    stock_id = (
        stock_id
        .replace(".TW", "")
        .replace(".TWO", "")
        .strip()
    )

    if not stock_id.isdigit():

        raise ValueError(
            f"股票代號【{stock_id}】格式錯誤。"
        )

    try:

        stock = twstock.Stock(stock_id)

    except Exception as e:

        raise ValueError(
            f"twstock 無法建立股票物件：{e}"
        )

    today = datetime.now()

    all_data = []

    # 抓近幾個月
    for i in range(4):

        target_date = (
            today
            - timedelta(days=i * 28)
        )

        try:

            month_data = stock.fetch(
                target_date.year,
                target_date.month
            )

            all_data.extend(
                month_data
            )

        except Exception:
            continue

    if not all_data:

        raise ValueError(
            f"查無股票代號【{stock_id}】的交易數據。"
        )

    # 去除重複日期
    unique_dict = {
        item.date: item
        for item in all_data
    }

    sorted_data = sorted(
        unique_dict.values(),
        key=lambda x: x.date
    )

    records = []

    for item in sorted_data:

        if (
            item.open is None
            or item.high is None
            or item.low is None
            or item.close is None
        ):
            continue

        records.append(
            {
                "Date": item.date,
                "Open": float(item.open),
                "High": float(item.high),
                "Low": float(item.low),
                "Close": float(item.close),
            }
        )

    if not records:

        raise ValueError(
            f"股票【{stock_id}】沒有有效 OHLC 資料。"
        )

    df = pd.DataFrame(records)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df.set_index(
        "Date",
        inplace=True
    )

    df.sort_index(
        inplace=True
    )

    return df[
        [
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ]


# ============================================================
# 5. 判斷三關價位置
# ============================================================

def judge_position(row):

    close = row["Close"]

    up = row["上關(空方防守)"]
    mid = row["中關(日關)"]
    down = row["下關(多方防守)"]

    if any(
        pd.isna(value)
        for value in [
            up,
            mid,
            down
        ]
    ):
        return "計算中"

    if close > up:

        return "強勢：漲破空方防守"

    elif close >= mid:

        return "偏多：介於中關與上關之間"

    elif close >= down:

        return "偏空：小於中關但未破多方防守"

    else:

        return "弱勢：跌破多方防守"


# ============================================================
# 6. 三關價核心運算
# ============================================================

def calculate_three_passes(
    df: pd.DataFrame,
    days: int = 30
):

    df = df.copy()

    # ========================================================
    # 昨收
    # ========================================================

    df["昨收"] = df["Close"].shift(1)

    # ========================================================
    # 漲跌
    # ========================================================

    df["漲跌"] = (
        df["Close"]
        - df["昨收"]
    )

    # ========================================================
    # 漲跌幅
    # ========================================================

    df["漲跌幅(%)"] = (
        df["漲跌"]
        / df["昨收"]
        * 100
    )

    # ========================================================
    # 振幅
    # ========================================================

    df["振幅(%)"] = (
        (
            df["High"]
            - df["Low"]
        )
        / df["昨收"]
        * 100
    )

    # ========================================================
    # 前一天高低
    # ========================================================

    prev_high = df["High"].shift(1)
    prev_low = df["Low"].shift(1)

    diff = (
        prev_high
        - prev_low
    )

    # ========================================================
    # 三關價
    #
    # 上關 = 昨高 + (昨高 - 昨低) × 0.382
    #
    # 中關 = (昨高 + 昨低) / 2
    #
    # 下關 = 昨低 - (昨高 - 昨低) × 0.382
    # ========================================================

    df["上關(空方防守)"] = (
        prev_high
        + diff * FIBO
    )

    df["中關(日關)"] = (
        prev_high
        + prev_low
    ) / 2

    df["下關(多方防守)"] = (
        prev_low
        - diff * FIBO
    )

    # ========================================================
    # 判斷
    # ========================================================

    df["說明"] = df.apply(
        judge_position,
        axis=1
    )

    # ========================================================
    # 明日三關價
    # ========================================================

    latest_row = df.iloc[-1]

    current_high = latest_row["High"]
    current_low = latest_row["Low"]

    current_diff = (
        current_high
        - current_low
    )

    next_up = (
        current_high
        + current_diff * FIBO
    )

    next_mid = (
        current_high
        + current_low
    ) / 2

    next_down = (
        current_low
        - current_diff * FIBO
    )

    next_info = {

        "latest_date":
            df.index[-1].strftime(
                "%Y-%m-%d"
            ),

        "latest_open":
            latest_row["Open"],

        "latest_high":
            latest_row["High"],

        "latest_low":
            latest_row["Low"],

        "latest_close":
            latest_row["Close"],

        "latest_change":
            latest_row["漲跌"],

        "latest_pct":
            latest_row["漲跌幅(%)"],

        "next_up":
            next_up,

        "next_mid":
            next_mid,

        "next_down":
            next_down,

        "today_note":
            latest_row["說明"],
    }

    # ========================================================
    # 欄位名稱
    # ========================================================

    rename_map = {

        "Open": "開盤",
        "High": "最高",
        "Low": "最低",
        "Close": "收盤",

    }

    df.rename(
        columns=rename_map,
        inplace=True
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

    result = df[cols].tail(
        days
    ).copy()

    result.sort_index(
        ascending=False,
        inplace=True
    )

    result.index = (
        result.index
        .strftime("%Y-%m-%d")
    )

    result.index.name = "日期"

    return (
        result,
        next_info
    )


# ============================================================
# 7. 取得標的資料
# ============================================================

@st.cache_data(ttl=300)
def get_stock_three_passes(
    user_input: str,
    days: int = 30
):

    query = (
        user_input
        .strip()
        .upper()
    )

    try:

        # ====================================================
        # 加權指數
        # ====================================================

        if query in [
            "大盤",
            "加權",
            "加權指數",
            "^TWII",
            "0000",
            "TWA00",
        ]:

            df = fetch_twse_index()

            target_name = (
                "加權指數 (大盤)"
            )

        # ====================================================
        # 櫃買
        # ====================================================

        elif query in [
            "櫃買",
            "上櫃",
            "OTC",
            "櫃買指數",
            "店頭",
            "^TWO",
            "^TWOII",
            "OTC0",
        ]:

            df = fetch_tpex_index()

            target_name = (
                "櫃買指數 (TPEx)"
            )

        # ====================================================
        # 台指期
        # ====================================================

        elif query in [
            "台指期",
            "台指",
            "期貨",
            "TX",
            "TX00",
            "台指近月",
        ]:

            df = fetch_tx_futures()

            target_name = (
                "台指期貨 (近月連續)"
            )

        # ====================================================
        # 個股
        # ====================================================

        else:

            clean_id = (
                query
                .replace(".TW", "")
                .replace(".TWO", "")
            )

            df = fetch_stock_data(
                clean_id
            )

            target_name = (
                f"個股 {clean_id}"
            )

    except Exception as e:

        return (
            None,
            None,
            None,
            f"資料抓取失敗：{e}"
        )

    if df.empty or len(df) < 2:

        return (
            None,
            None,
            None,
            "查無交易數據或交易日不足。"
        )

    try:

        result, next_info = (
            calculate_three_passes(
                df,
                days=days
            )
        )

    except Exception as e:

        return (
            None,
            None,
            None,
            f"三關價計算失敗：{e}"
        )

    return (
        result,
        next_info,
        target_name,
        None
    )


# ============================================================
# 8. 顏色
# ============================================================

def color_change(value):

    if pd.isna(value):
        return ""

    try:

        value = float(value)

        if value > 0:

            return (
                "color: #ff4b4b; "
                "font-weight: 600;"
            )

        elif value < 0:

            return (
                "color: #09ab3b; "
                "font-weight: 600;"
            )

    except Exception:

        pass

    return ""


def color_description(value):

    if not isinstance(value, str):
        return ""

    if "漲破" in value:

        return (
            "color: #d32f2f; "
            "font-weight: bold;"
        )

    elif "偏多" in value:

        return (
            "color: #e65100; "
            "font-weight: 600;"
        )

    elif "偏空" in value:

        return (
            "color: #1565c0; "
            "font-weight: 600;"
        )

    elif "跌破" in value:

        return (
            "color: #1b5e20; "
            "font-weight: bold;"
        )

    return ""


# ============================================================
# 9. Streamlit 介面
# ============================================================

st.title(
    "📊 三關價分析看板"
)

st.caption(
    "支援：個股代號、加權指數、櫃買指數、台指期近月連續"
)


# ============================================================
# 搜尋區
# ============================================================

col1, col2 = st.columns(
    [3, 1]
)

with col1:

    user_input = st.text_input(
        "請輸入股號或標的名稱",
        value="大盤",
        placeholder=(
            "例如：2330、8069、加權、櫃買、台指期"
        )
    )

with col2:

    days_to_show = st.number_input(
        "顯示天數",
        min_value=5,
        max_value=60,
        value=30,
        step=5
    )


# ============================================================
# 快速按鈕
# ============================================================

st.write("快速查詢")

quick_col1, quick_col2, quick_col3, quick_col4, quick_col5 = (
    st.columns(5)
)

if quick_col1.button(
    "📈 加權",
    use_container_width=True
):

    st.session_state["quick_symbol"] = "大盤"

if quick_col2.button(
    "📊 櫃買",
    use_container_width=True
):

    st.session_state["quick_symbol"] = "櫃買"

if quick_col3.button(
    "📉 台指期",
    use_container_width=True
):

    st.session_state["quick_symbol"] = "台指期"

if quick_col4.button(
    "2330",
    use_container_width=True
):

    st.session_state["quick_symbol"] = "2330"

if quick_col5.button(
    "8069",
    use_container_width=True
):

    st.session_state["quick_symbol"] = "8069"


# ============================================================
# 如果按快速按鈕
# ============================================================

if "quick_symbol" in st.session_state:

    user_input = st.session_state[
        "quick_symbol"
    ]

    # 清除，避免每次 rerun 都卡住
    del st.session_state[
        "quick_symbol"
    ]


# ============================================================
# 查詢
# ============================================================

if user_input:

    with st.spinner(
        "正在取得最新交易資料..."
    ):

        (
            df_result,
            next_info,
            target_name,
            err
        ) = get_stock_three_passes(
            user_input,
            days=int(days_to_show)
        )

    # ========================================================
    # 錯誤
    # ========================================================

    if err:

        st.error(err)

        st.info(
            "如果是櫃買或台指期，請稍後重新查詢。"
            "交易所 API 偶爾會有連線或資料更新延遲。"
        )

    # ========================================================
    # 成功
    # ========================================================

    else:

        st.subheader(
            f"📊 【{target_name}】"
            f" 最新收盤 "
            f"({next_info['latest_date']})"
            " 與預計明日三關價"
        )

        # ====================================================
        # 最新價格
        # ====================================================

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric(
            "最新收盤",
            f"{next_info['latest_close']:.2f}",
            (
                f"{next_info['latest_change']:+.2f} "
                f"({next_info['latest_pct']:+.2f}%)"
            )
        )

        kpi2.metric(
            "預計明日 上關",
            f"{next_info['next_up']:.2f}",
            help="明日空方防守價位"
        )

        kpi3.metric(
            "預計明日 中關",
            f"{next_info['next_mid']:.2f}",
            help="明日多空強弱分水嶺"
        )

        kpi4.metric(
            "預計明日 下關",
            f"{next_info['next_down']:.2f}",
            help="明日多方防守價位"
        )

        # ====================================================
        # 今日 OHLC
        # ====================================================

        st.markdown(
            f"""
            ### 今日行情

            **開盤：** {next_info['latest_open']:.2f}  
            **最高：** {next_info['latest_high']:.2f}  
            **最低：** {next_info['latest_low']:.2f}  
            **收盤：** {next_info['latest_close']:.2f}
            """
        )

        # ====================================================
        # 今日三關價判斷
        # ====================================================

        note = next_info["today_note"]

        if "漲破" in note:

            st.success(
                f"📈 **最新交易日型態：{note}**"
            )

        elif "偏多" in note:

            st.warning(
                f"🟠 **最新交易日型態：{note}**"
            )

        elif "偏空" in note:

            st.info(
                f"🔵 **最新交易日型態：{note}**"
            )

        elif "跌破" in note:

            st.error(
                f"📉 **最新交易日型態：{note}**"
            )

        else:

            st.write(
                f"最新交易日型態：{note}"
            )

        # ====================================================
        # 詳細資料
        # ====================================================

        st.write(
            "### 最近交易日歷史資料"
        )

        styled_df = (
            df_result.style
            .format(
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
            .map(
                color_change,
                subset=[
                    "漲跌",
                    "漲跌幅(%)"
                ]
            )
            .map(
                color_description,
                subset=[
                    "說明"
                ]
            )
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=650
        )

        # ====================================================
        # 三關價說明
        # ====================================================

        with st.expander(
            "📖 三關價計算方式"
        ):

            st.markdown(
                """
                ### 三關價公式

                **上關（空方防守）**

                > 昨日最高 + (昨日最高 - 昨日最低) × 0.382

                **中關（日關）**

                > (昨日最高 + 昨日最低) ÷ 2

                **下關（多方防守）**

                > 昨日最低 - (昨日最高 - 昨日最低) × 0.382

                ### 收盤位置判斷

                - 🔴 收盤 > 上關 → **強勢：漲破空方防守**
                - 🟠 中關 ≤ 收盤 ≤ 上關 → **偏多：介於中關與上關之間**
                - 🔵 下關 ≤ 收盤 < 中關 → **偏空：小於中關但未破多方防守**
                - 🟢 收盤 < 下關 → **弱勢：跌破多方防守**
                """
            )
