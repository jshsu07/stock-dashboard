"""
股票技術指標監控儀表板 v4.0
- 隱藏 Streamlit 系統 UI（選單、頁尾、頁眉）
- 自選股改為 ticker + 🗑️ 垃圾桶刪除
- 台股輸入純數字自動補 .TW
- 台股名稱改繁體中文（TWSE API + 備用對照表）
- 技術建議說明表格放在頁面最下方
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ════════════════════════════════════════════════════
# 0. 頁面設定
# ════════════════════════════════════════════════════
st.set_page_config(
    page_title="📈 股票技術指標儀表板",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    /* 隱藏 Streamlit 系統 UI */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    .stApp { font-family: 'Segoe UI', 'Microsoft JhengHei', Arial, sans-serif; }

    .gradient-title {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #0066cc 0%, #00aa44 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .legend-row {
        background: #f0f4f8; border-radius: 8px; padding: 0.6rem 1rem;
        font-size: 0.82rem; margin-top: 0.8rem; border-left: 4px solid #0066cc;
    }
    /* 垃圾桶按鈕縮小 */
    div[data-testid="stButton"] button[title^="移除"] {
        padding: 0.15rem 0.4rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 1. 常數
# ════════════════════════════════════════════════════
DEFAULT_US_STOCKS = ["NVDA", "TSLA", "QQQ", "MU", "AVGO"]
DEFAULT_TW_STOCKS = ["2330.TW", "0050.TW"]
RSI_PERIOD        = 14
KD_K_PERIOD, KD_K_SMOOTH, KD_D_SMOOTH = 14, 3, 3
HISTORY_PERIOD    = "6mo"

# 台股繁體中文名稱備用對照表（TWSE API 無法取得時使用）
_TW_NAMES_FALLBACK = {
    # ETF
    "0050": "元大台灣50",      "0056": "元大高股息",
    "006208": "富邦台50",      "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息","00929": "復華台灣科技優息",
    "00940": "元大台灣價值高息","00900": "富邦特選高股息30",
    "00713": "元大台灣高息低波","00850": "元大臺灣ESG永續",
    "00881": "國泰台灣5G+",    "00892": "富邦台灣半導體",
    # 上市大型股
    "2330": "台積電",  "2317": "鴻海",    "2454": "聯發科",
    "2382": "廣達",    "2308": "台達電",   "2303": "聯電",
    "3711": "日月光投控","2327": "國巨",   "2344": "華邦電",
    "3034": "聯詠",    "2395": "研華",     "2379": "瑞昱",
    "2408": "南亞科",  "2357": "華碩",     "2353": "宏碁",
    "2376": "技嘉",    "2377": "微星",     "2409": "友達",
    "3481": "群創",    "2356": "英業達",   "2382": "廣達",
    "2474": "可成",    "2360": "致茂",
    "2603": "長榮",    "2609": "陽明",     "2615": "萬海",
    "2618": "長榮航",  "2633": "台灣高鐵",
    "2412": "中華電信","4904": "遠傳",     "3045": "台灣大",
    "2002": "中鋼",    "1301": "台塑",     "1303": "南亞",
    "1326": "台化",    "6505": "台塑化",   "1216": "統一",
    "2912": "統一超",  "2207": "和泰車",   "9910": "豐泰",
    "3008": "大立光",  "5871": "中租-KY",
    "1101": "台泥",    "1102": "亞泥",
    # 金控
    "2881": "富邦金",  "2882": "國泰金",  "2886": "兆豐金",
    "2891": "中信金",  "2884": "玉山金",  "2892": "第一金",
    "2880": "華南金",  "2885": "元大金",  "2883": "開發金",
    "2887": "台新金",  "2890": "永豐金",  "5880": "合庫金",
    "2801": "彰銀",
    # 上櫃常見
    "3661": "世芯-KY", "6415": "矽力-KY", "3443": "創意",
    "5274": "信驊",    "6669": "緯穎",    "3406": "玉晶光",
    "8046": "南電",    "3533": "嘉澤",    "3105": "穩懋",
    "6271": "同欣電",  "3037": "欣興",    "4966": "譜瑞-KY",
}


# ════════════════════════════════════════════════════
# 2. 台股繁體中文名稱查詢（TWSE API + 備用表）
# ════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def _get_tw_name_map() -> dict:
    """
    從台灣證交所 OpenAPI 取得全部上市股票繁體中文名稱。
    每 24 小時更新一次，失敗時使用備用對照表。
    """
    import requests
    names = dict(_TW_NAMES_FALLBACK)   # 先載入備用表
    try:
        r = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
            timeout=8,
        )
        for item in r.json():
            code = str(item.get("Code", "")).strip()
            name = str(item.get("Name", "")).strip()
            if code and name:
                names[code] = name
    except Exception:
        pass
    return names


def _tw_chinese_name(code: str, fallback: str) -> str:
    """查詢台股繁體中文名稱，找不到回傳 fallback（英文名或代號）"""
    return _get_tw_name_map().get(code, fallback)


# ════════════════════════════════════════════════════
# 3. Supabase 資料庫
# ════════════════════════════════════════════════════
@st.cache_resource
def _get_supabase():
    try:
        from supabase import create_client
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        return None


def _load_watchlist(username: str):
    sb = _get_supabase()
    if not sb:
        return DEFAULT_US_STOCKS.copy(), DEFAULT_TW_STOCKS.copy()
    try:
        res = sb.table("watchlists").select("us_stocks,tw_stocks").eq("user_name", username).execute()
        if res.data:
            return res.data[0]["us_stocks"], res.data[0]["tw_stocks"]
    except Exception:
        pass
    return DEFAULT_US_STOCKS.copy(), DEFAULT_TW_STOCKS.copy()


def _save_watchlist():
    sb = _get_supabase()
    if not sb:
        return
    try:
        sb.table("watchlists").upsert({
            "user_name": st.session_state.username,
            "us_stocks": st.session_state.us_watchlist,
            "tw_stocks": st.session_state.tw_watchlist,
        }).execute()
    except Exception:
        pass


# ════════════════════════════════════════════════════
# 4. 登入系統（URL ?u= 書籤方式）
# ════════════════════════════════════════════════════
def _check_url_login():
    if "username" in st.session_state:
        return
    name = st.query_params.get("u", "").strip()
    if name:
        us, tw = _load_watchlist(name)
        st.session_state.username     = name
        st.session_state.us_watchlist = us
        st.session_state.tw_watchlist = tw
        st.session_state.last_refresh = datetime.now()


def _do_login(name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    us, tw = _load_watchlist(name)
    st.session_state.username     = name
    st.session_state.us_watchlist = us
    st.session_state.tw_watchlist = tw
    st.session_state.last_refresh = datetime.now()
    st.query_params["u"] = name
    return True


def _do_logout():
    if "u" in st.query_params:
        del st.query_params["u"]
    for k in ["username", "us_watchlist", "tw_watchlist", "last_refresh"]:
        st.session_state.pop(k, None)
    st.rerun()


def render_login_screen():
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div style='text-align:center; padding:3.5rem 0 1.5rem 0;'>
            <div style='font-size:3.5rem;'>📈</div>
            <div style='font-size:1.5rem; font-weight:700; color:#1e2a3a; margin-top:0.6rem;'>
                股票技術指標儀表板
            </div>
        </div>
        """, unsafe_allow_html=True)
        name = st.text_input("name", placeholder="輸入你的名字", label_visibility="collapsed")
        if st.button("進入 →", type="primary", use_container_width=True):
            if _do_login(name):
                st.session_state._show_bookmark_hint = True
                st.rerun()
            else:
                st.error("請輸入名字")


# ════════════════════════════════════════════════════
# 5. 技術指標計算
# ════════════════════════════════════════════════════
def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_kd(high, low, close,
                 k_period=KD_K_PERIOD, k_smooth=KD_K_SMOOTH, d_smooth=KD_D_SMOOTH):
    lowest   = low.rolling(k_period).min()
    highest  = high.rolling(k_period).max()
    hl_range = (highest - lowest).replace(0, np.nan)
    rsv = (100 * (close - lowest) / hl_range).fillna(50)
    k   = rsv.rolling(k_smooth).mean()
    d   = k.rolling(d_smooth).mean()
    return k, d


# ════════════════════════════════════════════════════
# 6. 技術建議
# ════════════════════════════════════════════════════
def _get_signal(rsi: float, k: float, d: float,
                kd_golden_cross: bool, kd_dead_cross: bool) -> str:
    if rsi < 30 and kd_golden_cross:    return "🔥 強力買進"
    if rsi > 70 and kd_dead_cross:      return "🔴 強力賣出"
    if rsi < 30 or kd_golden_cross:     return "🟢 考慮買進"
    if rsi > 70 or kd_dead_cross:       return "🟠 考慮賣出"
    if rsi < 45 and k < 40:             return "🔵 偏低留意"
    if rsi > 55 and k > 60:             return "🟡 偏高留意"
    return "⚪ 中性觀望"


# ════════════════════════════════════════════════════
# 7. 數據獲取（快取 5 分鐘）
# ════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=HISTORY_PERIOD)
        if hist is None or hist.empty or len(hist) < 30:
            return {"ticker": ticker, "error": "歷史數據不足或代號不存在"}

        rsi_s    = calculate_rsi(hist["Close"])
        k_s, d_s = calculate_kd(hist["High"], hist["Low"], hist["Close"])

        latest_rsi   = float(rsi_s.dropna().iloc[-1])
        latest_k     = float(k_s.dropna().iloc[-1])
        latest_d     = float(d_s.dropna().iloc[-1])
        latest_close = float(hist["Close"].iloc[-1])
        prev_close   = float(hist["Close"].iloc[-2])
        change_pct   = (latest_close - prev_close) / prev_close * 100

        # 英文名（yfinance 回傳）
        try:
            info = stock.info
            yf_name = (info.get("longName") or info.get("shortName") or ticker)[:30]
        except Exception:
            yf_name = ticker

        # 台股改繁體中文名稱
        is_tw = ticker.endswith(".TW") or ticker.endswith(".TWO")
        if is_tw:
            code = ticker.split(".")[0]
            name = _tw_chinese_name(code, yf_name)
        else:
            name = yf_name

        # KD 訊號偵測
        kd_golden_cross = kd_dead_cross = False
        k_clean, d_clean = k_s.dropna(), d_s.dropna()
        if len(k_clean) >= 2 and len(d_clean) >= 2:
            k_prev, k_now = float(k_clean.iloc[-2]), float(k_clean.iloc[-1])
            d_prev, d_now = float(d_clean.iloc[-2]), float(d_clean.iloc[-1])
            if k_prev < d_prev and k_now >= d_now and k_now < 20 and d_now < 20:
                kd_golden_cross = True
            if k_prev > d_prev and k_now <= d_now and k_now > 80 and d_now > 80:
                kd_dead_cross = True

        return {
            "ticker": ticker, "name": name,
            "close": latest_close, "change_pct": change_pct,
            "rsi": latest_rsi, "k": latest_k, "d": latest_d,
            "kd_golden_cross": kd_golden_cross,
            "kd_dead_cross":   kd_dead_cross,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ════════════════════════════════════════════════════
# 8. 表格樣式
# ════════════════════════════════════════════════════
def _style_rsi(s):
    return [
        "background-color:#c6efce;color:#276221;font-weight:700" if v < 30
        else "background-color:#ffc7ce;color:#9c0006;font-weight:700" if v > 70
        else "" for v in s
    ]

def _style_kd(s):
    return [
        "background-color:#e2efda;color:#375623" if v < 20
        else "background-color:#fce4d6;color:#833c0b" if v > 80
        else "" for v in s
    ]

def _style_change(s):
    return [
        "color:#c00000;font-weight:700" if v > 0
        else "color:#375623;font-weight:700" if v < 0
        else "color:gray" for v in s
    ]

def _style_signal(s):
    out = []
    for v in s:
        v = str(v)
        if   "強力買進" in v: out.append("background-color:#00897b;color:white;font-weight:700")
        elif "考慮買進" in v: out.append("background-color:#c6efce;color:#1b5e20;font-weight:700")
        elif "強力賣出" in v: out.append("background-color:#c62828;color:white;font-weight:700")
        elif "考慮賣出" in v: out.append("background-color:#ffc7ce;color:#7f0000;font-weight:700")
        elif "偏低留意" in v: out.append("background-color:#e3f2fd;color:#0d47a1")
        elif "偏高留意" in v: out.append("background-color:#fff8e1;color:#e65100")
        else:                  out.append("")
    return out

def _fmt_change(v):
    if pd.isna(v): return "-"
    return f"▲ {v:.2f}%" if v > 0 else f"▼ {abs(v):.2f}%" if v < 0 else f"{v:.2f}%"

def build_styled_df(df: pd.DataFrame):
    return (
        df.style
        .apply(_style_rsi,    subset=["RSI(14)"])
        .apply(_style_kd,     subset=["K值", "D值"])
        .apply(_style_change, subset=["漲跌幅(%)"])
        .apply(_style_signal, subset=["技術建議"])
        .format({
            "最新收盤價": "{:.2f}", "漲跌幅(%)": _fmt_change,
            "RSI(14)": "{:.1f}", "K值": "{:.1f}", "D值": "{:.1f}",
        }, na_rep="-")
    )


# ════════════════════════════════════════════════════
# 9. 自選股管理 UI
#    - 台股：純數字自動補 .TW
#    - 刪除：垃圾桶 🗑️ 在代號右側，不另起一排
# ════════════════════════════════════════════════════
def _normalize_ticker(raw: str, is_tw: bool) -> str:
    """台股輸入純數字時自動補 .TW 後綴"""
    raw = raw.strip().upper()
    if not is_tw:
        return raw
    if raw.endswith(".TW") or raw.endswith(".TWO"):
        return raw
    if raw.isdigit():
        return f"{raw}.TW"
    return raw


def render_watchlist_manager(watchlist_key: str, placeholder: str, is_tw: bool = False):
    col_inp, col_add, col_reset = st.columns([3, 1, 1])
    with col_inp:
        new_ticker = st.text_input(
            "ticker", placeholder=placeholder,
            key=f"input_{watchlist_key}", label_visibility="collapsed",
        )
    with col_add:
        if st.button("➕ 新增", key=f"add_{watchlist_key}", use_container_width=True):
            ticker = _normalize_ticker(new_ticker, is_tw)
            if not ticker:
                st.toast("請先輸入股票代號", icon="⚠️")
            elif ticker in st.session_state[watchlist_key]:
                st.toast(f"{ticker} 已在清單中", icon="ℹ️")
            else:
                st.session_state[watchlist_key].append(ticker)
                _save_watchlist()
                st.cache_data.clear()
                st.rerun()
    with col_reset:
        defaults = DEFAULT_US_STOCKS if "us" in watchlist_key else DEFAULT_TW_STOCKS
        if st.button("↺ 重置", key=f"reset_{watchlist_key}", use_container_width=True):
            st.session_state[watchlist_key] = defaults.copy()
            _save_watchlist()
            st.cache_data.clear()
            st.rerun()

    # ── 自選股標籤（代號 + 🗑️ 同列）──────────────
    watchlist = st.session_state[watchlist_key]
    if not watchlist:
        st.caption("清單為空，請新增股票代號")
        return

    ITEMS_PER_ROW = 6
    for start in range(0, len(watchlist), ITEMS_PER_ROW):
        batch = watchlist[start:start + ITEMS_PER_ROW]
        # 每個 ticker 佔 [ticker欄, 垃圾桶欄] = [2.5, 0.4]
        ratios = sum([[2.5, 0.4] for _ in batch], [])
        cols   = st.columns(ratios)
        for j, tkr in enumerate(batch):
            with cols[j * 2]:
                st.markdown(
                    f"<div style='padding-top:6px; font-weight:600;'>{tkr}</div>",
                    unsafe_allow_html=True,
                )
            with cols[j * 2 + 1]:
                if st.button("🗑️", key=f"del_{watchlist_key}_{tkr}", help=f"移除 {tkr}"):
                    st.session_state[watchlist_key].remove(tkr)
                    _save_watchlist()
                    st.cache_data.clear()
                    st.rerun()


# ════════════════════════════════════════════════════
# 10. 股票資料表格
# ════════════════════════════════════════════════════
def render_stock_table(watchlist: list):
    if not watchlist:
        st.info("⚠️ 自選股清單為空，請在上方新增股票代號")
        return

    rows, errors = [], []
    status = st.empty()
    prog   = st.progress(0)

    for i, ticker in enumerate(watchlist):
        status.caption(f"📡 正在載入：{ticker}  ({i+1}/{len(watchlist)})")
        prog.progress((i + 1) / len(watchlist))
        result = fetch_stock_data(ticker)
        if "error" in result:
            errors.append(f"**{result['ticker']}**：{result['error']}")
        else:
            rows.append({
                "股票代號":   result["ticker"],
                "股票名稱":   result["name"],
                "最新收盤價": result["close"],
                "漲跌幅(%)":  result["change_pct"],
                "RSI(14)":    result["rsi"],
                "K值":        result["k"],
                "D值":        result["d"],
                "技術建議":   _get_signal(
                    result["rsi"], result["k"], result["d"],
                    result["kd_golden_cross"], result["kd_dead_cross"],
                ),
            })

    status.empty()
    prog.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} 個代號異常"):
            for m in errors:
                st.warning(m)

    if not rows:
        st.error("所有代號均無法獲取數據，請確認格式並重試。")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        build_styled_df(df),
        use_container_width=True,
        hide_index=True,
        height=min(600, 55 + 38 * len(df)),
        column_config={
            "股票代號": st.column_config.TextColumn("股票代號", width="small"),
            "股票名稱": st.column_config.TextColumn("股票名稱", width="medium"),
            "技術建議": st.column_config.TextColumn("技術建議", width="medium"),
        },
    )
    st.markdown("""
    <div class="legend-row">
        <b>RSI / KD 顏色：</b>&ensp;
        <span style='background:#c6efce;color:#276221;padding:2px 7px;border-radius:4px;font-weight:700'>RSI &lt; 30</span> 超賣&ensp;
        <span style='background:#ffc7ce;color:#9c0006;padding:2px 7px;border-radius:4px;font-weight:700'>RSI &gt; 70</span> 超買&ensp;
        <span style='background:#e2efda;color:#375623;padding:2px 7px;border-radius:4px'>KD &lt; 20</span> 低檔&ensp;
        <span style='background:#fce4d6;color:#833c0b;padding:2px 7px;border-radius:4px'>KD &gt; 80</span> 高檔
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 11. 技術建議說明（頁面最下方）
# ════════════════════════════════════════════════════
def render_signal_legend():
    st.markdown("---")
    st.markdown("#### 📖 技術建議說明")
    st.markdown("""
| 訊號 | 觸發條件 |
|:----:|---------|
| 🔥 強力買進 | RSI < 30 **且** KD 在 20 以下黃金交叉（K 由下往上穿越 D） |
| 🟢 考慮買進 | RSI < 30 **或** KD 在 20 以下黃金交叉（單一訊號） |
| 🔴 強力賣出 | RSI > 70 **且** KD 在 80 以上死亡交叉（K 由上往下穿越 D） |
| 🟠 考慮賣出 | RSI > 70 **或** KD 在 80 以上死亡交叉（單一訊號） |
| 🔵 偏低留意 | RSI 介於 30～45 且 K 值 < 40（相對低檔，尚未超賣） |
| 🟡 偏高留意 | RSI 介於 55～70 且 K 值 > 60（相對高檔，尚未超買） |
| ⚪ 中性觀望 | 以上條件均不符合 |

> **RSI（相對強弱指標）**：衡量近期漲跌幅強弱，14 日為標準週期。
> **KD（隨機指標）**：K 值反映收盤位置在近期高低區間的相對強度，D 值為 K 值平均線。
> ⚠️ 技術指標僅供輔助參考，不構成投資建議，投資人應自行判斷。
""")


# ════════════════════════════════════════════════════
# 12. 主程式
# ════════════════════════════════════════════════════
def main():
    _check_url_login()

    if "username" not in st.session_state:
        render_login_screen()
        st.stop()

    if st.session_state.pop("_show_bookmark_hint", False):
        st.toast("✅ 登入成功！請將目前網址加入書籤，下次點擊直接進入", icon="📎")

    # ── Sidebar ──────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("登出", use_container_width=True):
            _do_logout()
        st.divider()
        st.caption("數據來源：Yahoo Finance\n數據延遲約 15 分鐘\n不構成投資建議")

    # ── Header ───────────────────────────────────
    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown('<div class="gradient-title">📈 股票技術指標儀表板</div>', unsafe_allow_html=True)
        st.caption(f"最後更新：{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}｜數據僅供參考，不構成投資建議")
    with hdr_r:
        st.write(""); st.write("")
        if st.button("🔄 更新數據", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    st.divider()

    # ── 美股 ──────────────────────────────────────
    st.markdown("### 🇺🇸 美股")
    render_watchlist_manager("us_watchlist", "輸入美股代號，例如：AAPL、MSFT", is_tw=False)
    render_stock_table(st.session_state.us_watchlist)

    st.divider()

    # ── 台股 ──────────────────────────────────────
    st.markdown("### 🇹🇼 台股")
    st.info("💡 直接輸入股票代號數字即可（如 `2330`），系統自動補 `.TW`；上櫃股請手動加 `.TWO`（如 `3661.TWO`）")
    render_watchlist_manager("tw_watchlist", "輸入台股代號，例如：2330、0056", is_tw=True)
    render_stock_table(st.session_state.tw_watchlist)

    # ── 技術建議說明（頁尾）─────────────────────
    render_signal_legend()

    st.caption("⚠️ 免責聲明：本儀表板提供之數據與技術指標僅供輔助分析，不構成任何投資建議。投資有風險，請審慎評估，自負盈虧。")


if __name__ == "__main__":
    main()
