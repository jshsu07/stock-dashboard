"""
股票技術指標監控儀表板 v5.0
- 刪除欄移至表格最後一欄（勾選即刪）
- 修正台積電 None：close 取 dropna() 最後一筆
- Yahoo Finance 限流錯誤改顯示中文說明
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 1. 常數
# ════════════════════════════════════════════════════
DEFAULT_US_STOCKS = ["NVDA", "TSLA", "QQQ", "MU", "AVGO"]
DEFAULT_TW_STOCKS = ["2330.TW", "0050.TW"]
RSI_PERIOD = 14
KD_K_PERIOD, KD_K_SMOOTH, KD_D_SMOOTH = 14, 3, 3
HISTORY_PERIOD = "1y"

_TW_NAMES_FALLBACK = {
    "0050": "元大台灣50",       "0056": "元大高股息",
    "006208": "富邦台50",       "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息", "00929": "復華台灣科技優息",
    "00940": "元大台灣價值高息", "00900": "富邦特選高股息30",
    "00713": "元大台灣高息低波", "00850": "元大臺灣ESG永續",
    "00881": "國泰台灣5G+",     "00892": "富邦台灣半導體",
    "2330": "台積電",  "2317": "鴻海",    "2454": "聯發科",
    "2382": "廣達",    "2308": "台達電",   "2303": "聯電",
    "3711": "日月光投控","2327": "國巨",   "2344": "華邦電",
    "3034": "聯詠",    "2395": "研華",     "2379": "瑞昱",
    "2408": "南亞科",  "2357": "華碩",     "2353": "宏碁",
    "2376": "技嘉",    "2377": "微星",     "2409": "友達",
    "3481": "群創",    "2356": "英業達",   "2474": "可成",
    "2360": "致茂",    "2603": "長榮",     "2609": "陽明",
    "2615": "萬海",    "2618": "長榮航",   "2633": "台灣高鐵",
    "2412": "中華電信","4904": "遠傳",     "3045": "台灣大",
    "2002": "中鋼",    "1301": "台塑",     "1303": "南亞",
    "1326": "台化",    "6505": "台塑化",   "1216": "統一",
    "2912": "統一超",  "2207": "和泰車",   "9910": "豐泰",
    "3008": "大立光",  "5871": "中租-KY",  "1101": "台泥",
    "1102": "亞泥",
    "2881": "富邦金",  "2882": "國泰金",  "2886": "兆豐金",
    "2891": "中信金",  "2884": "玉山金",  "2892": "第一金",
    "2880": "華南金",  "2885": "元大金",  "2883": "開發金",
    "2887": "台新金",  "2890": "永豐金",  "5880": "合庫金",
    "2801": "彰銀",
    "3661": "世芯-KY", "6415": "矽力-KY", "3443": "創意",
    "5274": "信驊",    "6669": "緯穎",    "3406": "玉晶光",
    "8046": "南電",    "3533": "嘉澤",    "3105": "穩懋",
    "6271": "同欣電",  "3037": "欣興",    "4966": "譜瑞-KY",
}


# ════════════════════════════════════════════════════
# 2. 台股繁體中文名稱
# ════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def _get_tw_name_map() -> dict:
    import requests
    names = dict(_TW_NAMES_FALLBACK)
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
    return _get_tw_name_map().get(code, fallback)


# ════════════════════════════════════════════════════
# 3. Supabase
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
                kd_golden_cross_mid: bool,
                kd_golden_cross_any: bool,
                kd_dead_cross_high: bool) -> str:
    """
    右側順勢突破策略訊號（優先順序由強到弱）

    強勢過熱：RSI > 75 且 KD 高檔（>80）死亡交叉 → 果斷分批獲利
    注意利潤：RSI > 70 或 KD 高檔死亡交叉 → 調整停損，防守持股
    強勢發動：RSI 48~55 且 KD 在 30~60 黃金交叉 → 主升段最佳起漲買點
    突破試單：RSI > 50 或 KD 黃金交叉（單一訊號）→ 試單建立部位
    潛在打底：RSI 35~45 且 K、D 均 < 35 → 醞釀期，列入觀察名單
    中性觀望：以上條件均不符合
    """
    if rsi > 75 and kd_dead_cross_high:
        return "🔔 強勢過熱"
    if rsi > 70 or kd_dead_cross_high:
        return "🟠 注意利潤"
    if 48 <= rsi <= 55 and kd_golden_cross_mid:
        return "🔥 強勢發動"
    if rsi > 50 or kd_golden_cross_any:
        return "🟢 突破試單"
    if 35 <= rsi <= 45 and k < 35 and d < 35:
        return "🔵 潛在打底"
    return "⚪ 中性觀望"


# ════════════════════════════════════════════════════
# 7. 數據獲取（快取 5 分鐘）
# ════════════════════════════════════════════════════
def _fmt_change(v) -> str:
    """漲跌幅格式化，含 None / NaN 防護；🟢 漲 / 🔴 跌（美股慣例）"""
    try:
        f = float(v)
        if np.isnan(f):
            return "-"
        if f > 0:
            return f"🟢 ▲{f:.2f}%"
        elif f < 0:
            return f"🔴 ▼{abs(f):.2f}%"
        else:
            return "⚪ 0.00%"
    except (TypeError, ValueError):
        return "-"


def _humanize_error(msg: str) -> str:
    """將 yfinance 常見錯誤轉為中文說明"""
    m = msg.lower()
    if "too many requests" in m or "rate limit" in m or "429" in m:
        return "⏳ Yahoo Finance 請求次數超限，請等 1~2 分鐘後點「更新數據」"
    if "no data found" in m or "歷史數據不足" in m:
        return "❌ 找不到此代號，請確認格式（美股：AAPL；台股：2330 或 2330.TW）"
    if "connection" in m or "timeout" in m:
        return "🌐 網路連線逾時，請稍後重試"
    return msg


@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=HISTORY_PERIOD)
        if hist is None or hist.empty or len(hist) < 30:
            return {"ticker": ticker, "error": "歷史數據不足或代號不存在"}

        # 用 dropna() 取最後兩筆有效收盤價，避免末行為 None/NaN 顯示為 None
        close_valid = hist["Close"].replace({None: np.nan}).dropna()
        if len(close_valid) < 2:
            return {"ticker": ticker, "error": "有效收盤價不足，無法計算"}
        latest_close = float(close_valid.iloc[-1])
        prev_close   = float(close_valid.iloc[-2])
        change_pct   = (latest_close - prev_close) / prev_close * 100

        rsi_s    = calculate_rsi(hist["Close"])
        k_s, d_s = calculate_kd(hist["High"], hist["Low"], hist["Close"])

        latest_rsi = float(rsi_s.dropna().iloc[-1])
        latest_k   = float(k_s.dropna().iloc[-1])
        latest_d   = float(d_s.dropna().iloc[-1])

        try:
            info = stock.info
            yf_name = (info.get("longName") or info.get("shortName") or ticker)[:30]
        except Exception:
            yf_name = ticker

        is_tw = ticker.endswith(".TW") or ticker.endswith(".TWO")
        name  = _tw_chinese_name(ticker.split(".")[0], yf_name) if is_tw else yf_name

        kd_golden_cross_any = kd_golden_cross_mid = kd_dead_cross_high = False
        k_clean, d_clean = k_s.dropna(), d_s.dropna()
        if len(k_clean) >= 2 and len(d_clean) >= 2:
            k_prev, k_now = float(k_clean.iloc[-2]), float(k_clean.iloc[-1])
            d_prev, d_now = float(d_clean.iloc[-2]), float(d_clean.iloc[-1])
            # 黃金交叉：K 由下往上穿越 D
            if k_prev < d_prev and k_now >= d_now:
                kd_golden_cross_any = True
                # 中段黃金交叉（30~60）→ 強勢發動訊號
                if 30 <= k_now <= 60 and 30 <= d_now <= 60:
                    kd_golden_cross_mid = True
            # 高檔死亡交叉：K 由上往下穿越 D，且 K、D 均 > 80
            if k_prev > d_prev and k_now <= d_now and k_now > 80 and d_now > 80:
                kd_dead_cross_high = True

        # OHLC 資料（K 線圖用，跟指標共用同一次 API 請求，無額外消耗）
        ohlc = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        if ohlc.index.tz is not None:
            ohlc.index = ohlc.index.tz_convert(None)
        ohlc = ohlc.dropna(subset=["Open", "High", "Low", "Close"])

        return {
            "ticker": ticker, "name": name,
            "close": latest_close, "change_pct": change_pct,
            "rsi": latest_rsi, "k": latest_k, "d": latest_d,
            "kd_golden_cross_any": kd_golden_cross_any,
            "kd_golden_cross_mid": kd_golden_cross_mid,
            "kd_dead_cross_high":  kd_dead_cross_high,
            "ohlc": ohlc,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ════════════════════════════════════════════════════
# 8. 自選股管理（新增 / 重置，刪除改至表格內）
# ════════════════════════════════════════════════════
def _normalize_ticker(raw: str, is_tw: bool) -> str:
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


# ════════════════════════════════════════════════════
# 9. 股票資料表格（最後一欄：勾選刪除）
# ════════════════════════════════════════════════════
def render_stock_table(watchlist: list, watchlist_key: str):
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
            errors.append(f"**{result['ticker']}**：{_humanize_error(result['error'])}")
        else:
            rows.append({
                "股票代號":   result["ticker"],
                "股票名稱":   result["name"],
                "最新收盤價": round(result["close"], 2),
                "漲跌幅":     _fmt_change(result["change_pct"]),
                "RSI(14)":    round(result["rsi"], 1),
                "K值":        round(result["k"], 1),
                "D值":        round(result["d"], 1),
                "技術建議":   _get_signal(
                    result["rsi"], result["k"], result["d"],
                    result["kd_golden_cross_mid"],
                    result["kd_golden_cross_any"],
                    result["kd_dead_cross_high"],
                ),
                "🗑️": False,
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

    edited = st.data_editor(
        df,
        key=f"editor_{watchlist_key}",
        column_config={
            "股票代號":   st.column_config.TextColumn("股票代號", width="small"),
            "股票名稱":   st.column_config.TextColumn("股票名稱", width="medium"),
            "最新收盤價": st.column_config.NumberColumn("最新收盤價", format="%.2f", width="small"),
            "漲跌幅":     st.column_config.TextColumn("漲跌幅", width="small"),
            "RSI(14)":    st.column_config.NumberColumn(
                "RSI(14)", format="%.1f", width="small",
                help="< 30 超賣  |  > 70 超買",
            ),
            "K值":        st.column_config.NumberColumn(
                "K值", format="%.1f", width="small",
                help="< 20 低檔  |  > 80 高檔",
            ),
            "D值":        st.column_config.NumberColumn("D值", format="%.1f", width="small"),
            "技術建議":   st.column_config.TextColumn("技術建議", width="medium"),
            "🗑️":        st.column_config.CheckboxColumn(
                "刪除",
                help="勾選後自動從自選股移除",
                default=False,
                width="small",
            ),
        },
        disabled=["股票代號", "股票名稱", "最新收盤價", "漲跌幅",
                  "RSI(14)", "K值", "D值", "技術建議"],
        hide_index=True,
        use_container_width=True,
        height=min(600, 56 + 36 * (len(df) + 1)),
    )

    # 勾選即刪除
    to_delete = edited[edited["🗑️"] == True]["股票代號"].tolist()
    if to_delete:
        for t in to_delete:
            if t in st.session_state[watchlist_key]:
                st.session_state[watchlist_key].remove(t)
        _save_watchlist()
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='background:#f0f4f8; border-radius:8px; padding:0.5rem 1rem;
                font-size:0.82rem; margin-top:0.5rem; border-left:4px solid #0066cc;'>
        <b>RSI / KD 參考：</b>&ensp;
        RSI &lt; 30 超賣 &ensp;｜&ensp; RSI &gt; 70 超買 &ensp;｜&ensp;
        KD &lt; 20 低檔 &ensp;｜&ensp; KD &gt; 80 高檔
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 10. K 線走勢圖（點擊展開，可切換股票與區間）
# ════════════════════════════════════════════════════
def render_chart_section(watchlist: list, watchlist_key: str):
    if not watchlist:
        return

    with st.expander("📊 K 線走勢圖", expanded=False):
        col_sel, col_range = st.columns([2, 3])
        with col_sel:
            selected = st.selectbox(
                "股票",
                options=watchlist,
                key=f"chart_sel_{watchlist_key}",
                label_visibility="collapsed",
            )
        with col_range:
            period_label = st.radio(
                "區間",
                options=["30 天", "3 個月", "1 年"],
                horizontal=True,
                key=f"chart_period_{watchlist_key}",
                label_visibility="collapsed",
            )

        n_days = {"30 天": 30, "3 個月": 90, "1 年": 365}[period_label]

        result = fetch_stock_data(selected)
        if "error" in result or result.get("ohlc") is None:
            st.warning(f"無法取得 {selected} 的走勢圖資料")
            return

        df = result["ohlc"].tail(n_days).copy()
        if len(df) < 5:
            st.warning("資料筆數不足，無法顯示走勢圖")
            return

        name = result.get("name", selected)
        df["MA5"]  = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()

        vol_colors = [
            "#26a69a" if float(c) >= float(o) else "#ef5350"
            for c, o in zip(df["Close"], df["Open"])
        ]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
        )

        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
            showlegend=False, name=selected,
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA5"],
            name="MA5", line=dict(color="#ff9800", width=1.3),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["MA20"],
            name="MA20", line=dict(color="#2196f3", width=1.3),
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=vol_colors,
            showlegend=False, name="成交量",
        ), row=2, col=1)

        fig.update_layout(
            title=dict(
                text=f"<b>{selected}</b>　{name}",
                font=dict(size=14, color="#1e2a3a"),
                x=0, xanchor="left",
            ),
            height=430,
            margin=dict(l=0, r=10, t=45, b=0),
            xaxis_rangeslider_visible=False,
            plot_bgcolor="#fafafa",
            paper_bgcolor="white",
            legend=dict(
                orientation="h", x=0.01, y=1.05,
                font=dict(size=11), bgcolor="rgba(0,0,0,0)",
            ),
            yaxis=dict(
                gridcolor="#eeeeee", side="right",
                showgrid=True, tickformat=".2f",
            ),
            yaxis2=dict(gridcolor="#eeeeee", side="right", showgrid=False),
            xaxis2=dict(gridcolor="#eeeeee", showgrid=True),
        )

        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════
# 11. 技術建議說明（頁面最下方）
# ════════════════════════════════════════════════════
def render_signal_legend():
    st.markdown("---")
    st.markdown("#### 📖 技術建議說明（右側順勢突破策略）")
    st.markdown("""
| 訊號 | 觸發條件 | 策略實務意義 |
|:----:|---------|------------|
| 🔥 強勢發動 | RSI **48～55** 且 KD 在 **30～60** 之間黃金交叉 | 最完美的「主升段起漲點」。代表股價拉回整理結束，多方成功奪回主導權，動能同時點火，屬於高勝率的效率買點。 |
| 🟢 突破試單 | RSI > 50 **或** KD 剛完成黃金交叉（單一訊號） | 趨勢已開始向多頭傾斜，但兩大指標尚未完全共振。適合先建立基本部位（試單），等待另一個指標到位再加碼。 |
| 🔔 強勢過熱 | RSI > 75 **且** KD 在 80 以上出現死亡交叉 | 股價進入極度瘋狂的超買區，且短線過熱動能已開始失速。右側交易者應在此時果斷分批獲利入袋。 |
| 🟠 注意利潤 | RSI > 70 **或** KD 在 80 以上死亡交叉（單一訊號） | 股價已進入高檔區，隨時可能面臨震盪。此時不宜追高，應開始調整移動停損點（防守性持股）。 |
| 🔵 潛在打底 | RSI **35～45** 且 KD 正在低檔打底（K、D 均 < 35） | 股價正在進行修正或橫盤。這通常是「下一波起漲前的醞釀期」，列入觀察名單。 |
| ⚪ 中性觀望 | 以上條件均不符合 | 市場目前沒有明顯趨勢，可能處於洗盤或盤整期，資金應保留實力，不盲目出手。 |

> **RSI**（相對強弱指標）：14 日週期，衡量近期漲跌幅強弱。
> **KD**（隨機指標）：14,3,3 設定。K 值反映收盤在近期高低區間的位置，D 值為 K 的平均線。
> ⚠️ 技術指標僅供輔助參考，不構成投資建議，請自行判斷風險。
""")


# ════════════════════════════════════════════════════
# 11. 主程式
# ════════════════════════════════════════════════════
def main():
    _check_url_login()

    if "username" not in st.session_state:
        render_login_screen()
        st.stop()

    if st.session_state.pop("_show_bookmark_hint", False):
        st.toast("✅ 登入成功！請將目前網址加入書籤，下次點擊直接進入", icon="📎")

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        if st.button("登出", use_container_width=True):
            _do_logout()
        st.divider()
        st.caption("數據來源：Yahoo Finance\n數據延遲約 15 分鐘\n不構成投資建議")

    hdr_l, hdr_r = st.columns([5, 1])
    with hdr_l:
        st.markdown('<div class="gradient-title">📈 股票技術指標儀表板</div>', unsafe_allow_html=True)
        st.caption(f"最後更新：{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}｜數據僅供參考")
    with hdr_r:
        st.write(""); st.write("")
        if st.button("🔄 更新數據", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    st.divider()

    st.markdown("### 🇺🇸 美股")
    render_watchlist_manager("us_watchlist", "輸入美股代號，例如：AAPL、MSFT", is_tw=False)
    render_stock_table(st.session_state.us_watchlist, "us_watchlist")
    render_chart_section(st.session_state.us_watchlist, "us_watchlist")

    st.divider()

    st.markdown("### 🇹🇼 台股")
    st.info("💡 直接輸入數字即可（如 `2330`），系統自動補 `.TW`；上櫃股請手動加 `.TWO`（如 `3661.TWO`）")
    render_watchlist_manager("tw_watchlist", "輸入台股代號，例如：2330、0056", is_tw=True)
    render_stock_table(st.session_state.tw_watchlist, "tw_watchlist")
    render_chart_section(st.session_state.tw_watchlist, "tw_watchlist")

    render_signal_legend()
    st.caption("⚠️ 免責聲明：本儀表板提供之數據與技術指標僅供輔助分析，不構成任何投資建議。投資有風險，請審慎評估，自負盈虧。")


if __name__ == "__main__":
    main()
