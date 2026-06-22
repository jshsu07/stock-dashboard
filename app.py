"""
股票技術指標監控儀表板 v3.0
- 登入：輸入名字 → URL 含 ?u=名字 → 書籤此頁 → 下次免登入
- 美股 + 台股同頁顯示
- 技術建議欄位（RSI + KD 組合訊號）
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
    .section-header {
        font-size: 1.2rem; font-weight: 700; margin: 0.5rem 0;
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


# ════════════════════════════════════════════════════
# 2. Supabase 資料庫
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
# 3. 登入系統（URL 書籤方式）
#
# 運作原理：
#   登入後 URL 變成 ?u=你的名字
#   把這個完整網址存成書籤
#   下次點書籤 → URL 帶有 ?u= → 自動登入，不需要再輸入
#   換電腦：輸入名字一次 → 再存書籤即可
# ════════════════════════════════════════════════════
def _check_url_login():
    """從 URL 的 ?u= 參數自動登入（書籤方式）"""
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
    st.query_params["u"] = name   # 將名字寫入 URL，讓使用者可以書籤存起來
    return True


def _do_logout():
    if "u" in st.query_params:
        del st.query_params["u"]
    for k in ["username", "us_watchlist", "tw_watchlist", "last_refresh"]:
        st.session_state.pop(k, None)
    st.rerun()


def render_login_screen():
    """登入頁面：輸入名字即可進入"""
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

        name = st.text_input(
            "name",
            placeholder="輸入你的名字",
            label_visibility="collapsed",
        )
        if st.button("進入 →", type="primary", use_container_width=True):
            if _do_login(name):
                st.session_state._show_bookmark_hint = True
                st.rerun()
            else:
                st.error("請輸入名字")


# ════════════════════════════════════════════════════
# 4. 技術指標計算
# ════════════════════════════════════════════════════
def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI — Wilder's Smoothing Method"""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_kd(high, low, close,
                 k_period=KD_K_PERIOD, k_smooth=KD_K_SMOOTH, d_smooth=KD_D_SMOOTH):
    """慢速 KD(14,3,3)"""
    lowest   = low.rolling(k_period).min()
    highest  = high.rolling(k_period).max()
    hl_range = (highest - lowest).replace(0, np.nan)
    rsv = (100 * (close - lowest) / hl_range).fillna(50)
    k   = rsv.rolling(k_smooth).mean()
    d   = k.rolling(d_smooth).mean()
    return k, d


# ════════════════════════════════════════════════════
# 5. 技術建議函數
# ════════════════════════════════════════════════════
def _get_signal(rsi: float, k: float, d: float,
                kd_golden_cross: bool, kd_dead_cross: bool) -> str:
    """
    根據 RSI 和 KD 組合給出技術分析建議，優先順序由強到弱：

    強力買進：RSI < 30 且 KD 低檔黃金交叉（雙重超賣確認）
    強力賣出：RSI > 70 且 KD 高檔死亡交叉（雙重超買確認）
    考慮買進：RSI < 30 或 KD 低檔黃金交叉（單一超賣訊號）
    考慮賣出：RSI > 70 或 KD 高檔死亡交叉（單一超買訊號）
    偏低留意：RSI 在 30~45 且 KD 低於 40（相對低檔但未到超賣）
    偏高留意：RSI 在 55~70 且 KD 高於 60（相對高檔但未到超買）
    中性觀望：其餘情況
    """
    if rsi < 30 and kd_golden_cross:
        return "🔥 強力買進"
    if rsi > 70 and kd_dead_cross:
        return "🔴 強力賣出"
    if rsi < 30 or kd_golden_cross:
        return "🟢 考慮買進"
    if rsi > 70 or kd_dead_cross:
        return "🟠 考慮賣出"
    if rsi < 45 and k < 40:
        return "🔵 偏低留意"
    if rsi > 55 and k > 60:
        return "🟡 偏高留意"
    return "⚪ 中性觀望"


# ════════════════════════════════════════════════════
# 6. 數據獲取（快取 5 分鐘）
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

        try:
            info = stock.info
            name = (info.get("longName") or info.get("shortName") or ticker)[:25]
        except Exception:
            name = ticker

        # KD 黃金交叉（低檔 K 上穿 D）
        kd_golden_cross = False
        # KD 死亡交叉（高檔 K 下穿 D）
        kd_dead_cross   = False
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
# 7. 表格樣式
# ════════════════════════════════════════════════════
def _style_rsi(s):
    out = []
    for v in s:
        if pd.isna(v):  out.append("")
        elif v < 30:    out.append("background-color:#c6efce;color:#276221;font-weight:700")
        elif v > 70:    out.append("background-color:#ffc7ce;color:#9c0006;font-weight:700")
        else:           out.append("")
    return out

def _style_kd(s):
    out = []
    for v in s:
        if pd.isna(v):  out.append("")
        elif v < 20:    out.append("background-color:#e2efda;color:#375623")
        elif v > 80:    out.append("background-color:#fce4d6;color:#833c0b")
        else:           out.append("")
    return out

def _style_change(s):
    out = []
    for v in s:
        if pd.isna(v):  out.append("")
        elif v > 0:     out.append("color:#c00000;font-weight:700")
        elif v < 0:     out.append("color:#375623;font-weight:700")
        else:           out.append("color:gray")
    return out

def _style_signal(s):
    out = []
    for v in s:
        v = str(v)
        if "強力買進" in v:   out.append("background-color:#00897b;color:white;font-weight:700")
        elif "考慮買進" in v:  out.append("background-color:#c6efce;color:#1b5e20;font-weight:700")
        elif "強力賣出" in v:  out.append("background-color:#c62828;color:white;font-weight:700")
        elif "考慮賣出" in v:  out.append("background-color:#ffc7ce;color:#7f0000;font-weight:700")
        elif "偏低留意" in v:  out.append("background-color:#e3f2fd;color:#0d47a1")
        elif "偏高留意" in v:  out.append("background-color:#fff8e1;color:#e65100")
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
# 8. 自選股管理 UI
# ════════════════════════════════════════════════════
def render_watchlist_manager(watchlist_key: str, placeholder: str):
    col_inp, col_add, col_reset = st.columns([3, 1, 1])
    with col_inp:
        new_ticker = st.text_input(
            "ticker", placeholder=placeholder,
            key=f"input_{watchlist_key}", label_visibility="collapsed",
        )
    with col_add:
        if st.button("➕ 新增", key=f"add_{watchlist_key}", use_container_width=True):
            ticker = str(new_ticker).strip().upper()
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

    watchlist = st.session_state[watchlist_key]
    if watchlist:
        cols = st.columns(min(len(watchlist), 8))
        for i, tkr in enumerate(list(watchlist)):
            with cols[i % 8]:
                if st.button(f"✕ {tkr}", key=f"del_{watchlist_key}_{tkr}", use_container_width=True):
                    st.session_state[watchlist_key].remove(tkr)
                    _save_watchlist()
                    st.cache_data.clear()
                    st.rerun()
    else:
        st.caption("清單為空，請新增股票代號")


# ════════════════════════════════════════════════════
# 9. 股票資料表格
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
        <b>顏色圖例：</b>&ensp;
        <span style='background:#c6efce;color:#276221;padding:2px 7px;border-radius:4px;font-weight:700'>RSI &lt; 30</span> 超賣&ensp;
        <span style='background:#ffc7ce;color:#9c0006;padding:2px 7px;border-radius:4px;font-weight:700'>RSI &gt; 70</span> 超買&ensp;
        <span style='background:#e2efda;color:#375623;padding:2px 7px;border-radius:4px'>KD &lt; 20</span> 低檔&ensp;
        <span style='background:#fce4d6;color:#833c0b;padding:2px 7px;border-radius:4px'>KD &gt; 80</span> 高檔&nbsp;｜&nbsp;
        <b>技術建議：</b>
        <span style='background:#00897b;color:white;padding:2px 7px;border-radius:4px'>🔥 強力買進</span>&ensp;
        <span style='background:#c6efce;color:#1b5e20;padding:2px 7px;border-radius:4px'>🟢 考慮買進</span>&ensp;
        <span style='background:#c62828;color:white;padding:2px 7px;border-radius:4px'>🔴 強力賣出</span>&ensp;
        <span style='background:#ffc7ce;color:#7f0000;padding:2px 7px;border-radius:4px'>🟠 考慮賣出</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# 10. 主程式
# ════════════════════════════════════════════════════
def main():
    # 嘗試從 URL ?u= 自動登入
    _check_url_login()

    # 未登入 → 顯示登入畫面
    if "username" not in st.session_state:
        render_login_screen()
        st.stop()

    # 第一次登入後提示書籤
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
    render_watchlist_manager("us_watchlist", "輸入美股代號，例如：AAPL、MSFT、SPY")
    render_stock_table(st.session_state.us_watchlist)

    st.divider()

    # ── 台股 ──────────────────────────────────────
    st.markdown("### 🇹🇼 台股")
    st.info("💡 上市加 `.TW`（如 `2330.TW`）；上櫃加 `.TWO`（如 `3661.TWO`）")
    render_watchlist_manager("tw_watchlist", "輸入台股代號，例如：2330.TW、0056.TW")
    render_stock_table(st.session_state.tw_watchlist)

    st.divider()
    st.caption("⚠️ 免責聲明：本儀表板提供之數據與技術指標僅供輔助分析，不構成任何投資建議。投資有風險，請審慎評估，自負盈虧。")


if __name__ == "__main__":
    main()
