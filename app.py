"""
股票技術指標監控儀表板 (Stock Technical Indicator Dashboard)
────────────────────────────────────────────────────
Tech Stack: Python 3.10+ | Streamlit | yfinance | Pandas | NumPy
Features  : RSI(14), 慢速 KD(14,3,3), 超買超賣警示, 自選股管理
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ═══════════════════════════════════════════════════
# 0. 頁面基本設定（必須是第一個 Streamlit 呼叫）
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="📈 股票技術指標儀表板",
    page_icon="📈",
    layout="wide",
)

# 注入自訂 CSS 提升視覺質感
st.markdown("""
<style>
    .stApp { font-family: 'Segoe UI', 'Microsoft JhengHei', Arial, sans-serif; }

    .gradient-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0066cc 0%, #00aa44 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }

    .legend-row {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        margin-top: 0.8rem;
        border-left: 4px solid #0066cc;
    }

    /* 分頁字體加粗 */
    button[data-baseweb="tab"] p { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# 1. 常數設定
# ═══════════════════════════════════════════════════
DEFAULT_US_STOCKS = ["NVDA", "TSLA", "QQQ", "MU", "AVGO"]
DEFAULT_TW_STOCKS = ["2330.TW", "0050.TW"]

RSI_PERIOD    = 14
KD_K_PERIOD   = 14
KD_K_SMOOTH   = 3
KD_D_SMOOTH   = 3
HISTORY_PERIOD = "6mo"  # yfinance 抓取區間（6個月日K，足夠計算指標）


# ═══════════════════════════════════════════════════
# 2. Session State 初始化（儲存自選股清單與更新時間）
# ═══════════════════════════════════════════════════
def _init_session():
    defaults = {
        "us_watchlist":  DEFAULT_US_STOCKS.copy(),
        "tw_watchlist":  DEFAULT_TW_STOCKS.copy(),
        "last_refresh":  datetime.now(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_session()


# ═══════════════════════════════════════════════════
# 3. 技術指標計算函數
# ═══════════════════════════════════════════════════

def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI (相對強弱指標) — Wilder's Smoothing Method

    公式：
      delta   = close.diff()
      gain    = clip(delta, 0, +∞)
      loss    = clip(-delta, 0, +∞)
      avg_g/l = EWM(com = period-1, min_periods=period)
      RSI     = 100 − 100 / (1 + avg_gain / avg_loss)
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)   # 避免除以零
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_kd(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    k_period: int = KD_K_PERIOD,
    k_smooth: int = KD_K_SMOOTH,
    d_smooth: int = KD_D_SMOOTH,
):
    """
    慢速 KD (Slow Stochastic Oscillator) KD(14, 3, 3)

    步驟：
      1. RSV  = (close − N日最低) / (N日最高 − N日最低) × 100
      2. K 值 = RSV 的 k_smooth 日 SMA（代表當前收盤在區間的相對位置）
      3. D 值 = K 值的 d_smooth 日 SMA（K 值的信號線）

    返回：(K 值 Series, D 值 Series)
    """
    lowest  = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()

    hl_range = (highest - lowest).replace(0, np.nan)   # 避免平盤時除以零
    rsv = 100 * (close - lowest) / hl_range
    rsv = rsv.fillna(50)    # 無法計算時（ex: 連續停板）預設中性值 50

    k = rsv.rolling(window=k_smooth).mean()
    d = k.rolling(window=d_smooth).mean()
    return k, d


# ═══════════════════════════════════════════════════
# 4. 數據獲取（帶 Cache，每 5 分鐘自動失效）
# ═══════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str) -> dict:
    """
    從 yfinance 抓取歷史日K並計算最新技術指標。

    快取 5 分鐘（ttl=300s），避免重複 API 請求。
    點擊「更新數據」按鈕時會呼叫 st.cache_data.clear() 強制清除快取。

    返回 dict：
      成功 → {ticker, name, close, change_pct, rsi, k, d, kd_golden_cross}
      失敗 → {ticker, error: "錯誤訊息"}
    """
    try:
        stock = yf.Ticker(ticker)
        hist  = stock.history(period=HISTORY_PERIOD)

        if hist is None or hist.empty or len(hist) < 30:
            return {"ticker": ticker, "error": "歷史數據不足或代號不存在"}

        # ── 計算技術指標 ──────────────────────────────
        rsi_s = calculate_rsi(hist["Close"])
        k_s, d_s = calculate_kd(hist["High"], hist["Low"], hist["Close"])

        # 取最新有效值
        latest_rsi = float(rsi_s.dropna().iloc[-1])
        latest_k   = float(k_s.dropna().iloc[-1])
        latest_d   = float(d_s.dropna().iloc[-1])

        latest_close = float(hist["Close"].iloc[-1])
        prev_close   = float(hist["Close"].iloc[-2])
        change_pct   = (latest_close - prev_close) / prev_close * 100

        # ── 取公司名稱（info 請求較慢，做容錯）────────
        try:
            info = stock.info
            name = (
                info.get("longName")
                or info.get("shortName")
                or info.get("symbol")
                or ticker
            )
            name = str(name)[:25]   # 截斷過長名稱避免破版
        except Exception:
            name = ticker

        # ── 偵測 KD 黃金交叉（在 20 以下 K 上穿 D）──
        kd_golden_cross = False
        k_clean = k_s.dropna()
        d_clean = d_s.dropna()
        if len(k_clean) >= 2 and len(d_clean) >= 2:
            k_prev, k_now = float(k_clean.iloc[-2]), float(k_clean.iloc[-1])
            d_prev, d_now = float(d_clean.iloc[-2]), float(d_clean.iloc[-1])
            if (
                k_prev < d_prev          # 昨日 K < D（尚未交叉）
                and k_now >= d_now       # 今日 K >= D（完成交叉）
                and k_now < 20           # 交叉發生在低檔區（< 20）
                and d_now < 20
            ):
                kd_golden_cross = True

        return {
            "ticker":          ticker,
            "name":            name,
            "close":           latest_close,
            "change_pct":      change_pct,
            "rsi":             latest_rsi,
            "k":               latest_k,
            "d":               latest_d,
            "kd_golden_cross": kd_golden_cross,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ═══════════════════════════════════════════════════
# 5. DataFrame 樣式（Pandas Styler）
# ═══════════════════════════════════════════════════

def _col_rsi(series: pd.Series):
    """RSI 欄位：< 30 綠底（超賣）、> 70 紅底（超買）"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append("")
        elif v < 30:
            out.append("background-color:#c6efce; color:#276221; font-weight:700")
        elif v > 70:
            out.append("background-color:#ffc7ce; color:#9c0006; font-weight:700")
        else:
            out.append("")
    return out


def _col_kd(series: pd.Series):
    """K/D 欄位：< 20 淡綠（低檔）、> 80 淡紅（高檔）"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append("")
        elif v < 20:
            out.append("background-color:#e2efda; color:#375623")
        elif v > 80:
            out.append("background-color:#fce4d6; color:#833c0b")
        else:
            out.append("")
    return out


def _col_change(series: pd.Series):
    """
    漲跌幅欄位（台灣慣例：漲紅跌綠）
    注意：此顏色慣例與美股（漲綠跌紅）相反，以台灣使用者習慣為主。
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append("")
        elif v > 0:
            out.append("color:#c00000; font-weight:700")   # 漲：紅色
        elif v < 0:
            out.append("color:#375623; font-weight:700")   # 跌：綠色
        else:
            out.append("color:gray")
    return out


def _fmt_change(v):
    """漲跌幅加上箭頭符號的顯示格式"""
    if pd.isna(v):
        return "-"
    if v > 0:
        return f"▲ {v:.2f}%"
    if v < 0:
        return f"▼ {abs(v):.2f}%"
    return f"{v:.2f}%"


def build_styled_df(df: pd.DataFrame):
    """將 DataFrame 套用全部顏色與格式規則，回傳 Pandas Styler"""
    return (
        df.style
        .apply(_col_rsi,    subset=["RSI(14)"])
        .apply(_col_kd,     subset=["K值", "D值"])
        .apply(_col_change, subset=["漲跌幅(%)"])
        .format(
            {
                "最新收盤價": "{:.2f}",
                "漲跌幅(%)":  _fmt_change,
                "RSI(14)":    "{:.1f}",
                "K值":        "{:.1f}",
                "D值":        "{:.1f}",
            },
            na_rep="-",   # 若有 NaN 顯示為 "-"
        )
    )


# ═══════════════════════════════════════════════════
# 6. 自選股管理 UI 元件
# ═══════════════════════════════════════════════════

def render_watchlist_manager(watchlist_key: str, placeholder: str):
    """
    渲染自選股管理介面：
      • 輸入框 + 新增按鈕
      • 以標籤按鈕顯示現有清單（點 ✕ 可刪除）
      • 重置按鈕恢復預設清單
    """
    col_inp, col_add, col_reset = st.columns([3, 1, 1])

    with col_inp:
        new_ticker = st.text_input(
            "ticker_input",
            placeholder=placeholder,
            key=f"input_{watchlist_key}",
            label_visibility="collapsed",
        )

    with col_add:
        if st.button("➕ 新增", key=f"add_{watchlist_key}", use_container_width=True):
            ticker = str(new_ticker).strip().upper()
            if ticker == "":
                st.toast("請先輸入股票代號", icon="⚠️")
            elif ticker in st.session_state[watchlist_key]:
                st.toast(f"{ticker} 已在清單中", icon="ℹ️")
            else:
                st.session_state[watchlist_key].append(ticker)
                st.cache_data.clear()   # 新增代號後強制重新抓取
                st.rerun()

    with col_reset:
        defaults = DEFAULT_US_STOCKS if "us" in watchlist_key else DEFAULT_TW_STOCKS
        if st.button("↺ 重置", key=f"reset_{watchlist_key}", use_container_width=True,
                     help="恢復預設自選股清單"):
            st.session_state[watchlist_key] = defaults.copy()
            st.cache_data.clear()
            st.rerun()

    # ── 顯示現有清單（標籤式 + 刪除按鈕）──────────
    watchlist = st.session_state[watchlist_key]
    if watchlist:
        n_cols = min(len(watchlist), 8)
        tag_cols = st.columns(n_cols)
        for i, tkr in enumerate(list(watchlist)):   # list() 避免迭代中修改
            with tag_cols[i % n_cols]:
                if st.button(
                    f"✕ {tkr}",
                    key=f"del_{watchlist_key}_{tkr}",
                    use_container_width=True,
                    help=f"從清單移除 {tkr}",
                ):
                    st.session_state[watchlist_key].remove(tkr)
                    st.cache_data.clear()
                    st.rerun()
    else:
        st.caption("清單目前為空，請在上方輸入代號後點擊「➕ 新增」")


# ═══════════════════════════════════════════════════
# 7. 股票資料表格渲染
# ═══════════════════════════════════════════════════

def render_stock_table(watchlist: list):
    """
    遍歷自選股清單 → 批量獲取數據 → 組裝 DataFrame → 顯示格式化表格。
    使用進度條讓使用者了解載入進度。
    """
    if not watchlist:
        st.info("⚠️ 自選股清單為空，請在上方輸入股票代號後點擊「➕ 新增」")
        return

    rows  = []
    errors = []
    total  = len(watchlist)

    # 進度顯示
    status_text = st.empty()
    prog_bar    = st.progress(0)

    for i, ticker in enumerate(watchlist):
        status_text.caption(f"📡 正在載入：{ticker}  ({i + 1}/{total})")
        prog_bar.progress((i + 1) / total)

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
                "KD 訊號":    "🌟 黃金交叉" if result["kd_golden_cross"] else "",
            })

    # 清除進度顯示
    status_text.empty()
    prog_bar.empty()

    # ── 顯示異常代號（摺疊）──────────────────────
    if errors:
        with st.expander(f"⚠️ {len(errors)} 個代號異常（點擊展開查看詳情）"):
            for msg in errors:
                st.warning(msg)

    if not rows:
        st.error("所有代號均無法獲取數據。請確認代號格式正確並檢查網路連線。")
        return

    # ── 建立 DataFrame 並套用樣式 ────────────────
    df     = pd.DataFrame(rows)
    styled = build_styled_df(df)

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(600, 55 + 38 * len(df)),
        column_config={
            "股票代號": st.column_config.TextColumn("股票代號", width="small"),
            "股票名稱": st.column_config.TextColumn("股票名稱", width="medium"),
            "KD 訊號":  st.column_config.TextColumn("KD 訊號",  width="small"),
        },
    )

    # ── 圖例說明 ─────────────────────────────────
    st.markdown("""
    <div class="legend-row">
        <b>顏色圖例：</b>&ensp;
        <span style='background:#c6efce;color:#276221;padding:2px 8px;border-radius:4px;font-weight:700'>RSI &lt; 30</span>
        &thinsp;超賣（潛在買入訊號）&ensp;
        <span style='background:#ffc7ce;color:#9c0006;padding:2px 8px;border-radius:4px;font-weight:700'>RSI &gt; 70</span>
        &thinsp;超買（潛在賣出訊號）&ensp;
        <span style='background:#e2efda;color:#375623;padding:2px 8px;border-radius:4px'>KD &lt; 20</span>
        &thinsp;低檔區&ensp;
        <span style='background:#fce4d6;color:#833c0b;padding:2px 8px;border-radius:4px'>KD &gt; 80</span>
        &thinsp;高檔區&ensp;
        🌟&thinsp;KD 黃金交叉（兩線在 20 以下，K 由下往上穿越 D）
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# 8. 主程式入口
# ═══════════════════════════════════════════════════

def main():
    # ── 頂部標題列 ────────────────────────────────
    hdr_l, hdr_r = st.columns([5, 1])

    with hdr_l:
        st.markdown(
            '<div class="gradient-title">📈 股票技術指標監控儀表板</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"數據來源：Yahoo Finance（yfinance）｜"
            f"最後更新：{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}｜"
            "本儀表板僅供參考，不構成投資建議"
        )

    with hdr_r:
        st.write("")
        st.write("")
        if st.button("🔄 更新數據", type="primary", use_container_width=True,
                     help="清除快取並重新抓取最新數據"):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    st.divider()

    # ── 美股 / 台股 分頁 ──────────────────────────
    tab_us, tab_tw = st.tabs(["🇺🇸  美股 (US Stocks)", "🇹🇼  台股 (TW Stocks)"])

    with tab_us:
        st.subheader("自選股管理")
        render_watchlist_manager(
            watchlist_key="us_watchlist",
            placeholder="輸入美股代號，例如：AAPL、MSFT、SPY",
        )
        st.divider()
        render_stock_table(st.session_state.us_watchlist)

    with tab_tw:
        st.subheader("自選股管理")
        st.info(
            "💡 **代號格式說明**：上市股票（TWSE）加 `.TW`，例如 `2330.TW`、`0050.TW`；"
            "上櫃股票（TPEx）加 `.TWO`，例如 `3661.TWO`、`6415.TWO`"
        )
        render_watchlist_manager(
            watchlist_key="tw_watchlist",
            placeholder="輸入台股代號，例如：2330.TW、0056.TW",
        )
        st.divider()
        render_stock_table(st.session_state.tw_watchlist)

    # ── 頁尾免責聲明 ──────────────────────────────
    st.divider()
    st.caption(
        "⚠️ 免責聲明：本儀表板提供之數據與技術指標僅供輔助分析，不構成任何投資建議。"
        "投資有風險，請審慎評估，自負盈虧。數據延遲最多 15 分鐘。"
    )


if __name__ == "__main__":
    main()
