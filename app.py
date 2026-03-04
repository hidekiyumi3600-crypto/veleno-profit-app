import streamlit as st
import pandas as pd
import altair as alt
import os

# --- 設定 ---
CHANNEL_CONFIG = {
    "楽天": {"price_col": "rakuten_price", "list_price_col": "rakuten_list_price", "discount1_col": "rakuten_discount1", "fee_col": "rakuten_fee", "ship_col": "rakuten_shipping", "default_fee_rate": 0.16},
    "Amazon": {"price_col": "amazon_price", "list_price_col": "amazon_list_price", "discount1_col": "amazon_discount1", "fee_col": "amazon_fee", "ship_col": "amazon_shipping", "default_fee_rate": 0.15},
    "Yahoo": {"price_col": "yahoo_price", "list_price_col": "yahoo_list_price", "discount1_col": "yahoo_discount1", "fee_col": "yahoo_fee", "ship_col": "yahoo_shipping", "default_fee_rate": 0.16},
    "業販": {"price_col": "wholesale_price", "list_price_col": "wholesale_list_price", "discount1_col": "wholesale_discount1", "fee_col": None, "ship_col": "wholesale_shipping", "default_fee_rate": 0.0},
}

st.set_page_config(page_title="VELENO 利益計算", page_icon="🚗", layout="wide")


# =====================================================
# カスタムCSS注入
# =====================================================
def inject_custom_css():
    st.markdown("""
    <style>
    /* サイドバー: ダークグラデーション */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
    }

    /* ページタイトル: 赤いアクセント下線 */
    h1 {
        border-bottom: 3px solid #e53935;
        padding-bottom: 0.3em;
    }

    /* セクションタイトル: 左赤線 */
    .section-title {
        border-left: 4px solid #e53935;
        padding-left: 12px;
        font-size: 1.3em;
        font-weight: 700;
        margin: 1.5em 0 0.8em 0;
        color: #1a1a2e;
    }

    /* メトリクスカード */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 0.82em;
        color: #888;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-card .value {
        font-size: 1.6em;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-card .delta {
        font-size: 0.8em;
        font-weight: 500;
    }
    .metric-card .value.positive { color: #2e7d32; }
    .metric-card .value.negative { color: #c62828; }
    .metric-card .value.warning { color: #f9a825; }
    .metric-card .value.neutral { color: #1a1a2e; }
    .metric-card .delta.positive { color: #2e7d32; }
    .metric-card .delta.negative { color: #c62828; }

    /* フォームセクション */
    .form-section {
        background: #f8f9fa;
        border-left: 3px solid #1976d2;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .form-section-title {
        font-weight: 700;
        font-size: 1.05em;
        color: #1976d2;
        margin-bottom: 8px;
    }

    /* テーブルヘッダー */
    .stDataFrame thead th {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }

    /* Streamlitブランディング非表示（サイドバー開閉ボタンは残す） */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {
        visibility: hidden;
    }
    /* サイドバー展開ボタン: 1.38+は stExpandSidebarButton、旧は collapsedControl */
    [data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
    }

    /* ログインカード */
    .login-card {
        max-width: 400px;
        margin: 80px auto;
        background: #ffffff;
        border-radius: 16px;
        padding: 40px 36px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
        text-align: center;
    }
    .login-card .brand {
        font-size: 2.2em;
        font-weight: 800;
        color: #e53935;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .login-card .subtitle {
        font-size: 0.95em;
        color: #888;
        margin-bottom: 24px;
    }

    /* 削除警告カード */
    .delete-card {
        background: #fff5f5;
        border: 2px solid #e53935;
        border-radius: 10px;
        padding: 20px;
        margin-top: 12px;
    }
    .delete-card .title {
        color: #c62828;
        font-weight: 700;
        font-size: 1.05em;
        margin-bottom: 8px;
    }

    /* スマホ対応 */
    @media (max-width: 768px) {
        .main .block-container { padding: 0.5rem !important; }
        h1 { font-size: 1.3em; }
        .metric-card { padding: 10px 8px; }
        .metric-card .value { font-size: 1.2em; }
        .metric-card .label { font-size: 0.72em; }
        .metric-card .delta { font-size: 0.7em; }
        .section-title { font-size: 1.1em; }
        .metric-card [style*="display:flex"][style*="gap:32px"],
        .metric-card [style*="display: flex"][style*="gap: 32px"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        .login-card { margin: 40px auto; padding: 24px 20px; }
        .stDataFrame { font-size: 0.8em; }
        .form-section { padding: 12px; }
        .delete-card { padding: 14px; }
    }

    /* 割引フロー: ステップバー */
    .price-flow {
        display: flex;
        align-items: center;
        gap: 0;
        margin: 12px 0;
        flex-wrap: wrap;
    }
    .price-flow-step {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 0.85em;
        font-weight: 600;
        color: #333;
        text-align: center;
        position: relative;
        min-width: 80px;
    }
    .price-flow-step.start { background: #e3f2fd; color: #1565c0; }
    .price-flow-step.discount { background: #fff3e0; color: #e65100; }
    .price-flow-step.final { background: #e8f5e9; color: #2e7d32; }
    .price-flow-arrow {
        font-size: 1.3em;
        color: #999;
        margin: 0 6px;
    }

    /* ゲージバー */
    .gauge-bar {
        position: relative;
        height: 24px;
        background: #e0e0e0;
        border-radius: 12px;
        overflow: visible;
        margin: 8px 0 4px 0;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 12px;
        transition: width 0.3s;
    }
    .gauge-bar-target {
        position: absolute;
        top: -4px;
        width: 3px;
        height: 32px;
        background: #333;
        border-radius: 2px;
    }
    .gauge-bar-target-label {
        position: absolute;
        top: -20px;
        font-size: 0.7em;
        font-weight: 600;
        color: #333;
        transform: translateX(-50%);
        white-space: nowrap;
    }

    /* コスト内訳: 積み上げ棒グラフ */
    .cost-stack-bar {
        display: flex;
        height: 32px;
        border-radius: 8px;
        overflow: hidden;
        margin: 8px 0;
    }
    .cost-stack-bar > div {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7em;
        font-weight: 600;
        color: #fff;
        white-space: nowrap;
        overflow: hidden;
        min-width: 2px;
    }

    /* 現在 vs シミュレーション比較行 */
    .comparison-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        border: 1px solid #e8e8e8;
    }
    .comparison-row .comp-label {
        font-size: 0.82em;
        color: #888;
        font-weight: 500;
    }
    .comparison-row .comp-values {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
    }
    .comparison-row .comp-arrow { color: #999; }
    .comparison-row .comp-diff {
        font-size: 0.82em;
        font-weight: 600;
        margin-left: 8px;
    }
    .comp-diff.positive { color: #2e7d32; }
    .comp-diff.negative { color: #c62828; }

    /* ステータスバッジ */
    .status-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.82em;
        font-weight: 600;
        line-height: 1.4;
    }
    .status-badge.green { background: #e8f5e9; color: #2e7d32; }
    .status-badge.red { background: #ffebee; color: #c62828; }
    .status-badge.yellow { background: #fff8e1; color: #f57f17; }

    </style>
    """, unsafe_allow_html=True)


# =====================================================
# ヘルパー関数
# =====================================================
def html_metric_card(label, value, delta="", status="neutral"):
    """カード型KPI表示。status: positive/negative/warning/neutral"""
    delta_class = "positive" if status == "positive" else ("negative" if status == "negative" else "")
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value {status}">{value}</div>
        <div class="delta {delta_class}">{delta}</div>
    </div>
    """


def html_section_title(text):
    """アクセント付きセクション見出し"""
    return f'<div class="section-title">{text}</div>'


def profit_indicator(value, formatted_str):
    """利益値にCSSバッジを追加"""
    if value < 0:
        return f'<span class="status-badge red">{formatted_str}</span>'
    return f'<span class="status-badge green">{formatted_str}</span>'


def margin_indicator(margin, formatted_str, target=56):
    """粗利率にCSSバッジを追加"""
    if margin < 0:
        return f'<span class="status-badge red">{formatted_str}</span>'
    if margin < target:
        return f'<span class="status-badge yellow">{formatted_str}</span>'
    return f'<span class="status-badge green">{formatted_str}</span>'


# =====================================================
# パスワード認証（localStorage で 7日間保持）
# =====================================================
import hashlib, time

AUTH_TOKEN_DAYS = 7

def _make_token(password):
    return hashlib.sha256(f"veleno_{password}_salt".encode()).hexdigest()

def _inject_auth_js(token, days):
    """ログイン成功時にlocalStorageへトークンを保存するJSを注入"""
    expires_ms = int(time.time() * 1000) + days * 86400 * 1000
    st.markdown(f"""
    <script>
    localStorage.setItem('veleno_auth', '{token}');
    localStorage.setItem('veleno_auth_exp', '{expires_ms}');
    </script>
    """, unsafe_allow_html=True)

def _inject_auth_check_js():
    """localStorage からトークンを読み出し、query param に渡すJSを注入"""
    st.markdown("""
    <script>
    (function() {
        const token = localStorage.getItem('veleno_auth');
        const exp = localStorage.getItem('veleno_auth_exp');
        if (token && exp && Date.now() < parseInt(exp)) {
            const url = new URL(window.location);
            if (url.searchParams.get('auth') !== token) {
                url.searchParams.set('auth', token);
                window.location.replace(url.toString());
            }
        }
    })();
    </script>
    """, unsafe_allow_html=True)

def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # query param 経由の自動ログインチェック
    if not st.session_state.authenticated:
        params = st.query_params
        auth_token = params.get("auth", "")
        if auth_token == _make_token(st.secrets["password"]):
            st.session_state.authenticated = True

    if st.session_state.authenticated:
        return True

    inject_custom_css()

    # localStorage チェック用JSを注入
    _inject_auth_check_js()

    st.markdown("""
    <div class="login-card">
        <div class="brand">VELENO</div>
        <div class="subtitle">利益計算システム</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        password = st.text_input("パスワードを入力してください", type="password", key="login_pw")
        if st.button("ログイン", type="primary", use_container_width=True):
            if password == st.secrets["password"]:
                st.session_state.authenticated = True
                _inject_auth_js(_make_token(password), AUTH_TOKEN_DAYS)
                st.rerun()
            else:
                st.error("パスワードが正しくありません")
    return False


if not check_password():
    st.stop()

# --- CSS注入（ログイン後） ---
inject_custom_css()

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "products.csv")

NUMERIC_COLS = [
    "usd_cost", "cost_jpy", "import_tax", "overseas_shipping", "exchange_rate", "tariff",
    "pcs_per_unit",
    "rakuten_price", "amazon_price", "yahoo_price", "wholesale_price",
    "rakuten_list_price", "amazon_list_price", "yahoo_list_price", "wholesale_list_price",
    "rakuten_discount1", "amazon_discount1", "yahoo_discount1", "wholesale_discount1",
    "rakuten_discount2", "amazon_discount2", "yahoo_discount2", "wholesale_discount2",
    "rakuten_shipping", "amazon_shipping", "yahoo_shipping", "wholesale_shipping",
    "rakuten_fee", "amazon_fee", "yahoo_fee",
]


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.fillna({"name": "", "rank": "", "product_id": ""})
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def save_data(df):
    df.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()


def calc_profit(price_tax_incl, cost_jpy, overseas_shipping, fee_amount, domestic_shipping):
    if price_tax_incl <= 0:
        return 0, 0, 0
    price_ex_tax = price_tax_incl / 1.1
    total_cost = cost_jpy + overseas_shipping + fee_amount + domestic_shipping
    profit = price_ex_tax - total_cost
    margin = (profit / price_ex_tax * 100) if price_ex_tax > 0 else 0
    return round(profit), round(margin, 1), round(total_cost)


def calc_channel_profit(row, channel, fee_rate_override=None):
    cfg = CHANNEL_CONFIG[channel]
    price = row[cfg["price_col"]]
    ship = row[cfg["ship_col"]]

    if fee_rate_override is not None:
        fee_amount = price * fee_rate_override
    elif cfg["fee_col"] and row.get(cfg["fee_col"], 0) > 0:
        fee_amount = row[cfg["fee_col"]]
    else:
        fee_amount = price * cfg["default_fee_rate"]

    return calc_profit(price, row["cost_jpy"], row["overseas_shipping"], fee_amount, ship)


def add_profit_columns(df, channel):
    profits, margins, costs = [], [], []
    for _, row in df.iterrows():
        p, m, c = calc_channel_profit(row, channel)
        profits.append(p)
        margins.append(m)
        costs.append(c)
    df[f"{channel}_利益"] = profits
    df[f"{channel}_粗利率"] = margins
    df[f"{channel}_コスト"] = costs
    return df


# --- データ読み込み ---
df = load_data()

# --- サイドバー ---
st.sidebar.markdown("""
<div style="text-align:center; padding: 16px 0 8px 0;">
    <div style="font-size:1.8em; font-weight:800; color:#e53935 !important; letter-spacing:0.08em;">VELENO</div>
    <div style="font-size:0.85em; color:#aaa !important; margin-top:2px;">利益計算システム</div>
    <hr style="border-color: rgba(255,255,255,0.15); margin: 12px 0;">
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "ナビゲーション",
    ["📊 商品一覧", "🔍 商品詳細・チャネル比較", "📈 価格シミュレーション", "⚙️ 商品管理"],
    label_visibility="collapsed",
)

st.sidebar.divider()
TARGET_MARGIN = st.sidebar.number_input("目標粗利率 (%)", min_value=0.0, max_value=99.0, value=56.0, step=1.0, key="target_margin_global")

# =====================================================
# 画面1: 商品一覧 & 利益ダッシュボード
# =====================================================
if page == "📊 商品一覧":
    st.title("商品一覧 & 利益ダッシュボード")

    # チャネル選択
    channel = st.selectbox("チャネル", list(CHANNEL_CONFIG.keys()))

    # フィルター
    col1, col2, col3 = st.columns(3)
    with col1:
        ranks = ["すべて"] + sorted(df["rank"].unique().tolist())
        rank_filter = st.selectbox("ランク", ranks)
    with col2:
        search = st.text_input("商品名・管理番号で検索")
    with col3:
        show_loss = st.toggle("🔴 赤字商品のみ表示", value=False)

    # データ準備
    view = df.copy()
    view = add_profit_columns(view, channel)

    price_col = CHANNEL_CONFIG[channel]["price_col"]
    view = view[view[price_col] > 0]

    if rank_filter != "すべて":
        view = view[view["rank"] == rank_filter]
    if search:
        view = view[
            view["name"].str.contains(search, case=False, na=False)
            | view["product_id"].astype(str).str.contains(search, case=False, na=False)
        ]
    if show_loss:
        view = view[view[f"{channel}_利益"] < 0]

    # サマリー計算
    total = len(view)
    loss_count = len(view[view[f"{channel}_利益"] < 0])
    profit_count = total - loss_count
    avg_margin = view[f"{channel}_粗利率"].mean() if total > 0 else 0
    target_met = len(view[view[f"{channel}_粗利率"] >= TARGET_MARGIN]) if total > 0 else 0
    total_profit = view[f"{channel}_利益"].sum() if total > 0 else 0
    profit_rate = (profit_count / total * 100) if total > 0 else 0

    # KPIカード (6指標 → 3列×2行)
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(html_metric_card("商品数", f"{total:,}", "", "neutral"), unsafe_allow_html=True)
    with k2:
        st.markdown(html_metric_card(
            "赤字商品", f"{loss_count}",
            f"全体の{loss_count/total*100:.0f}%" if total > 0 else "",
            "negative" if loss_count > 0 else "positive"
        ), unsafe_allow_html=True)
    with k3:
        margin_status = "positive" if avg_margin >= TARGET_MARGIN else ("warning" if avg_margin >= TARGET_MARGIN * 0.7 else "negative")
        st.markdown(html_metric_card("平均粗利率", f"{avg_margin:.1f}%", f"目標: {TARGET_MARGIN:.0f}%", margin_status), unsafe_allow_html=True)
    k4, k5, k6 = st.columns(3)
    with k4:
        st.markdown(html_metric_card(
            "目標達成数", f"{target_met}",
            f"粗利率{TARGET_MARGIN:.0f}%以上",
            "positive" if target_met > total // 2 else "warning"
        ), unsafe_allow_html=True)
    with k5:
        profit_status = "positive" if total_profit > 0 else "negative"
        st.markdown(html_metric_card("合計利益", f"¥{total_profit:,.0f}", "", profit_status), unsafe_allow_html=True)
    with k6:
        rate_status = "positive" if profit_rate >= 80 else ("warning" if profit_rate >= 50 else "negative")
        st.markdown(html_metric_card("黒字率", f"{profit_rate:.0f}%", f"{profit_count}/{total}", rate_status), unsafe_allow_html=True)

    # テーブル表示
    st.markdown(html_section_title("商品テーブル"), unsafe_allow_html=True)

    list_price_col = CHANNEL_CONFIG[channel]["list_price_col"]
    discount1_col = CHANNEL_CONFIG[channel]["discount1_col"]
    display_cols = ["no", "product_id", "rank", "name", list_price_col, discount1_col, price_col, "cost_jpy", f"{channel}_利益", f"{channel}_粗利率"]
    display_names = {"no": "No", "product_id": "管理番号", "rank": "ランク", "name": "商品名",
                     list_price_col: "定価(税込)", discount1_col: "割引率",
                     price_col: "販売価格(税込)", "cost_jpy": "原価",
                     f"{channel}_利益": "利益(税抜)", f"{channel}_粗利率": "粗利率(%)"}

    sort_col = st.selectbox("並び替え", [f"{channel}_利益", f"{channel}_粗利率", price_col, "cost_jpy"], format_func=lambda x: display_names.get(x, x))
    sort_asc = st.checkbox("昇順（低い順）", value=True)
    view = view.sort_values(sort_col, ascending=sort_asc)

    # #5 CSVエクスポート
    csv_export = view[display_cols].copy()
    csv_export = csv_export.rename(columns=display_names)
    csv_data = csv_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 CSVダウンロード", data=csv_data, file_name=f"veleno_{channel}_products.csv", mime="text/csv")

    # #6 ページネーション
    ITEMS_PER_PAGE = 20
    total_items = len(view)
    if total_items > ITEMS_PER_PAGE:
        total_pages = (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
        current_page = st.number_input("ページ", min_value=1, max_value=total_pages, value=1, step=1, key="table_page")
        start_idx = (current_page - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, total_items)
        st.caption(f"{total_items}件中 {start_idx+1}〜{end_idx}件を表示（全{total_pages}ページ）")
        page_view = view.iloc[start_idx:end_idx]
    else:
        page_view = view

    disp = page_view[display_cols].copy()
    disp[list_price_col] = disp[list_price_col].apply(lambda x: f"¥{x:,.0f}")
    disp[discount1_col] = disp[discount1_col].apply(lambda x: f"{x:.0%}" if x > 0 else "-")
    disp[price_col] = disp[price_col].apply(lambda x: f"¥{x:,.0f}")
    disp["cost_jpy"] = disp["cost_jpy"].apply(lambda x: f"¥{x:,.0f}")
    # 利益・粗利率にインジケーター追加
    profit_raw = page_view[f"{channel}_利益"]
    margin_raw = page_view[f"{channel}_粗利率"]
    disp[f"{channel}_利益"] = [profit_indicator(p, f"¥{p:,.0f}") for p in profit_raw]
    disp[f"{channel}_粗利率"] = [margin_indicator(m, f"{m:.1f}%", TARGET_MARGIN) for m in margin_raw]

    # #13 赤字行ハイライト用スタイル
    loss_indices = page_view[page_view[f"{channel}_利益"] < 0].index.tolist()

    disp = disp.rename(columns=display_names)
    st.dataframe(
        disp,
        height=min(600, max(200, len(disp) * 35 + 50)),
        use_container_width=True,
        column_config={
            "No": st.column_config.NumberColumn(width="small"),
            "管理番号": st.column_config.TextColumn(width="small"),
            "ランク": st.column_config.TextColumn(width="small"),
            "商品名": st.column_config.TextColumn(width="large"),
            "利益(税抜)": st.column_config.TextColumn(width="medium"),
            "粗利率(%)": st.column_config.TextColumn(width="medium"),
        },
    )

    # TOP/WORST 横並び
    def fmt_ranking(src, profit_col, margin_col):
        d = src[["name", price_col, profit_col, margin_col]].copy()
        d.columns = ["商品名", "販売価格", "利益", "粗利率(%)"]
        d["販売価格"] = d["販売価格"].apply(lambda x: f"¥{x:,.0f}")
        d["利益"] = [profit_indicator(p, f"¥{p:,.0f}") for p in src[profit_col]]
        d["粗利率(%)"] = [margin_indicator(m, f"{m:.1f}%", TARGET_MARGIN) for m in src[margin_col]]
        return d

    top_col, worst_col = st.columns(2)
    with top_col:
        st.markdown(html_section_title("🏆 利益 TOP10"), unsafe_allow_html=True)
        st.dataframe(fmt_ranking(view.nlargest(10, f"{channel}_利益"), f"{channel}_利益", f"{channel}_粗利率"), use_container_width=True)
    with worst_col:
        st.markdown(html_section_title("⚠️ 利益 WORST10"), unsafe_allow_html=True)
        st.dataframe(fmt_ranking(view.nsmallest(10, f"{channel}_利益"), f"{channel}_利益", f"{channel}_粗利率"), use_container_width=True)

# =====================================================
# 画面2: 商品詳細 & チャネル比較
# =====================================================
elif page == "🔍 商品詳細・チャネル比較":
    st.title("商品詳細 & チャネル比較")

    # 商品選択
    active = df[df["rakuten_price"] > 0].copy()
    active["label"] = active["no"].astype(str) + " | " + active["product_id"].astype(str) + " | " + active["name"]

    detail_search = st.text_input("管理番号・商品名で検索", key="detail_search")
    if detail_search:
        mask = (
            active["product_id"].astype(str).str.contains(detail_search, case=False, na=False)
            | active["name"].str.contains(detail_search, case=False, na=False)
        )
        filtered_active = active[mask]
    else:
        filtered_active = active
    if len(filtered_active) == 0:
        st.warning("該当する商品がありません")
        st.stop()
    selected_label = st.selectbox("商品を選択", filtered_active["label"].tolist(), key="detail_select")
    idx = active[active["label"] == selected_label].index[0]
    row = df.loc[idx]

    # 商品ヘッダー（カード型）
    st.markdown(f"""
    <div class="metric-card" style="text-align:left; padding:20px 28px;">
        <div style="font-size:1.4em; font-weight:700; color:#1a1a2e; margin-bottom:8px;">
            {row['name']}
        </div>
        <div style="display:flex; gap:32px; font-size:0.95em; color:#666;">
            <span>ランク: <strong>{row['rank']}</strong></span>
            <span>管理番号: <strong>{row['product_id']}</strong></span>
            <span>商品原価: <strong>¥{row['cost_jpy']:,.0f}</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 原価内訳
    with st.expander("原価の内訳"):
        st.write(f"- ドル原価: ${row['usd_cost']:.2f}")
        st.write(f"- 為替レート: ¥{row['exchange_rate']:.1f}")
        st.write(f"- 関税: {row['tariff']:.2f}")
        st.write(f"- 商品原価（円）: ¥{row['cost_jpy']:,.0f}")
        st.write(f"- 輸入消費税: ¥{row['import_tax']:,.0f}")
        st.write(f"- 海外送料: ¥{row['overseas_shipping']:,.0f}")

    # 4チャネル比較
    st.markdown(html_section_title("チャネル別利益比較"), unsafe_allow_html=True)

    results = []
    profit_vals = {}
    for ch in CHANNEL_CONFIG:
        cfg = CHANNEL_CONFIG[ch]
        list_price = row[cfg["list_price_col"]]
        price = row[cfg["price_col"]]
        disc1 = row[cfg["discount1_col"]]
        if price <= 0:
            results.append({"チャネル": ch, "定価(税込)": 0, "割引率": "-", "販売価格(税込)": 0,
                          "手数料率": f"{cfg['default_fee_rate']*100:.0f}%",
                          "手数料": 0, "国内送料": 0, "販売コスト合計": 0, "利益": 0, "粗利率(%)": 0})
            profit_vals[ch] = 0
            continue
        if cfg["fee_col"] and row.get(cfg["fee_col"], 0) > 0:
            fee = row[cfg["fee_col"]]
        else:
            fee = price * cfg["default_fee_rate"]
        ship = row[cfg["ship_col"]]
        profit, margin, total_cost = calc_profit(price, row["cost_jpy"], row["overseas_shipping"], fee, ship)
        profit_vals[ch] = profit
        results.append({
            "チャネル": ch,
            "定価(税込)": f"¥{list_price:,.0f}",
            "割引率": f"{disc1:.0%}" if disc1 > 0 else "-",
            "販売価格(税込)": f"¥{price:,.0f}",
            "手数料率": f"{cfg['default_fee_rate']*100:.0f}%",
            "手数料": f"¥{fee:,.0f}",
            "国内送料": f"¥{ship:,.0f}",
            "販売コスト合計": f"¥{total_cost:,.0f}",
            "利益": profit,
            "粗利率(%)": margin,
        })

    # 最高/最低利益チャネルの表示
    active_channels = {ch: pv for ch, pv in profit_vals.items() if row[CHANNEL_CONFIG[ch]["price_col"]] > 0}
    if active_channels:
        best_ch = max(active_channels, key=active_channels.get)
        worst_ch = min(active_channels, key=active_channels.get)
        bc1, bc2 = st.columns(2)
        with bc1:
            st.success(f"🏆 最高利益: **{best_ch}** ¥{active_channels[best_ch]:,.0f}")
        with bc2:
            if active_channels[worst_ch] < 0:
                st.error(f"⚠️ 最低利益: **{worst_ch}** ¥{active_channels[worst_ch]:,.0f}")
            else:
                st.warning(f"📉 最低利益: **{worst_ch}** ¥{active_channels[worst_ch]:,.0f}")

    result_df = pd.DataFrame(results)
    result_df["利益"] = result_df["利益"].apply(lambda x: f"¥{x:,.0f}" if isinstance(x, (int, float)) else x)
    result_df["粗利率(%)"] = result_df["粗利率(%)"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    st.dataframe(result_df, use_container_width=True)

    # Altair横棒グラフ（黒字=緑、赤字=赤）
    chart_data = pd.DataFrame({
        "チャネル": [r["チャネル"] for r in results],
        "利益": [r["利益"] if isinstance(r["利益"], (int, float)) else 0 for r in results]
    })
    chart_data["色"] = chart_data["利益"].apply(lambda x: "黒字" if x >= 0 else "赤字")

    bars = alt.Chart(chart_data).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("チャネル:N", sort="-x", title=None),
        x=alt.X("利益:Q", title="利益 (円)"),
        color=alt.Color("色:N",
            scale=alt.Scale(domain=["黒字", "赤字"], range=["#2e7d32", "#c62828"]),
            legend=None
        ),
        tooltip=[
            alt.Tooltip("チャネル:N"),
            alt.Tooltip("利益:Q", format=",.0f", title="利益(円)")
        ]
    )
    text = alt.Chart(chart_data).mark_text(
        align="left", dx=4, fontSize=12, fontWeight="bold"
    ).encode(
        y=alt.Y("チャネル:N", sort="-x"),
        x=alt.X("利益:Q"),
        text=alt.Text("利益:Q", format=",.0f"),
        color=alt.condition(
            alt.datum["利益"] >= 0,
            alt.value("#2e7d32"),
            alt.value("#c62828")
        )
    )
    chart = (bars + text).properties(height=200)
    st.altair_chart(chart, use_container_width=True)

# =====================================================
# 画面3: 価格シミュレーション
# =====================================================
elif page == "📈 価格シミュレーション":
    st.title("価格シミュレーション")

    # 商品選択
    active = df[df["rakuten_price"] > 0].copy()
    active["label"] = active["no"].astype(str) + " | " + active["product_id"].astype(str) + " | " + active["name"]

    sim_search = st.text_input("管理番号・商品名で検索", key="sim_search")
    if sim_search:
        mask = (
            active["product_id"].astype(str).str.contains(sim_search, case=False, na=False)
            | active["name"].str.contains(sim_search, case=False, na=False)
        )
        filtered_active = active[mask]
    else:
        filtered_active = active
    if len(filtered_active) == 0:
        st.warning("該当する商品がありません")
        st.stop()

    sort_options = {"管理番号": "product_id", "No": "no", "商品名": "name", "商品原価": "cost_jpy"}
    sim_c1, sim_c2 = st.columns([3, 1])
    with sim_c2:
        sort_key = st.selectbox("並び替え", list(sort_options.keys()), key="sim_sort")
    sorted_active = filtered_active.sort_values(sort_options[sort_key])
    with sim_c1:
        selected_label = st.selectbox("商品を選択", sorted_active["label"].tolist())
    idx = active[active["label"] == selected_label].index[0]
    row = df.loc[idx]

    st.markdown(f"""
    <div class="metric-card" style="text-align:left; padding:16px 24px;">
        <div style="font-size:1.2em; font-weight:700; color:#1a1a2e;">{row['name']}</div>
        <div style="color:#666; font-size:0.9em; margin-top:4px;">
            現在の原価: ¥{row['cost_jpy']:,.0f}　|　輸入消費税: ¥{row['import_tax']:,.0f}　|　海外送料: ¥{row['overseas_shipping']:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    channel = st.selectbox("シミュレーション対象チャネル", list(CHANNEL_CONFIG.keys()))
    cfg = CHANNEL_CONFIG[channel]
    list_price = row[cfg["list_price_col"]]
    current_price = row[cfg["price_col"]]
    existing_disc = row[cfg["discount1_col"]]

    if existing_disc > 0:
        st.info(f"この商品には既存の割引があります: 定価 ¥{list_price:,.0f} → {existing_disc:.0%} OFF → 実売 ¥{current_price:,.0f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # 価格設定セクション
        st.markdown('<div class="form-section"><div class="form-section-title">💰 価格設定</div></div>', unsafe_allow_html=True)
        new_price = st.number_input("販売価格（税込）", min_value=0, max_value=100000,
                                     value=int(current_price), step=100)

        disc1_type = st.radio("1段階目の割引方式", ["割引率(%)", "割引額(円)"], horizontal=True, key="disc1_type")
        if disc1_type == "割引率(%)":
            discount_pct = st.slider("割引率 (%)", 0, 80, 0)
            price_after_1st = int(new_price * (1 - discount_pct / 100))
            disc1_yen_ref = new_price - price_after_1st
            st.caption(f"= ¥{disc1_yen_ref:,}引き")
        else:
            discount_yen = st.number_input("割引額（円）", min_value=0, max_value=100000, value=0, step=100, key="disc1_yen")
            price_after_1st = max(0, int(new_price - discount_yen))
            disc1_pct_ref = (discount_yen / new_price * 100) if new_price > 0 else 0
            st.caption(f"= {disc1_pct_ref:.1f}% OFF")
        st.write(f"1段階目の割引後: **¥{price_after_1st:,}**")

        # 追加割引セクション
        st.markdown('<div class="form-section"><div class="form-section-title">🏷️ 追加割引</div></div>', unsafe_allow_html=True)
        disc2_type = st.radio("2段階目の割引方式", ["割引率(%)", "割引額(円)"], horizontal=True, key="disc2_type")
        if disc2_type == "割引率(%)":
            extra_pct = st.slider("追加割引率 (%)", 0, 50, 0)
            discounted_price = max(0, int(price_after_1st * (1 - extra_pct / 100)))
            disc2_yen_ref = price_after_1st - discounted_price
            st.caption(f"= ¥{disc2_yen_ref:,}引き")
        else:
            extra_yen = st.number_input("追加割引額（円）", min_value=0, max_value=50000, value=0, step=100, key="disc2_yen")
            discounted_price = max(0, int(price_after_1st - extra_yen))
            disc2_pct_ref = (extra_yen / price_after_1st * 100) if price_after_1st > 0 else 0
            st.caption(f"= {disc2_pct_ref:.1f}% OFF")
        st.write(f"最終販売価格: **¥{discounted_price:,}**")
        total_off = new_price - discounted_price
        total_off_pct = (total_off / new_price * 100) if new_price > 0 else 0
        st.caption(f"合計値引: ¥{total_off:,}（{total_off_pct:.1f}% OFF）")

        # コスト変更セクション
        st.markdown('<div class="form-section"><div class="form-section-title">🔧 コスト変更</div></div>', unsafe_allow_html=True)
        new_fee_rate = st.slider("手数料率 (%)", 0.0, 30.0, cfg["default_fee_rate"] * 100, 0.5) / 100
        new_exchange = st.number_input("為替レート (円/ドル)", min_value=80.0, max_value=200.0,
                                        value=float(row["exchange_rate"]) if row["exchange_rate"] > 0 else 150.0, step=1.0)

    with col_right:
        st.markdown(html_section_title("シミュレーション結果"), unsafe_allow_html=True)

        # 為替変更で原価再計算
        if row["exchange_rate"] > 0 and row["usd_cost"] > 0:
            adjusted_cost = row["usd_cost"] * new_exchange * row["pcs_per_unit"]
        else:
            adjusted_cost = row["cost_jpy"]

        price_ex = discounted_price / 1.1
        fee = discounted_price * new_fee_rate
        ship = row[cfg["ship_col"]]
        total_cost = adjusted_cost + row["overseas_shipping"] + fee + ship
        profit = price_ex - total_cost
        margin = (profit / price_ex * 100) if price_ex > 0 else 0

        # #3 現在 vs シミュレーション比較
        cur_profit, cur_margin, cur_total_cost = calc_channel_profit(row, channel)
        profit_diff = profit - cur_profit
        margin_diff = margin - cur_margin
        profit_diff_sign = "▲" if profit_diff >= 0 else "▼"
        margin_diff_sign = "▲" if margin_diff >= 0 else "▼"
        profit_diff_class = "positive" if profit_diff >= 0 else "negative"
        margin_diff_class = "positive" if margin_diff >= 0 else "negative"
        st.markdown(f"""
        <div class="comparison-row">
            <div class="comp-label">利益</div>
            <div class="comp-values">
                <span>¥{cur_profit:,.0f}</span>
                <span class="comp-arrow">→</span>
                <span>¥{profit:,.0f}</span>
                <span class="comp-diff {profit_diff_class}">{profit_diff_sign} ¥{abs(profit_diff):,.0f}</span>
            </div>
        </div>
        <div class="comparison-row">
            <div class="comp-label">粗利率</div>
            <div class="comp-values">
                <span>{cur_margin:.1f}%</span>
                <span class="comp-arrow">→</span>
                <span>{margin:.1f}%</span>
                <span class="comp-diff {margin_diff_class}">{margin_diff_sign} {abs(margin_diff):.1f}pt</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 結果カード（利益正負で色変更）
        profit_status = "positive" if profit >= 0 else "negative"
        margin_status = "positive" if margin >= TARGET_MARGIN else ("warning" if margin >= 0 else "negative")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(html_metric_card("利益（税抜）", f"¥{profit:,.0f}",
                "🟢 黒字" if profit >= 0 else "🔴 赤字", profit_status), unsafe_allow_html=True)
        with r2:
            st.markdown(html_metric_card("粗利率", f"{margin:.1f}%",
                f"目標達成" if margin >= TARGET_MARGIN else "目標未達", margin_status), unsafe_allow_html=True)

        # #9 ゲージバー（粗利率）
        gauge_pct = max(0, min(margin, 100))
        if margin >= TARGET_MARGIN:
            gauge_color = "#2e7d32"
        elif margin >= TARGET_MARGIN * 0.7:
            gauge_color = "#f9a825"
        else:
            gauge_color = "#c62828"
        target_pos = min(TARGET_MARGIN, 100)
        st.markdown(f"""
        <div class="gauge-bar">
            <div class="gauge-bar-fill" style="width:{gauge_pct}%; background:{gauge_color};"></div>
            <div class="gauge-bar-target" style="left:{target_pos}%;">
                <div class="gauge-bar-target-label">目標 {TARGET_MARGIN:.0f}%</div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.72em; color:#999;">
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
        """, unsafe_allow_html=True)

        # #8 割引フローの視覚化
        st.markdown(html_section_title("割引フロー"), unsafe_allow_html=True)
        flow_html = '<div class="price-flow">'
        flow_html += f'<div class="price-flow-step start">販売価格<br>¥{new_price:,}</div>'
        if new_price != price_after_1st:
            flow_html += '<span class="price-flow-arrow">→</span>'
            flow_html += f'<div class="price-flow-step discount">1段階目<br>¥{price_after_1st:,}</div>'
        if price_after_1st != discounted_price:
            flow_html += '<span class="price-flow-arrow">→</span>'
            flow_html += f'<div class="price-flow-step discount">2段階目<br>¥{discounted_price:,}</div>'
        if new_price != discounted_price:
            flow_html += '<span class="price-flow-arrow">→</span>'
            flow_html += f'<div class="price-flow-step final">最終価格<br>¥{discounted_price:,}</div>'
        else:
            flow_html += '<span class="price-flow-arrow">→</span>'
            flow_html += f'<div class="price-flow-step final">割引なし</div>'
        flow_html += '</div>'
        st.markdown(flow_html, unsafe_allow_html=True)

        # #10 コスト内訳 — 積み上げ棒グラフで常時表示
        st.markdown(html_section_title("コスト内訳"), unsafe_allow_html=True)
        cost_items = [
            ("商品原価", adjusted_cost, "#1565c0"),
            ("海外送料", row["overseas_shipping"], "#00897b"),
            (f"手数料({new_fee_rate*100:.1f}%)", fee, "#e65100"),
            ("国内送料", ship, "#6a1b9a"),
        ]
        if total_cost > 0:
            bar_html = '<div class="cost-stack-bar">'
            for label, amount, color in cost_items:
                pct = amount / total_cost * 100
                bar_html += f'<div style="width:{pct}%; background:{color};" title="{label}: ¥{amount:,.0f}">'
                if pct > 10:
                    bar_html += f'¥{amount:,.0f}'
                bar_html += '</div>'
            bar_html += '</div>'
            st.markdown(bar_html, unsafe_allow_html=True)
            # 凡例
            legend_html = '<div style="display:flex; flex-wrap:wrap; gap:12px; font-size:0.78em; margin-top:4px;">'
            for label, amount, color in cost_items:
                pct = amount / total_cost * 100
                legend_html += f'<span><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:4px;"></span>{label}: ¥{amount:,.0f} ({pct:.0f}%)</span>'
            legend_html += '</div>'
            st.markdown(legend_html, unsafe_allow_html=True)
        st.markdown(html_metric_card("販売コスト合計", f"¥{total_cost:,.0f}", "", "neutral"), unsafe_allow_html=True)
        st.caption("※輸入消費税は仕入税額控除で回収可能のため、コストに含めていません")

        # 逆算セクション
        st.divider()
        st.markdown(html_section_title("目標からの逆算"), unsafe_allow_html=True)

        # 目標粗利率からの逆算
        target_margin = st.number_input("目標粗利率 (%)", min_value=0.0, max_value=90.0, value=float(TARGET_MARGIN), step=1.0)
        base_cost = adjusted_cost + row["overseas_shipping"] + ship
        denom = 1/1.1 - new_fee_rate - target_margin / 100 / 1.1
        if denom > 0:
            required_price = base_cost / denom
            st.success(f"粗利率 {target_margin:.0f}% に必要な販売価格: **¥{required_price:,.0f}**")
        else:
            st.error("この手数料率と目標粗利率の組み合わせでは達成不可能です")

        # #2 目標利益額からの逆算
        target_profit_yen = st.number_input("目標利益額（円）", min_value=0, max_value=100000, value=0, step=100, key="target_profit_yen")
        if target_profit_yen > 0:
            denom2 = 1/1.1 - new_fee_rate
            if denom2 > 0:
                required_price2 = (base_cost + target_profit_yen) / denom2
                result_margin2 = (target_profit_yen / (required_price2 / 1.1) * 100) if required_price2 > 0 else 0
                st.success(f"利益 ¥{target_profit_yen:,} に必要な販売価格: **¥{required_price2:,.0f}**（粗利率: {result_margin2:.1f}%）")
            else:
                st.error("この手数料率では達成不可能です")

# =====================================================
# 画面4: 商品管理
# =====================================================
elif page == "⚙️ 商品管理":
    st.title("商品管理")

    tab_edit, tab_add = st.tabs(["商品を編集", "新規追加"])

    # --- 共通: チャネル別フィールドを描画するヘルパー ---
    def render_channel_fields(prefix, label, default_fee_rate, defaults=None, key_prefix=""):
        kp = key_prefix
        d = defaults or {}
        vals = {}
        vals[f"{prefix}_list_price"] = st.number_input(
            f"定価(税込)", min_value=0, value=int(d.get(f"{prefix}_list_price", 0)),
            step=100, key=f"{kp}{prefix}_lp")
        vals[f"{prefix}_price"] = st.number_input(
            f"販売価格(税込)", min_value=0, value=int(d.get(f"{prefix}_price", 0)),
            step=100, key=f"{kp}{prefix}_p")
        vals[f"{prefix}_discount1"] = st.number_input(
            f"割引率", min_value=0.0, max_value=1.0,
            value=float(d.get(f"{prefix}_discount1", 0.0)),
            step=0.05, format="%.2f", key=f"{kp}{prefix}_d1")
        vals[f"{prefix}_discount2"] = st.number_input(
            f"追加割引率", min_value=0.0, max_value=1.0,
            value=float(d.get(f"{prefix}_discount2", 0.0)),
            step=0.05, format="%.2f", key=f"{kp}{prefix}_d2")
        vals[f"{prefix}_shipping"] = st.number_input(
            f"国内送料", min_value=0, value=int(d.get(f"{prefix}_shipping", 0)),
            step=10, key=f"{kp}{prefix}_s")
        auto_fee = round(vals[f"{prefix}_price"] * default_fee_rate)
        current_fee = int(d.get(f"{prefix}_fee", auto_fee)) if d else auto_fee
        if prefix != "wholesale":
            vals[f"{prefix}_fee"] = st.number_input(
                f"手数料({default_fee_rate*100:.0f}%)", min_value=0,
                value=current_fee, step=10, key=f"{kp}{prefix}_f")
        return vals

    # ===================
    # タブ: 商品を編集
    # ===================
    with tab_edit:
        all_items = df.copy()
        all_items["label"] = all_items["no"].astype(str) + " | " + all_items["product_id"].astype(str) + " | " + all_items["name"]

        edit_search = st.text_input("管理番号・商品名で検索", key="edit_search")
        if edit_search:
            mask = (
                all_items["product_id"].astype(str).str.contains(edit_search, case=False, na=False)
                | all_items["name"].str.contains(edit_search, case=False, na=False)
            )
            filtered = all_items[mask]
        else:
            filtered = all_items
        if len(filtered) == 0:
            st.warning("該当する商品がありません")
            st.stop()
        selected = st.selectbox("編集する商品を選択", filtered["label"].tolist(), key="edit_select")
        edit_idx = all_items[all_items["label"] == selected].index[0]
        row = df.loc[edit_idx]

        with st.form("edit_form"):
            # 基本情報: 3列 + 2列の2段構成
            st.markdown(html_section_title("基本情報"), unsafe_allow_html=True)
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                e_rank = st.text_input("ランク", value=str(row["rank"]), key="e_rank")
            with bc2:
                e_pid = st.text_input("管理番号", value=str(row["product_id"]), key="e_pid")
            with bc3:
                e_name = st.text_input("商品名", value=str(row["name"]), key="e_name")

            st.markdown(html_section_title("原価"), unsafe_allow_html=True)
            oc1, oc2, oc3, oc4 = st.columns(4)
            with oc1:
                e_usd = st.number_input("ドル原価", min_value=0.0, value=float(row["usd_cost"]), step=0.1, key="e_usd")
                e_tariff = st.number_input("関税", min_value=0.0, value=float(row["tariff"]), step=0.01, key="e_tariff")
            with oc2:
                e_rate = st.number_input("為替レート", min_value=0.0, value=float(row["exchange_rate"]), step=1.0, key="e_rate")
                e_pcs = st.number_input("PCS/商品", min_value=0.0, value=float(row["pcs_per_unit"]), step=1.0, key="e_pcs")
            with oc3:
                e_cost = st.number_input("商品原価(円)", min_value=0, value=int(row["cost_jpy"]), step=10, key="e_cost")
                e_tax = st.number_input("輸入消費税", min_value=0, value=int(row["import_tax"]), step=10, key="e_tax")
            with oc4:
                e_oship = st.number_input("海外送料", min_value=0.0, value=float(row["overseas_shipping"]), step=1.0, key="e_oship")

            # チャネル別: タブ化
            st.markdown(html_section_title("チャネル別設定"), unsafe_allow_html=True)
            ch_tab1, ch_tab2, ch_tab3, ch_tab4 = st.tabs(["楽天", "Amazon", "Yahoo", "業販"])
            channel_vals = {}
            defaults = row.to_dict()
            with ch_tab1:
                channel_vals.update(render_channel_fields("rakuten", "楽天", 0.16, defaults, "e_"))
            with ch_tab2:
                channel_vals.update(render_channel_fields("amazon", "Amazon", 0.15, defaults, "e_"))
            with ch_tab3:
                channel_vals.update(render_channel_fields("yahoo", "Yahoo", 0.16, defaults, "e_"))
            with ch_tab4:
                channel_vals.update(render_channel_fields("wholesale", "業販", 0.0, defaults, "e_"))

            submitted = st.form_submit_button("保存", type="primary", use_container_width=True)

        if submitted:
            df.at[edit_idx, "rank"] = e_rank
            df.at[edit_idx, "product_id"] = e_pid
            df.at[edit_idx, "name"] = e_name
            df.at[edit_idx, "usd_cost"] = e_usd
            df.at[edit_idx, "tariff"] = e_tariff
            df.at[edit_idx, "exchange_rate"] = e_rate
            df.at[edit_idx, "pcs_per_unit"] = e_pcs
            df.at[edit_idx, "cost_jpy"] = e_cost
            df.at[edit_idx, "import_tax"] = e_tax
            df.at[edit_idx, "overseas_shipping"] = e_oship
            for k, v in channel_vals.items():
                df.at[edit_idx, k] = v
            save_data(df)
            st.success(f"「{e_name}」を保存しました")
            st.rerun()

        # 削除: 赤い警告カード
        st.divider()
        st.markdown(f"""
        <div class="delete-card">
            <div class="title">⚠️ 商品の削除</div>
            <div>この操作は取り消せません。慎重に操作してください。</div>
        </div>
        """, unsafe_allow_html=True)
        confirm = st.checkbox(f"「{row['name']}」を本当に削除する", key="del_confirm")
        if st.button("削除を実行", disabled=not confirm, type="secondary"):
            df = df.drop(edit_idx).reset_index(drop=True)
            save_data(df)
            st.success("削除しました")
            st.rerun()

    # ===================
    # タブ: 新規追加
    # ===================
    with tab_add:
        with st.form("add_form"):
            new_no = int(df["no"].max()) + 1 if len(df) > 0 else 1
            st.caption(f"No: {new_no}（自動付番）")

            st.markdown(html_section_title("基本情報"), unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                a_rank = st.text_input("ランク", value="", key="a_rank")
            with ac2:
                a_pid = st.text_input("管理番号", value="", key="a_pid")
            with ac3:
                a_name = st.text_input("商品名", value="", key="a_name")

            st.markdown(html_section_title("原価"), unsafe_allow_html=True)
            nc1, nc2, nc3, nc4 = st.columns(4)
            with nc1:
                a_usd = st.number_input("ドル原価", min_value=0.0, value=0.0, step=0.1, key="a_usd")
                a_tariff = st.number_input("関税", min_value=0.0, value=0.0, step=0.01, key="a_tariff")
            with nc2:
                a_rate = st.number_input("為替レート", min_value=0.0, value=150.0, step=1.0, key="a_rate")
                a_pcs = st.number_input("PCS/商品", min_value=0.0, value=1.0, step=1.0, key="a_pcs")
            with nc3:
                a_cost = st.number_input("商品原価(円)", min_value=0, value=0, step=10, key="a_cost")
                a_tax = st.number_input("輸入消費税", min_value=0, value=0, step=10, key="a_tax")
            with nc4:
                a_oship = st.number_input("海外送料", min_value=0.0, value=0.0, step=1.0, key="a_oship")

            st.markdown(html_section_title("チャネル別設定"), unsafe_allow_html=True)
            nch_tab1, nch_tab2, nch_tab3, nch_tab4 = st.tabs(["楽天", "Amazon", "Yahoo", "業販"])
            new_ch_vals = {}
            with nch_tab1:
                new_ch_vals.update(render_channel_fields("rakuten", "楽天", 0.16, key_prefix="a_"))
            with nch_tab2:
                new_ch_vals.update(render_channel_fields("amazon", "Amazon", 0.15, key_prefix="a_"))
            with nch_tab3:
                new_ch_vals.update(render_channel_fields("yahoo", "Yahoo", 0.16, key_prefix="a_"))
            with nch_tab4:
                new_ch_vals.update(render_channel_fields("wholesale", "業販", 0.0, key_prefix="a_"))

            add_submitted = st.form_submit_button("追加", type="primary", use_container_width=True)

        if add_submitted:
            if not a_name:
                st.error("商品名は必須です")
            else:
                new_row = {
                    "no": new_no, "rank": a_rank, "product_id": a_pid,
                    "name": a_name,
                    "usd_cost": a_usd, "tariff": a_tariff, "exchange_rate": a_rate,
                    "pcs_per_unit": a_pcs, "cost_jpy": a_cost, "import_tax": a_tax,
                    "overseas_shipping": a_oship,
                }
                new_row.update(new_ch_vals)
                for col in df.columns:
                    if col not in new_row:
                        new_row[col] = 0
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(df)
                st.success(f"「{a_name}」を追加しました（No: {new_no}）")
                st.rerun()
