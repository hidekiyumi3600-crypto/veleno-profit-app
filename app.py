import streamlit as st
import pandas as pd
import os

# --- 設定 ---
CHANNEL_CONFIG = {
    "楽天": {"price_col": "rakuten_price", "list_price_col": "rakuten_list_price", "discount1_col": "rakuten_discount1", "fee_col": "rakuten_fee", "ship_col": "rakuten_shipping", "default_fee_rate": 0.16},
    "Amazon": {"price_col": "amazon_price", "list_price_col": "amazon_list_price", "discount1_col": "amazon_discount1", "fee_col": "amazon_fee", "ship_col": "amazon_shipping", "default_fee_rate": 0.15},
    "Yahoo": {"price_col": "yahoo_price", "list_price_col": "yahoo_list_price", "discount1_col": "yahoo_discount1", "fee_col": "yahoo_fee", "ship_col": "yahoo_shipping", "default_fee_rate": 0.16},
    "業販": {"price_col": "wholesale_price", "list_price_col": "wholesale_list_price", "discount1_col": "wholesale_discount1", "fee_col": None, "ship_col": "wholesale_shipping", "default_fee_rate": 0.0},
}

st.set_page_config(page_title="VELENO 利益計算", page_icon="🚗", layout="wide")

# --- パスワード認証 ---
def check_password():
    """パスワード認証。正しければTrueを返す。"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 VELENO 利益計算")
    password = st.text_input("パスワードを入力してください", type="password")
    if st.button("ログイン", type="primary"):
        if password == st.secrets["password"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが正しくありません")
    return False

if not check_password():
    st.stop()

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
    df = df.fillna({"name": "", "size": "", "color": "", "rank": "", "product_id": ""})
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def save_data(df):
    """CSVに保存してキャッシュをクリア。"""
    df.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()


def calc_profit(price_tax_incl, cost_jpy, overseas_shipping, fee_amount, domestic_shipping):
    """利益を計算する。販売価格は税込→税抜に変換。
    ※輸入消費税は仕入税額控除で回収可能のため、コストに含めない（Excel準拠）。
    """
    if price_tax_incl <= 0:
        return 0, 0, 0
    price_ex_tax = price_tax_incl / 1.1  # 税抜き
    total_cost = cost_jpy + overseas_shipping + fee_amount + domestic_shipping
    profit = price_ex_tax - total_cost
    margin = (profit / price_ex_tax * 100) if price_ex_tax > 0 else 0
    return round(profit), round(margin, 1), round(total_cost)


def calc_channel_profit(row, channel, fee_rate_override=None):
    """指定チャネルの利益を計算する。
    CSVに手数料額がある場合はそれを使用、なければ税込価格×手数料率で計算。
    """
    cfg = CHANNEL_CONFIG[channel]
    price = row[cfg["price_col"]]
    ship = row[cfg["ship_col"]]

    if fee_rate_override is not None:
        # シミュレーション時: 税込価格ベースで手数料計算
        fee_amount = price * fee_rate_override
    elif cfg["fee_col"] and row.get(cfg["fee_col"], 0) > 0:
        # CSVに手数料額がある場合はそのまま使用（Excel準拠）
        fee_amount = row[cfg["fee_col"]]
    else:
        # デフォルト: 税込価格 × 手数料率（Excel準拠）
        fee_amount = price * cfg["default_fee_rate"]

    return calc_profit(price, row["cost_jpy"], row["overseas_shipping"], fee_amount, ship)


def add_profit_columns(df, channel):
    """DataFrameにチャネル別利益列を追加。"""
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
st.sidebar.title("VELENO 利益計算")
page = st.sidebar.radio("ページ", ["商品一覧", "商品詳細・チャネル比較", "価格シミュレーション", "商品管理"])

# =====================================================
# 画面1: 商品一覧 & 利益ダッシュボード
# =====================================================
if page == "商品一覧":
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
        show_loss = st.checkbox("赤字商品のみ表示")

    # データ準備
    view = df.copy()
    view = add_profit_columns(view, channel)

    price_col = CHANNEL_CONFIG[channel]["price_col"]
    view = view[view[price_col] > 0]  # 販売価格0を除外

    if rank_filter != "すべて":
        view = view[view["rank"] == rank_filter]
    if search:
        view = view[
            view["name"].str.contains(search, case=False, na=False)
            | view["product_id"].astype(str).str.contains(search, case=False, na=False)
        ]
    if show_loss:
        view = view[view[f"{channel}_利益"] < 0]

    # サマリー
    total = len(view)
    loss_count = len(view[view[f"{channel}_利益"] < 0])
    avg_margin = view[f"{channel}_粗利率"].mean() if total > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("商品数", f"{total:,}")
    m2.metric("赤字商品", f"{loss_count}", delta=f"-{loss_count}" if loss_count > 0 else "0", delta_color="inverse")
    m3.metric("平均粗利率", f"{avg_margin:.1f}%")
    m4.metric("粗利率目標", "56%")

    # テーブル表示
    list_price_col = CHANNEL_CONFIG[channel]["list_price_col"]
    discount1_col = CHANNEL_CONFIG[channel]["discount1_col"]
    display_cols = ["no", "product_id", "rank", "name", "size", "color", list_price_col, discount1_col, price_col, "cost_jpy", f"{channel}_利益", f"{channel}_粗利率"]
    display_names = {"no": "No", "product_id": "管理番号", "rank": "ランク", "name": "商品名", "size": "サイズ",
                     "color": "色", list_price_col: "定価(税込)", discount1_col: "割引率",
                     price_col: "販売価格(税込)", "cost_jpy": "原価",
                     f"{channel}_利益": "利益(税抜)", f"{channel}_粗利率": "粗利率(%)"}

    sort_col = st.selectbox("並び替え", [f"{channel}_利益", f"{channel}_粗利率", price_col, "cost_jpy"], format_func=lambda x: display_names.get(x, x))
    sort_asc = st.checkbox("昇順（低い順）", value=True)
    view = view.sort_values(sort_col, ascending=sort_asc)

    disp = view[display_cols].copy()
    disp[list_price_col] = disp[list_price_col].apply(lambda x: f"¥{x:,.0f}")
    disp[discount1_col] = disp[discount1_col].apply(lambda x: f"{x:.0%}" if x > 0 else "-")
    disp[price_col] = disp[price_col].apply(lambda x: f"¥{x:,.0f}")
    disp["cost_jpy"] = disp["cost_jpy"].apply(lambda x: f"¥{x:,.0f}")
    disp[f"{channel}_利益"] = disp[f"{channel}_利益"].apply(lambda x: f"¥{x:,.0f}")
    disp[f"{channel}_粗利率"] = disp[f"{channel}_粗利率"].apply(lambda x: f"{x:.1f}%")
    disp = disp.rename(columns=display_names)
    st.dataframe(disp, height=600, use_container_width=True)

    # TOP/WORST
    def fmt_ranking(src, profit_col, margin_col):
        d = src[["name", "size", "color", price_col, profit_col, margin_col]].copy()
        d.columns = ["商品名", "サイズ", "色", "販売価格", "利益", "粗利率(%)"]
        d["販売価格"] = d["販売価格"].apply(lambda x: f"¥{x:,.0f}")
        d["利益"] = d["利益"].apply(lambda x: f"¥{x:,.0f}")
        d["粗利率(%)"] = d["粗利率(%)"].apply(lambda x: f"{x:.1f}%")
        return d

    st.subheader("利益 TOP10")
    st.dataframe(fmt_ranking(view.nlargest(10, f"{channel}_利益"), f"{channel}_利益", f"{channel}_粗利率"), use_container_width=True)

    st.subheader("利益 WORST10（赤字順）")
    st.dataframe(fmt_ranking(view.nsmallest(10, f"{channel}_利益"), f"{channel}_利益", f"{channel}_粗利率"), use_container_width=True)

# =====================================================
# 画面2: 商品詳細 & チャネル比較
# =====================================================
elif page == "商品詳細・チャネル比較":
    st.title("商品詳細 & チャネル比較")

    # 商品選択
    active = df[df["rakuten_price"] > 0].copy()
    active["label"] = active["no"].astype(str) + " | " + active["product_id"].astype(str) + " | " + active["name"] + " " + active["size"].astype(str) + " " + active["color"].astype(str)

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

    # 商品情報
    st.subheader(f"{row['name']}　{row['size']}　{row['color']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("ランク", row["rank"])
    c2.metric("管理番号", row["product_id"])
    c3.metric("商品原価", f"¥{row['cost_jpy']:,.0f}")

    # 原価内訳
    with st.expander("原価の内訳"):
        st.write(f"- ドル原価: ${row['usd_cost']:.2f}")
        st.write(f"- 為替レート: ¥{row['exchange_rate']:.1f}")
        st.write(f"- 関税: {row['tariff']:.2f}")
        st.write(f"- 商品原価（円）: ¥{row['cost_jpy']:,.0f}")
        st.write(f"- 輸入消費税: ¥{row['import_tax']:,.0f}")
        st.write(f"- 海外送料: ¥{row['overseas_shipping']:,.0f}")

    # 4チャネル比較
    st.subheader("チャネル別利益比較")
    results = []
    for ch in CHANNEL_CONFIG:
        cfg = CHANNEL_CONFIG[ch]
        list_price = row[cfg["list_price_col"]]
        price = row[cfg["price_col"]]
        disc1 = row[cfg["discount1_col"]]
        if price <= 0:
            results.append({"チャネル": ch, "定価(税込)": 0, "割引率": "-", "販売価格(税込)": 0,
                          "手数料率": f"{cfg['default_fee_rate']*100:.0f}%",
                          "手数料": 0, "国内送料": 0, "販売コスト合計": 0, "利益": 0, "粗利率(%)": 0})
            continue
        # 手数料: CSVの値があればそれを使用、なければ税込価格×手数料率
        if cfg["fee_col"] and row.get(cfg["fee_col"], 0) > 0:
            fee = row[cfg["fee_col"]]
        else:
            fee = price * cfg["default_fee_rate"]
        ship = row[cfg["ship_col"]]
        profit, margin, total_cost = calc_profit(price, row["cost_jpy"], row["overseas_shipping"], fee, ship)
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

    result_df = pd.DataFrame(results)
    result_df["利益"] = result_df["利益"].apply(lambda x: f"¥{x:,.0f}" if isinstance(x, (int, float)) else x)
    result_df["粗利率(%)"] = result_df["粗利率(%)"].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x)
    st.dataframe(result_df, use_container_width=True)

    # 棒グラフ
    chart_data = pd.DataFrame({
        "チャネル": [r["チャネル"] for r in results],
        "利益": [r["利益"] if isinstance(r["利益"], (int, float)) else 0 for r in results]
    })
    st.bar_chart(chart_data.set_index("チャネル"))

# =====================================================
# 画面3: 価格シミュレーション
# =====================================================
elif page == "価格シミュレーション":
    st.title("価格シミュレーション")

    # 商品選択
    active = df[df["rakuten_price"] > 0].copy()
    active["label"] = active["no"].astype(str) + " | " + active["product_id"].astype(str) + " | " + active["name"] + " " + active["size"].astype(str) + " " + active["color"].astype(str)

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

    st.subheader(f"{row['name']}　{row['size']}　{row['color']}")
    st.caption(f"現在の原価: ¥{row['cost_jpy']:,.0f}　輸入消費税: ¥{row['import_tax']:,.0f}　海外送料: ¥{row['overseas_shipping']:,.0f}")

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
        st.subheader("価格を変更")
        new_price = st.number_input("販売価格（税込）", min_value=0, max_value=100000,
                                     value=int(current_price), step=100)

        discount_pct = st.slider("割引率 (%)", 0, 80, 0)
        price_after_1st = int(new_price * (1 - discount_pct / 100))
        st.write(f"1段階目の割引後: **¥{price_after_1st:,}**")

        st.divider()
        st.subheader("追加割引")
        extra_pct = st.slider("追加割引率 (%)", 0, 50, 0)
        extra_yen = st.number_input("追加割引額（円）", min_value=0, max_value=50000, value=0, step=100)
        discounted_price = int(price_after_1st * (1 - extra_pct / 100) - extra_yen)
        if discounted_price < 0:
            discounted_price = 0
        st.write(f"最終販売価格: **¥{discounted_price:,}**")
        total_off = new_price - discounted_price
        total_off_pct = (total_off / new_price * 100) if new_price > 0 else 0
        st.caption(f"合計値引: ¥{total_off:,}（{total_off_pct:.1f}% OFF）")

        st.divider()
        st.subheader("コスト変更")
        new_fee_rate = st.slider("手数料率 (%)", 0.0, 30.0, cfg["default_fee_rate"] * 100, 0.5) / 100
        new_exchange = st.number_input("為替レート (円/ドル)", min_value=80.0, max_value=200.0,
                                        value=float(row["exchange_rate"]) if row["exchange_rate"] > 0 else 150.0, step=1.0)

    with col_right:
        st.subheader("シミュレーション結果")

        # 為替変更で原価再計算
        if row["exchange_rate"] > 0 and row["usd_cost"] > 0:
            adjusted_cost = row["usd_cost"] * new_exchange * row["pcs_per_unit"]
        else:
            adjusted_cost = row["cost_jpy"]

        price_ex = discounted_price / 1.1
        fee = discounted_price * new_fee_rate  # 税込価格ベースで手数料計算（Excel準拠）
        ship = row[cfg["ship_col"]]
        total_cost = adjusted_cost + row["overseas_shipping"] + fee + ship
        profit = price_ex - total_cost
        margin = (profit / price_ex * 100) if price_ex > 0 else 0

        color = "🔴" if profit < 0 else "🟢" if margin >= 56 else "🟡"
        st.metric("利益（税抜）", f"¥{profit:,.0f}", delta=f"{color}")
        st.metric("粗利率", f"{margin:.1f}%", delta="目標56%以上" if margin >= 56 else "目標未達")
        st.metric("販売コスト合計", f"¥{total_cost:,.0f}")

        st.divider()
        st.caption("コスト内訳")
        st.write(f"- 商品原価: ¥{adjusted_cost:,.0f}")
        st.write(f"- 海外送料: ¥{row['overseas_shipping']:,.0f}")
        st.write(f"- 手数料({new_fee_rate*100:.1f}%): ¥{fee:,.0f}")
        st.write(f"- 国内送料: ¥{ship:,.0f}")
        st.caption("※輸入消費税は仕入税額控除で回収可能のため、コストに含めていません")

        # 逆算: 目標粗利率を達成する価格
        st.divider()
        st.subheader("目標粗利率からの逆算")
        target_margin = st.number_input("目標粗利率 (%)", min_value=0.0, max_value=90.0, value=56.0, step=1.0)
        base_cost = adjusted_cost + row["overseas_shipping"] + ship
        # profit = price_tax_incl/1.1 - base_cost - price_tax_incl * fee_rate
        # margin = profit / (price_tax_incl/1.1) = target_margin/100
        # price_tax_incl * (1/1.1 - fee_rate) - base_cost = price_tax_incl/1.1 * target_margin/100
        # price_tax_incl * (1/1.1 - fee_rate - target_margin/100/1.1) = base_cost
        denom = 1/1.1 - new_fee_rate - target_margin / 100 / 1.1
        if denom > 0:
            required_price = base_cost / denom
            st.success(f"必要な販売価格（税込）: **¥{required_price:,.0f}**")
        else:
            st.error("この手数料率と目標粗利率の組み合わせでは達成不可能です")

# =====================================================
# 画面4: 商品管理
# =====================================================
elif page == "商品管理":
    st.title("商品管理")

    tab_edit, tab_add = st.tabs(["商品を編集", "新規追加"])

    # --- 共通: チャネル別フィールドを描画するヘルパー ---
    def render_channel_fields(prefix, label, default_fee_rate, defaults=None, key_prefix=""):
        """チャネル列の入力フィールドを描画し、値の辞書を返す。"""
        kp = key_prefix
        d = defaults or {}
        vals = {}
        st.markdown(f"**{label}**")
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
        # 手数料: 販売価格から自動計算をデフォルト表示
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
        all_items["label"] = all_items["no"].astype(str) + " | " + all_items["product_id"].astype(str) + " | " + all_items["name"] + " " + all_items["size"].astype(str) + " " + all_items["color"].astype(str)

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
            st.subheader("基本情報")
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            with bc1:
                e_rank = st.text_input("ランク", value=str(row["rank"]), key="e_rank")
            with bc2:
                e_pid = st.text_input("管理番号", value=str(row["product_id"]), key="e_pid")
            with bc3:
                e_name = st.text_input("商品名", value=str(row["name"]), key="e_name")
            with bc4:
                e_size = st.text_input("サイズ", value=str(row["size"]), key="e_size")
            with bc5:
                e_color = st.text_input("色", value=str(row["color"]), key="e_color")

            st.subheader("原価")
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

            st.subheader("チャネル別")
            ch1, ch2, ch3, ch4 = st.columns(4)
            channel_vals = {}
            defaults = row.to_dict()
            with ch1:
                channel_vals.update(render_channel_fields("rakuten", "楽天", 0.16, defaults, "e_"))
            with ch2:
                channel_vals.update(render_channel_fields("amazon", "Amazon", 0.15, defaults, "e_"))
            with ch3:
                channel_vals.update(render_channel_fields("yahoo", "Yahoo", 0.16, defaults, "e_"))
            with ch4:
                channel_vals.update(render_channel_fields("wholesale", "業販", 0.0, defaults, "e_"))

            submitted = st.form_submit_button("保存", type="primary", use_container_width=True)

        if submitted:
            df.at[edit_idx, "rank"] = e_rank
            df.at[edit_idx, "product_id"] = e_pid
            df.at[edit_idx, "name"] = e_name
            df.at[edit_idx, "size"] = e_size
            df.at[edit_idx, "color"] = e_color
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

        # 削除
        st.divider()
        with st.expander("この商品を削除"):
            confirm = st.checkbox(f"「{row['name']} {row['size']} {row['color']}」を本当に削除する", key="del_confirm")
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

            st.subheader("基本情報")
            ac1, ac2, ac3, ac4, ac5 = st.columns(5)
            with ac1:
                a_rank = st.text_input("ランク", value="", key="a_rank")
            with ac2:
                a_pid = st.text_input("管理番号", value="", key="a_pid")
            with ac3:
                a_name = st.text_input("商品名", value="", key="a_name")
            with ac4:
                a_size = st.text_input("サイズ", value="", key="a_size")
            with ac5:
                a_color = st.text_input("色", value="", key="a_color")

            st.subheader("原価")
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

            st.subheader("チャネル別")
            nch1, nch2, nch3, nch4 = st.columns(4)
            new_ch_vals = {}
            with nch1:
                new_ch_vals.update(render_channel_fields("rakuten", "楽天", 0.16, key_prefix="a_"))
            with nch2:
                new_ch_vals.update(render_channel_fields("amazon", "Amazon", 0.15, key_prefix="a_"))
            with nch3:
                new_ch_vals.update(render_channel_fields("yahoo", "Yahoo", 0.16, key_prefix="a_"))
            with nch4:
                new_ch_vals.update(render_channel_fields("wholesale", "業販", 0.0, key_prefix="a_"))

            add_submitted = st.form_submit_button("追加", type="primary", use_container_width=True)

        if add_submitted:
            if not a_name:
                st.error("商品名は必須です")
            else:
                new_row = {
                    "no": new_no, "rank": a_rank, "product_id": a_pid,
                    "name": a_name, "size": a_size, "color": a_color,
                    "usd_cost": a_usd, "tariff": a_tariff, "exchange_rate": a_rate,
                    "pcs_per_unit": a_pcs, "cost_jpy": a_cost, "import_tax": a_tax,
                    "overseas_shipping": a_oship,
                }
                new_row.update(new_ch_vals)
                # CSVの全列を揃える
                for col in df.columns:
                    if col not in new_row:
                        new_row[col] = 0
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                save_data(df)
                st.success(f"「{a_name}」を追加しました（No: {new_no}）")
                st.rerun()
