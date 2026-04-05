import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import matplotlib
# Streamlit環境でのMatplotlibの警告・エラー防止
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from matplotlib import font_manager

# --- 日本語フォント対応 ---
def set_japanese_font():
    # 1. まず japanize_matplotlib を試す（入っていれば一番確実）
    try:
        import japanize_matplotlib
        japanize_matplotlib.japanize()  # ← ここで明示的にフォント設定を適用
        return
    except ImportError:
        pass
    
    # 2. 入っていない場合はOSに入っている標準的な日本語フォントを探す
    candidates = ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP',
                  'Hiragino Sans', 'Hiragino Maru Gothic Pro', 'MS Gothic', 
                  'Yu Gothic', 'Meiryo']
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            return

# 起動時に一度フォントを設定
set_japanese_font()

# --- アルファベット付与(CLD)の共通関数 ---
def get_cld_letters(df, target, group, tukey_summary):
    means = df.groupby(group)[target].mean().sort_values(ascending=False)
    groups = means.index.astype(str).tolist()
    
    cld = {g: [] for g in groups}
    current_letter = ord('a')
    
    for i in range(len(groups)):
        g1 = groups[i]
        non_diff = [g1]
        for j in range(i + 1, len(groups)):
            g2 = groups[j]
            mask = ((tukey_summary['group1'].astype(str) == g1) & (tukey_summary['group2'].astype(str) == g2)) | \
                   ((tukey_summary['group1'].astype(str) == g2) & (tukey_summary['group2'].astype(str) == g1))
            reject = tukey_summary.loc[mask, 'reject'].values
            if len(reject) > 0 and not reject[0]: 
                non_diff.append(g2)
        
        shared_letters = set(cld[non_diff[0]])
        for g in non_diff[1:]:
            shared_letters = shared_letters.intersection(cld[g])
        
        if not shared_letters:
            for g in non_diff:
                cld[g].append(chr(current_letter))
            current_letter += 1
    
    return pd.DataFrame({
        'groups_name': list(cld.keys()),
        'letters': ["".join(l) for l in cld.values()]
    }), groups

# ==========================================
# ページ設定とタイトル
# ==========================================
st.set_page_config(page_title="第3章 t検定・分散分析・多重比較", layout="wide")

st.title("第3章 t検定・分散分析・多重比較「平均の差を正しく検定する」")
st.markdown("""
このアプリでは、架空の試験データを用いたシミュレーターで**t検定**と**分散分析**の仕組みを直感的に学ぶことができます。
また、「実データ解析ツール」タブでは、**ご自身の実験データ（CSV）を読み込んで本格的な統計解析とグラフ作成**を自動で行うことができます。
""")

# タブの作成
tab1, tab2, tab3, tab4 = st.tabs([
    "⚖️ 1. t検定シミュレーター (2群)", 
    "🌾 2. 分散分析シミュレーター (3群以上)", 
    "📊 3. 実データ解析ツール (CSV対応)", 
    "📖 4. 統計の基礎知識とコード"
])

# ==========================================
# タブ1: t検定
# ==========================================
with tab1:
    st.header("🥔 t検定：新系統と標準品種の収量を比較する")
    st.markdown("**帰無仮説（$H_0$）**：「新系統と標準品種の収量に差はない」と仮定します。")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("データ設定")
        st.info("スライダーを動かすたびにデータが再生成されます。バラツキを増減させて、p値がどう変化するか観察してください。")
        n_samples_t = st.slider("サンプルサイズ（株数）", min_value=3, max_value=50, value=10, key="n_t")
        mean_a = st.slider("標準品種の平均収量 (g/株)", 500, 1500, 1000, step=50)
        mean_b = st.slider("新系統の平均収量 (g/株)", 500, 1500, 1100, step=50)
        std_dev = st.slider("データのバラツキ（標準偏差）", 50, 300, 100, step=10, key="std_t")
        
    with col2:
        # データ生成
        data_a = np.random.normal(mean_a, std_dev, n_samples_t)
        data_b = np.random.normal(mean_b, std_dev, n_samples_t)
        
        df_t = pd.DataFrame({
            "品種": ["標準品種"] * n_samples_t + ["新系統"] * n_samples_t,
            "収量 (g)": np.concatenate([data_a, data_b])
        })
        
        # 可視化 (Plotly)
        fig_t = px.box(df_t, x="品種", y="収量 (g)", color="品種", points="all", title="収量データの分布（箱ひげ図）")
        fig_t.update_yaxes(range=[0, 2000]) # Y軸を固定し、バラツキの変化を視覚的に分かりやすく
        st.plotly_chart(fig_t, use_container_width=True)
        
        # t検定の実行（現代の主流である「ウェルチのt検定 (equal_var=False)」）
        t_stat, p_value_t = stats.ttest_ind(data_a, data_b, equal_var=False)
        
        st.subheader("📊 検定結果 (ウェルチのt検定)")
        st.write(f"**t値:** {t_stat:.3f}")
        st.write(f"**p値:** {p_value_t:.4f}")
        
        if p_value_t < 0.05:
            st.success("🎉 **有意差あり (p < 0.05)**：帰無仮説は棄却されました。「新系統と標準品種の収量には有意な差がある」と言えます。")
        else:
            st.warning("🤔 **有意差なし (p >= 0.05)**：帰無仮説は棄却されません。「収量に差があるとは言えない（たまたまのバラツキの範囲内）」という結論になります。")

# ==========================================
# タブ2: 分散分析 (ANOVA)
# ==========================================
with tab2:
    st.header("🌾 分散分析（F検定）：3つの施肥区の収量を比較する")
    st.markdown("**帰無仮説（$H_0$）**：「無施肥区、標準区、多肥区の3つのグループ間で、平均収量に差はない」と仮定します。")
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.subheader("データ設定")
        n_samples_f = st.slider("サンプルサイズ（株数）", min_value=3, max_value=50, value=10, key="n_f")
        mean_c = st.slider("無施肥区の平均収量", 500, 1500, 800, step=50)
        mean_d = st.slider("標準区の平均収量", 500, 1500, 1000, step=50)
        mean_e = st.slider("多肥区の平均収量", 500, 1500, 1050, step=50)
        std_dev_f = st.slider("データのバラツキ（標準偏差）", 50, 300, 150, step=10, key="std_f")

    with col4:
        # データ生成
        data_c = np.random.normal(mean_c, std_dev_f, n_samples_f)
        data_d = np.random.normal(mean_d, std_dev_f, n_samples_f)
        data_e = np.random.normal(mean_e, std_dev_f, n_samples_f)
        
        df_f = pd.DataFrame({
            "処理区": ["無施肥区"] * n_samples_f + ["標準区"] * n_samples_f + ["多肥区"] * n_samples_f,
            "収量 (g)": np.concatenate([data_c, data_d, data_e])
        })
        
        # 可視化 (Plotly)
        fig_f = px.box(df_f, x="処理区", y="収量 (g)", color="処理区", points="all", title="施肥区ごとの収量分布（箱ひげ図）")
        fig_f.update_yaxes(range=[0, 2000]) # Y軸を固定
        st.plotly_chart(fig_f, use_container_width=True)
        
        # 一元配置分散分析の実行
        f_stat, p_value_f = stats.f_oneway(data_c, data_d, data_e)
        
        st.subheader("📊 検定結果 (分散分析)")
        st.write(f"**F値:** {f_stat:.3f}")
        st.write(f"**p値:** {p_value_f:.4f}")
        
        if p_value_f < 0.05:
            st.success("🎉 **有意差あり (p < 0.05)**：少なくとも1つの処理区に有意な差があります！")
            
            # 分散分析で有意差が出た場合のみ、Tukeyの多重比較を自動で実行
            st.markdown("#### 🔍 続けて多重比較 (Tukey法) を実行：具体的にどの区に差があるのか？")
            tukey = pairwise_tukeyhsd(endog=df_f['収量 (g)'], groups=df_f['処理区'], alpha=0.05)
            
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            tukey_df['reject'] = tukey_df['reject'].map({True: '有意差あり', False: '差なし'})
            tukey_df = tukey_df.rename(columns={'group1': '比較A', 'group2': '比較B', 'meandiff': '平均値の差', 'p-adj': '調整済p値', 'reject': '判定'})
            
            st.dataframe(tukey_df[['比較A', '比較B', '平均値の差', '調整済p値', '判定']], use_container_width=True)
            
        else:
            st.warning("🤔 **有意差なし (p >= 0.05)**：処理区の間に有意な差があるとは言えません。")

# ==========================================
# タブ3: 実データ解析ツール (CSV対応)
# ==========================================
with tab3:
    st.header("📊 栽培試験データ・分散分析解析ツール")
    st.markdown("自身の実験データ（CSV）を読み込んで、分散分析からグラフ作成までを一括で行います。")

    data_source = st.radio(
        "データの入力方法を選んでください：", 
        ["🥔 組み込みのサンプルデータで試す", "📁 自分のCSVをアップロードする"],
        key="data_source"
    )

    df_real = None

    if data_source == "🥔 組み込みのサンプルデータで試す":
        df_real = pd.DataFrame({
            '品種': ['とうや', 'とうや', 'とうや', 'ニシユタカ', 'ニシユタカ', 'ニシユタカ', 'デジマ', 'デジマ', 'デジマ'] * 2,
            '施肥量': [0, 10, 20, 0, 10, 20, 0, 10, 20] * 2,
            '収量': [210, 250, 260, 230, 280, 310, 190, 240, 250, 215, 255, 265, 235, 285, 315, 195, 245, 255]
        })
        st.success("✅ サンプルデータを読み込みました。（施肥量を数値データとして扱います）")
    else:
        uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv", key="uploader")
        if uploaded_file is not None:
            try:
                df_real = pd.read_csv(uploaded_file, encoding='shift_jis')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df_real = pd.read_csv(uploaded_file, encoding='utf-8')
            
            if len(df_real.columns) == 1:
                uploaded_file.seek(0)
                df_real = pd.read_csv(uploaded_file, encoding='shift_jis', sep='\t')
            st.success("✅ CSVを読み込みました。")

    if df_real is not None:
        st.subheader("1. データプレビュー")
        st.dataframe(df_real.head(10))

        col1, col2, col3 = st.columns(3)
        with col1:
            numeric_cols = df_real.select_dtypes(include=['number']).columns.tolist()
            yld_idx = df_real.columns.get_loc('収量') if '収量' in df_real.columns else (df_real.columns.get_loc(numeric_cols[0]) if numeric_cols else 0)
            target_col = st.selectbox("目的変数（収量など：数値）", df_real.columns, index=yld_idx)
        with col2:
            var_idx = df_real.columns.get_loc('品種') if '品種' in df_real.columns else 0
            factor_x = st.selectbox("主要因（品種など：カテゴリ扱いにします）", df_real.columns, index=var_idx)
        with col3:
            options_sub = ["なし"] + list(df_real.columns)
            fer_idx = df_real.columns.get_loc('施肥量') + 1 if '施肥量' in df_real.columns else 0
            factor_sub = st.selectbox("副要因（交互作用を解析可能）", options_sub, index=fer_idx)

        if st.button("解析を実行する", type="primary"):
            try:
                df_clean = df_real.copy()
                df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
                df_clean = df_clean.dropna(subset=[target_col]).copy()

                if df_clean.empty:
                    st.error("【エラー】目的変数に数値がありません。列の選択を見直してください。")
                else:
                    st.header("📈 解析結果報告書")
                    
                    # ======= 【重要】グラフのフォントリセット対策 =======
                    sns.set_theme(style="whitegrid") # ここで一度フォントが英語にリセットされてしまう
                    set_japanese_font()              # すかさず日本語フォントを再設定する
                    # ====================================================

                    df_clean[factor_x] = df_clean[factor_x].astype(str)
                    if factor_sub != "なし":
                        df_clean[factor_sub] = df_clean[factor_sub].astype(str)

                    # ==========================================
                    # 1. 分散分析 (ANOVA)
                    # ==========================================
                    if factor_sub == "なし":
                        formula = f'Q("{target_col}") ~ C(Q("{factor_x}"))'
                        model = ols(formula, data=df_clean).fit()
                        anova_res = anova_lm(model, typ=2)
                        anova_res.index = [f"{factor_x} (主効果)", '残差']
                        st.subheader("1. 一元配置分散分析表 (ANOVA)")
                        
                        py_anova_code = f"""
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

formula = 'Q("{target_col}") ~ C(Q("{factor_x}"))'
model = ols(formula, data=df).fit()
anova_res = anova_lm(model, typ=2)
anova_res['寄与率'] = anova_res['sum_sq'] / anova_res['sum_sq'].sum()
print(anova_res)
                        """
                        r_anova_code = f"""
model <- aov({target_col} ~ as.factor({factor_x}), data=df)
summary(model)
library(effectsize)
eta_squared(model)
                        """
                        interaction_p = np.nan
                    else:
                        formula = f'Q("{target_col}") ~ C(Q("{factor_x}")) * C(Q("{factor_sub}"))'
                        model = ols(formula, data=df_clean).fit()
                        anova_res = anova_lm(model, typ=2)
                        
                        idx_interaction = f"{factor_x} x {factor_sub} (交互作用)"
                        new_index = [
                            f"{factor_x} (主効果)", 
                            f"{factor_sub} (主効果)", 
                            idx_interaction,
                            '残差'
                        ]
                        if len(anova_res.index) == len(new_index):
                            anova_res.index = new_index
                        else:
                            st.warning("⚠️ モデルの縮退により、一部の効果が計算されませんでした。ANOVA表のインデックスは自動設定されます。")

                        st.subheader("1. 二元配置分散分析表 (主効果＋交互作用) (Typ II ANOVA)")

                        interaction_p = np.nan
                        if idx_interaction in anova_res.index:
                            interaction_p = anova_res.loc[idx_interaction, 'PR(>F)']
                            
                        py_anova_code = f"""
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

formula = 'Q("{target_col}") ~ C(Q("{factor_x}")) * C(Q("{factor_sub}"))'
model = ols(formula, data=df).fit()
anova_res = anova_lm(model, typ=2)
anova_res['寄与率'] = anova_res['sum_sq'] / anova_res['sum_sq'].sum()
print(anova_res)
                        """
                        r_anova_code = f"""
model <- aov({target_col} ~ as.factor({factor_x}) * as.factor({factor_sub}), data=df)
summary(model)
library(effectsize)
eta_squared(model)
                        """

                    # 寄与率の計算
                    anova_res['寄与率'] = anova_res['sum_sq'] / anova_res['sum_sq'].sum()

                    st.dataframe(anova_res.style.format({
                        'sum_sq': '{:.2f}', 
                        'df': '{:.0f}', 
                        'F': '{:.3f}', 
                        'PR(>F)': '{:.4f}',
                        '寄与率': '{:.3f}'
                    }))

                    with st.expander("💻 この解析のコードを見る (Python / R)"):
                        sub_tab1, sub_tab2 = st.tabs(["🐍 Python", "🔵 R"])
                        with sub_tab1: st.code(py_anova_code.strip(), language="python")
                        with sub_tab2: st.code(r_anova_code.strip(), language="r")

                    st.divider()

                    # ==========================================
                    # 2. 主要因の多重比較
                    # ==========================================
                    st.subheader(f"2. {factor_x} ごとの多重比較 (Tukey HSD)")
                    
                    if factor_sub != "なし" and not np.isnan(interaction_p) and interaction_p < 0.05:
                        st.warning(f"⚠️ 交互作用（{idx_interaction}）が有意（p={interaction_p:.4f}）であるため、主効果の検定結果には注意が必要です。")

                    tukey_x = pairwise_tukeyhsd(endog=df_clean[target_col], groups=df_clean[factor_x], alpha=0.05)
                    tukey_summary_x = pd.DataFrame(data=tukey_x._results_table.data[1:], columns=tukey_x._results_table.data[0])
                    letters_df_x, groups_order_x = get_cld_letters(df_clean, target_col, factor_x, tukey_summary_x)
                    
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.dataframe(tukey_summary_x)
                    with col_b:
                        st.dataframe(letters_df_x.rename(columns={'groups_name': factor_x, 'letters': '有意差(abc)'}))

                    fig_x, ax_x = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=factor_x, y=target_col, data=df_clean, order=groups_order_x, ax=ax_x, color='#f0f0f0', showfliers=False)
                    sns.stripplot(x=factor_x, y=target_col, data=df_clean, order=groups_order_x, ax=ax_x, color='black', alpha=0.5)
                    
                    for group_name in groups_order_x:
                        letter = letters_df_x.loc[letters_df_x['groups_name'] == group_name, 'letters'].values[0]
                        x_pos = groups_order_x.index(group_name)
                        y_max = df_clean[df_clean[factor_x] == group_name][target_col].max()
                        ax_x.text(x_pos, y_max * 1.05, letter, ha='center', va='bottom', fontweight='bold', fontsize=16, color='#d62728')
                    
                    ax_x.set_ylabel(target_col, fontsize=12)
                    ax_x.set_xlabel(factor_x, fontsize=12)
                    st.pyplot(fig_x)

                    with st.expander("💻 この解析のコードを見る (Python / R)"):
                        sub_tab3, sub_tab4 = st.tabs(["🐍 Python", "🔵 R"])
                        with sub_tab3:
                            st.code(f"""
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog=df['{target_col}'], groups=df['{factor_x}'], alpha=0.05)
print(tukey)
                            """, language="python")
                        with sub_tab4:
                            st.code(f"""
library(multcompView)
model <- aov({target_col} ~ as.factor({factor_x}), data=df)
tukey <- TukeyHSD(model)
cld <- multcompLetters4(model, tukey)
print(cld)
                            """, language="r")

                    st.divider()

                    # ==========================================
                    # 3. 交互作用図
                    # ==========================================
                    if factor_sub != "なし":
                        st.subheader(f"3. {idx_interaction} の解析と交互作用図")
                        
                        if np.isnan(interaction_p):
                            st.warning("⚠️ データが足りず、交互作用を計算できませんでした。")
                        else:
                            if interaction_p < 0.05:
                                st.error(f"⚠️ 交互作用（{idx_interaction}）は有意（p={interaction_p:.4f} < 0.05）です。")
                            else:
                                st.success(f"✅ 交互作用（{idx_interaction}）は有意ではありません（p={interaction_p:.4f} > 0.05）。主効果を単独で解釈しても比較的安全です。")

                        col_int_a, col_int_b = st.columns([1, 2])
                        with col_int_a:
                            st.info("💡 **ワンポイント：交互作用図の読み方**\n\n- 折れ線が**平行**に近い場合、交互作用はありません。\n- 折れ線が**交差**していたり、**傾きが異なる**場合、交互作用が存在します（特定の組み合わせで効果が変わります）。")

                        with col_int_b:
                            fig_int, ax_int = plt.subplots(figsize=(10, 6))
                            sns.pointplot(x=factor_x, y=target_col, hue=factor_sub, data=df_clean, ax=ax_int, 
                                         dodge=True, capsize=.2, errwidth=1, markers=['o', 's', '^', 'D', 'v', '<', '>'], 
                                         palette='colorblind')
                            
                            ax_int.set_ylabel(target_col, fontsize=12)
                            ax_int.set_xlabel(factor_x, fontsize=12)
                            ax_int.set_title(f"交互作用図：{idx_interaction}", fontsize=14)
                            st.pyplot(fig_int)

                        st.divider()

                    # ==========================================
                    # 4. 副要因の多重比較
                    # ==========================================
                    if factor_sub != "なし":
                        st.subheader(f"4. {factor_sub} ごとの多重比較 (ANOVA視点：カテゴリとして比較)")
                        
                        tukey_sub = pairwise_tukeyhsd(endog=df_clean[target_col], groups=df_clean[factor_sub], alpha=0.05)
                        tukey_summary_sub = pd.DataFrame(data=tukey_sub._results_table.data[1:], columns=tukey_sub._results_table.data[0])
                        letters_df_sub, groups_order_sub = get_cld_letters(df_clean, target_col, factor_sub, tukey_summary_sub)
                        
                        col_c, col_d = st.columns([2, 1])
                        with col_c:
                            st.dataframe(tukey_summary_sub)
                        with col_d:
                            st.dataframe(letters_df_sub.rename(columns={'groups_name': factor_sub, 'letters': '有意差(abc)'}))

                        fig_sub, ax_sub = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x=factor_sub, y=target_col, data=df_clean, order=groups_order_sub, ax=ax_sub, color='#e6f3ff', showfliers=False)
                        sns.stripplot(x=factor_sub, y=target_col, data=df_clean, order=groups_order_sub, ax=ax_sub, color='black', alpha=0.5)
                        
                        for group_name in groups_order_sub:
                            letter = letters_df_sub.loc[letters_df_sub['groups_name'] == group_name, 'letters'].values[0]
                            x_pos = groups_order_sub.index(group_name)
                            y_max = df_clean[df_clean[factor_sub] == group_name][target_col].max()
                            ax_sub.text(x_pos, y_max * 1.05, letter, ha='center', va='bottom', fontweight='bold', fontsize=16, color='#d62728')
                        
                        ax_sub.set_ylabel(target_col, fontsize=12)
                        ax_sub.set_xlabel(factor_sub, fontsize=12)
                        st.pyplot(fig_sub)
                        st.divider()

                        # ==========================================
                        # 5. 副要因の単回帰分析
                        # ==========================================
                        st.subheader(f"5. {factor_sub} を量的変数とした単回帰分析 (回帰視点)")
                        st.info("💡 **ワンポイント：分散分析と回帰分析の関係**\n\n分散分析（ANOVA）は、内部的にはカテゴリをダミー変数（0と1）に変換した「回帰分析」として計算されています。どちらも「一般線形モデル（GLM）」という同じ数学的枠組みの仲間です。")

                        df_reg = df_clean.copy()
                        df_reg[factor_sub] = pd.to_numeric(df_reg[factor_sub], errors='coerce')

                        if df_reg[factor_sub].isna().all():
                            st.warning(f"『{factor_sub}』列が数値データではないため、単回帰分析はスキップしました。")
                        else:
                            df_reg = df_reg.dropna(subset=[factor_sub, target_col])
                            formula_reg = f'Q("{target_col}") ~ Q("{factor_sub}")'
                            model_reg = ols(formula_reg, data=df_reg).fit()
                            
                            r2 = model_reg.rsquared
                            p_val = model_reg.f_pvalue
                            intercept = model_reg.params['Intercept']
                            slope = model_reg.params[f'Q("{factor_sub}")']
                            
                            col_e, col_f = st.columns([1, 2])
                            with col_e:
                                st.write("▼ 単回帰モデルの要約")
                                st.write(f"- **決定係数 ($R^2$)**: {r2:.3f}")
                                st.write(f"- **モデルのP値**: {p_val:.3e}")
                                st.write(f"- **回帰式**: Y = {slope:.2f}X + {intercept:.2f}")
                                
                            with col_f:
                                fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
                                sns.regplot(x=factor_sub, y=target_col, data=df_reg, ax=ax_reg, 
                                            scatter_kws={'alpha':0.6, 'color':'black', 's': 50}, line_kws={'color':'#d62728'})
                                
                                textstr = f'$R^2 = {r2:.3f}$\n$y = {slope:.2f}x + {intercept:.2f}$'
                                props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
                                ax_reg.text(0.05, 0.95, textstr, transform=ax_reg.transAxes, fontsize=14,
                                            verticalalignment='top', bbox=props)
                                
                                ax_reg.set_ylabel(target_col, fontsize=12)
                                ax_reg.set_xlabel(f"{factor_sub} (量的変数)", fontsize=12)
                                st.pyplot(fig_reg)

            except Exception as e:
                st.error(f"解析中にエラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

# ==========================================
# タブ4: 統計用語の解説とコード
# ==========================================
with tab4:
    st.header("📖 押さえておきたい基礎知識")
    
    st.markdown("### 帰無仮説とは？")
    st.info("「証明したいこと（例：新しい肥料は効果がある）」の逆、「効果がない（差がない）」という仮説のこと。この仮説が**「確率的にあり得ない（珍しい）」**ことを示すことで、間接的に証明したいことを主張する背理法のアプローチです。")
    
    st.markdown("### p値（p-value）とは？")
    st.info("帰無仮説が正しい（本当に差がない）と仮定したとき、今回得られたデータ、あるいはそれ以上に極端なデータが「たまたま偶然」発生する確率のこと。農業分野では一般的に **5%（0.05）** を基準（有意水準）とし、これより小さければ「偶然とは考えにくい＝有意差あり」と判定します。")

    st.markdown("---")
    st.markdown("### 📊 箱ひげ図（Box plot）の見方")
    st.info("""
    本アプリのグラフでも使用されている「箱ひげ図」は、データの**ばらつき（分布）**を視覚的に把握するのに便利なグラフです。
    
    * **箱の真ん中の線**：**中央値（メジアン）**。データを小さい順に並べたときに、ちょうど真ん中にくる値です。（※平均値とは異なる場合があります）
    * **箱の上下の端**：データの**真ん中50%**がこの箱の中に収まっています。箱が縦に長いほど、データのばらつきが大きいことを意味します。
    * **上下に伸びる線（ひげ）**：通常データの範囲を示します。
    * **点（ドット）**：本アプリでは、横に並ぶ点は**実際のデータ1つ1つ（各株の収量など）**を表しています。ひげの外側にある点は「外れ値（極端な値）」とされることもあります。
    """)

    st.markdown("---")
    st.markdown("### 💡 2つの「t検定」の違い（スチューデント vs ウェルチ）")
    st.info("""
    統計の教科書やソフトには「t検定」が2種類存在します。本アプリでは実務の標準である**「ウェルチのt検定」**を採用しています。

    **1. スチューデントのt検定（Student's t-test）**
    * **弱点**: 比較するA群とB群の**「データのばらつき（分散）が等しい」という厳しい前提条件**があります。

    **2. ウェルチのt検定（Welch's t-test）**
    * **特徴**: **「データのばらつきが等しくても、異なっていても、どちらでも使える」**万能型のt検定です。
    * **実務でのポイント**: R言語の `t.test()` 関数なども、デフォルトでこのウェルチのt検定が選ばれるようになっています。
    """)
    
    st.markdown("---")
    st.markdown("### なぜ3群以上の比較でt検定を繰り返してはいけないのか？（検定の多重性）")
    st.error("A, B, Cの3品種があるとき、t検定を「A-B」「B-C」「A-C」と3回繰り返すと、「1回の検定で5%の確率で間違った結論を出すリスク」が雪だるま式に増え、**全体のミスの確率が約15%まで上昇**してしまいます。これを防ぐために、まずは全体を評価する**分散分析（F検定）**を使用します。")

    st.markdown("### 🔍 Tukey（テューキー）検定：全当たりの比較")
    st.success("""
    分散分析で「有意差あり」となった後、**具体的にどのグループ間に差があるか**をすべての組み合わせで調べる手法です。
    最大のメリットは、比較する数が増えても**全体のミス（第一種の過誤）の確率を5%以内にピタリと抑えてくれる**点です。
    """)

    st.markdown("---")
    st.markdown("### 💻 【参考】R言語での実行スクリプト")
    st.info("実際の研究データ解析で広く使われる「R」でのコード例です。アプリで学んだ理論を、実際のデータ解析に当てはめる際のリファレンスとして活用してください。")
    
    st.code("""
# 1. t検定 (ウェルチのt検定: var.equal = FALSE)
t_result <- t.test(Yield ~ Variety, data = data_t, var.equal = FALSE)
print(t_result)

# 2. 分散分析 (F検定)
data_f$Treatment <- as.factor(data_f$Treatment)
anova_model <- aov(Yield ~ Treatment, data = data_f)
summary(anova_model)

# 3. Tukey（テューキー）検定 (多重比較)
tukey_result <- TukeyHSD(anova_model, conf.level = 0.95)
print(tukey_result)
plot(tukey_result)
    """, language="r")
