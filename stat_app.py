import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import itertools
import scikit_posthocs as sp

# --- 共通関数：有意差のabc（CLD）を生成 ---
def get_cld_letters(df, target, group, tukey_summary):
    df_clean = df.copy()
    df_clean[group] = df_clean[group].astype(str)
    
    # 平均値の降順でグループを並べる
    means = df_clean.groupby(group)[target].mean().sort_values(ascending=False)
    groups = means.index.tolist()
    
    # 隣接行列の作成（有意差がない場合に1）
    n = len(groups)
    adj = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            g1, g2 = groups[i], groups[j]
            mask = ((tukey_summary['group1'].astype(str) == g1) & (tukey_summary['group2'].astype(str) == g2)) | \
                   ((tukey_summary['group1'].astype(str) == g2) & (tukey_summary['group2'].astype(str) == g1))
            reject = tukey_summary.loc[mask, 'reject'].values
            if len(reject) > 0 and reject[0]:
                adj[i, j] = adj[j, i] = 0
                
    # 簡略化されたCLDアルゴリズム
    letters = {g: [] for g in groups}
    found_sets = []
    for i in range(n):
        current_set = {i}
        for j in range(i + 1, n):
            if all(adj[j, k] == 1 for k in current_set):
                current_set.add(j)
        
        is_subset = False
        for s in found_sets:
            if current_set.issubset(s):
                is_subset = True
                break
        if not is_subset:
            found_sets = [s for s in found_sets if not s.issubset(current_set)]
            found_sets.append(current_set)
            
    for i, s in enumerate(sorted(found_sets, key=lambda x: min(x))):
        char = chr(ord('a') + i)
        for idx in s:
            letters[groups[idx]].append(char)
            
    return pd.DataFrame({
        'groups_name': list(letters.keys()), 
        'letters': ["".join(l) for l in letters.values()]
    }), groups

# --- ページ設定 ---
st.set_page_config(page_title="農業統計解析ツール", layout="wide")
st.title("📊 農業データ・自動統計解析システム (修正版)")

# --- 🎓 統計用語ガイド（サイドバー） ---
with st.sidebar.expander("🎓 統計用語ガイド"):
    st.markdown("""
    **💡 P値 (P-value)**
    「偶然起きた確率」。**0.05未満**で有意差あり。
    
    **💡 アルファベット (a, b, c)**
    多重比較の結果。**文字が重ならない**グループ間に有意差があります。
    
    **💡 分布の選び方**
    * **正規分布:** 連続値（収量、草丈）
    * **二項分布:** 割合（発芽率、生存率）
    * **ポアソン分布:** 個数（虫数、病斑数）
    * **ノンパラ:** ランク（発生程度1〜4）
    """)

# --- 1. サイドバー：設定 ---
st.sidebar.header("⚙️ 1. 解析の設定")
analysis_purpose = st.sidebar.radio(
    "解析目的",
    ["📊 要件解析 (比較)", "📈 回帰解析 (予測)"]
)

mode = st.sidebar.selectbox(
    "データの種類", 
    ["📦 正規分布 (収量など)", 
     "🌱 二項分布 (発芽率など)", 
     "🐛 ポアソン分布 (虫数など)",
     "🗂️ 順序データ (発生程度など)"]
)

include_interaction = st.sidebar.checkbox("交互作用を含める", value=False, help="要因Aと要因Bの組み合わせによる相乗効果を解析します。")

st.sidebar.header("⚙️ 2. データの入力")
data_source = st.sidebar.radio("入力方法", ["🧪 サンプルデータ", "📁 CSVアップロード"])

# --- 2. データの読み込み (キャッシュ利用) ---
@st.cache_data
def get_sample_data(purpose, mode_str):
    np.random.seed(42)
    if "要件解析" in purpose:
        n_grp = 15
        df = pd.DataFrame({
            '品種': ['品種A']*n_grp + ['品種B']*n_grp + ['品種C']*n_grp,
            '温度': ['10度']*15 + ['20度']*15 + ['30度']*15,
            '処理': ['標準', '多量', '少量'] * 15,
            '収量': np.concatenate([np.random.normal(280, 15, n_grp), np.random.normal(230, 15, n_grp), np.random.normal(180, 15, n_grp)]).astype(int),
            '全数': [100] * 45,
            '発芽数': np.concatenate([np.random.binomial(100, 0.90, n_grp), np.random.binomial(100, 0.50, n_grp), np.random.binomial(100, 0.15, n_grp)]),
            '腐敗数': np.concatenate([np.random.poisson(2, n_grp), np.random.poisson(10, n_grp), np.random.poisson(25, n_grp)]),
            '発生程度': np.concatenate([
                np.random.choice([1, 2], n_grp, p=[0.8, 0.2]),       
                np.random.choice([1, 2, 3], n_grp, p=[0.2, 0.6, 0.2]), 
                np.random.choice([3, 4], n_grp, p=[0.4, 0.6])        
            ])
        })
    else:
        n = 45
        x = np.linspace(10, 40, n)
        y_ols = 4.5 * x + 100 + np.random.normal(0, 15, n)
        p = 1 / (1 + np.exp(-0.3 * (x - 25)))
        sp_data = np.random.binomial(100, p)
        y_poi = np.random.poisson(np.exp(0.09 * x))
        df = pd.DataFrame({'温度': x, '品種': ['品種A']*15 + ['品種B']*15 + ['品種C']*15, '収量': y_ols, '全数': 100, '発芽数': sp_data, '腐敗数': y_poi})
    return df

if data_source == "🧪 サンプルデータ":
    df = get_sample_data(analysis_purpose, mode)
    st.success("✅ サンプルデータを読み込みました。")
else:
    uploaded_file = st.sidebar.file_uploader("CSVファイルを選択", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='shift-jis')
    else:
        df = None

# --- 3. メイン解析 ---
if df is not None:
    st.header("🔍 解析設定")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "正規分布" in mode: target = st.selectbox("目的変数 (Y)", df.columns, index=df.columns.get_loc('収量') if '収量' in df.columns else 0)
        elif "二項分布" in mode:
            sp_col = st.selectbox("成功数 (分子)", df.columns, index=df.columns.get_loc('発芽数') if '発芽数' in df.columns else 0)
            tt_col = st.selectbox("全試行数 (分母)", df.columns, index=df.columns.get_loc('全数') if '全数' in df.columns else 0)
            df['ratio'] = df[sp_col] / df[tt_col]
            target = 'ratio'
        elif "順序データ" in mode: target = st.selectbox("順序データ", df.columns, index=df.columns.get_loc('発生程度') if '発生程度' in df.columns else 0)
        else: target = st.selectbox("カウント数 (Y)", df.columns, index=df.columns.get_loc('腐敗数') if '腐敗数' in df.columns else 0)

    with col2:
        fx = st.selectbox("主要因 (X)", df.columns, index=df.columns.get_loc('温度') if '温度' in df.columns else 0)

    with col3:
        if "要件解析" in analysis_purpose and "順序データ" not in mode:
            fs = st.selectbox("副要因 (色分け)", ["なし"] + list(df.columns), index=0)
            fs = None if fs == "なし" else fs
        else:
            fs = None
            st.info("※このモードでは副要因は使用しません。")

    if st.button("🚀 解析実行"):
        cols_to_clean = [target, fx]
        if fs: cols_to_clean.append(fs)
        if "二項分布" in mode: cols_to_clean.extend([sp_col, tt_col])
        df_clean = df.dropna(subset=[c for c in cols_to_clean if c in df.columns]).copy()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        try:
            if "要件解析" in analysis_purpose:
                df_clean[fx] = df_clean[fx].astype(str)
                
                if "順序データ" in mode:
                    st.subheader("1. Kruskal-Wallis 検定")
                    groups_data = [df_clean[df_clean[fx] == g][target] for g in df_clean[fx].unique()]
                    stat, p_val = stats.kruskal(*groups_data)
                    st.write(f"統計量: {stat:.3f}, **P値: {p_val:.4e}**")

                    st.subheader("2. 多重比較 (Steel-Dwass)")
                    res_matrix = sp.posthoc_dscf(df_clean, val_col=target, group_col=fx)
                    groups = df_clean[fx].unique()
                    res = []
                    for g1, g2 in itertools.combinations(groups, 2):
                        res.append({'group1': g1, 'group2': g2, 'pval_adj': res_matrix.loc[g1, g2], 'reject': res_matrix.loc[g1, g2] < 0.05})
                    res_df = pd.DataFrame(res)
                    
                    let_df, grp_ord = get_cld_letters(df_clean, target, fx, res_df)
                    st.dataframe(res_df.style.format({'pval_adj': '{:.4f}'}))
                    
                    sns.boxplot(x=fx, y=target, data=df_clean, ax=ax, order=grp_ord)
                    sns.stripplot(x=fx, y=target, data=df_clean, ax=ax, color=".3", alpha=0.5, order=grp_ord)
                    
                else:
                    if fs:
                        formula = f'Q("{target}") ~ C(Q("{fx}")) * C(Q("{fs}"))' if include_interaction else f'Q("{target}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                    else:
                        formula = f'Q("{target}") ~ C(Q("{fx}"))'
                    
                    if "正規分布" in mode:
                        model = smf.ols(formula, data=df_clean).fit()
                        st.subheader("1. 分散分析表 (ANOVA)")
                        st.table(sm.stats.anova_lm(model, typ=2))
                    else:
                        if "二項分布" in mode:
                            df_clean['not_sp'] = df_clean[tt_col] - df_clean[sp_col]
                            formula = formula.replace(f'Q("{target}")', f'Q("{sp_col}") + not_sp')
                            family = sm.families.Binomial()
                        else:
                            family = sm.families.Poisson()
                        
                        model = smf.glm(formula, data=df_clean, family=family).fit()
                        st.subheader("1. 偏差分析 (Wald検定)")
                        st.table(model.wald_test_terms().summary_frame())

                    st.subheader("2. 多重比較 (Tukey HSD)")
                    tukey = pairwise_tukeyhsd(df_clean[target], df_clean[fx])
                    tk_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                    let_df, grp_ord = get_cld_letters(df_clean, target, fx, tk_df)
                    st.dataframe(tk_df)
                    
                    if fs:
                        sns.boxplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, order=grp_ord)
                    else:
                        sns.barplot(x=fx, y=target, data=df_clean, ax=ax, order=grp_ord, capsize=.1)

                for i, g in enumerate(grp_ord):
                    label = let_df.loc[let_df['groups_name'] == g, 'letters'].values[0]
                    y_max = df_clean[df_clean[fx] == g][target].max()
                    ax.text(i, y_max * 1.05, label, ha='center', color='red', weight='bold', size=14)

            else:
                if not np.issubdtype(df_clean[fx].dtype, np.number):
                    st.error("❌ 主要因(X)には数値列を選択してください。")
                    st.stop()
                
                x_range = np.linspace(df_clean[fx].min(), df_clean[fx].max(), 100)
                if "正規分布" in mode:
                    model = smf.ols(f'Q("{target}") ~ Q("{fx}")', data=df_clean).fit()
                    sns.regplot(x=fx, y=target, data=df_clean, ax=ax, line_kws={'color':'red'})
                elif "二項分布" in mode:
                    df_clean['not_sp'] = df_clean[tt_col] - df_clean[sp_col]
                    model = smf.glm(f'Q("{sp_col}") + not_sp ~ Q("{fx}")', data=df_clean, family=sm.families.Binomial()).fit()
                    y_pred = 1 / (1 + np.exp(-(model.params[1] * x_range + model.params[0])))
                    ax.plot(x_range, y_pred, color='red', lw=2)
                    ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)
                else:
                    model = smf.glm(f'Q("{target}") ~ Q("{fx}")', data=df_clean, family=sm.families.Poisson()).fit()
                    y_pred = np.exp(model.params[1] * x_range + model.params[0])
                    ax.plot(x_range, y_pred, color='red', lw=2)
                    ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)
                
                st.subheader("1. モデル要約")
                st.text(model.summary())

            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"解析中にエラーが発生しました: {e}")
            st.exception(e)
