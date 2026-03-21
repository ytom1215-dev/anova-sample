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
    
    means = df_clean.groupby(group)[target].mean().sort_values(ascending=False)
    groups = means.index.tolist()
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
    return pd.DataFrame({'groups_name': list(cld.keys()), 'letters': ["".join(l) for l in cld.values()]}), groups

# --- ページ設定 ---
st.set_page_config(page_title="農業統計解析ツール", layout="wide")
st.title("📊 農業データ・自動統計解析システム (研修対応版)")

# --- 🎓 新人研修用：統計用語ガイド（サイドバー） ---
with st.sidebar.expander("🎓 新人研修向け：統計用語ガイド"):
    st.markdown("""
    **💡 P値 (P-value) とは？**
    「この結果が偶然起きた確率」です。**0.05 (5%) 未満**なら「意味のある差（有意差）がある」と判断します。
    
    **💡 アルファベット (a, b, c) の意味**
    グラフの上の文字は「多重比較」の結果です。**違う文字**がついているグループ同士には有意差があります。
    
    **💡 どの分布を選ぶべき？**
    * **正規分布:** 収量(kg)、草丈(cm)など「連続した数字」
    * **二項分布:** 発芽率、病害発生率など「全体の何%か」
    * **ポアソン分布:** 虫害数、腐敗数など「数えられるもの（0以上の整数）」
    * **ノンパラ (順序):** 発生程度(1〜4)など「ランク分けされたもの」
    """)

# --- 1. サイドバー：解析の目的とモード選択 ---
st.sidebar.header("⚙️ 1. 解析の目的")
analysis_purpose = st.sidebar.radio(
    "どちらの解析を行いますか？",
    ["📊 要因解析 (カテゴリ比較・多重比較)", "📈 回帰解析 (数値からの予測・回帰曲線)"],
    help="グループ間の違いを見たいなら「要因解析」、温度や量から結果を予測したいなら「回帰解析」を選びます。"
)

st.sidebar.header("⚙️ 2. データ分布の選択")
mode = st.sidebar.selectbox(
    "データの種類", 
    ["📦 収量など (正規分布/OLS)", 
     "🌱 発芽率など (二項分布/Logit)", 
     "🐛 虫害・腐敗数など (ポアソン/Log)",
     "🗂️ 発生程度1-4など順序データ (ノンパラ/Kruskal-Wallis)"]
)

st.sidebar.header("⚙️ 3. データの入力")
data_source = st.sidebar.radio("入力方法：", ["🧪 サンプルデータで試す", "📁 CSVアップロード"])

# --- 2. サンプルデータの自動生成 ---
if data_source == "🧪 サンプルデータで試す":
    np.random.seed(42)
    if "要因解析" in analysis_purpose:
        n_grp = 15
        df = pd.DataFrame({
            '品種': ['品種A']*n_grp + ['品種B']*n_grp + ['品種C']*n_grp,
            '温度': ['10度']*15 + ['20度']*15 + ['30度']*15,
            '処理': ['標準', '多量', '少量'] * 15,
            '収量': np.concatenate([np.random.normal(280, 10, n_grp), np.random.normal(230, 10, n_grp), np.random.normal(180, 10, n_grp)]).astype(int),
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
        n = 30
        x = np.linspace(10, 40, n)
        y_ols = 4.5 * x + 100 + np.random.normal(0, 8, n)
        p = 1 / (1 + np.exp(-0.3 * (x - 25)))
        sp_counts = np.random.binomial(100, p)
        y_poi = np.random.poisson(np.exp(0.09 * x))
        df = pd.DataFrame({'温度': x, '品種': ['品種A']*15 + ['品種B']*15, '収量': y_ols, '全数': 100, '発芽数': sp_counts, '腐敗数': y_poi})
    st.success(f"✅ 研修用のサンプルデータ（{mode.split()[1]}用）を読み込みました。")
else:
    uploaded_file = st.sidebar.file_uploader("CSV選択", type="csv")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

# --- 3. メイン解析UI ---
if df is not None:
    st.header(f"🔍 設定：{analysis_purpose.split()[1]} × {mode.split()[1]}")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "収量" in mode: 
            target = st.selectbox("目的変数 (Y軸)", df.columns, index=df.columns.get_loc('収量') if '収量' in df.columns else len(df.columns)-1)
        elif "発芽率" in mode:
            sp_col = st.selectbox("カウント数 (分子)", df.columns, index=df.columns.get_loc('発芽数') if '発芽数' in df.columns else 0)
            tt_col = st.selectbox("全数 (分母)", df.columns, index=df.columns.get_loc('全数') if '全数' in df.columns else 0)
            df['ratio'] = df[sp_col] / df[tt_col]
            target = 'ratio'
        elif "順序データ" in mode: 
            target = st.selectbox("順序データ (1,2,3...)", df.columns, index=df.columns.get_loc('発生程度') if '発生程度' in df.columns else len(df.columns)-1)
        else: 
            target = st.selectbox("カウント数 (Y軸)", df.columns, index=df.columns.get_loc('腐敗数') if '腐敗数' in df.columns else len(df.columns)-1)

    with col2:
        if "要因解析" in analysis_purpose:
            fx = st.selectbox("主要因 (比較カテゴリ)", df.columns, index=df.columns.get_loc('温度') if '温度' in df.columns else 0)
        else:
            fx = st.selectbox("主要因 (X軸の数値)", df.columns, index=0)
            
    with col3:
        if "要因解析" in analysis_purpose:
            if "順序データ" in mode:
                st.info("※ノンパラ検定では副要因による色分けは行いません。")
                fs = None
            else:
                fs = st.selectbox("副要因 (色分け/処理など)", df.columns, index=1 if len(df.columns)>1 else 0)
        else:
            st.info("※回帰解析モードでは全体の傾向を算出します。")
            fs = None

    if st.button("🚀 解析を実行"):
        use_cols = [target, fx] if fs is None else [target, fx, fs]
        if "発芽率" in mode: use_cols.extend([sp_col, tt_col])
        df_clean = df.dropna(subset=[c for c in use_cols if c in df.columns]).copy()

        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        fig, ax = plt.subplots(figsize=(10, 6))
        
        code_snippet = ""
        
        try:
            # ==========================================
            # A: 要因解析モード
            # ==========================================
            if "要因解析" in analysis_purpose:
                df_clean[fx] = df_clean[fx].astype(str)
                formula_str = f'Q("{target}") ~ C(Q("{fx}")) + C(Q("{fs}"))' if fs else f'Q("{target}") ~ C(Q("{fx}"))'
                
                if "順序データ" in mode:
                    st.subheader("1. Kruskal-Wallis 検定 (全体に差があるか)")
                    groups_data = [df_clean[df_clean[fx] == g][target] for g in df_clean[fx].unique()]
                    stat, p_val = stats.kruskal(*groups_data)
                    st.info(f"📝 統計量 (H) = {stat:.3f}, **P値 = {p_val:.4e}**")

                    st.subheader(f"2. {fx} の多重比較 (Steel-Dwass検定)")
                    res_matrix = sp.posthoc_dscf(df_clean, val_col=target, group_col=fx)
                    
                    groups = df_clean[fx].unique()
                    pairs = list(itertools.combinations(groups, 2))
                    res = []
                    for g1, g2 in pairs:
                        p_adj = res_matrix.loc[g1, g2]
                        res.append({'group1': g1, 'group2': g2, 'pval_adj': p_adj})
                    
                    res_df = pd.DataFrame(res)
                    res_df['reject'] = res_df['pval_adj'] < 0.05
                    let_df, grp_ord = get_cld_letters(df_clean, target, fx, res_df)
                    c_l, c_r = st.columns([2, 1])
                    c_l.dataframe(res_df.style.format({'pval_adj': '{:.4f}'}))
                    c_r.dataframe(let_df.rename(columns={'groups_name': fx}))
                    
                    sns.boxplot(x=fx, y=target, data=df_clean, ax=ax, palette="Set2", order=grp_ord, showfliers=False)
                    sns.stripplot(x=fx, y=target, data=df_clean, ax=ax, color=".3", alpha=0.6, jitter=True, order=grp_ord)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    
                    code_snippet = f"""import pandas as pd\nimport scipy.stats as stats\nimport scikit_posthocs as sp\n\ndf = pd.read_csv("your_data.csv")\ngroups_data = [df[df['{fx}'] == g]['{target}'] for g in df['{fx}'].unique()]\nstat, p_val = stats.kruskal(*groups_data)\nprint(f"Kruskal-Wallis P値: {{p_val}}")\nres_matrix = sp.posthoc_dscf(df, val_col='{target}', group_col='{fx}')\nprint(res_matrix)"""

                else:
                    if "収量" in mode:
                        model = smf.ols(formula_str, data=df_clean).fit()
                        st.subheader("1. 分散分析表 (ANOVA)")
                        st.table(sm.stats.anova_lm(model, typ=2))
                        code_snippet = f"""import pandas as pd\nimport statsmodels.api as sm\nimport statsmodels.formula.api as smf\nfrom statsmodels.stats.multicomp import pairwise_tukeyhsd\n\ndf = pd.read_csv("your_data.csv")\nmodel = smf.ols('{formula_str.replace('Q("', '').replace('")', '')}', data=df).fit()\nprint(sm.stats.anova_lm(model, typ=2))\ntukey = pairwise_tukeyhsd(df['{target}'], df['{fx}'])\nprint(tukey.summary())"""
                    else:
                        if "発芽率" in mode:
                            df_clean['not_sp'] = df_clean[tt_col] - df_clean[sp_col]
                            formula = f'Q("{sp_col}") + not_sp ~ C(Q("{fx}")) + C(Q("{fs}"))' if fs else f'Q("{sp_col}") + not_sp ~ C(Q("{fx}"))'
                            family = sm.families.Binomial()
                            family_str = "sm.families.Binomial()"
                            code_formula = f"'{sp_col} + not_sp ~ {fx} + {fs}'" if fs else f"'{sp_col} + not_sp ~ {fx}'"
                        else:
                            formula = formula_str
                            family = sm.families.Poisson()
                            family_str = "sm.families.Poisson()"
                            code_formula = f"'{target} ~ {fx} + {fs}'" if fs else f"'{target} ~ {fx}'"
                            
                        model = smf.glm(formula, data=df_clean, family=family).fit()
                        st.subheader("1. 偏差分析 (カイ二乗検定)")
                        w_res = model.wald_test_terms().summary_frame()
                        st.table(w_res.apply(lambda c: c.map(lambda x: np.asarray(x).item() if hasattr(x, "__len__") else x)))
                        code_snippet = f"""import pandas as pd\nimport statsmodels.api as sm\nimport statsmodels.formula.api as smf\nfrom statsmodels.stats.multicomp import pairwise_tukeyhsd\n\ndf = pd.read_csv("your_data.csv")\n{'df["not_sp"] = df["' + tt_col + '"] - df["' + sp_col + '"]' if "発芽率" in mode else ''}\nmodel = smf.glm({code_formula}, data=df, family={family_str}).fit()\nprint(model.wald_test_terms().summary_frame())\ntukey = pairwise_tukeyhsd(df['{target}'], df['{fx}'])\nprint(tukey.summary())"""

                    st.subheader(f"2. {fx} の多重比較 (Tukey HSD)")
                    tukey = pairwise_tukeyhsd(df_clean[target], df_clean[fx])
                    tk_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                    let_df, grp_ord = get_cld_letters(df_clean, target, fx, tk_df)
                    
                    c_l, c_r = st.columns([2, 1])
                    c_l.dataframe(tk_df); c_r.dataframe(let_df.rename(columns={'groups_name': fx}))
                    
                    if "収量" in mode: sns.boxplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, palette="Set2", order=grp_ord)
                    else: sns.barplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, capsize=.1, palette="viridis", order=grp_ord)
                
                # 📊 グラフの仕上げ (要因解析時のみアルファベットを付与)
                for i, g in enumerate(grp_ord):
                    l = let_df.loc[let_df['groups_name'] == g, 'letters'].values[0]
                    y_val = df_clean[df_clean[fx] == g][target].max()
                    y_offset = df_clean[target].max() * 0.05 if df_clean[target].max() > 0 else 0.05
                    ax.text(i, y_val + y_offset, l, ha='center', color='red', weight='bold', size=15)

            # ==========================================
            # B: 回帰解析モード
            # ==========================================
            else:
                is_num = np.issubdtype(df_clean[fx].dtype, np.number)
                if not is_num:
                    st.warning(f"⚠️ {fx} は数値データではないため、回帰曲線は描画できません。数値の列を選択してください。")
                    st.stop()
                    
                if "収量" in mode:
                    model = smf.ols(f'Q("{target}") ~ Q("{fx}")', data=df_clean).fit()
                    st.subheader("1. 予測モデル (直線回帰)")
                    sns.regplot(x=fx, y=target, data=df_clean, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    code_snippet = f"""import pandas as pd\nimport statsmodels.formula.api as smf\n\ndf = pd.read_csv("your_data.csv")\nmodel = smf.ols('{target} ~ {fx}', data=df).fit()\nprint(model.summary())"""
                
                elif "発芽率" in mode:
                    df_clean['not_sp'] = df_clean[tt_col] - df_clean[sp_col]
                    model = smf.glm(f'Q("{sp_col}") + not_sp ~ Q("{fx}")', data=df_clean, family=sm.families.Binomial()).fit()
                    st.subheader("1. 予測モデル (ロジスティック回帰)")
                    x_range = np.linspace(df_clean[fx].min(), df_clean[fx].max(), 100)
                    
                    # predictを使って安全に曲線を生成
                    pred_df = pd.DataFrame({fx: x_range})
                    y_pred = model.predict(pred_df)
                    
                    ax.plot(x_range, y_pred, color='red', lw=3)
                    ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)
                    code_snippet = f"""import pandas as pd\nimport statsmodels.api as sm\nimport statsmodels.formula.api as smf\n\ndf = pd.read_csv("your_data.csv")\ndf['not_sp'] = df['{tt_col}'] - df['{sp_col}']\nmodel = smf.glm('{sp_col} + not_sp ~ {fx}', data=df, family=sm.families.Binomial()).fit()\nprint(model.summary())"""
                
                else:
                    model = smf.glm(f'Q("{target}") ~ Q("{fx}")', data=df_clean, family=sm.families.Poisson()).fit()
                    st.subheader("1. 予測モデル (ポアソン回帰)")
                    x_range = np.linspace(df_clean[fx].min(), df_clean[fx].max(), 100)
                    
                    # predictを使って安全に曲線を生成
                    pred_df = pd.DataFrame({fx: x_range})
                    y_pred = model.predict(pred_df)
                    
                    ax.plot(x_range, y_pred, color='red', lw=3)
                    ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)
                    code_snippet = f"""import pandas as pd\nimport statsmodels.api as sm\nimport statsmodels.formula.api as smf\n\ndf = pd.read_csv("your_data.csv")\nmodel = smf.glm('{target} ~ {fx}', data=df, family=sm.families.Poisson()).fit()\nprint(model.summary())"""

                st.write("### 2. モデルの詳細要約")
                with st.expander("詳細を表示"):
                    st.text(model.summary())
            
            # --- グラフの描画 ---
            st.pyplot(fig)

            # ==========================================
            # 👩‍💻 中級者向けスクリプト表示エリア
            # ==========================================
            st.divider()
            with st.expander("👩‍💻 中級者向け：この解析のPythonスクリプトを確認する"):
                st.markdown("このアプリの裏側で実行されている、Pythonライブラリを用いた統計解析のコードです。")
                st.code(code_snippet, language="python")

        except Exception as e:
            st.error(f"解析エラー: {e}")
