import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --- 共通関数：有意差のabc（CLD）を生成 ---
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
    return pd.DataFrame({'groups_name': list(cld.keys()), 'letters': ["".join(l) for l in cld.values()]}), groups

# --- ページ設定 ---
st.set_page_config(page_title="農業統計解析ツール", layout="wide")
st.title("📊 農業データ・自動統計解析システム")

# --- 1. データ読み込み ---
data_source = st.sidebar.radio("データの入力方法：", ["🥔 サンプルデータ", "📁 CSVアップロード"])

if data_source == "🥔 サンプルデータ":
    df = pd.DataFrame({
        '品種': ['品種A', '品種A', '品種B', '品種B', '品種C', '品種C'] * 3,
        '処理': ['標準', '多量'] * 9,
        '収量': [210, 250, 180, 200, 150, 170, 215, 255, 185, 205, 155, 175, 205, 245, 175, 195, 145, 165],
        '発芽数': [88, 95, 45, 60, 15, 30, 85, 92, 42, 58, 12, 28, 90, 96, 40, 62, 18, 35],
        '全数': [100] * 18,
        '腐敗数': [2, 1, 8, 12, 25, 30, 3, 2, 7, 10, 22, 28, 1, 3, 9, 11, 24, 32]
    })
else:
    uploaded_file = st.sidebar.file_uploader("CSVファイルを選択", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='shift_jis')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')
    else:
        df = None

if df is not None:
    # --- 2. 解析モードの選択 ---
    st.sidebar.divider()
    mode = st.sidebar.selectbox(
        "解析モードの選択",
        ["📦 収量 (正規分布/ANOVA)", "🌱 発芽率 (二項分布/GLM)", "🐛 腐敗数 (ポアソン分布/GLM)"]
    )

    st.header(f"🔍 設定：{mode}")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "収量" in mode:
            target_col = st.selectbox("解析する数値（収量など）", df.columns)
        elif "発芽率" in mode:
            sp_col = st.selectbox("カウント数（発芽数など）", df.columns)
            tt_col = st.selectbox("全数（調査数）", df.columns)
        else:
            target_col = st.selectbox("カウント数（腐敗数・虫数など）", df.columns)

    with col2:
        fx = st.selectbox("主要因（品種など）", df.columns)
    with col3:
        fs = st.selectbox("副要因（処理・かん水など）", df.columns)

    if st.button("🚀 解析を実行"):
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        
        try:
            # --- 解析ロジック ---
            if "収量" in mode:
                # ANOVA
                formula = f'Q("{target_col}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                model = smf.ols(formula, data=df).fit()
                st.subheader("1. 分散分析表 (ANOVA)")
                st.table(sm.stats.anova_lm(model, typ=2))
                plot_target = target_col
            
            elif "発芽率" in mode:
                # GLM (Binomial)
                df['not_sp'] = df[tt_col] - df[sp_col]
                df['ratio'] = df[sp_col] / df[tt_col]
                formula = f'Q("{sp_col}") + not_sp ~ C(Q("{fx}")) + C(Q("{fs}"))'
                model = smf.glm(formula, data=df, family=sm.families.Binomial()).fit()
                st.subheader("1. 偏差分析 (カイ二乗検定)")
                w_res = model.wald_test_terms().summary_frame()
                st.table(w_res.apply(lambda c: c.map(lambda x: np.asarray(x).item() if hasattr(x, "__len__") else x)))
                plot_target = 'ratio'
            
            else:
                # GLM (Poisson)
                formula = f'Q("{target_col}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                st.subheader("1. 偏差分析 (カイ二乗検定)")
                w_res = model.wald_test_terms().summary_frame()
                st.table(w_res.apply(lambda c: c.map(lambda x: np.asarray(x).item() if hasattr(x, "__len__") else x)))
                plot_target = target_col

            # --- Tukey多重比較 ---
            st.subheader(f"2. {fx} の多重比較 (Tukey)")
            tukey = pairwise_tukeyhsd(df[plot_target], df[fx])
            tk_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            let_df, grp_ord = get_cld_letters(df, plot_target, fx, tk_df)
            
            c_left, c_right = st.columns([2, 1])
            c_left.dataframe(tk_df)
            c_right.dataframe(let_df.rename(columns={'groups_name': fx}))

            # --- グラフ ---
            fig, ax = plt.subplots(figsize=(8, 5))
            if "収量" in mode:
                sns.boxplot(x=fx, y=plot_target, hue=fs, data=df, ax=ax)
            else:
                sns.barplot(x=fx, y=plot_target, hue=fs, data=df, ax=ax, capsize=.1)
            
            # abcラベル
            for i, g in enumerate(grp_ord):
                l = let_df.loc[let_df['groups_name'] == g, 'letters'].values[0]
                y_pos = df[df[fx] == g][plot_target].max()
                ax.text(i, y_pos * 1.05 if "収量" in mode else y_pos + 0.05, l, ha='center', color='red', weight='bold', size=14)
            
            st.pyplot(fig)

        except Exception as e:
            st.error(f"解析エラーが発生しました。設定を見直してください： {e}")
