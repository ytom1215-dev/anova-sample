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

# --- 1. モード選択とデータ生成 ---
st.sidebar.header("⚙️ 解析設定")
mode = st.sidebar.selectbox("解析モード", ["📦 収量解析 (正規/OLS)", "🌱 発芽率解析 (二項/Logit)", "🐛 腐敗・虫害数解析 (ポアソン/Log)"])
data_source = st.sidebar.radio("データ：", ["🧪 数値回帰用サンプル", "📁 CSVアップロード"])

if data_source == "🧪 数値回帰用サンプル":
    n = 30
    x = np.linspace(10, 40, n) # 温度や日数などの数値要因
    if "収量解析" in mode:
        y = 3.17 * x + 217.5 + np.random.normal(0, 10, n)
        df = pd.DataFrame({'温度': x, '品種': ['品種A']*15 + ['品種B']*15, '収量': y})
    elif "発芽率" in mode:
        # S字カーブ（ロジスティック曲線）
        z = 0.2 * (x - 25)
        p = 1 / (1 + np.exp(-z))
        sp = np.random.binomial(100, p)
        df = pd.DataFrame({'温度': x, '品種': ['品種A']*15 + ['品種B']*15, '全数': 100, '発芽数': sp})
    else:
        # 指数関数的増加（ポアソン）
        y_poi = np.random.poisson(np.exp(0.08 * x))
        df = pd.DataFrame({'保存日数': x, '品種': ['品種A']*15 + ['品種B']*15, '腐敗数': y_poi})
    st.success("✅ 回帰分析に適した数値を読み込みました。")
else:
    uploaded_file = st.sidebar.file_uploader("CSV選択", type="csv")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

# --- 2. 解析実行 ---
if df is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        if "収量解析" in mode: target = st.selectbox("目的変数", df.columns, index=len(df.columns)-1)
        elif "発芽率" in mode:
            sp_col = st.selectbox("発芽数", df.columns, index=len(df.columns)-1)
            tt_col = st.selectbox("全数", df.columns, index=len(df.columns)-2)
            df['ratio'] = df[sp_col] / df[tt_col]
            target = 'ratio'
        else: target = st.selectbox("カウント数", df.columns, index=len(df.columns)-1)
    with col2: fx = st.selectbox("主要因 (X軸/数値推奨)", df.columns, index=0)
    with col3: fs = st.selectbox("副要因 (色分け)", df.columns, index=1)

    if st.button("🚀 回帰解析を実行"):
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            # --- 統計モデルと回帰式の算出 ---
            is_numeric = np.issubdtype(df[fx].dtype, np.number)
            
            if "収量解析" in mode:
                model = smf.ols(f'Q("{target}") ~ Q("{fx}")', data=df).fit()
                st.info(f"📝 直線回帰式: **Y = {model.params[1]:.3f}X + {model.params[0]:.3f}**")
                if is_numeric: sns.regplot(x=fx, y=target, data=df, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            
            elif "発芽率" in mode:
                df['not_sp'] = df[tt_col] - df[sp_col]
                model = smf.glm(f'Q("{sp_col}") + not_sp ~ Q("{fx}")', data=df, family=sm.families.Binomial()).fit()
                a, b = model.params[1], model.params[0]
                st.info(f"📝 ロジスティック予測式: **P = 1 / (1 + exp(-({a:.3f}X + {b:.3f})))**")
                if is_numeric:
                    x_range = np.linspace(df[fx].min(), df[fx].max(), 100)
                    y_pred = 1 / (1 + np.exp(-(a * x_range + b)))
                    ax.plot(x_range, y_pred, color='red', lw=3); ax.scatter(df[fx], df[target], alpha=0.5)

            else:
                model = smf.glm(f'Q("{target}") ~ Q("{fx}")', data=df, family=sm.families.Poisson()).fit()
                a, b = model.params[1], model.params[0]
                st.info(f"📝 ポアソン回帰式 (指数): **Y = exp({a:.3f}X + {b:.3f})**")
                if is_numeric:
                    x_range = np.linspace(df[fx].min(), df[fx].max(), 100)
                    y_pred = np.exp(a * x_range + b)
                    ax.plot(x_range, y_pred, color='red', lw=3); ax.scatter(df[fx], df[target], alpha=0.5)

            st.write("### モデル詳細 (Summary)")
            st.text(model.summary())
            ax.set_title(f"{mode} の回帰曲線")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"解析エラー: {e}")
