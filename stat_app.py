import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ページ設定
st.set_page_config(page_title="農業データ統計解析アプリ", layout="wide")
st.title("📊 栽培試験データ・統合解析ツール")

# 1. データ入力セクション
data_source = st.radio("データの入力方法：", ["🥔 サンプルデータで試す", "📁 自分のCSVをアップロードする"])

if data_source == "🥔 サンプルデータで試す":
    # 発芽試験のサンプルデータを生成
    df = pd.DataFrame({
        '品種': ['品種A', '品種A', '品種B', '品種B', '品種C', '品種C'] * 3,
        'かん水': ['少', '多'] * 9,
        '全数': [100] * 18,
        '発芽数': [85, 92, 40, 55, 10, 25, 88, 90, 42, 58, 12, 28, 84, 95, 38, 60, 15, 30],
        '収量': [210, 250, 180, 200, 150, 170, 215, 255, 185, 205, 155, 175, 205, 245, 175, 195, 145, 165]
    })
    st.success("✅ サンプルデータを読み込みました。（発芽率解析にも対応しています）")
else:
    uploaded_file = st.file_uploader("CSVアップロード", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='shift_jis')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        st.success("✅ CSVを読み込みました。")
    else:
        df = None

if df is not None:
    st.divider()
    
    # 2. 解析モードの選択
    analysis_mode = st.sidebar.selectbox(
        "解析モードを選択してください",
        ["📦 収量解析 (正規分布/ANOVA)", "🌱 発芽・発病率解析 (二項分布/GLM)"]
    )

    # 3. 変数選択のUI
    st.subheader(f"設定: {analysis_mode}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "収量解析" in analysis_mode:
            target_col = st.selectbox("目的変数（数値）", df.columns, index=df.columns.get_loc('収量') if '収量' in df.columns else 0)
        else:
            sprouted_col = st.selectbox("発芽数・発病数（分子）", df.columns, index=df.columns.get_loc('発芽数') if '発芽数' in df.columns else 0)
            total_col = st.selectbox("全数・調査数（分母）", df.columns, index=df.columns.get_loc('全数') if '全数' in df.columns else 0)
    
    with col2:
        factor_x = st.selectbox("主要因（品種など）", df.columns, index=df.columns.get_loc('品種') if '品種' in df.columns else 0)
    
    with col3:
        factor_sub = st.selectbox("副要因（かん水・施肥量など）", df.columns, index=df.columns.get_loc('かん水') if 'かん水' in df.columns else 1)

    if st.button("解析を実行"):
        # グラフ共通設定
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        
        try:
            # --- モード1：収量解析 (ANOVA) ---
            if "収量解析" in analysis_mode:
                formula = f'Q("{target_col}") ~ C(Q("{factor_x}")) + C(Q("{factor_sub}"))'
                model = smf.ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                st.write("### 分散分析表 (ANOVA)")
                st.table(anova_table)
                
                # 多重比較（Tukey）
                tukey = pairwise_tukeyhsd(df[target_col], df[factor_x])
                st.write(f"### {factor_x} の多重比較")
                st.write(pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0]))
                
                # グラフ
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(x=factor_x, y=target_col, hue=factor_sub, data=df, ax=ax)
                st.pyplot(fig)

           # --- モード2：発芽・発病率解析 (GLM) ---
            else:
                df['not_sprouted'] = df[total_col] - df[sprouted_col]
                
                # GLM実行 (Binomial)
                model = smf.glm(formula=f'Q("{sprouted_col}") + not_sprouted ~ C(Q("{factor_x}")) + C(Q("{factor_sub}"))',
                                data=df, family=sm.families.Binomial()).fit()
                
                st.write("### GLM 解析結果要約 (ロジスティック回帰)")
                st.text(model.summary())
                
                # 🌟ここを修正：table_summary ではなく summary_frame() を使う🌟
                st.write("### 偏差分析 (カイ二乗検定)")
                try:
                    wald_res = model.wald_test_terms().summary_frame()
                    st.table(wald_res)
                except:
                    # 万が一の予備コード
                    st.write(model.wald_test_terms())
                
                # グラフ（比率を表示）
                df['ratio'] = df[sprouted_col] / df[total_col]
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(x=factor_x, y='ratio', hue=factor_sub, data=df, ax=ax)
                ax.set_ylabel("比率 (0-1.0)")
                ax.set_ylim(0, 1.1)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"解析エラー: {e}")
