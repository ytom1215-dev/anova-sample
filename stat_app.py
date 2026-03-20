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
    
    return pd.DataFrame({
        'groups_name': list(cld.keys()),
        'letters': ["".join(l) for l in cld.values()]
    }), groups

# --- ページ設定 ---
st.set_page_config(page_title="農業統計解析ツール", layout="wide")
st.title("📊 農業データ・自動統計解析システム")
st.caption("正規分布（収量）から二項分布（発芽率・発病率）まで、Rレベルの解析をブラウザで。")

# --- 1. データ読み込み ---
data_source = st.radio("データの入力方法：", ["🥔 サンプルデータで試す", "📁 自分のCSVをアップロードする"])

if data_source == "🥔 サンプルデータで試す":
    df = pd.DataFrame({
        '品種': ['品種A', '品種A', '品種B', '品種B', '品種C', '品種C'] * 3,
        '処理': ['標準', '多量'] * 9,
        '全数': [100] * 18,
        '発芽数': [88, 95, 45, 60, 15, 30, 85, 92, 42, 58, 12, 28, 90, 96, 40, 62, 18, 35],
        '収量': [210, 250, 180, 200, 150, 170, 215, 255, 185, 205, 155, 175, 205, 245, 175, 195, 145, 165]
    })
    st.success("✅ サンプルデータを読み込みました。")
else:
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
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
    st.sidebar.header("⚙️ 解析設定")
    analysis_mode = st.sidebar.selectbox(
        "解析モード",
        ["📦 収量解析 (正規分布/ANOVA)", "🌱 発芽・発病率解析 (二項分布/GLM)"]
    )

    # 変数選択UI
    col1, col2, col3 = st.columns(3)
    with col1:
        if "収量解析" in analysis_mode:
            target_col = st.selectbox("目的変数 (数値)", df.columns, index=df.columns.get_loc('収量') if '収量' in df.columns else 0)
        else:
            sprouted_col = st.selectbox("発芽・発病数 (分子)", df.columns, index=df.columns.get_loc('発芽数') if '発芽数' in df.columns else 0)
            total_col = st.selectbox("全数・調査数 (分母)", df.columns, index=df.columns.get_loc('全数') if '全数' in df.columns else 0)
    with col2:
        factor_x = st.selectbox("主要因 (品種など)", df.columns, index=df.columns.get_loc('品種') if '品種' in df.columns else 0)
    with col3:
        factor_sub = st.selectbox("副要因 (処理・施肥など)", df.columns, index=df.columns.get_loc('処理') if '処理' in df.columns else 1)

    if st.button("🚀 解析を実行"):
        # グラフスタイル設定
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        
        try:
            st.header("📈 解析結果レポート")
            
            # --- モード1：収量解析 (ANOVA) ---
            if "収量解析" in analysis_mode:
                # 統計計算
                formula = f'Q("{target_col}") ~ C(Q("{factor_x}")) + C(Q("{factor_sub}"))'
                model = smf.ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                st.subheader("1. 二元配置分散分析表 (ANOVA)")
                st.table(anova_table)
                
                st.subheader(f"2. {factor_x} ごとの多重比較 (Tukey HSD)")
                tukey = pairwise_tukeyhsd(df[target_col], df[factor_x])
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                letters_df, groups_order = get_cld_letters(df, target_col, factor_x, tukey_df)
                
                c_a, c_b = st.columns([2, 1])
                c_a.dataframe(tukey_df)
                c_b.write("▼ 有意差(abc)")
                c_b.dataframe(letters_df.rename(columns={'groups_name': factor_x}))
                
                # グラフ描画
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=factor_x, y=target_col, hue=factor_sub, data=df, ax=ax, palette="Set3")
                # abcラベル追加
                for i, grp in enumerate(groups_order):
                    letter = letters_df.loc[letters_df['groups_name'] == grp, 'letters'].values[0]
                    y_val = df[df[factor_x] == grp][target_col].max()
                    ax.text(i, y_val * 1.05, letter, ha='center', color='red', weight='bold', size=14)
                st.pyplot(fig)

            # --- モード2：発芽・発病率解析 (GLM) ---
            else:
                df['not_sprouted'] = df[total_col] - df[sprouted_col]
                df['ratio'] = df[sprouted_col] / df[total_col]
                
                # GLM実行
                model = smf.glm(formula=f'Q("{sprouted_col}") + not_sprouted ~ C(Q("{factor_x}")) + C(Q("{factor_sub}"))',
                                data=df, family=sm.families.Binomial()).fit()
                
                st.subheader("1. GLM 解析結果 (ロジスティック回帰)")
                with st.expander("詳細なモデル要約を表示"):
                    st.text(model.summary())
                
                st.subheader("2. 偏差分析 (カイ二乗検定)")
                st.table(model.wald_test_terms().summary_frame())
                
                st.subheader(f"3. {factor_x} ごとの多重比較 (比率ベース)")
                tukey = pairwise_tukeyhsd(df['ratio'], df[factor_x])
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                letters_df, groups_order = get_cld_letters(df, 'ratio', factor_x, tukey_df)
                
                c_c, c_d = st.columns([2, 1])
                c_c.dataframe(tukey_df)
                c_d.write("▼ 有意差(abc)")
                c_d.dataframe(letters_df.rename(columns={'groups_name': factor_x}))
                
                # グラフ描画
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=factor_x, y='ratio', hue=factor_sub, data=df, ax=ax, palette="viridis", capsize=.1)
                for i, grp in enumerate(groups_order):
                    letter = letters_df.loc[letters_df['groups_name'] == grp, 'letters'].values[0]
                    y_val = df[df[factor_x] == grp]['ratio'].max()
                    ax.text(i, y_val + 0.05, letter, ha='center', color='red', weight='bold', size=14)
                ax.set_ylim(0, 1.2)
                ax.set_ylabel("比率 (0.0 - 1.0)")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"解析エラーが発生しました。列の選択が正しいか確認してください： {e}")
