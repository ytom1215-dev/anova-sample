import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib # 日本語化の救世主
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ページ設定
st.set_page_config(page_title="農業データ統計解析アプリ", layout="wide")
st.title("📊 栽培試験データ・分散分析解析ツール")

data_source = st.radio(
    "データの入力方法を選んでください：", 
    ["🥔 組み込みのサンプルデータで試す", "📁 自分のCSVをアップロードする"]
)

df = None

if data_source == "🥔 組み込みのサンプルデータで試す":
    df = pd.DataFrame({
        '品種': ['とうや', 'とうや', 'とうや', 'ニシユタカ', 'ニシユタカ', 'ニシユタカ', 'デジマ', 'デジマ', 'デジマ'] * 2,
        '施肥量': [0, 10, 20, 0, 10, 20, 0, 10, 20] * 2,
        '収量': [210, 250, 260, 230, 280, 310, 190, 240, 250, 215, 255, 265, 235, 285, 315, 195, 245, 255]
    })
    st.success("✅ サンプルデータを読み込みました。（施肥量を数値データとして扱います）")
else:
    uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='shift_jis')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        
        if len(df.columns) == 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='shift_jis', sep='\t')
        st.success("✅ CSVを読み込みました。")

if df is not None:
    st.subheader("1. データプレビュー")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    with col1:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        yld_idx = df.columns.get_loc('収量') if '収量' in df.columns else (df.columns.get_loc(numeric_cols[0]) if numeric_cols else 0)
        target_col = st.selectbox("目的変数（収量など：数値）", df.columns, index=yld_idx)
    with col2:
        var_idx = df.columns.get_loc('品種') if '品種' in df.columns else 0
        factor_x = st.selectbox("主要因（品種など：カテゴリ扱いにします）", df.columns, index=var_idx)
    with col3:
        fer_idx = df.columns.get_loc('施肥量') if '施肥量' in df.columns else (1 if len(df.columns)>1 else 0)
        factor_sub = st.selectbox("副要因（施肥量など：カテゴリ・連続変数の両方で解析します）", df.columns, index=fer_idx)

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

    if st.button("解析を実行する"):
        try:
            df_clean = df.copy()
            df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[target_col]).copy()

            if df_clean.empty:
                st.error("【エラー】目的変数に数値がありません。列の選択を見直してください。")
            else:
                st.header("📈 解析結果報告書")
                
                # 🌟すべてのグラフの背景とフォントを「一括」で設定（ここで豆腐を完全ブロック）🌟
                sns.set_theme(style="whitegrid")
                plt.rcParams['font.family'] = 'IPAexGothic'

                df_clean[factor_x] = df_clean[factor_x].astype(str)
                df_clean[factor_sub] = df_clean[factor_sub].astype(str)

                # 1. 二元配置分散分析 (ANOVA)
                formula = f'Q("{target_col}") ~ C(Q("{factor_x}")) + C(Q("{factor_sub}"))'
                model = ols(formula, data=df_clean).fit()
                anova_res = anova_lm(model, typ=2)
                anova_res.index = [f"{factor_x} (主効果)", f"{factor_sub} (副効果)", '残差']

                st.subheader("1. 二元配置分散分析表 (ANOVA)")
                st.write(anova_res)
                st.divider()

                # 2. 主要因（品種など）の多重比較
                st.subheader(f"2. {factor_x} ごとの多重比較 (Tukey HSD)")
                tukey_x = pairwise_tukeyhsd(endog=df_clean[target_col], groups=df_clean[factor_x], alpha=0.05)
                tukey_summary_x = pd.DataFrame(data=tukey_x._results_table.data[1:], columns=tukey_x._results_table.data[0])
                letters_df_x, groups_order_x = get_cld_letters(df_clean, target_col, factor_x, tukey_summary_x)
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.dataframe(tukey_summary_x)
                with col_b:
                    st.dataframe(letters_df_x.rename(columns={'groups_name': factor_x, 'letters': '有意差(abc)'}))

                # 個別の設定を削除し、一括設定に従わせる
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
                st.divider()

                # 3. 副要因（施肥量など）の多重比較
                st.subheader(f"3. {factor_sub} ごとの多重比較 (ANOVA視点：カテゴリとして比較)")
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

                # 4. 単回帰分析
                st.subheader(f"4. {factor_sub} を量的変数とした単回帰分析 (回帰視点)")
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
else:
    st.info("CSVファイルをアップロードしてください。")