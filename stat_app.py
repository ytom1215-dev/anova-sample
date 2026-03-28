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
        # 副要因に「なし」を追加
        options_sub = ["なし"] + list(df.columns)
        fer_idx = df.columns.get_loc('施肥量') + 1 if '施肥量' in df.columns else 0
        factor_sub = st.selectbox("副要因（交互作用を解析可能）", options_sub, index=fer_idx)

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
                
                # 🌟すべてのグラフの背景とフォントを「一括」で設定🌟
                sns.set_theme(style="whitegrid")
                plt.rcParams['font.family'] = 'IPAexGothic'

                df_clean[factor_x] = df_clean[factor_x].astype(str)
                if factor_sub != "なし":
                    df_clean[factor_sub] = df_clean[factor_sub].astype(str)

                # ==========================================
                # 1. 分散分析 (ANOVA) - 🌟交互作用に対応🌟
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

# 一元配置モデルの作成
formula = 'Q("{target_col}") ~ C(Q("{factor_x}"))'
model = ols(formula, data=df).fit()

# 分散分析表を出力
anova_res = anova_lm(model, typ=2)

# 寄与率（η2）の計算
anova_res['寄与率'] = anova_res['sum_sq'] / anova_res['sum_sq'].sum()
print(anova_res)
                    """
                    r_anova_code = f"""
# Rでの一元配置分散分析
model <- aov({target_col} ~ as.factor({factor_x}), data=df)
summary(model)

# 寄与率（η2）の計算
library(effectsize)
eta_squared(model)
                    """

                else:
                    # 🌟モデル式を主効果＋交互作用（*）に変更🌟
                    formula = f'Q("{target_col}") ~ C(Q("{factor_x}")) * C(Q("{factor_sub}"))'
                    model = ols(formula, data=df_clean).fit()
                    anova_res = anova_lm(model, typ=2)
                    
                    # 🌟インデックス名を適切に設定🌟
                    idx_interaction = f"{factor_x} x {factor_sub} (交互作用)"
                    new_index = [
                        f"{factor_x} (主効果)", 
                        f"{factor_sub} (主効果)", 
                        idx_interaction,
                        '残差'
                    ]
                    # anova_resの行数がnew_indexと一致することを確認（欠損などでモデルが縮退した場合を除く）
                    if len(anova_res.index) == len(new_index):
                        anova_res.index = new_index
                    else:
                        st.warning("⚠️ モデルの縮退により、一部の効果が計算されませんでした。ANOVA表のインデックスは自動設定されます。")

                    st.subheader("1. 二元配置分散分析表 (主効果＋交互作用) (Typ II ANOVA)")

                    # 🌟交互作用のP値を取得して有意性を判定🌟
                    interaction_p = np.nan
                    if idx_interaction in anova_res.index:
                        interaction_p = anova_res.loc[idx_interaction, 'PR(>F)']
                        
                    py_anova_code = f"""
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# 🌟二元配置モデルの作成（主効果＋交互作用: *）🌟
formula = 'Q("{target_col}") ~ C(Q("{factor_x}")) * C(Q("{factor_sub}"))'
model = ols(formula, data=df).fit()

# 分散分析表を出力 (Typ II ANOVA)
anova_res = anova_lm(model, typ=2)

# 寄与率（η2）の計算
anova_res['寄与率'] = anova_res['sum_sq'] / anova_res['sum_sq'].sum()
print(anova_res)
                    """
                    r_anova_code = f"""
# 🌟Rでの二元配置分散分析（主効果＋交互作用: *）🌟
model <- aov({target_col} ~ as.factor({factor_x}) * as.factor({factor_sub}), data=df)
summary(model)

# 寄与率（η2）の計算
library(effectsize)
eta_squared(model)
                    """

                # 寄与率（η2：イータ二乗）の計算
                anova_res['寄与率'] = anova_res['sum_sq'] / anova_res['sum_sq'].sum()

                # 数値が見やすいようにフォーマットを適用
                st.dataframe(anova_res.style.format({
                    'sum_sq': '{:.2f}', 
                    'df': '{:.0f}', 
                    'F': '{:.3f}', 
                    'PR(>F)': '{:.4f}',
                    '寄与率': '{:.3f}'
                }))

                # --- コードリファレンス (ANOVA) ---
                with st.expander("💻 この解析のコードを見る (Python / R)"):
                    tab1, tab2 = st.tabs(["🐍 Python", "🔵 R"])
                    with tab1:
                        st.code(py_anova_code.strip(), language="python")
                    with tab2:
                        st.code(r_anova_code.strip(), language="r")

                st.divider()

                # ==========================================
                # 2. 主要因（品種など）の多重比較
                # ==========================================
                st.subheader(f"2. {factor_x} ごとの多重比較 (Tukey HSD)")
                
                # 有意差の有無を判定
                main_effect_p = anova_res.loc[f"{factor_x} (主効果)", 'PR(>F)']
                if factor_sub != "なし" and not np.isnan(interaction_p) and interaction_p < 0.05:
                    st.warning(f"⚠️ 交互作用（{idx_interaction}）が有意（p={interaction_p:.4f}）であるため、主効果（品種単独の効果）の検定結果には注意が必要です。交互作用図を見て、組み合わせによる効果の変化を確認してください。")

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

                # --- コードリファレンス (Tukey: 主要因) ---
                with st.expander("💻 この解析のコードを見る (Python / R)"):
                    tab1, tab2 = st.tabs(["🐍 Python", "🔵 R"])
                    with tab1:
                        st.code(f"""
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey HSD検定の実行
tukey = pairwise_tukeyhsd(endog=df['{target_col}'], groups=df['{factor_x}'], alpha=0.05)
print(tukey)
                        """, language="python")
                    with tab2:
                        st.code(f"""
# RでのTukeyの多重比較とCLD（アルファベット付与）
library(multcompView)

model <- aov({target_col} ~ as.factor({factor_x}), data=df)
tukey <- TukeyHSD(model)
print(tukey)

# アルファベットの付与 (CLD)
cld <- multcompLetters4(model, tukey)
print(cld)
                        """, language="r")

                st.divider()

                # ==========================================
                # 🌟3. 交互作用の解析と交互作用図 (🌟新規🌟)
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

                    # 🌟交互作用図の作成（Pointplot）🌟
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

                    # --- コードリファレンス (交互作用図) ---
                    with st.expander("💻 この解析のコードを見る (Python / R)"):
                        tab1, tab2 = st.tabs(["🐍 Python", "🔵 R"])
                        with tab1:
                            st.code(f"""
import seaborn as sns
import matplotlib.pyplot as plt

# 🌟交互作用図の作成（Pointplot）🌟
fig, ax = plt.subplots(figsize=(10, 6))
sns.pointplot(x='{factor_x}', y='{target_col}', hue='{factor_sub}', data=df, ax=ax, 
             dodge=True, capsize=.2, palette='colorblind')
ax.set_title(f"交互作用図：{factor_x} x {factor_sub}")
plt.show()
                            """, language="python")
                        with tab2:
                            st.code(f"""
# 🌟Rでの交互作用図の作成（ggplot2パッケージ）🌟
# install.packages("ggplot2")
library(ggplot2)

# 各組み合わせの平均値を計算
df_means <- aggregate({target_col} ~ {factor_x} + {factor_sub}, data=df, mean)

# プロットの作成
ggplot(df_means, aes(x=as.factor({factor_x}), y={target_col}, group=as.factor({factor_sub}), color=as.factor({factor_sub}))) + 
  geom_line() + geom_point() +
  labs(x="{factor_x}", y="{target_col}", color="{factor_sub}", title=f"交互作用図：{factor_x} x {factor_sub}") +
  theme_minimal()
                            """, language="r")

                    st.divider()

                # ==========================================
                # 4 & 5. 副要因のごとの多重比較と単回帰分析（セクション番号を変更）
                # ==========================================
                # ==========================================
                # 4. 副要因（施肥量など）の多重比較
                # ==========================================
                if factor_sub != "なし":
                    st.subheader(f"4. {factor_sub} ごとの多重比較 (ANOVA視点：カテゴリとして比較)")
                    
                    if not np.isnan(interaction_p) and interaction_p < 0.05:
                        st.warning(f"⚠️ 交互作用（{idx_interaction}）が有意（p={interaction_p:.4f}）であるため、この多重比較結果の解釈には注意が必要です。交互作用図を見て、組み合わせによる効果の変化を確認してください。")

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

                    # 🌟5. 単回帰分析（セクション番号を変更）🌟
                    st.subheader(f"5. {factor_sub} を量的変数とした単回帰分析 (回帰視点)")
                    # ...（以下は前回のコードと同じため省略、セクション番号だけ変更しています）
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

                        # --- コードリファレンス (単回帰分析) ---
                        with st.expander("💻 この解析のコードを見る (Python / R)"):
                            tab1, tab2 = st.tabs(["🐍 Python", "🔵 R"])
                            with tab1:
                                st.code(f"""
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt

# 単回帰モデルの構築
formula_reg = 'Q("{target_col}") ~ Q("{factor_sub}")'
model_reg = ols(formula_reg, data=df).fit()
print(model_reg.summary())

# 回帰直線のプロット
sns.regplot(x='{factor_sub}', y='{target_col}', data=df)
plt.show()
                                """, language="python")
                            with tab2:
                                st.code(f"""
# Rでの単回帰分析 (要因を連続変数として扱う)
model_reg <- lm({target_col} ~ {factor_sub}, data=df)
summary(model_reg)

# プロットと回帰直線の追加
plot(df${factor_sub}, df${target_col}, 
     xlab="{factor_sub}", ylab="{target_col}")
abline(model_reg, col="red")
                                """, language="r")

        except Exception as e:
            st.error(f"解析中にエラーが発生しました: {e}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("CSVファイルをアップロードしてください。")
