import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib  # インポートするだけで日本語化されます
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
    cld = {g:[] for g in groups}
    current_letter = ord('a')
    
    # 文字列/Bool どちらが来ても安全に判定する関数
    def is_rejected(val):
        if isinstance(val, bool) or isinstance(val, np.bool_): return val
        return str(val).lower() == 'true'

    for i in range(len(groups)):
        g1 = groups[i]
        non_diff = [g1]
        for j in range(i + 1, len(groups)):
            g2 = groups[j]
            mask = ((tukey_summary['group1'].astype(str) == g1) & (tukey_summary['group2'].astype(str) == g2)) | \
                   ((tukey_summary['group1'].astype(str) == g2) & (tukey_summary['group2'].astype(str) == g1))
            reject_vals = tukey_summary.loc[mask, 'reject'].values
            if len(reject_vals) > 0 and not is_rejected(reject_vals[0]): 
                non_diff.append(g2)
        shared_letters = set(cld[non_diff[0]])
        for g in non_diff[1:]:
            shared_letters = shared_letters.intersection(cld[g])
        if not shared_letters:
            for g in non_diff:
                cld[g].append(chr(current_letter))
            current_letter += 1
    return pd.DataFrame({'groups_name': list(cld.keys()), 'letters':["".join(l) for l in cld.values()]}), groups

# --- ページ設定 ---
st.set_page_config(page_title="農業統計解析ツール", layout="wide")
st.title("📊 農業データ・自動統計解析システム (研修対応版)")

with st.sidebar.expander("🎓 新人研修向け：統計用語ガイド"):
    st.markdown("""
    **💡 P値 (P-value)**: 偶然起きた確率。**0.05未満**なら有意差あり。
    **💡 アルファベット**: 違う文字(a, b)がついているグループ同士は有意差あり。
    **💡 分布の選び方**:
    * **正規分布:** 連続した数字 (収量など)
    * **二項分布:** 割合 (発芽率など)
    * **ポアソン:** 0以上の整数・カウント (虫数など)
    * **ノンパラ:** ランク分けされたもの (発生程度など)
    """)

# ==========================================
# ⚙️ サイドバー（設定パネル）
# ==========================================
st.sidebar.header("⚙️ 1. データの入力")
data_source = st.sidebar.radio("入力方法：",["🧪 サンプルデータで試す", "📁 CSVアップロード"])

st.sidebar.header("⚙️ 2. 解析の目的")
analysis_purpose = st.sidebar.radio("目的：",["📊 要因解析 (カテゴリ比較・多重比較)", "📈 回帰解析 (数値からの予測・回帰曲線)"])

st.sidebar.header("⚙️ 3. 目的変数の性質")
mode = st.sidebar.selectbox("データの種類",["📦 連続値：収量・草丈・重量など (正規分布/OLS)", 
                                           "🌱 割合：発芽率・発病率など (二項分布/Logit)", 
                                           "🐛 カウント：虫数・腐敗数など (ポアソン/Log)",
                                           "🗂️ 順序データ：発生程度スコアなど (ノンパラ/Kruskal-Wallis)"])

# ==========================================
# 📊 データの読み込み (文字化け対策付き)
# ==========================================
df = None
uploaded_file = None
if data_source == "🧪 サンプルデータで試す":
    np.random.seed(42)
    if "要因解析" in analysis_purpose:
        n_grp = 15
        df = pd.DataFrame({
            '品種': ['品種A']*n_grp +['品種B']*n_grp + ['品種C']*n_grp,
            '温度':['10度']*15 +['20度']*15 + ['30度']*15,
            '処理':['標準', '多量', '少量'] * 15,
            '収量': np.concatenate([np.random.normal(280, 10, n_grp), np.random.normal(230, 10, n_grp), np.random.normal(180, 10, n_grp)]).astype(int),
            '全数':[100] * 45,
            '発芽数': np.concatenate([np.random.binomial(100, 0.90, n_grp), np.random.binomial(100, 0.50, n_grp), np.random.binomial(100, 0.15, n_grp)]),
            '腐敗数': np.concatenate([np.random.poisson(2, n_grp), np.random.poisson(10, n_grp), np.random.poisson(25, n_grp)]),
            '発生程度': np.concatenate([np.random.choice([1, 2], n_grp, p=[0.8, 0.2]), np.random.choice([1, 2, 3], n_grp, p=[0.2, 0.6, 0.2]), np.random.choice([3, 4], n_grp, p=[0.4, 0.6])])
        })
    else:
        n = 30; x = np.linspace(10, 40, n)
        df = pd.DataFrame({'温度': x, '品種': ['品種A']*15 + ['品種B']*15, '収量': 4.5 * x + 100 + np.random.normal(0, 8, n), '全数': 100, '発芽数': np.random.binomial(100, 1 / (1 + np.exp(-0.3 * (x - 25)))), '腐敗数': np.random.poisson(np.exp(0.09 * x))})
else:
    uploaded_file = st.sidebar.file_uploader("分析するCSVファイルを選択", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp932')

if df is None:
    st.info("📂 左のサイドバーから分析したいCSVファイルをアップロードしてください。")
    st.stop()

# ==========================================
# 📋 メイン解析UI：Step 1 & Step 2
# ==========================================
st.markdown("### 📋 1. 読み込みデータの確認")
c1, c2 = st.columns(2)
with c1: st.markdown("**▼ 先頭5行**"); st.dataframe(df.head(5), use_container_width=True)
with c2: st.markdown("**▼ データの要約**"); st.dataframe(df.describe(include='all').fillna(""), use_container_width=True)
st.divider()

st.markdown(f"### ⚙️ 2. 分析変数の設定")
col1, col2, col3 = st.columns(3)

with col1:
    if "連続値" in mode: target = st.selectbox("目的変数 (Y軸)", df.columns, index=df.columns.get_loc('収量') if '収量' in df.columns else len(df.columns)-1)
    elif "割合" in mode:
        sp_col = st.selectbox("カウント数 (分子)", df.columns, index=df.columns.get_loc('発芽数') if '発芽数' in df.columns else 0)
        tt_col = st.selectbox("全数 (分母)", df.columns, index=df.columns.get_loc('全数') if '全数' in df.columns else 0)
        df['ratio'] = df[sp_col] / df[tt_col]
        target = 'ratio'
    elif "順序" in mode: target = st.selectbox("順序データ (1,2,3...)", df.columns, index=df.columns.get_loc('発生程度') if '発生程度' in df.columns else len(df.columns)-1)
    else: target = st.selectbox("カウント数 (Y軸)", df.columns, index=df.columns.get_loc('腐敗数') if '腐敗数' in df.columns else len(df.columns)-1)

with col2:
    if "要因解析" in analysis_purpose: fx = st.selectbox("主要因 (比較カテゴリ)", df.columns, index=df.columns.get_loc('温度') if '温度' in df.columns else 0)
    else: fx = st.selectbox("主要因 (X軸の数値)", df.columns, index=0)
        
with col3:
    interaction_op = "+"
    if "要因解析" in analysis_purpose:
        if "順序" in mode:
            fs = None
        else:
            fs = st.selectbox("副要因 (色分け/処理など)", ["なし"] + list(df.columns), index=0)
            if fs == "なし": fs = None
            else:
                interaction_choice = st.radio("交互作用の考慮", ["なし (+)", "あり (*)"], horizontal=True)
                interaction_op = "*" if "あり" in interaction_choice else "+"
    else: fs = None

# ==========================================
# 💡 変数が変わったら結果をリセットする仕組み
# ==========================================
current_settings = {
    "data_source": data_source,
    "file_name": uploaded_file.name if data_source == "📁 CSVアップロード" and uploaded_file is not None else None,
    "analysis_purpose": analysis_purpose,
    "mode": mode, "target": target, "fx": fx, "fs": fs, "interaction": interaction_op,
    "sp_col": sp_col if "割合" in mode else None, "tt_col": tt_col if "割合" in mode else None
}

if 'last_settings' not in st.session_state:
    st.session_state.last_settings = current_settings
    st.session_state.run_clicked = False

if st.session_state.last_settings != current_settings:
    st.session_state.run_clicked = False
    st.session_state.last_settings = current_settings

def handle_click(): st.session_state.run_clicked = True

st.markdown("<br>", unsafe_allow_html=True)
st.button("🚀 解析を実行", type="primary", use_container_width=True, on_click=handle_click)

# ==========================================
# 🚀 実行結果 (Step 3)
# ==========================================
if st.session_state.run_clicked:
    st.divider()
    st.markdown("### 📊 3. 解析結果")
    
    use_cols =[target, fx] if fs is None else [target, fx, fs]
    if "割合" in mode: use_cols.extend([sp_col, tt_col])
    df_clean = df.dropna(subset=[c for c in use_cols if c in df.columns]).copy()

    # 🛠️ 【超重要】日本語カラム名によるエラーを防ぐための安全な内部リネーム処理
    rename_dict = {target: 'Y_val', fx: 'X_factor'}
    if fs: rename_dict[fs] = 'F_factor'
    if "割合" in mode: rename_dict[tt_col] = 'Total_n'
    df_model = df_clean.rename(columns=rename_dict)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    code_snippet = ""
    download_df = None
    
    try:
        # ------------------------------------------
        # A: 要因解析モード
        # ------------------------------------------
        if "要因解析" in analysis_purpose:
            df_model['X_factor'] = df_model['X_factor'].astype(str)
            if fs: df_model['F_factor'] = df_model['F_factor'].astype(str)
            formula_str = f'Y_val ~ C(X_factor) {interaction_op} C(F_factor)' if fs else 'Y_val ~ C(X_factor)'
            
            # ① 順序データ
            if "順序" in mode:
                st.subheader("1. 全体の検定結果 (Kruskal-Wallis検定)")
                groups_data =[df_model[df_model['X_factor'] == g]['Y_val'] for g in df_model['X_factor'].unique()]
                stat, p_val = stats.kruskal(*groups_data)
                st.info(f"📝 統計量 (H) = {stat:.3f}, **P値 = {p_val:.4e}**")

                st.subheader(f"2. {fx} の多重比較 (Steel-Dwass検定)")
                res_matrix = sp.posthoc_dscf(df_model, val_col='Y_val', group_col='X_factor')
                
                groups = df_model['X_factor'].unique()
                pairs = list(itertools.combinations(groups, 2))
                res =[]
                for g1, g2 in pairs:
                    p_adj = res_matrix.loc[g1, g2]
                    res.append({'group1': str(g1), 'group2': str(g2), 'p_value': p_adj, 'reject': p_adj < 0.05})
                
                res_df = pd.DataFrame(res)
                let_df, grp_ord = get_cld_letters(df_model, 'Y_val', 'X_factor', res_df)
                
                # 出力用に名前を元に戻す
                download_df = res_df.rename(columns={'group1': f'{fx}_1', 'group2': f'{fx}_2'})
                
                c_l, c_r = st.columns([2, 1])
                c_l.dataframe(download_df.style.format({'p_value': '{:.4f}'}))
                c_r.dataframe(let_df.rename(columns={'groups_name': fx}))
                
                sns.boxplot(x=fx, y=target, data=df_clean, ax=ax, palette="Set2", order=grp_ord, showfliers=False)
                sns.stripplot(x=fx, y=target, data=df_clean, ax=ax, color=".3", alpha=0.6, jitter=True, order=grp_ord)
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            # ② それ以外
            else:
                if "連続値" in mode:
                    model = smf.ols(formula_str, data=df_model).fit()
                    st.subheader("1. 全体の検定結果 (分散分析: ANOVA)")
                    anova_table = sm.stats.anova_lm(model, typ=2)
                    anova_table.rename(index={'C(X_factor)': fx, 'C(F_factor)': fs, 'C(X_factor):C(F_factor)': f'{fx}:{fs}'}, inplace=True)
                    st.table(anova_table)
                else:
                    if "割合" in mode:
                        family = sm.families.Binomial()
                        model = smf.glm(formula_str, data=df_model, family=family, var_weights=np.asarray(df_model['Total_n'])).fit()
                    else:
                        family = sm.families.Poisson()
                        model = smf.glm(formula_str, data=df_model, family=family).fit()
                        
                    st.subheader("1. 全体の検定結果 (尤度比 / カイ二乗検定)")
                    w_res = model.wald_test_terms().summary_frame()
                    w_res.rename(index={'C(X_factor)': fx, 'C(F_factor)': fs, 'C(X_factor):C(F_factor)': f'{fx}:{fs}'}, inplace=True)
                    st.table(w_res.apply(lambda c: c.map(lambda x: np.asarray(x).item() if hasattr(x, "__len__") else x)))

                st.subheader(f"2. {fx} の多重比較 (Tukey HSD検定)")
                if fs is not None:
                    st.info(f"💡 【補足】多重比較（Tukey検定）は全体の傾向を見るため、主要因「{fx}」間のみで実施しています。")

                tukey = pairwise_tukeyhsd(df_model['Y_val'], df_model['X_factor'])
                tk_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                let_df, grp_ord = get_cld_letters(df_model, 'Y_val', 'X_factor', tk_df)
                
                # 出力用に名前を元に戻す
                download_df = tk_df.rename(columns={'group1': f'{fx}_1', 'group2': f'{fx}_2'})
                
                c_l, c_r = st.columns([2, 1])
                c_l.dataframe(download_df)
                c_r.dataframe(let_df.rename(columns={'groups_name': fx}))
                
                if "連続値" in mode: 
                    sns.boxplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, palette="Set2", order=grp_ord)
                else: 
                    sns.barplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, capsize=.1, palette="viridis", order=grp_ord)
            
            # アルファベット付与 (副要因がない場合のみ描画)
            if fs is None:
                for i, g in enumerate(grp_ord):
                    l = let_df.loc[let_df['groups_name'] == g, 'letters'].values[0]
                    y_val = df_clean[df_clean[fx] == str(g)][target].max()
                    y_offset = df_clean[target].max() * 0.05 if df_clean[target].max() > 0 else 0.05
                    ax.text(i, y_val + y_offset, l, ha='center', color='red', weight='bold', size=15)

            st.pyplot(fig)

            if download_df is not None:
                csv = download_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(label="📥 多重比較の結果をCSVでダウンロード", data=csv, file_name='tukey_results.csv', mime='text/csv')

        # ------------------------------------------
        # B: 回帰解析モード
        # ------------------------------------------
        else:
            if not np.issubdtype(df_clean[fx].dtype, np.number):
                st.warning(f"⚠️ エラー：{fx} は数値データではないため回帰曲線は描画できません。X軸には数値データを選択してください。")
                st.stop()
                
            x_range = np.linspace(df_model['X_factor'].min(), df_model['X_factor'].max(), 100)
            pred_df = pd.DataFrame({'X_factor': x_range})

            if "連続値" in mode:
                model = smf.ols('Y_val ~ X_factor', data=df_model).fit()
                sns.regplot(x=fx, y=target, data=df_clean, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            elif "割合" in mode:
                model = smf.glm('Y_val ~ X_factor', data=df_model, family=sm.families.Binomial(), var_weights=np.asarray(df_model['Total_n'])).fit()
                ax.plot(x_range, model.predict(pred_df), color='red', lw=3)
                ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)
            else:
                model = smf.glm('Y_val ~ X_factor', data=df_model, family=sm.families.Poisson()).fit()
                ax.plot(x_range, model.predict(pred_df), color='red', lw=3)
                ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)

            st.subheader("1. 予測モデルの回帰グラフ")
            st.pyplot(fig)
            
            st.write("### 2. モデルの詳細要約")
            with st.expander("詳細を表示"): st.text(model.summary())
            
            summary_csv = model.summary().as_csv()
            st.download_button("📥 サマリー結果をダウンロード", data=summary_csv.encode('utf-8-sig'), file_name='regression_summary.csv', mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ 解析中にエラーが発生しました。\n選択した変数やデータの形式が正しいか確認してください。\n\n詳細: {e}")
