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
st.title("📊 農業データ・自動統計解析システム")

# --- 1. サイドバー：解析の目的とモード選択 ---
st.sidebar.header("⚙️ 1. 解析の目的")
analysis_purpose = st.sidebar.radio(
    "どちらの解析を行いますか？",
    ["📊 要因解析 (カテゴリ比較・Tukey検定)", "📈 回帰解析 (数値からの予測・回帰曲線)"]
)

st.sidebar.header("⚙️ 2. データ分布の選択")
mode = st.sidebar.selectbox(
    "データの種類", 
    ["📦 収量など (正規分布/OLS)", "🌱 発芽率など (二項分布/Logit)", "🐛 虫害・腐敗数など (ポアソン/Log)"]
)

st.sidebar.header("⚙️ 3. データの入力")
data_source = st.sidebar.radio("入力方法：", ["🧪 サンプルデータで試す", "📁 CSVアップロード"])

# --- 2. サンプルデータの自動生成（★確実に有意差が出るように強化★） ---
if data_source == "🧪 サンプルデータで試す":
    np.random.seed(42) # データが毎回変わらないように固定
    
    if "要因解析" in analysis_purpose:
        n_grp = 10
        # a, b, cが綺麗に分かれるように平均値を大きく離す
        df = pd.DataFrame({
            '品種': ['品種A(優)']*n_grp + ['品種B(中)']*n_grp + ['品種C(劣)']*n_grp,
            '処理': ['標準', '多量'] * 15,
            '収量': np.concatenate([
                np.random.normal(280, 10, n_grp), # Aは高収量
                np.random.normal(230, 10, n_grp), # Bは中程度
                np.random.normal(180, 10, n_grp)  # Cは低収量
            ]).astype(int),
            '全数': [100] * (n_grp * 3),
            '発芽数': np.concatenate([
                np.random.binomial(100, 0.90, n_grp), # Aは90%発芽
                np.random.binomial(100, 0.50, n_grp), # Bは50%発芽
                np.random.binomial(100, 0.15, n_grp)  # Cは15%発芽
            ]),
            '腐敗数': np.concatenate([
                np.random.poisson(2, n_grp),  # Aは腐りにくい
                np.random.poisson(10, n_grp), # Bは普通
                np.random.poisson(25, n_grp)  # Cはすぐ腐る
            ])
        })
    else:
        n = 30
        x = np.linspace(10, 40, n)
        # グラフの線に沿うようにノイズを小さく設定
        y_ols = 4.5 * x + 100 + np.random.normal(0, 8, n)
        p = 1 / (1 + np.exp(-0.3 * (x - 25)))
        sp = np.random.binomial(100, p)
        y_poi = np.random.poisson(np.exp(0.09 * x))
        
        df = pd.DataFrame({
            '温度': x, 
            '品種': ['品種A']*15 + ['品種B']*15, 
            '収量': y_ols, 
            '全数': 100, 
            '発芽数': sp, 
            '腐敗数': y_poi
        })
    st.success(f"✅ テスト用データ（有意差保証版）を読み込みました。美しい結果を確認できます。")
else:
    uploaded_file = st.sidebar.file_uploader("CSV選択", type="csv")
    df = pd.read_csv(uploaded_file) if uploaded_file else None

# --- 3. メイン解析UI ---
if df is not None:
    st.header(f"🔍 設定：{analysis_purpose.split()[1]} × {mode.split()[1]}")
    col1, col2, col3 = st.columns(3)

    with col1:
        if "収量" in mode: target = st.selectbox("目的変数 (Y軸)", df.columns, index=df.columns.get_loc('収量') if '収量' in df.columns else len(df.columns)-1)
        elif "発芽率" in mode:
            sp_col = st.selectbox("カウント数 (分子)", df.columns, index=df.columns.get_loc('発芽数') if '発芽数' in df.columns else 0)
            tt_col = st.selectbox("全数 (分母)", df.columns, index=df.columns.get_loc('全数') if '全数' in df.columns else 0)
            df['ratio'] = df[sp_col] / df[tt_col]
            target = 'ratio'
        else: target = st.selectbox("カウント数 (Y軸)", df.columns, index=df.columns.get_loc('腐敗数') if '腐敗数' in df.columns else len(df.columns)-1)

    with col2:
        if "要因解析" in analysis_purpose:
            fx = st.selectbox("主要因 (比較するカテゴリ)", df.columns, index=0)
        else:
            fx = st.selectbox("主要因 (X軸の数値)", df.columns, index=0)
            
    with col3:
        if "要因解析" in analysis_purpose:
            fs = st.selectbox("副要因 (色分け/処理など)", df.columns, index=1 if len(df.columns)>1 else 0)
        else:
            st.info("※回帰解析モードでは、全体の傾向を単回帰モデルで算出します。")
            fs = None

    if st.button("🚀 解析を実行"):
        use_cols = [target, fx] if fs is None else [target, fx, fs]
        if "発芽率" in mode: use_cols.extend([sp_col, tt_col])
        df_clean = df.dropna(subset=[c for c in use_cols if c in df.columns]).copy()

        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if "要因解析" in analysis_purpose:
                df_clean[fx] = df_clean[fx].astype(str)
                
                if "収量" in mode:
                    formula = f'Q("{target}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                    model = smf.ols(formula, data=df_clean).fit()
                    st.subheader("1. 分散分析表 (ANOVA)")
                    st.table(sm.stats.anova_lm(model, typ=2))
                else:
                    if "発芽率" in mode:
                        df_clean['not_sp'] = df_clean[tt_col] - df_clean[sp_col]
                        formula = f'Q("{sp_col}") + not_sp ~ C(Q("{fx}")) + C(Q("{fs}"))'
                        family = sm.families.Binomial()
                    else:
                        formula = f'Q("{target}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                        family = sm.families.Poisson()
                        
                    model = smf.glm(formula, data=df_clean, family=family).fit()
                    st.subheader("1. 偏差分析 (カイ二乗検定)")
                    w_res = model.wald_test_terms().summary_frame()
                    st.table(w_res.apply(lambda c: c.map(lambda x: np.asarray(x).item() if hasattr(x, "__len__") else x)))

                st.subheader(f"2. {fx} の多重比較 (Tukey HSD)")
                tukey = pairwise_tukeyhsd(df_clean[target], df_clean[fx])
                tk_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                let_df, grp_ord = get_cld_letters(df_clean, target, fx, tk_df)
                
                c_l, c_r = st.columns([2, 1])
                c_l.dataframe(tk_df); c_r.dataframe(let_df.rename(columns={'groups_name': fx}))
                
                if "収量" in mode: 
                    sns.boxplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, palette="Set2", order=grp_ord)
                else: 
                    sns.barplot(x=fx, y=target, hue=fs, data=df_clean, ax=ax, capsize=.1, palette="viridis", order=grp_ord)
                
                y_offset = df_clean[target].max() * 0.05
                if y_offset == 0: y_offset = 0.05
                
                for i, g in enumerate(grp_ord):
                    l = let_df.loc[let_df['groups_name'] == g, 'letters'].values[0]
                    y_val = df_clean[df_clean[fx] == g][target].max()
                    ax.text(i, y_val + y_offset, l, ha='center', color='red', weight='bold', size=15)
                st.pyplot(fig)

            else:
                is_num = np.issubdtype(df_clean[fx].dtype, np.number)
                if not is_num:
                    st.warning(f"⚠️ {fx} は数値データではないため、回帰曲線は描画できません。数値の列を選択してください。")
                    st.stop()
                    
                if "収量" in mode:
                    model = smf.ols(f'Q("{target}") ~ Q("{fx}")', data=df_clean).fit()
                    st.subheader("1. 予測モデル (直線回帰)")
                    st.info(f"📝 式: **Y = {model.params[1]:.3f}X + {model.params[0]:.3f}** (R²={model.rsquared:.3f})")
                    sns.regplot(x=fx, y=target, data=df_clean, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    
                elif "発芽率" in mode:
                    df_clean['not_sp'] = df_clean[tt_col] - df_clean[sp_col]
                    model = smf.glm(f'Q("{sp_col}") + not_sp ~ Q("{fx}")', data=df_clean, family=sm.families.Binomial()).fit()
                    a, b = model.params[1], model.params[0]
                    st.subheader("1. 予測モデル (ロジスティック回帰)")
                    st.info(f"📝 式: **P = 1 / (1 + exp(-({a:.3f}X + {b:.3f})))**")
                    x_range = np.linspace(df_clean[fx].min(), df_clean[fx].max(), 100)
                    ax.plot(x_range, 1 / (1 + np.exp(-(a * x_range + b))), color='red', lw=3)
                    ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)
                    
                else:
                    model = smf.glm(f'Q("{target}") ~ Q("{fx}")', data=df_clean, family=sm.families.Poisson()).fit()
                    a, b = model.params[1], model.params[0]
                    st.subheader("1. 予測モデル (ポアソン回帰)")
                    st.info(f"📝 式: **Y = exp({a:.3f}X + {b:.3f})**")
                    x_range = np.linspace(df_clean[fx].min(), df_clean[fx].max(), 100)
                    ax.plot(x_range, np.exp(a * x_range + b), color='red', lw=3)
                    ax.scatter(df_clean[fx], df_clean[target], alpha=0.5)

                st.write("### 2. モデルの詳細要約")
                with st.expander("詳細を表示"):
                    st.text(model.summary())
                
                ax.set_title(f"{mode.split()[1]} の回帰曲線")
                ax.set_xlabel(fx); ax.set_ylabel("比率" if "発芽率" in mode else target)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"解析エラー: {e}")
