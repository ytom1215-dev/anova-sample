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

# --- 2. サンプルデータの自動生成（目的に応じて分岐） ---
if data_source == "🧪 サンプルデータで試す":
    if "要因解析" in analysis_purpose:
        # カテゴリ（品種・処理）のデータ
        df = pd.DataFrame({
            '品種': ['品種A']*6 + ['品種B']*6 + ['品種C']*6,
            '処理': ['標準', '多量'] * 9,
            '収量': [210, 250, 180, 200, 150, 170, 215, 255, 185, 205, 155, 175, 205, 245, 175, 195, 145, 165],
            '発芽数': [88, 95, 45, 60, 15, 30, 85, 92, 42, 58, 12, 28, 90, 96, 40, 62, 18, 35],
            '全数': [100] * 18,
            '腐敗数': [2, 1, 8, 12, 25, 30, 3, 2, 7, 10, 22, 28, 1, 3, 9, 11, 24, 32]
        })
    else:
        # 数値（温度・施肥量・日数）のデータ
        n = 30
        x = np.linspace(10, 40, n)
        y_ols = 3.17 * x + 217.5 + np.random.normal(0, 15, n)
        p = 1 / (1 + np.exp(-0.2 * (x - 25)))
        sp = np.random.binomial(100, p)
        y_poi = np.random.poisson(np.exp(0.08 * x))
        df = pd.DataFrame({'温度': x, '品種': ['品種A']*15 + ['品種B']*15, '収量': y_ols, '全数': 100, '発芽数': sp, '腐敗数': y_poi})
    
    st.success(f"✅ {analysis_purpose.split()[1]} 用のサンプルデータを読み込みました。")
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
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'IPAexGothic'
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            # ==========================================
            # A: 要因解析モード (Tukey & abc付け)
            # ==========================================
            if "要因解析" in analysis_purpose:
                if "収量" in mode:
                    formula = f'Q("{target}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                    model = smf.ols(formula, data=df).fit()
                    st.subheader("1. 分散分析表 (ANOVA)")
                    st.table(sm.stats.anova_lm(model, typ=2))
                else:
                    if "発芽率" in mode:
                        df['not_sp'] = df[tt_col] - df[sp_col]
                        formula = f'Q("{sp_col}") + not_sp ~ C(Q("{fx}")) + C(Q("{fs}"))'
                        family = sm.families.Binomial()
                    else:
                        formula = f'Q("{target}") ~ C(Q("{fx}")) + C(Q("{fs}"))'
                        family = sm.families.Poisson()
                        
                    model = smf.glm(formula, data=df, family=family).fit()
                    st.subheader("1. 偏差分析 (カイ二乗検定)")
                    w_res = model.wald_test_terms().summary_frame()
                    st.table(w_res.apply(lambda c: c.map(lambda x: np.asarray(x).item() if hasattr(x, "__len__") else x)))

                # Tukey & グラフ
                st.subheader(f"2. {fx} の多重比較 (Tukey HSD)")
                tukey = pairwise_tukeyhsd(df[target], df[fx])
                tk_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                let_df, grp_ord = get_cld_letters(df, target, fx, tk_df)
                
                c_l, c_r = st.columns([2, 1])
                c_l.dataframe(tk_df); c_r.dataframe(let_df.rename(columns={'groups_name': fx}))
                
                if "収量" in mode: sns.boxplot(x=fx, y=target, hue=fs, data=df, ax=ax, palette="Set2")
                else: sns.barplot(x=fx, y=target, hue=fs, data=df, ax=ax, capsize=.1, palette="viridis")
                
                for i, g in enumerate(grp_ord):
                    l = let_df.loc[let_df['groups_name'] == g, 'letters'].values[0]
                    y_val = df[df[fx] == g][target].max()
                    ax.text(i, y_val * 1.05 if "収量" in mode else y_val + 0.05, l, ha='center', color='red', weight='bold', size=15)
                st.pyplot(fig)

            # ==========================================
            # B: 回帰解析モード (予測曲線と数式)
            # ==========================================
            else:
                is_num = np.issubdtype(df[fx].dtype, np.number)
                if not is_num:
                    st.warning(f"⚠️ {fx} は数値データではないため、回帰曲線は描画できません。数値の列を選択してください。")
                    st.stop()
                    
                if "収量" in mode:
                    model = smf.ols(f'Q("{target}") ~ Q("{fx}")', data=df).fit()
                    st.subheader("1. 予測モデル (直線回帰)")
                    st.info(f"📝 式: **Y = {model.params[1]:.3f}X + {model.params[0]:.3f}** (R²={model.rsquared:.3f})")
                    sns.regplot(x=fx, y=target, data=df, ax=ax, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    
                elif "発芽率" in mode:
                    df['not_sp'] = df[tt_col] - df[sp_col]
                    model = smf.glm(f'Q("{sp_col}") + not_sp ~ Q("{fx}")', data=df, family=sm.families.Binomial()).fit()
                    a, b = model.params[1], model.params[0]
                    st.subheader("1. 予測モデル (ロジスティック回帰)")
                    st.info(f"📝 式: **P = 1 / (1 + exp(-({a:.3f}X + {b:.3f})))**")
                    x_range = np.linspace(df[fx].min(), df[fx].max(), 100)
                    ax.plot(x_range, 1 / (1 + np.exp(-(a * x_range + b))), color='red', lw=3)
                    ax.scatter(df[fx], df[target], alpha=0.5)
                    
                else:
                    model = smf.glm(f'Q("{target}") ~ Q("{fx}")', data=df, family=sm.families.Poisson()).fit()
                    a, b = model.params[1], model.params[0]
                    st.subheader("1. 予測モデル (ポアソン回帰)")
                    st.info(f"📝 式: **Y = exp({a:.3f}X + {b:.3f})**")
                    x_range = np.linspace(df[fx].min(), df[fx].max(), 100)
                    ax.plot(x_range, np.np.exp(a * x_range + b), color='red', lw=3)
                    ax.scatter(df[fx], df[target], alpha=0.5)

                st.write("### 2. モデルの詳細要約")
                with st.expander("詳細を表示"):
                    st.text(model.summary())
                
                ax.set_title(f"{mode.split()[1]} の回帰曲線")
                ax.set_xlabel(fx); ax.set_ylabel("比率" if "発芽率" in mode else target)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"解析エラー: {e}")
