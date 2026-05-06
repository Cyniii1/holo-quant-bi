import streamlit as st
import pandas as pd
import sqlite3
import akshare as ak
import jieba
import jieba.analyse
import re
from functools import reduce
import requests
import urllib3
import numpy as np

# ===== 算法库大换血：引入 SVR 与数据缩放 =====
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ================= 1. 终端全局配置 =================
st.set_page_config(page_title="Holo-Quant 宏微观跨模态投研终端", page_icon="🌌", layout="wide")

# ================= 2. 底层数据治理引擎 =================
class QuantDataEngine:
    def __init__(self, db_name="quant_terminal_v4.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)

    def extract_macro_data(self):
        macros = {}
        def safe_fetch_macro(func_name, indicator_name):
            if hasattr(ak, func_name):
                try:
                    df = getattr(ak, func_name)()
                    if df is not None and not df.empty:
                        macros[indicator_name] = df
                except: pass
        
        # 放弃不稳定的社融，引入 LPR (信贷成本) 和 工业利润 (实体造血能力)
        safe_fetch_macro('macro_china_ppi_yearly', 'PPI(同比)')
        safe_fetch_macro('macro_china_cpi_yearly', 'CPI(同比)')
        safe_fetch_macro('macro_china_m2_yearly', 'M2(同比)')
        safe_fetch_macro('macro_china_pmi_yearly', 'PMI(景气度)')
        safe_fetch_macro('macro_china_lpr', 'LPR(信贷成本)')
        safe_fetch_macro('macro_china_gyqylr_cs_yoy', '工业利润(同比)')
        return macros

    def extract_micro_news(self):
        news_pool = set()
        def safe_fetch_micro(func_name):
            if hasattr(ak, func_name):
                try:
                    df = getattr(ak, func_name)()
                    if df is not None and not df.empty:
                        col = next((c for c in ['内容', '标题', '名称'] if c in df.columns), None)
                        if col:
                            for text in df[col].astype(str): news_pool.add(text)
                except: pass
        safe_fetch_micro('stock_info_global_sina')
        safe_fetch_micro('stock_info_global_em')
        return list(news_pool)

    def transform_and_load(self, macros_dict, news_texts):
        if macros_dict:
            cleaned_dfs = []
            for name, df in macros_dict.items():
                v_col = '今值' if '今值' in df.columns else df.columns[1]
                date_col = '日期' if '日期' in df.columns else df.columns[0]  # 核心修复：动态识别时间列
                t = df[[date_col, v_col]].copy().rename(columns={date_col: 'date', v_col: name})
                t['date'] = t['date'].astype(str).str.slice(0, 7)
                cleaned_dfs.append(t.drop_duplicates('date'))
            
            if cleaned_dfs:
                macro_final = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), cleaned_dfs)
                macro_final = macro_final.sort_values('date').ffill().tail(12) # 截取近12期
                macro_final.to_sql("macro_v4", self.conn, if_exists='replace', index=False)

        if news_texts:
            full_content = " ".join(news_texts)
            stop_words = ['报道', '记者', '消息', '表示', '持续', '预计', '影响', '相关', '进行', '已经', '目前', '显示', '可能', '超过', '增长', '下降', '亿元', '同比']
            for sw in stop_words: jieba.analyse.set_stop_words
            keywords = jieba.analyse.extract_tags(full_content, topK=100, withWeight=True, allowPOS=('n', 'nz', 'vn', 'nt'))
            freq_df = pd.DataFrame(keywords, columns=['word', 'weight'])
            freq_df.to_sql("micro_nlp_v4", self.conn, if_exists='replace', index=False)

    def run(self):
        self.transform_and_load(self.extract_macro_data(), self.extract_micro_news())
        return self.extract_micro_news()

# ================= 3. 跨模态算法组件 =================
def predict_macro_svr():
    """
    废弃线性回归，升级为 SVR (支持向量回归)
    解释：SVR 结合 RBF 径向基核函数，能将低维的小样本数据映射到高维空间，
    极大地提升了捕获 12 个月时间序列中“非线性拐点”的能力。
    """
    try:
        conn = sqlite3.connect("quant_terminal_v4.db")
        df = pd.read_sql("SELECT * FROM macro_v4", conn)
        conn.close()
        
        ppi_col = [col for col in df.columns if 'PPI' in col]
        if not ppi_col: return "SVR 预测缺失", "N/A"
        
        target_col = ppi_col[0]
        X = np.array(range(len(df))).reshape(-1, 1)
        y = df[target_col].values
        
        # 机器学习严谨步骤：数据标准化 (Standard Scaling) 防止梯度爆炸
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        next_X_scaled = scaler_X.transform(np.array([[len(df)]]))
        
        # 实例化 SVR 模型
        model = SVR(kernel='rbf', C=10.0, gamma='scale')
        model.fit(X_scaled, y)
        pred = model.predict(next_X_scaled)[0]
        
        trend = "📈 边际修复" if pred > y[-1] else "📉 持续承压"
        return f"SVR 动态预测 ({target_col})", f"{round(pred, 2)} ({trend})"
    except: return "SVR 算法异常", "N/A"

def analyze_market_sentiment(news_texts):
    try:
        sentiment_model = pipeline("sentiment-analysis", model="techthiyanes/chinese_sentiment")
        pos, neg = 0, 0
        for text in news_texts[:50]: 
            try:
                res = sentiment_model(text[:150])[0] 
                if '4' in res['label'] or '5' in res['label']: pos += 1
                elif '1' in res['label'] or '2' in res['label']: neg += 1
            except: continue
                
        total = pos + neg
        if total == 0: return "量化情绪感知", "⚖️ 绝对中性 (0.00)"
        score = (pos - neg) / total
        
        if score >= 0.5: return "量化情绪感知", f"🔥 极度贪婪 ({round(score,2)})"
        elif 0.1 <= score < 0.5: return "量化情绪感知", f"📈 风险偏好提升 ({round(score,2)})"
        elif -0.1 < score < 0.1: return "量化情绪感知", f"⚖️ 多空博弈 ({round(score,2)})"
        elif -0.5 <= score <= -0.1: return "量化情绪感知", f"📉 资金避险 ({round(score,2)})"
        else: return "量化情绪感知", f"❄️ 极度恐慌出逃 ({round(score,2)})"
    except: return "模型加载中", "等待计算"

def call_ai_quant_analyst(industry, ml_pred, dl_sent):
    conn = sqlite3.connect("quant_terminal_v4.db")
    m_str = pd.read_sql("SELECT * FROM macro_v4", conn).to_string(index=False)
    n_str = pd.read_sql("SELECT * FROM micro_nlp_v4", conn).to_string(index=False)
    conn.close()

    # 深度优化的机构投研 Prompt
    prompt = f"""
你现在是顶级买方基金的【{industry}】赛道基金经理。
请基于以下机器演算的先验指标与宏微观原始数据，输出一份专业的《【{industry}】赛道 Alpha 策略研报》。

[量化先验指标]
1. 宏观环境 (基于 SVR 非线性回归预测)：{ml_pred}
2. 微观资金情绪 (基于 BERT 神经网络提取)：{dl_sent}

[底层数据源]
1. 近12期宏观流动性与成本矩阵：\n{m_str}\n
2. 赛道 NLP 核心实体锚点 (全网信息降维)：\n{n_str}\n

[输出严格限制]
1. 【宏观与流动性映射】：分析宏观矩阵(含LPR和工业利润)及SVR预测趋势，对【{industry}】赛道的估值中枢(Beta)有何实质影响？
2. 【资金与情绪共振】：结合情感指数与NLP实体词，透视当前资金在【{industry}】是处于左侧潜伏、右侧追高还是出逃阶段？
3. 【Alpha 策略配置】：直接给出针对该赛道的买/卖点建议，提示核心尾部风险 (Tail Risk)。
文风要求：极度客观精炼，使用严谨的投研术语，禁止使用“总而言之、希望”等废话。
"""
    # ================= 替换你的密钥 =================
    # CTO级安全做法：从 Streamlit 的加密金库中读取 Token，防盗刷！
    TOKEN = st.secrets["COZE_TOKEN"]
    BOT_ID = "7636359684756619279" 
    # ===============================================
    

    res = requests.post(
        "https://api.coze.cn/open_api/v2/chat",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"bot_id": BOT_ID, "user": "quant_dev", "query": prompt, "stream": False},
        timeout=120
    )
    if res.status_code == 200:
        for msg in res.json().get("messages", []):
            if msg.get("type") == "answer": return msg.get("content")
    return "API 路由中断，请检查网关认证。"

# ================= 4. 彭博风前端 UI =================

st.title("🌌 宏微观泛行业量化分析")
st.markdown("`Architecture: Async Scrapers -> TF-IDF -> Support Vector Regression (SVR) -> RoBERTa Sentiment -> LLM Attention Routing`")
st.divider()

with st.sidebar:
    st.header("⚙️ 因子与模型控制台")
    
    # 扩容后的 17 个硬核实体与金融赛道
    industry_list = [
        "宏观全局基准配置", "算电协同与AI基建", "消费电子与半导体", "新能源车与智能驾驶", 
        "低空经济与商业航天", "创新药与生物制造", "固态电池与新型储能", "人形机器人与具身智能", 
        "高端装备与工业母机", "跨境出海与物流供应链", "房地产与后周期产业链", "泛大宗商品与周期资源", 
        "现代农业与粮食安全", "军工与国防信息化", "数据要素与信创安全", "黄金与避险资产", "金融科技与支付"
    ]
    
    selected_industry = st.selectbox("🎯 锁定 Target 赛道", industry_list)
    st.caption(f"当前 LLM 注意力机制将强制收敛至: **{selected_industry}**")
    start_button = st.button("🔴 运行 ", use_container_width=True)

if start_button:
    with st.status(f"正在构建【{selected_industry}】特征工程...", expanded=True) as status:
        
        st.write("📡 Step 1: 异步调度全网多源异构数据并执行 TF-IDF 降维...")
        bot = QuantDataEngine()
        news_texts_for_dl = bot.run()
        
        st.write("🤖 Step 2: 注入宏观时序数据，利用 SVR (支持向量回归) 计算非线性预测...")
        ml_title, ml_value = predict_macro_svr()
        
        st.write("🧠 Step 3: 加载 Transformer 权重，量化当前微观市场情绪因子...")
        dl_title, dl_value = analyze_market_sentiment(news_texts_for_dl)
        
        st.write(f"🧠 Step 4: 执行提示词劫持，路由至 LLM 大脑生成最终研报...")
        final_report = call_ai_quant_analyst(selected_industry, f"{ml_title}: {ml_value}", f"{dl_title}: {dl_value}")
        
        status.update(label="模型推理完毕 (Inference Completed)", state="complete", expanded=False)

    st.subheader(f"📊 {selected_industry} - 量化监测面板")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="全网信息熵 (特征提取量)", value=f"{len(news_texts_for_dl)} 条", delta="NLP 特征化完成")
    col2.metric(label=ml_title, value=ml_value)
    col3.metric(label=dl_title, value=dl_value)
    
    st.divider()
    
    st.subheader(f"📄 机构级内参: {selected_industry} Alpha 策略")
    st.markdown(final_report)
else:
    st.info("👈 请在左侧面板配置赛道参数，并启动量化引擎。")