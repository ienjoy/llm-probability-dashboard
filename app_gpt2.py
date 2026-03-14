import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import platform

# --- 字体配置（虽然场景换成英文，但 UI 标签仍保留中文，确保兼容） ---
if platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
elif platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 页面配置 ---
st.set_page_config(page_title="AI 概率纠偏看板 (GPT-2版)", layout="wide")

st.title("🚀 LLM 概率修正演示 (GPT-2 引擎)")
st.markdown("""
由于切换至轻量化 **GPT-2** 模型，本系统已自动转为**英文逻辑推理**（GPT-2 对中文支持较弱）。
您将观察到 **Evidence d (证据)** 如何在英文语境下重塑 Token 的分布。
""")

# --- 加载模型 (GPT-2) ---
@st.cache_resource
def load_model():
    model_name = "gpt2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# --- 核心函数：适配 GPT-2 的空格敏感型概率抓取 ---
def get_probs(prompt, target_words):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
    
    res = []
    for word in target_words:
        # GPT-2 核心技巧：必须加上前导空格，因为在句子中间的单词通常带空格
        # 比如 "is Ibuprofen" 中的 Ibuprofen Token 实际是 " Ibuprofen"
        token_id = tokenizer.encode(" " + word.strip())[0]
        res.append(probs[token_id].item())
    return res

# --- 侧边栏 ---
industry = st.sidebar.selectbox(
    "选择应用场景",
    ["医疗健康 (Medical)", "法律合规 (Legal)", "教育咨询 (Education)"]
)

# --- 场景数据配置 (全面回归英文以适配 GPT-2) ---
config = {
    "医疗健康 (Medical)": {
        "base_prompt": "Patient has a headache and fever. The most likely medicine is",
        "evidence": "Medical Manual: For viral headache with fever, Ibuprofen is the primary choice.",
        "targets": ["Paris", "Ibuprofen", "Surgery", "Water"],
        "labels": ["Paris(Noise)", "Ibuprofen(Target)", "Surgery(Aggressive)", "Water(Neutral)"]
    },
    "法律合规 (Legal)": {
        "base_prompt": "According to the new safety statute, the company",
        "evidence": "Statute 402: Under high-risk conditions, the company SHALL be held liable.",
        "targets": ["may", "shall", "maybe", "not"],
        "labels": ["May(Vague)", "Shall(Precise)", "Maybe(Uncertain)", "Not(Negative)"]
    },
    "教育咨询 (Education)": {
        "base_prompt": "Explain gravity to a 5-year-old. It's like a magical invisible",
        "evidence": "Teaching Guide: Use simple analogies. Gravity is like magic invisible glue.",
        "targets": ["force", "glue", "mass", "theory"],
        "labels": ["Force(Complex)", "Glue(Analogy)", "Mass(Abstract)", "Theory(Formal)"]
    }
}

data = config[industry]

# --- 页面布局 ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🛠️ 配置提示词 (Prompt)")
    base_text = st.text_area("原始 Prompt:", data["base_prompt"], height=100)
    
    use_evidence = st.toggle("🧪 注入行业证据 (Evidence d)", value=False)
    
    if use_evidence:
        # GPT-2 需要更清晰的上下文分隔，所以加了 Context: 前缀
        final_prompt = "Context: " + data["evidence"] + "\n\nQuestion: " + base_text
        st.info(f"**已注入证据:** \n{data['evidence']}")
    else:
        final_prompt = base_text
        st.warning("原生权重状态（盲猜中）")
    
    st.markdown("---")
    st.write("**最终发送给模型的大脑信号:**")
    st.code(final_prompt)

# --- 计算与绘图 ---
current_probs = get_probs(final_prompt, data["targets"])

with col2:
    st.subheader("📊 实时概率分布")
    
    fig, ax = plt.subplots()
    colors = ['#1E90FF' if use_evidence else '#D3D3D3'] * len(data['targets'])
    
    bars = ax.bar(data['labels'], current_probs, color=colors)
    
    # GPT-2 的概率比较分散，我们将 Y 轴设为 0.5 更有视觉冲击力
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Probability (0-1)")
    ax.bar_label(bars, fmt='%.3f', padding=3)
    
    st.pyplot(fig)

st.divider()
if use_evidence:
    st.success("**结论:** GPT-2 虽然规模小，但在接收到 Context 证据后，对应 Token 的概率依然发生了显著偏移，验证了 RAG 的有效性。")
else:
    st.info("提示：开启开关，观察 GPT-2 如何在证据指导下放弃‘噪声’转向‘专业词汇’。")