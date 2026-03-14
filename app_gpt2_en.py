import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="LLM Bayesian Probability Visualizer", layout="wide")

st.title("🚀 Bayesian Inference in LLMs: Visualizing Probability Shifts")
st.markdown("""
### The Core Logic: Prompts as Bayesian Evidence
In the world of Large Language Models, a **Prompt** isn't just a question—it is **Evidence ($d$)** that updates the model's **Prior Knowledge ($W$)** to produce a **Posterior Probability Distribution** over the vocabulary.
""")

# --- Model Loading (Optimized for GPT-2) ---
@st.cache_resource
def load_model():
    model_name = "gpt2" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# --- Core Logic: Extracting Token Probabilities ---
def get_probs(prompt, target_words):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
    
    res = []
    for word in target_words:
        # GPT-2 specific: target tokens usually require a leading space in context
        token_id = tokenizer.encode(" " + word.strip())[0]
        res.append(probs[token_id].item())
    return res

# --- Sidebar: Domain Selection ---
st.sidebar.header("Select Application Domain")
industry = st.sidebar.selectbox(
    "Choose a scenario:",
    ["Healthcare (Diagnostic Bias)", "Legal (Term Precision)", "Education (Adaptive Analogy)"]
)

# --- Scenario Configuration (English) ---
config = {
    "Healthcare (Diagnostic Bias)": {
        "base_prompt": "The patient has a headache and fever. The recommended drug is",
        "evidence": "Medical Protocol: For viral symptoms, Ibuprofen is the first-line treatment.",
        "targets": ["Paris", "Ibuprofen", "Surgery", "Rest"],
        "labels": ["Paris (Noise)", "Ibuprofen (Target)", "Surgery (Aggressive)", "Rest (General)"]
    },
    "Legal (Term Precision)": {
        "base_prompt": "According to the safety clause, the contractor",
        "evidence": "Statute 402: Under high-risk conditions, the contractor SHALL be liable.",
        "targets": ["may", "shall", "maybe", "not"],
        "labels": ["May (Vague)", "Shall (Precise)", "Maybe (Uncertain)", "Not (Negative)"]
    },
    "Education (Adaptive Analogy)": {
        "base_prompt": "Explain gravity to a toddler. It is like a magical invisible",
        "evidence": "Teaching Guide: Use childhood analogies. Gravity is like magic invisible glue.",
        "targets": ["force", "glue", "mass", "down"],
        "labels": ["Force (Academic)", "Glue (Analogy)", "Mass (Abstract)", "Down (Simple)"]
    }
}

data = config[industry]

# --- Main UI Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🛠️ Prompt Configuration")
    base_text = st.text_area("Base Prompt (The 'Query'):", data["base_prompt"], height=100)
    
    use_evidence = st.toggle("🧪 Inject Evidence (Evidence $d$)", value=False)
    
    if use_evidence:
        # Structuring the prompt for better Bayesian guidance
        final_prompt = f"Context: {data['evidence']}\n\nQuery: {base_text}"
        st.info(f"**Evidence Injected:** \n{data['evidence']}")
    else:
        final_prompt = base_text
        st.warning("Running on Pre-trained Priors only (Zero Evidence).")
    
    st.markdown("---")
    st.write("**Full Tensor Input (Final String):**")
    st.code(final_prompt)

# --- Computation & Visualization ---
current_probs = get_probs(final_prompt, data["targets"])

with col2:
    st.subheader("📊 Real-time Token Probability Distribution")
    
    fig, ax = plt.subplots()
    # Blue for evidence-based, Grey for base model
    colors = ['#1E90FF' if use_evidence else '#BDBDBD'] * len(data['targets'])
    
    bars = ax.bar(data['labels'], current_probs, color=colors)
    
    # Adjusting Y-axis for better visibility of shifts
    ax.set_ylim(0, 0.5) 
    ax.set_ylabel("Probability Score")
    ax.bar_label(bars, fmt='%.3f', padding=3)
    
    st.pyplot(fig)

# --- Technical Insight Section ---
st.divider()
with st.expander("🔍 The Mathematical Logic behind the Jump"):
    st.latex(r"P(y | x, d) = \frac{P(d | y, x) P(y | x)}{P(d | x)}")
    st.markdown("""
    - **Prior $P(y|x)$:** The pre-trained weights of GPT-2. Without evidence, the model relies on general web-crawl statistics (e.g., "Paris" might have a high prior in some contexts).
    - **Evidence Injected ($d$):** Your specific industry data or context.
    - **Posterior $P(y|x, d)$:** The new probability distribution you see in the bars. By injecting **Evidence**, we force the Attention mechanism to re-weight the hidden states, effectively "shifting" the model's focus toward professional or contextually accurate tokens.
    
    **Observation:** Notice how the Target token (e.g., 'Ibuprofen' or 'Shall') jumps from near-zero to a dominant position once the Evidence switch is flipped.
    """)

st.success("This demonstrates that LLMs are not just 'stochastic parrots,' but dynamic Bayesian engines that can be steered by context.")