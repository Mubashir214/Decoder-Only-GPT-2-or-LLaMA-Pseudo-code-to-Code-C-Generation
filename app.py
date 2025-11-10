import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Pseudo-code to C++ Converter",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; color: #ff7f0e; margin-bottom: 1rem; }
    .generated-code { background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; font-family: 'Courier New', monospace; white-space: pre-wrap; }
    .info-box { background-color: #e8f4fd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    .success-message { background-color: #d4edda; color: #155724; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #c3e6cb; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned GPT-2 model and tokenizer"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(".")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = GPT2LMHeadModel.from_pretrained(".")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_code(pseudo_code, model, tokenizer, device, max_new_tokens=128, temperature=0.7):
    """Generate C++ code from pseudo-code"""
    try:
        prompt = f"### PSEUDOCODE:\n{pseudo_code.strip()}\n### C++ CODE:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### C++ CODE:" in generated_text:
            generated_code = generated_text.split("### C++ CODE:")[-1].strip()
        else:
            generated_code = generated_text.replace(prompt, "").strip()
        generated_code = generated_code.split("### PSEUDOCODE:")[0].strip()
        return generated_code
    except Exception as e:
        return f"Error generating code: {e}"

def main():
    st.markdown('<div class="main-header">🧠 Pseudo-code to C++ Code Generator</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Powered by GPT-2 • Fine-tuned on SPOC Dataset</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Generation Settings")
        max_tokens = st.slider("Max Tokens", 50, 256, 128)
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7)
        st.markdown("---")
        st.markdown("### 📝 Example Pseudo-code")
        example_pseudo = st.selectbox(
            "Try these examples:",
            [
                "Create a function to add two numbers",
                "Check if a number is prime",
                "Sort a list using bubble sort",
                "Calculate factorial of a number",
                "Find the maximum number in a list",
                "Reverse a string",
                "Count vowels in a string",
                "Check if a string is palindrome"
            ]
        )
        if st.button("📥 Load Example"):
            st.session_state.pseudo_code = example_pseudo
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("This AI model converts pseudo-code into working C++ code using GPT-2 fine-tuned on SPOC.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="sub-header">📝 Input Pseudo-code</div>', unsafe_allow_html=True)
        pseudo_code = st.text_area("Enter your pseudo-code:", height=200,
                                   value=st.session_state.get('pseudo_code', ''), key="pseudo_input")
        col1_1, col1_2 = st.columns([2, 1])
        with col1_1:
            generate_btn = st.button("🚀 Generate C++ Code", type="primary", use_container_width=True)
        with col1_2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.pseudo_code = ""
                st.rerun()

    with col2:
        st.markdown('<div class="sub-header">💻 Generated C++ Code</div>', unsafe_allow_html=True)
        if generate_btn and pseudo_code:
            if 'model' not in st.session_state:
                model, tokenizer, device = load_model()
                if model is None:
                    return
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
            else:
                model = st.session_state.model
                tokenizer = st.session_state.tokenizer
                device = st.session_state.device

            with st.spinner("🤖 Generating C++ code..."):
                generated_code = generate_code(
                    pseudo_code, model, tokenizer, device,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )

            st.markdown("### ✅ Generated Code:")
            st.markdown('<div class="generated-code">', unsafe_allow_html=True)
            st.code(generated_code, language='cpp')
            st.markdown('</div>', unsafe_allow_html=True)
        elif not pseudo_code and generate_btn:
            st.warning("⚠️ Please enter some pseudo-code first!")
        else:
            st.info("👆 Enter pseudo-code and click 'Generate C++ Code' to see the output.")

if __name__ == "__main__":
    main()
