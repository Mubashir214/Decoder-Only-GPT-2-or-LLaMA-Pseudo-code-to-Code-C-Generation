import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Pseudo-code to Python Converter",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .generated-code {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        font-family: 'Courier New', monospace;
        white-space: pre-wrap;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned GPT-2 model and tokenizer"""
    try:
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # For demo purposes, we'll use the base GPT-2 model
        # In production, you would load your fine-tuned model:
        # model = GPT2LMHeadModel.from_pretrained("./gpt2-pseudocode-final")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_code(pseudo_code, model, tokenizer, device, max_new_tokens=128, temperature=0.7):
    """Generate Python code from pseudo-code"""
    try:
        # Create prompt in the format the model was trained on
        prompt = f"### PSEUDOCODE:\n{pseudo_code.strip()}\n### CODE:\n"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate output
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
        
        # Decode and clean output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the code part after "### CODE:"
        if "### CODE:" in generated_text:
            generated_code = generated_text.split("### CODE:")[-1].strip()
        else:
            generated_code = generated_text.replace(prompt, "").strip()
        
        # Remove any subsequent pseudo-code prompts
        generated_code = generated_code.split("### PSEUDOCODE:")[0].strip()
        
        return generated_code
    except Exception as e:
        return f"Error generating code: {e}"

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Pseudo-code to Python Code Generator</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=256, value=128, help="Maximum length of generated code")
        temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, help="Higher values = more creative, Lower values = more deterministic")
        
        st.markdown("---")
        st.markdown("### 📝 Example Pseudo-code")
        example_pseudo = st.selectbox(
            "Try these examples:",
            [
                "Create a function to add two numbers",
                "Check if a number is prime",
                "Sort a list using bubble sort",
                "Calculate factorial of a number",
                "Find the maximum number in a list"
            ]
        )
        
        if st.button("Load Example"):
            st.session_state.pseudo_code = example_pseudo
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Input Pseudo-code</div>', unsafe_allow_html=True)
        
        # Text area for pseudo-code input
        pseudo_code = st.text_area(
            "Enter your pseudo-code:",
            height=200,
            placeholder="Example: Create a function that takes two numbers and returns their sum...",
            value=st.session_state.get('pseudo_code', '')
        )
        
        # Generate button
        generate_btn = st.button("🚀 Generate Python Code", type="primary", use_container_width=True)
        
        # Info box
        with st.expander("💡 Tips for better results"):
            st.markdown("""
            - **Be specific and clear** in your pseudo-code
            - **Use common programming terms** (if, for, while, function, etc.)
            - **Describe the inputs and outputs** clearly
            - **Break complex logic** into simple steps
            - **Example format:**
              ```
              FUNCTION add_numbers(a, b):
                  RETURN a + b
              ```
            """)
    
    with col2:
        st.markdown('<div class="sub-header">🐍 Generated Python Code</div>', unsafe_allow_html=True)
        
        if generate_btn and pseudo_code:
            with st.spinner("Generating Python code..."):
                # Load model if not already loaded
                if 'model' not in st.session_state:
                    with st.status("Loading AI model...", expanded=True) as status:
                        st.write("Initializing tokenizer...")
                        model, tokenizer, device = load_model()
                        if model and tokenizer:
                            st.session_state.model = model
                            st.session_state.tokenizer = tokenizer
                            st.session_state.device = device
                            st.write("Model loaded successfully!")
                            status.update(label="Model loaded!", state="complete")
                        else:
                            st.error("Failed to load model")
                            return
                else:
                    model = st.session_state.model
                    tokenizer = st.session_state.tokenizer
                    device = st.session_state.device
                
                # Generate code
                generated_code = generate_code(
                    pseudo_code, 
                    model, 
                    tokenizer, 
                    device, 
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Display generated code
                st.markdown('<div class="generated-code">', unsafe_allow_html=True)
                st.code(generated_code, language='python')
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Copy to clipboard button
                st.code(generated_code, language='python')
                
        elif not pseudo_code and generate_btn:
            st.warning("Please enter some pseudo-code first!")
        else:
            # Placeholder
            st.info("Enter pseudo-code and click 'Generate Python Code' to see the result here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with ❤️ using Streamlit & Hugging Face Transformers | 
            Fine-tuned GPT-2 model for pseudo-code to Python conversion
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()