import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Pseudo-code to C++ Converter",
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
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned GPT-2 model and tokenizer from local files"""
    try:
        # Show loading status
        with st.status("🚀 Loading your fine-tuned AI model...", expanded=True) as status:
            st.write("📥 Initializing tokenizer...")
            
            # Load tokenizer from your local files
            tokenizer = GPT2Tokenizer.from_pretrained(".")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
            st.write("🤖 Loading fine-tuned GPT-2 model...")
            
            # Load your fine-tuned model from local files
            model = GPT2LMHeadModel.from_pretrained(".")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            st.write(f"⚡ Moving model to: {device}")
            model.to(device)
            model.eval()
            
            st.write("✅ Model loaded successfully!")
            status.update(label="AI Model Ready!", state="complete")
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Make sure all model files are in the same directory as app.py")
        return None, None, None

def generate_code(pseudo_code, model, tokenizer, device, max_new_tokens=128, temperature=0.7):
    """Generate Python code from pseudo-code using the fine-tuned model"""
    try:
        # Create prompt in the exact format the model was trained on
        prompt = f"### PSEUDOCODE:\n{pseudo_code.strip()}\n### CODE:"
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate output with the same parameters used during training
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
        
        # Remove any subsequent pseudo-code prompts that might be generated
        generated_code = generated_code.split("### PSEUDOCODE:")[0].strip()
        
        return generated_code
    except Exception as e:
        return f"Error generating code: {e}"

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Pseudo-code to Python Code Generator</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Powered by Fine-tuned GPT-2 • Trained on SPOC Dataset</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Generation Settings")
        
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=256, value=128, 
                              help="Maximum length of generated code")
        temperature = st.slider("Creativity", min_value=0.1, max_value=1.0, value=0.7, 
                               help="Higher = more creative, Lower = more deterministic")
        
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
        st.markdown("""
        This AI model was fine-tuned on the SPOC dataset to convert 
        pseudo-code into working Python code.
        
        **Model:** GPT-2 Fine-tuned  
        **Training Data:** SPOC Dataset  
        **Purpose:** Educational Code Generation
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Input Pseudo-code</div>', unsafe_allow_html=True)
        
        # Text area for pseudo-code input
        pseudo_code = st.text_area(
            "Enter your pseudo-code:",
            height=200,
            placeholder="Example: \nCreate a function that takes two numbers and returns their sum...\n\nOr: \nIF number is divisible by 2 THEN print 'Even' ELSE print 'Odd'",
            value=st.session_state.get('pseudo_code', ''),
            key="pseudo_input"
        )
        
        # Generate button
        col1_1, col1_2 = st.columns([2, 1])
        with col1_1:
            generate_btn = st.button("🚀 Generate Python Code", type="primary", use_container_width=True)
        with col1_2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.pseudo_code = ""
                st.rerun()
        
        # Info box
        with st.expander("💡 Tips for Better Results", expanded=True):
            st.markdown("""
            **Best Practices:**
            - Use clear, step-by-step instructions
            - Specify inputs and outputs explicitly
            - Use common programming terms (if, for, while, function, return)
            - Be specific about variable names and operations
            
            **Example Format:**
            ```
            FUNCTION add_numbers(a, b):
                SET result = a + b
                RETURN result
            ```
            
            **Or:**
            ```
            FOR each number in list:
                IF number is even:
                    PRINT number
            ```
            """)
    
    with col2:
        st.markdown('<div class="sub-header">🐍 Generated Python Code</div>', unsafe_allow_html=True)
        
        if generate_btn and pseudo_code:
            # Load model if not already loaded
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
            
            # Generate code
            with st.spinner("🤖 AI is generating Python code..."):
                generated_code = generate_code(
                    pseudo_code, 
                    model, 
                    tokenizer, 
                    device, 
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
            
            # Display generated code in a nice box
            st.markdown("### ✅ Generated Code:")
            st.markdown('<div class="generated-code">', unsafe_allow_html=True)
            st.code(generated_code, language='python')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Copy functionality
            st.code(generated_code, language='python')
            
            # Success message
            st.markdown('<div class="success-message">🎉 Code generated successfully! You can copy the code above.</div>', unsafe_allow_html=True)
                
        elif not pseudo_code and generate_btn:
            st.warning("⚠️ Please enter some pseudo-code first!")
        else:
            # Placeholder with instructions
            st.info("👆 Enter pseudo-code on the left and click 'Generate Python Code' to see the AI-generated Python code here.")
            
            # Show sample output
            with st.expander("📋 See Sample Output", expanded=False):
                st.markdown("**Input Pseudo-code:**")
                st.code("Create a function to check if a number is prime", language='text')
                
                st.markdown("**Expected Output:**")
                st.code("""def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True""", language='python')
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
            🚀 Built with <strong>Streamlit</strong> & <strong>Hugging Face Transformers</strong> | 
            🤖 <strong>Fine-tuned GPT-2</strong> model trained on SPOC dataset |
            💡 Educational AI Tool for Code Generation
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

