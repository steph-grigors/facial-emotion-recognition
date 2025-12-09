"""
Facial Emotion Recognition - Streamlit Dark Theme

A sleek, modern dark-themed interface for real-time emotion detection.
Features automatic face detection and image preprocessing.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import io
from pathlib import Path

from src.inference import create_predictor

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HERO IMAGE (Demo Screenshot)
# ============================================================================

# Display demo image at the top
demo_image_path = Path("assets/dataset-cover.png")
if demo_image_path.exists():
    col1, col2, col3 = st.columns([0.333, 0.333, 0.333]) 
    with col2:
        st.image(str(demo_image_path))

# ============================================================================
# CUSTOM CSS - BEAUTIFUL DARK THEME
# ============================================================================

st.markdown("""
<style>
    /* Import sleek font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global dark theme */
    .stApp {
        background: #0a0e27;
        color: #e2e8f0;
    }
    
    * {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Main container */
    .main {
        background: #0a0e27;
    }
    
    /* Headers with gradient */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }
    
    h2 {
        color: #cbd5e0;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #a0aec0;
        font-weight: 500;
        font-size: 1.2rem !important;
    }
    
    /* Dark cards */
    .stApp > div > div > div > div {
        background: rgba(26, 32, 53, 0.8);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
    }
    
    /* Gradient buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5);
        text-transform: uppercase;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.7);
    }
    
    /* File uploader - dark */
    .stFileUploader {
        background: rgba(26, 32, 53, 0.6);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Radio buttons - make more visible and attractive */
    .stRadio > label {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }
    
    .stRadio > div {
        background: rgba(102, 126, 234, 0.15);
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .stRadio > div > label {
        background: rgba(26, 32, 53, 0.8);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        margin: 0.25rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stRadio > div > label:hover {
        background: rgba(102, 126, 234, 0.3);
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        background-color: #667eea !important;
    }
    
    /* Camera input */
    .stCamera {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Metrics - gradient cards */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Sidebar - dark gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2035 0%, #0a0e27 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #cbd5e0 !important;
        border-bottom: 2px solid #667eea;
    }
    
    /* Success/Warning boxes - dark */
    .stSuccess {
        background: rgba(72, 187, 120, 0.15);
        color: #68d391;
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stWarning {
        background: rgba(237, 137, 54, 0.15);
        color: #f6ad55;
        border: 1px solid rgba(237, 137, 54, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stInfo {
        background: rgba(66, 153, 225, 0.15);
        color: #63b3ed;
        border: 1px solid rgba(66, 153, 225, 0.3);
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Expander - dark */
    .streamlit-expanderHeader {
        background: rgba(26, 32, 53, 0.8);
        border-radius: 8px;
        font-weight: 600;
        color: #cbd5e0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .streamlit-expanderContent {
        background: rgba(26, 32, 53, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-top: none;
    }
    
    /* Text and labels */
    p, label, .stMarkdown {
        color: #cbd5e0;
    }
    
    /* Links */
    a {
        color: #667eea;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'models/final/best_model.pth'
MODEL_TYPE = 'enhanced'

EMOTION_EMOJIS = {
    'angry': '😠',
    'disgust': '🤢', 
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

EMOTION_COLORS = {
    'angry': '#ef4444',
    'disgust': '#22c55e',
    'fear': '#a855f7',
    'happy': '#fbbf24',
    'neutral': '#6b7280',
    'sad': '#3b82f6',
    'surprise': '#f97316'
}

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def resize_image_for_display(image, max_width=400):
    """Resize image for display while maintaining aspect ratio and quality."""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return image.resize((max_width, new_height), Image.LANCZOS)
    return image

def preprocess_for_model(image):
    """
    Preprocess image with face detection and enhancement.
    Returns processed PIL image and face detection status.
    """
    # Convert to numpy
    img_np = np.array(image)
    
    # Convert to RGB if needed
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_found = len(faces) > 0
    
    if face_found:
        # Get largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Add margin
        margin = int(0.3 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_np.shape[1], x + w + margin)
        y2 = min(img_np.shape[0], y + h + margin)
        
        # Crop face
        face_img = img_np[y1:y2, x1:x2]
    else:
        # Use center crop as fallback
        h, w = img_np.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        face_img = img_np[start_h:start_h+size, start_w:start_w+size]
    
    # Convert back to PIL
    face_pil = Image.fromarray(face_img)
    
    # Enhance
    enhancer = ImageEnhance.Contrast(face_pil)
    enhanced_image = enhancer.enhance(1.2)
    
    return enhanced_image, face_found

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_emotion_chart(probs, class_names, prediction):
    """Create beautiful dark-themed emotion chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dark theme
    fig.patch.set_facecolor('#0a0e27')
    ax.set_facecolor('#1a2035')
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    sorted_names = [class_names[i] for i in sorted_indices]
    
    colors = [EMOTION_COLORS.get(name, '#6b7280') for name in sorted_names]
    
    bars = ax.barh(sorted_names, sorted_probs, color=colors, alpha=0.9, 
                   edgecolor='#667eea', linewidth=2)
    
    # Add emoji labels
    for i, (name, prob) in enumerate(zip(sorted_names, sorted_probs)):
        emoji = EMOTION_EMOJIS.get(name, '')
        ax.text(prob + 2, i, f'{emoji} {prob:.1f}%', 
                va='center', fontsize=13, fontweight='700', color='#e2e8f0')
    
    ax.set_xlabel('Confidence (%)', fontsize=14, fontweight='600', color='#cbd5e0')
    ax.set_title('Emotion Probability Distribution', 
                 fontsize=16, fontweight='700', pad=20, color='#e2e8f0')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.15, linestyle='--', color='#667eea')
    ax.tick_params(colors='#a0aec0', labelsize=12)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load model with caching."""
    try:
        predictor = create_predictor(MODEL_PATH, model_type=MODEL_TYPE)
        return predictor
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

predictor = load_model()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Animated header with gradient
    st.markdown("""
    <h1 style='text-align: center; color: #a0aec0; font-size: 0.8rem; margin-top: 0.5rem;'>
        Deep Learning Powered<br>
        Facial Emotion Recognition
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 About")
        st.markdown("""
        Advanced emotion recognition using **ResNet50** architecture, 
        trained on **35,000+ facial expressions** from FER2013.
        """)
        
        st.markdown("---")
        
        st.markdown("## 💡 Best Practices")
        st.markdown("""
        ✅ Well-lit, clear photos<br>
        ✅ Face directly facing camera<br>
        ✅ Exaggerated expressions work better<br>
        ✅ Minimal background clutter
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("## 🛠️ Tech Stack")
        st.markdown("""
        ```
        PyTorch 2.0+
        ResNet50
        OpenCV
        Streamlit
        ``` 
        """)
    
    # Main content - two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("## 📸 Input")
        
        # Input method
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload Image", "📷 Take Photo"],
            horizontal=True
        )
        
        image = None
        
        if input_method == "📁 Upload Image":
            uploaded_file = st.file_uploader(
                "Drop an image or click to browse",
                type=['jpg', 'jpeg', 'png', 'webp'],
                help="Upload a photo with clear facial expression"
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
        else:
            camera_image = st.camera_input("Take a photo")
            if camera_image:
                image = Image.open(camera_image)
        
        if image:
            # Resize for display
            original_display = resize_image_for_display(image, max_width=500)
            st.image(original_display, caption="📷 Original Image")
            
            # Big analyze button
            if st.button("🔍 ANALYZE EMOTION", key="analyze_btn"):
                with st.spinner("🧠 Processing with AI..."):
                    # Preprocess
                    processed_image, face_found = preprocess_for_model(image)
                    
                    # Predict
                    prediction, confidence, probs = predictor.predict(processed_image)
                    
                    # Store results
                    st.session_state.prediction = prediction
                    st.session_state.confidence = confidence
                    st.session_state.probs = probs
                    st.session_state.processed_image = processed_image
                    st.session_state.face_found = face_found
                    
                    st.rerun()
    
    with col2:
        st.markdown("## 📊 Analysis")
        
        if 'prediction' in st.session_state:
            # Status
            if st.session_state.face_found:
                st.success("✅ Face detected and processed")
            else:
                st.warning("⚠️ No face detected - using center crop")
            
            # Show processed image - resized for display
            processed_display = resize_image_for_display(st.session_state.processed_image, max_width=400)
            st.image(
                processed_display,
                caption="🔧 Processed Image (Face-cropped & Enhanced)"
            )
            
            
            # Metrics - updated with smaller font size
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"<p style='font-size: 0.8rem; margin: 0;'><strong>Accuracy</strong><br>{st.session_state.confidence:.1f}%</p>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<p style='font-size: 0.8rem; margin: 0;'><strong>Model</strong><br>{MODEL_TYPE.upper()}</p>", unsafe_allow_html=True)
            with col_c:
                device_icon = "⚡" if "cuda" in str(predictor.device) else "🖥️"
                device_text = f"{device_icon} GPU" if "cuda" in str(predictor.device) else "🖥️ CPU"
                st.markdown(f"<p style='font-size: 0.8rem; margin: 0;'><strong>Device</strong><br>{device_text}</p>", unsafe_allow_html=True)
            
            # Chart
            st.markdown("### 📈 Probability Distribution")
            fig = create_emotion_chart(
                st.session_state.probs,
                predictor.class_names,
                st.session_state.prediction
            )
            st.pyplot(fig)
        
        else:
            st.info("👆 Upload an image or take a photo to begin")
            
            st.markdown("""
            ### 🎯 How It Works
            
            1. **Upload/Capture** 📸  
               Provide an image with facial expression
            
            2. **Face Detection** 🎯  
               AI automatically finds and crops the face
            
            3. **Preprocessing** 🔧  
               Image is enhanced for optimal accuracy
            
            4. **Analysis** 🧠  
               ResNet50 model predicts emotion
            
            5. **Results** 📊  
               View probabilities for all 7 emotions
            """)
    
    # ========================================
    # BOTTOM SECTION - Developer Info
    # ========================================
    st.divider()

    st.markdown("""
    <div style='text-align: center; padding: 0.8rem 0;'>
        <h3>👨‍💻 Developed by Stéphan Grigorescu</h3>
        <p style='color: #666;'>
            Data Scientist & AI Engineer | Building intelligent solutions with Python & AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Social Links (centered)
    social_col1, social_col2, social_col3, social_col4 = st.columns([1, 1, 1, 1])

    with social_col1:
        st.link_button("🌐 Portfolio", "https://www.stephan-gs.work", use_container_width=True)
    with social_col2:
        st.link_button("💼 LinkedIn", "https://linkedin.com/in/stéphan-grs", use_container_width=True)
    with social_col3:
        st.link_button("🐙 GitHub", "https://github.com/steph-grigors", use_container_width=True)
    with social_col4:
        st.link_button("📧 Contact", "mailto:stephan.grigorescu@gmail.com", use_container_width=True)

    # Disclaimer (Collapsed)
    with st.expander("⚠️ Disclaimer"):
        st.caption("""
        This application is provided as-is for educational and demonstration purposes.
        Emotion predictions are generated by a deep learning model trained on a rather small sample of facial expressions
        and may therefore not always be accurate.
        """)

if __name__ == "__main__":
    if predictor:
        main()
    else:
        st.error("❌ Failed to load model. Check MODEL_PATH in config.")