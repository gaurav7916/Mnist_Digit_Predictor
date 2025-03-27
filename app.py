import streamlit as st
import torch
from fastai.vision.all import *
import numpy as np
import plotly.express as px
import base64
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Set page configuration
st.set_page_config(page_title="MNIST Digit Predictor", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )

add_bg_from_local('wallpaper.jpg')

def load_model():
    """Load the trained MNIST model"""
    return load_learner('mnist_model.pkl')

def preprocess_image(image):
    """Preprocess the drawn image for prediction"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Invert colors (white background to black)
    image = ImageOps.invert(image)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Create a PILImage for FastAI compatibility
    return Image.fromarray((img_array * 255).astype(np.uint8))

def main():
    # Apply CSS styling
    apply_custom_styles()
    
    st.title('MNIST Digits Predictor')
    
    # Load the model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Initialize session state for canvas clear
    if 'clear_canvas' not in st.session_state:
        st.session_state.clear_canvas = False
    
    # Create a canvas for drawing
    st.write("Draw a digit using the cursor in the canvas:")
    
    # Conditional canvas rendering based on clear state
    canvas_key = "canvas_cleared" if st.session_state.clear_canvas else "canvas"
    canvas_result = st_canvas(
        fill_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key=canvas_key
    )
    
    if st.session_state.clear_canvas:
        st.session_state.clear_canvas = False
    
    # Create columns for buttons
    col1, col2 = st.columns(2)
    
    with col1:
        predict_button = st.button('Predict Digit', key='predict')
    
    with col2:
        if st.button('Clear Canvas', key='clear'):
            st.session_state.clear_canvas = True
            st.rerun()
    
    # Prediction logic
    if predict_button and canvas_result.image_data is not None:
        # Convert canvas to PIL Image
        input_image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        # Preprocess and predict
        processed_image = preprocess_image(input_image)
        pred, pred_idx, probs = model.predict(processed_image)
        
        # Display results
        st.success(f"Predicted Digit: **{pred}**")
        st.write("Confidence Probabilities:")
        
        # In the prediction logic section, replace the radar chart creation with this:

        # Create a radar chart of probabilities
        digits = [str(i) for i in range(10)]
        probabilities = [float(prob) for prob in probs]

        # Create a list to highlight the predicted digit
        highlight = [i == int(pred) for i in range(10)]

        fig = px.line_polar(
            r=probabilities,
            theta=digits,
            line_close=True,
            template="plotly_dark",
            title="Probability Distribution Across Digits"
        )

        # Update the trace to make the predicted value stand out
        fig.update_traces(
            fill='toself',
            line=dict(width=2, color='rgba(100, 100, 255, 0.5)'),
            marker=dict(
                size=18,
                color=['rgba(255, 0, 0, 0.8)' if h else 'rgba(100, 255, 100, 0.6)' for h in highlight],
                line=dict(
                    width=[10 if h else 1 for h in highlight],
                    color=['white' if h else 'rgba(200, 200, 200, 0.5)' for h in highlight]
                )
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Also show the probabilities in a table
        st.write("Detailed probabilities:")
        prob_data = {"Digit": digits, "Probability": [f"{p:.2%}" for p in probabilities]}
        st.table(prob_data)
        
    elif predict_button:
        st.warning("Please draw a digit first!!!")
        
    elif predict_button:
        st.warning("Please draw a digit first!!!")

def apply_custom_styles():
    st.markdown("""
    <style>
        /* General styling */
        .stTextArea, .stTextInput, .stButton, .stMarkdown {
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        
        /* Button styling */
        .stButton > button {
            font-family: 'Comic Sans MS', cursive, sans-serif;
            font-size: 1.2rem;
            border: 0;
            border-radius: 1rem;
            background-image: conic-gradient(
                from var(--border-angle-1) at 10% 15%, 
                transparent, 
                var(--bright-blue) 10%, 
                transparent 30%, 
                transparent
            ),
            conic-gradient(
                from var(--border-angle-2) at 70% 60%, 
                transparent, 
                var(--bright-green) 10%, 
                transparent 60%, 
                transparent
            ),
            conic-gradient(
                from var(--border-angle-3) at 50% 20%, 
                transparent, 
                var(--bright-red) 10%, 
                transparent 50%, 
                transparent
            );
            color: white;
            cursor: pointer;
            text-align: center;
            margin: 0.5rem 0;
            width: 100%;
            animation: rotateBackground 4s linear infinite,
                rotateBackground2 8s linear infinite,
                rotateBackground3 12s linear infinite;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        
        .stButton > button div {
            background: var(--background);
            padding: 0.5em 1em;
            border-radius: calc(var(--border-radius) - var(--border-size));
            color: var(--foreground);
        }
        
        /* Results styling */
        .stSuccess {
            font-size: 1.5rem;
            text-align: center;
            margin: 1rem 0;
        }
        
        /* CSS variables and animations */
        :root {
            --bright-blue: rgb(0, 100, 255);
            --bright-green: rgb(0, 255, 0);
            --bright-red: rgb(255, 0, 0);
            --background: rgba(0, 0, 0, 0.8);
            --foreground: white;
            --border-size: 1px;
            --border-radius: 0.5em;
        }
        
        @property --border-angle-1 {
            syntax: "<angle>";
            inherits: true;
            initial-value: 0deg;
        }
        
        @property --border-angle-2 {
            syntax: "<angle>";
            inherits: true;
            initial-value: 90deg;
        }
        
        @property --border-angle-3 {
            syntax: "<angle>";
            inherits: true;
            initial-value: 180deg;
        }
        
        @keyframes rotateBackground {
            to { --border-angle-1: 360deg; }
        }
        
        @keyframes rotateBackground2 {
            to { --border-angle-2: -270deg; }
        }
        
        @keyframes rotateBackground3 {
            to { --border-angle-3: 540deg; }
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()