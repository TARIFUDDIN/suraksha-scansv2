import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
import os
import tempfile
import time
import joblib
import json
import requests

# Remove problematic imports that may not be available on Streamlit Cloud
try:
    from tensorflow.keras.models import load_model
    from keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.preprocessing import image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not available. Brain tumor and lung cancer detection will be disabled.")

try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Jansevak - AI Health Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animations
def load_lottieurl(url: str):
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Add custom CSS with animations and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem !important;
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .feature-header {
        font-size: 2.2rem !important;
        color: #4a4a4a;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        animation: slideIn 0.8s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .feature-card {
        background-color: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border-left: 5px solid transparent;
        animation: cardAppear 0.5s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
        border-left: 5px solid #667eea;
    }
    
    @keyframes cardAppear {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .prediction-result {
        background: linear-gradient(to right, #f5f7fa 0%, #e4e8eb 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin-top: 1.5rem;
        border-left: 5px solid #667eea;
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(to right, #f5f7fa 0%, #e4e8eb 100%);
        margin-top: 3rem;
        border-radius: 1rem;
        box-shadow: 0 -5px 15px rgba(0,0,0,0.03);
    }
    
    .uploaded-image {
        max-width: 100%;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .uploaded-image:hover {
        transform: scale(1.02);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 0.8rem;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102, 126, 234, 0.4);
    }
    
    .success-box {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        border-left: 5px solid #4CAF50;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        border-left: 5px solid #FF9800;
    }
    
    .info-box {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        border-left: 5px solid #2196F3;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Load animations
if LOTTIE_AVAILABLE:
    brain_animation = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_obhph3sh.json")
    health_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_5tkzkblw.json")
    doctor_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_yyjaansa.json")
    scan_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_5nmikzps.json")
    lung_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_5nmikzps.json")
else:
    brain_animation = health_animation = doctor_animation = scan_animation = lung_animation = None

# Load brain tumor model
@st.cache_resource
def load_brain_tumor_model():
    if not TF_AVAILABLE:
        return None, None
    try:
        model_path = Path('./notebooks/Brain-tumor-prediction/model.h5')
        if not model_path.exists():
            st.warning(f"Brain tumor model not found at {model_path}")
            return None, None
            
        model = load_model(model_path)
        class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']
        return model, class_labels
    except Exception as e:
        st.warning(f"Failed to load brain tumor model: {str(e)}")
        return None, None

# Load lung cancer model
@st.cache_resource
def load_lung_cancer_model():
    try:
        model_path = Path('./notebooks/Lung-cancer-prediction/lung_cancer_model.pkl')
        if not model_path.exists():
            st.warning(f"Lung cancer model not found at {model_path}")
            return None
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.warning(f"Failed to load lung cancer model: {str(e)}")
        return None

# Initialize models
brain_tumor_model, brain_tumor_labels = load_brain_tumor_model()
lung_cancer_model = load_lung_cancer_model()

# Function to predict brain tumor
def predict_brain_tumor(image_path):
    if not TF_AVAILABLE or brain_tumor_model is None:
        return "Model not available", 0.0
    try:
        IMAGE_SIZE = 128
        img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = brain_tumor_model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]
        
        if brain_tumor_labels[predicted_class_index] == 'notumor':
            return "No Tumor Detected", confidence_score
        else:
            return f"{brain_tumor_labels[predicted_class_index].capitalize()} Tumor", confidence_score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Function to predict lung cancer
def predict_lung_cancer(image_path):
    if not CV2_AVAILABLE or lung_cancer_model is None:
        return "Model not available", 0.0
    try:
        import cv2
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img.flatten().reshape(1, -1)
        
        prediction = lung_cancer_model.predict(img)
        proba = lung_cancer_model.predict_proba(img)[0]
        
        predicted_class = "Lung Cancer Detected" if prediction[0] == 1 else "No Lung Cancer Detected"
        confidence_score = proba[1] * 100 if predicted_class == "Lung Cancer Detected" else proba[0] * 100
        
        return predicted_class, confidence_score
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Navigation
def create_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("🏥", unsafe_allow_html=True)
    with col2:
        st.markdown("<h1 style='color: #667eea; margin-bottom: 0;'>Jansevak</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6c757d; margin-top: 0; font-size: 1.2rem;'>AI-Powered Health Diagnostics</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    tabs = st.tabs(["🏠 Home", "🩺 Disease Prediction", "🧠 Brain Tumor", "🫁 Lung Cancer", "👁️ Cataract"])
    return tabs

tabs = create_header()

# Define file paths for disease prediction
MODEL_PATH = Path('notebooks/Disease-prediction/random_forest_disease_model.pkl')
SYMPTOMS_PATH = Path('notebooks/Disease-prediction/symptoms_list.pkl')
TRAINING_DATA_PATH = Path('notebooks/Disease-prediction/data/Training.csv')
TESTING_DATA_PATH = Path('notebooks/Disease-prediction/data/Testing.csv')

# Load disease prediction model data
@st.cache_data
def load_disease_data():
    try:
        if MODEL_PATH.exists() and SYMPTOMS_PATH.exists():
            with open(MODEL_PATH, 'rb') as model_file:
                clf = pickle.load(model_file)
            with open(SYMPTOMS_PATH, 'rb') as symptoms_file:
                symptoms = pickle.load(symptoms_file)
            dictionary = {symptom: index for index, symptom in enumerate(symptoms)}
            st.success("Loaded trained Random Forest model")
            return clf, symptoms, dictionary
        else:
            raise FileNotFoundError("Model files not found")
    
    except Exception as e:
        st.warning(f"Using fallback Decision Tree model. Error: {str(e)}")
        
        try:
            if not TRAINING_DATA_PATH.exists():
                # Create dummy data for demonstration
                symptoms = ['fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting', 'chest_pain', 'shortness_of_breath']
                data = {symptom: np.random.randint(0, 2, 100) for symptom in symptoms}
                data['prognosis'] = np.random.choice(['Common Cold', 'Flu', 'Pneumonia', 'Bronchitis'], 100)
                df = pd.DataFrame(data)
            else:
                df = pd.read_csv(TRAINING_DATA_PATH)
                
            cols = df.columns[:-1]  # All columns except the last one (prognosis)
            x = df[cols]
            y = df['prognosis']
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            
            dt = DecisionTreeClassifier()
            clf_dt = dt.fit(x_train, y_train)
            
            symptoms = list(cols)
            dictionary = {symptom: i for i, symptom in enumerate(symptoms)}
            
            return clf_dt, symptoms, dictionary
            
        except Exception as e:
            st.error(f"Failed to load any model: {str(e)}")
            # Return dummy data to prevent crashes
            symptoms = ['fever', 'cough', 'headache', 'fatigue']
            dictionary = {symptom: i for i, symptom in enumerate(symptoms)}
            return None, symptoms, dictionary

def predict_disease(symptom_list, clf, dictionary):
    if clf is None:
        return "Model not available", 0.0
    try:
        input_features = np.zeros(len(dictionary))
        for symptom in symptom_list:
            if symptom in dictionary:
                input_features[dictionary[symptom]] = 1

        input_features = input_features.reshape(1, -1)
        prediction = clf.predict(input_features)[0]
        probability = np.max(clf.predict_proba(input_features)) * 100
        
        return prediction, probability
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return "Error in prediction", 0.0

# Function to display home page
def show_home():
    st.markdown("<h1 class='main-header'>AI-Powered Health Diagnostics</h1>", unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div style='font-size: 1.2rem; line-height: 1.6; margin-bottom: 2rem;'>
            Jansevak provides advanced AI-powered diagnostic tools to help identify potential health issues early. 
            Our system can analyze symptoms and medical images to assist healthcare professionals in making accurate diagnoses.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%); 
                    padding: 1.5rem; border-radius: 1rem; margin-bottom: 2rem;'>
            <h3 style='color: #667eea;'>How it works:</h3>
            <ol style='line-height: 2;'>
                <li>Select the diagnostic tool you need</li>
                <li>Provide the required inputs (symptoms or medical images)</li>
                <li>Get AI-powered analysis and recommendations</li>
                <li>Consult with your healthcare provider for next steps</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        if health_animation and LOTTIE_AVAILABLE:
            st_lottie(health_animation, height=300, key="health")
    
    # Features section
    st.markdown("<h2 class='feature-header'>Our Diagnostic Tools</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("🩺")
        st.markdown("<h3 style='color: #667eea;'>Disease Prediction</h3>", unsafe_allow_html=True)
        st.write("Analyze your symptoms to get potential disease predictions with confidence scores.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("🫁")
        st.markdown("<h3 style='color: #667eea;'>Lung Cancer</h3>", unsafe_allow_html=True)
        st.write("Analyze CT scans for early detection of lung cancer with AI-powered imaging.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("🧠")
        st.markdown("<h3 style='color: #667eea;'>Brain Tumor</h3>", unsafe_allow_html=True)
        st.write("Upload MRI scans to detect potential brain tumors with AI analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.markdown("👁️")
        st.markdown("<h3 style='color: #667eea;'>Cataract</h3>", unsafe_allow_html=True)
        st.write("Coming soon: Analyze eye images for cataract detection.")
        st.markdown("</div>", unsafe_allow_html=True)

# Main app logic with tabs
with tabs[0]:  # Home tab
    show_home()

with tabs[1]:  # Disease Prediction tab
    st.markdown("<h1 class='main-header'>🩺 Disease Prediction System</h1>", unsafe_allow_html=True)

    try:
        clf, symptoms, dictionary = load_disease_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.subheader("Select Your Symptoms")
            
            if doctor_animation and LOTTIE_AVAILABLE:
                st_lottie(doctor_animation, height=200, key="doctor_anim")
            
            symptom1 = st.selectbox("Primary Symptom", [""] + symptoms, key="symptom1")
            symptom2 = st.selectbox("Secondary Symptom", [""] + symptoms, key="symptom2")
            symptom3 = st.selectbox("Additional Symptom 1", [""] + symptoms, key="symptom3")
            symptom4 = st.selectbox("Additional Symptom 2", [""] + symptoms, key="symptom4")
            symptom5 = st.selectbox("Additional Symptom 3", [""] + symptoms, key="symptom5")
            
            selected_symptoms = []
            for symptom in [symptom1, symptom2, symptom3, symptom4, symptom5]:
                if symptom != "" and symptom not in selected_symptoms:
                    selected_symptoms.append(symptom)
            
            if st.button("Predict Disease", key="predict_disease"):
                if not selected_symptoms:
                    st.error("Please select at least one symptom")
                else:
                    with st.spinner("🔍 Analyzing symptoms with AI..."):
                        progress_bar = st.progress(0)
                        for percent_complete in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(percent_complete + 1)
                        
                        disease, confidence_score = predict_disease(selected_symptoms, clf, dictionary)
                        
                        st.session_state.disease = disease
                        st.session_state.confidence_score = confidence_score
                        st.session_state.selected_symptoms = selected_symptoms
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.subheader("Prediction Results")
            
            if 'disease' in st.session_state and 'confidence_score' in st.session_state:
                st.markdown("<div class='prediction-result info-box'>", unsafe_allow_html=True)
                st.success(f"🩺 Predicted Condition: **{st.session_state.disease}**")
                st.info(f"📊 Confidence Score: **{st.session_state.confidence_score:.2f}%**")
                
                st.write("**Selected Symptoms:**")
                for i, symptom in enumerate(st.session_state.selected_symptoms, 1):
                    st.write(f"{i}. {symptom}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 0.8rem; margin-top: 1rem;'>
                    <h4>Recommended Actions</h4>
                    <ol>
                        <li>Schedule an appointment with your doctor</li>
                        <li>Monitor your symptoms</li>
                        <li>Seek emergency care if symptoms worsen</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Please select symptoms and click 'Predict Disease' to see results")
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in disease prediction: {str(e)}")

with tabs[2]:  # Brain Tumor tab
    st.markdown("<h1 class='main-header'>🧠 Brain Tumor Detection</h1>", unsafe_allow_html=True)
    
    if not TF_AVAILABLE:
        st.error("TensorFlow is required for brain tumor detection but is not available.")
        st.info("Please install TensorFlow to use this feature.")
    elif brain_tumor_model is None:
        st.error("Brain tumor prediction model failed to load. Please check the model file.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.subheader("Upload Brain MRI Scan")
            
            if brain_animation and LOTTIE_AVAILABLE:
                st_lottie(brain_animation, height=200, key="brain_anim")
            
            uploaded_file = st.file_uploader("Choose an MRI image (JPEG, PNG)", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Brain Scan", use_column_width=True)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    image.save(tmp_file.name)
                    tmp_file_path = tmp_file.name
                
                if st.button("Analyze Brain Scan", key="analyze_brain"):
                    with st.spinner("🧠 Analyzing the brain scan with AI..."):
                        result, confidence = predict_brain_tumor(tmp_file_path)
                        st.session_state.brain_result = result
                        st.session_state.brain_confidence = confidence
                        st.session_state.brain_image = image
                    
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            st.subheader("Analysis Results")
            
            if 'brain_result' in st.session_state:
                if "No Tumor" in st.session_state.brain_result:
                    st.success(f"✅ Result: {st.session_state.brain_result}")
                else:
                    st.warning(f"⚠️ Result: {st.session_state.brain_result}")
                
                st.info(f"🔍 Confidence: {st.session_state.brain_confidence * 100:.2f}%")
                st.image(st.session_state.brain_image, caption=f"Prediction: {st.session_state.brain_result}")
            else:
                st.info("ℹ️ Please upload a brain MRI scan and click 'Analyze Brain Scan' to see results")
            st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:  # Lung Cancer tab
    st.markdown("<h1 class='main-header'>🫁 Lung Cancer Detection</h1>", unsafe_allow_html=True)
    
    if not CV2_AVAILABLE:
        st.error("OpenCV is required for lung cancer detection but is not available.")
        st.info("Please install OpenCV to use this feature.")
    elif lung_cancer_model is None:
        st.error("Lung cancer prediction model failed to load. Please check the model file.")
    else:
        st.info("🚧 Lung cancer detection model is available but requires proper training data.")
        st.write("This feature will be fully functional once the model is properly trained.")

with tabs[4]:  # Cataract tab
    st.markdown("<h1 class='main-header'>👁️ Cataract Detection</h1>", unsafe_allow_html=True)
    st.warning("🚧 This feature is currently under development and will be available soon!")
    st.markdown("""
    <div style='background: #fff3cd; padding: 1.5rem; border-radius: 0.8rem;'>
        <h3 style='color: #856404;'>Coming Soon</h3>
        <p>Our cataract module will analyze eye images to identify clouding of the lens.</p>
        <p>Expected launch: Q4 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(to right, #f5f7fa 0%, #e4e8eb 100%); 
           margin-top: 3rem; border-radius: 1rem;'>
    <p style='margin: 0;'>© 2025 Suraksha - AI Health Assistant | All Rights Reserved</p>
    <p style='margin: 0; font-size: 0.9rem; color: #6c757d;'>
        Disclaimer: This tool does not provide medical advice. Always consult a healthcare professional.
    </p>
</div>
""", unsafe_allow_html=True)