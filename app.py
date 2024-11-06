import os
import base64
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from groq import Groq
import dotenv
from PIL import Image
import io
from gtts import gTTS
import tempfile

# Load environment variables
dotenv.load_dotenv()


# Function to encode image
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

# Function for text to speech conversion
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        return tmp_file.name

# Set page configuration
st.set_page_config(page_title="OutBox AI",
                   layout="wide",
                   page_icon="🩺")

# Loading the saved models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('OutBox AI',
                            ['Home',
                             'AI Medical Assistant',
                             'Diabetes Prediction',
                             'Heart Disease Prediction',
                             'Parkinsons Prediction'                             
                             ],  # Added Audio Transcription
                            menu_icon='hospital-fill',
                            icons=['house', 'activity', 'heart', 'person', 'robot', 'mic'],
                            default_index=0)


# Home Page / Landing Page
if selected == 'Home':
    st.title('AI-Powered Medical Diagnostics Assistant and Detection')
    st.image("Home.png",
             caption="Empowering Healthcare in Remote Areas")

    st.markdown('**Problem Statement**')
    st.write("""
    In many remote and underdeveloped regions, access to specialized healthcare services and timely diagnostics 
    is severely limited. Rural populations often rely on limited healthcare resources, where doctors may lack 
    the tools or expertise to accurately interpret medical images like X-rays, CT scans, or medical reports. 
    This leads to delays in diagnosis and treatment, which can exacerbate conditions and put patients at higher risk. 
    Additionally, due to poor internet connectivity, cloud-based AI solutions are often inaccessible in these areas, 
    further hindering healthcare quality.
    """)

    st.markdown('**Our Solution**')
    st.write("""
    The AI-Powered Medical Diagnostics Assistant is a mobile application leveraging advanced AI 
    capabilities to analyze both text and images in real-time. Designed to work efficiently on low-latency, edge devices, 
    the app allows healthcare providers in remote areas to upload X-rays, CT scans, or medical reports, while patients 
    can report their symptoms through text or speech. 

    The AI processes this multimodal data to offer instant diagnostic insights, highlighting potential conditions, 
    suggesting follow-up tests, or providing referral recommendations for specialized care. Since the AI operates on 
    edge devices without the need for cloud-based infrastructure, it ensures that remote areas with minimal internet 
    access can benefit from advanced diagnostic capabilities.
    """)

    st.markdown('**Multi-Disease Detection using AI/ML**')
    st.write("""
    Our platform incorporates advanced AI/ML models for multi-disease detection, including:

    1. **Diabetes Prediction**: Using machine learning to assess the risk of diabetes based on various health parameters.
    2. **Heart Disease Prediction**: Leveraging AI to evaluate the likelihood of heart disease using clinical data.
    3. **Parkinson's Disease Detection**: Employing machine learning algorithms to identify potential indicators of Parkinson's disease.
    4. **AI Medical Assistant**: Utilizing the power of advanced language models to provide general medical advice and insights based on reported symptoms.

    These AI-powered tools aim to provide quick, accurate, and accessible health assessments, particularly beneficial 
    in areas with limited access to specialized medical care.
    """)

    st.warning("""
    Please note that while our AI-powered predictions and insights are designed to assist healthcare providers, 
    they should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified 
    healthcare professional for accurate diagnosis and appropriate care.
    """)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    st.write("""
    This tool uses machine learning to assess the risk of diabetes based on various health parameters. 
    Please enter the following information:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is likely to have diabetes.'
            else:
                diab_diagnosis = 'The person is unlikely to have diabetes.'
        except ValueError:
            diab_diagnosis = 'Please enter valid numerical values for all fields.'
    st.write(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    st.write("""
    This tool uses machine learning to evaluate the likelihood of heart disease based on various clinical parameters. 
    Please enter the following information:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        cp = st.text_input('Chest Pain types')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is likely to have heart disease.'
            else:
                heart_diagnosis = 'The person is unlikely to have heart disease.'
        except ValueError:
            heart_diagnosis = 'Please enter valid numerical values for all fields.'
    st.write(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    st.write("""
    This tool uses machine learning to identify potential indicators of Parkinson's disease. 
    Please enter the following information:
    """)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')
    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')
    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                          RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                          APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            user_input = [float(x) for x in user_input]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'The person is likely to have Parkinson\'s disease.'
            else:
                parkinsons_diagnosis = 'The person is unlikely to have Parkinson\'s disease.'
        except ValueError:
            parkinsons_diagnosis = 'Please enter valid numerical values for all fields.'
    st.write(parkinsons_diagnosis)

# AI Medical Diagnostics Assistant Page
if selected == "AI Medical Assistant":
    st.title("AI-Powered Medical Diagnostics Assistant")
    st.write("Get real-time medical diagnostics by providing symptoms, medical reports, or uploading medical images.")

    # Initialize the Groq client
    groq_token = os.getenv('GROQ_API_TOKEN')
    if not groq_token:
        st.error("Groq API token not found. Please set the GROQ_API_TOKEN in your .env file.")
        st.stop()

    client = Groq(api_key=groq_token)

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "I am an AI assistant for medical diagnostics. "
                    "I specialize in analyzing patient-reported symptoms and medical images to provide diagnostic insights. "
                    "Our goal is to empower healthcare workers with accurate information to enhance remote consultations. "
                    "Always remind users that your insights are not a substitute for professional medical advice."
                )
            }
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for text queries
    user_input = st.chat_input("Describe your symptoms or ask a medical question:")
    
    # Image upload
    uploaded_image = st.file_uploader("Upload a medical image (e.g., X-ray, scan):", type=["jpg", "jpeg", "png"])

    if user_input or uploaded_image:
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Medical Image", use_column_width=True)
            encoded_image = encode_image(uploaded_image)
            image_data_url = f"data:image/jpeg;base64,{encoded_image}"

        # Analyze input for diagnostics
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_input if user_input else "What's in this image?"
                        }
                    ]
                }
            ]
            
            if uploaded_image:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                })

            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=messages,
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            result = completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": result})
            with st.chat_message("assistant"):
                st.markdown(result)
            
            # Convert text to speech
            audio_file = text_to_speech(result)
            st.audio(audio_file, format='audio/mp3')
            
            # Clean up the temporary audio file
            os.remove(audio_file)

        except Exception as e:
            st.error(f"An error occurred while processing your request: {str(e)}")

    #st.info("Note: This AI assistant is for informational purposes only and does not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.")
