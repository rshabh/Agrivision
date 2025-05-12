import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

import streamlit as st
from deep_translator import GoogleTranslator as Translator
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import torch.nn as nn
import os
import shutil
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from typing import List
from fpdf import FPDF

# my imports -
from torchvision import transforms
from PIL import Image
from io import BytesIO
from streamlit_geolocation import streamlit_geolocation
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import sqlite3
from translations import state_translation, district_translation
from pyngrok import ngrok
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.applications import ResNet50

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/"

# Load your pre-trained models
# plant_disease_model_weights_path = 'models for predictions/Crops/plant-disease-model.pth'
plant_disease_model_path = 'models for predictions/Crops/plant-disease-model-complete.pth'
soil_composition_model_path = 'models for predictions/Soil/composition/composition/two.h5'
soil_type_model_weights_path = 'models for predictions/Soil/soil type/soiltype/tycoon.weights.h5'

GENERATIVE_PROMPT_TEMPLATE = """
Generate a response to the following question, even if no direct context is provided:

Question: {question}
"""

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

st.set_page_config(page_title="AgriVision: farmer's MITRA", page_icon="🌾")


# Function to create or update the Chroma vector store
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", show_progress=True)
    documents = loader.load()
    return documents


def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def save_to_chroma(chunks: List[Document]):
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        # Retry mechanism
        for _ in range(5):
            try:
                shutil.rmtree(CHROMA_PATH)
                break
            except PermissionError as e:
                print(f"PermissionError: {e}. Retrying in 1 second...")
                time.sleep(1)  # Wait before retrying
            except Exception as e:
                print(f"Failed to remove directory: {e}")
                break

    # Use OpenAI embeddings
    embedding_function = OpenAIEmbeddings(openai_api_key=openai.api_key)

    # Create a new DB from the documents
    db = Chroma.from_documents(
        chunks, embedding_function, persist_directory=CHROMA_PATH
    )
    db.persist()


def reciprocal_rank_fusion(results: List[List], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = doc.json()
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = sorted(
        [(doc, score) for doc, score in fused_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return [doc for doc, _ in reranked_results]


# Function to process the query and generate a response using RAG Fusion
def process_query(query_text):
    embedding_function = OpenAIEmbeddings(openai_api_key=openai.api_key)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Perform similarity search to get relevant documents and their scores
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    if len(results) == 0 or results[0][1] < 0.7:
        # If no relevant context is found or relevance score is below threshold, use GPT for response
        source = t("Intelligently generated by our AI model- **Please verify the response as it is AI generated.")
        context_text = "N/A"
        generative_prompt_template = ChatPromptTemplate.from_template(GENERATIVE_PROMPT_TEMPLATE)
        generative_prompt = generative_prompt_template.format(question=query_text)
        model = ChatOpenAI(api_key=openai.api_key, model="gpt-4o")
        response = model.predict(generative_prompt)
        response_text = response.strip()

    else:
        # If relevant context is found, use RAG Fusion Chain
        source = t("Based upon the papers and publications by authorized bodies.")
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Set up RAG Fusion Chain
        prompt_chain = ChatPromptTemplate(
            input_variables=['question'],
            messages=[
                SystemMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=[],
                        template='You are a helpful assistant that generates multiple search queries based on a single input query.'
                    )
                ),
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=['question'],
                        template='Generate multiple search queries related to: {question} \n OUTPUT (4 queries):'
                    )
                )
            ]
        )

        generate_queries = (
                prompt_chain | ChatOpenAI(api_key=openai.api_key, model="gpt-4o") | StrOutputParser() | (
            lambda x: x.split("\n"))
        )

        ragfusion_chain = generate_queries | db.as_retriever().map() | reciprocal_rank_fusion

        full_rag_fusion_chain = (
                {
                    "context": ragfusion_chain,
                    "question": RunnablePassthrough()
                }
                | ChatPromptTemplate.from_template("""Answer the question based only on the following context:
            {context}

            Question: {question}
            """)
                | ChatOpenAI(api_key=openai.api_key, model="gpt-4o")
                | StrOutputParser()
        )

        # Get the response from the RAG Fusion chain
        response = full_rag_fusion_chain.invoke({"question": query_text})
        response_text = response.strip()

    # Formatting the response
    formatted_response = f"""

    1. *Source*: {source}

    2. *Response from the RAG model*: 
    {response_text}
    """

    return formatted_response.strip()


# Crop Disease Prediction Function
class ConvBlock(nn.Module):
    def _init_(self, in_channels, out_channels, kernel_size=3, padding=1, pool=False):
        super(ConvBlock, self)._init_()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResNet9(nn.Module):
    def _init_(self, in_channels, num_diseases):
        super()._init_()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True)
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

        self.conv3 = ConvBlock(128, 256, pool=True)
        self.conv4 = ConvBlock(256, 512, pool=True)
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_diseases))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Increase the input size
        transforms.ToTensor(),
    ])
    img = transform(image).unsqueeze(0)  # Add batch dimension
    return img


# Prediction function using the fully saved PyTorch model
def predict_crop_disease(image, model_path):
    try:
        # Load the complete model directly
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        # Preprocess the image
        img = preprocess_image(image)

        # Make predictions
        with torch.no_grad():
            predictions = model(img)
        predicted_class = torch.argmax(predictions, dim=1).item()

        # Map the prediction to class name
        return class_names[predicted_class]
    except Exception as e:
        print(f"Error in predicting crop disease: {e}")
        return None


# List of disease names corresponding to class indices
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___(including_sour)_Powdery_mildew",
    "Cherry___(including_sour)_healthy",
    "Corn___(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___(maize)Common_rust",
    "Corn___(maize)_Northern_Leaf_Blight",
    "Corn___(maize)_healthy",
    "Grape___Black_rot",
    "Grape___Esca(Black_Measles)",
    "Grape___Leaf_blight(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,bell___Bacterial_spot",
    "Pepper,bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# District soil type code ---------------------------------------------------------------------
# Connect to the SQLite database
def load_data():
    conn = sqlite3.connect('soiltype.db')
    query = "SELECT DISTINCT `State Name`, `District` FROM soiltype;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def get_soil_type(state, district):
    conn = sqlite3.connect('soiltype.db')
    query = """
    SELECT `Soil Type`
    FROM soiltype
    WHERE `State Name` = ? AND `District` = ?
    """
    df = pd.read_sql(query, conn, params=(state, district))
    conn.close()
    if not df.empty:
        return df.iloc[0]['Soil Type']
    else:
        return "Soil Type not found"


# Load data from the database
data = load_data()


# --------------------------------------------------------------------------------------------------------

# Soil Composition Prediction Function------------------------------------------------------------------------
def predict_soil_composition(img):
    img = Image.open(img)
    img = img.resize((224, 224))  # Resize to the expected input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    model = load_model(soil_composition_model_path)
    predictions = model.predict(img)
    return predictions[0]


# Define image dimensions
img_width, img_height = 224, 224

# Manually specify the class labels
class_labels = {0: 'Black Soil', 1: 'Cinder Soil', 2: 'Red Soil', 3: 'Yellow Soil'}


def load_and_preprocess_image(img):
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array


# Recreate the model architecture exactly as in training
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet',
                                            input_shape=(img_width, img_height, 3))

# Fine-tuning: Unfreeze the last 10 layers
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Add custom classifier layers with dropout and batch normalization for regularization
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model with the same configuration as training
initial_learning_rate = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load the weights
model.load_weights('models for predictions/Soil/soil type/soiltype/tycoon.weights.h5')


# Weather API Function----------------------------------------------------------------------------------
# Function to get weather data from Visual Crossing API
# Function to get weather data from the API
def get_weather_data(lat, lon, start_date, end_date, api_key):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"

    params = {
        'unitGroup': 'metric',  # Use 'us' for Fahrenheit and 'metric' for Celsius
        'key': api_key,
        'include': 'days',  # Include daily data
        'elements': 'tempmax,tempmin,precip,windspeed'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['days']
    elif response.status_code == 429:
        st.warning("You have exceeded the API rate limit. Please try again later.")
        return None
    else:
        st.error(f"Unable to fetch weather data. Error code: {response.status_code}")
        return None


# Function to create a summary string from the weather data
def create_summary(weather_data, period_name):
    if not weather_data:
        return f"No data available for the {period_name}."

    temps_max = [day['tempmax'] for day in weather_data if 'tempmax' in day]
    temps_min = [day['tempmin'] for day in weather_data if 'tempmin' in day]
    precip = [day.get('precip', 0) for day in weather_data]
    wind_speeds = [day['windspeed'] for day in weather_data if 'windspeed' in day]

    summary = (
        f"Weather Summary for {period_name} is that, "
        f"Max Temp: {max(temps_max, default='N/A')}°C, Min Temp: {min(temps_min, default='N/A')}°C, "
        f"Total Precipitation: {sum(precip)} mm\n and "
        f"Average Wind Speed: {sum(wind_speeds) / len(wind_speeds) if wind_speeds else 'N/A'} km/h\n"
    )
    return summary


# API Key
api_key = 'M47UVGQR6MBEMYSBXEJPTEYFE'  # Replace with your Visual Crossing API key

# ---------------------------------------------------------------------------------------------------------

# # Prediction variables
# # Initialize session state variables to store predictions and weather summary
if 'predicted_crop_disease' not in st.session_state:
    st.session_state.predicted_crop_disease = None
if 'predicted_soil_composition' not in st.session_state:
    st.session_state.predicted_soil_composition = None
if 'predicted_soil_type' not in st.session_state:
    st.session_state.predicted_soil_type = None
if 'weather_summary' not in st.session_state:
    st.session_state.weather_summary = None
if 'weather_button_clicked' not in st.session_state:
    st.session_state.weather_button_clicked = False
if 'formatted_response' not in st.session_state:
    st.session_state.formatted_response = ""
if 'crop_name' not in st.session_state:
    st.session_state.crop_name = "No prediction made"
if 'disease_name' not in st.session_state:
    st.session_state.disease_name = "No prediction made"

# # Streamlit UI Layout

# Initialize the translator
translator = Translator()

# Define supported languages
LANGUAGES = {
    "English": "en",
    "Telugu": "te",
    "Tamil": "ta",
    "Marathi": "mr",
    "Hindi": "hi"
}

# Language selection dropdown
selected_language = st.sidebar.selectbox("Select Language", options=list(LANGUAGES.keys()))


@st.cache_data  # Cache the translation results
def translate_text(text, lang):
    return Translator(source='auto', target=LANGUAGES[lang]).translate(text)


def t(text):
    return translate_text(text, selected_language)


# --------------------------------------------------------------------------------------------------------------------


# # Background image styling
page_bg_img = """
    <style>
        /* Background styling */
        [data-testid="stAppViewContainer"] {
            background-color: #c5d7ca;
opacity: 1;
background-image:  radial-gradient(#034125 0.5px, transparent 0.5px), radial-gradient(#034125 0.5px, #c5d7ca 0.5px);
background-size: 20px 20px;
background-position: 0 0,10px 10px;
        }

        /* Header styling */
        [data-testid="stHeader"] {
            background-color: #8bc49d;
            opacity: 0.8;
            background-image: repeating-radial-gradient(circle at 0 0, transparent 0, #8bc49d 10px),
                              repeating-linear-gradient(#03834a55, #03834a);
        }

        /* Text styling */
        h1, h2, h3, h4, h5, h6 {
            color: #183D74;
            text-shadow: 1px 1px 2px black;
        }
        /* Select the checkboxes using their data-testid attribute */
    /* Ensure the heading with the emoji is displayed as a block */
    [data-testid="stHeadingWithActionElements"] {
        display: block;
    }

    /* Select the checkboxes using their data-testid attribute and align them side by side */
    [data-testid="stCheckbox"] {
        display: block;
        margin-top: 20px; /* Adjust the space between checkboxes as needed */
        vertical-align: middle;
    }

    </style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title(t("AgriVision: Your Agricultural Assistant"))

# # Crop Disease Prediction Section
st.header(t("Predict Crop Disease"))
crop_image = st.file_uploader(t("Upload a Crop Image"), type=["jpg", "png", "jpeg", "JPG"])

# Crop disease prediction section
if crop_image:
    st.image(crop_image, caption=t('Uploaded Crop Image'), use_column_width=True)

    if st.button(t("Predict Disease")):
        image = Image.open(crop_image).convert('RGB')
        st.session_state.predicted_crop_disease = predict_crop_disease(image, plant_disease_model_path)

        if st.session_state.predicted_crop_disease is not None:
            # Split the predicted label into crop name and disease name
            parts = st.session_state.predicted_crop_disease.split("___")
            if len(parts) == 2:
                st.session_state.crop_name, st.session_state.disease_name = parts
            else:
                st.session_state.crop_name = parts[0]
                st.session_state.disease_name = t("None (Healthy)")

            # Create the summary message for translation
            predicted_crop_details = f"Predicted Crop Details:\n\nCrop Name: {st.session_state.crop_name}\n" \
                                     f"Found issue (Disease/Healthy): {st.session_state.disease_name}"

            # Translate the summary message
            translated_crop_details = t(predicted_crop_details)

            # Store the translated details in the session state
            st.session_state.translated_crop_details = translated_crop_details

            # Display the translated prediction result
            st.success(st.session_state.translated_crop_details)

        else:
            st.error(t("Error in predicting crop disease. Please check the logs."))


# Soil Composition Prediction Section
st.header(t("Predict Soil Composition🌱"))
soil_image_composition = st.file_uploader(t("Upload a Soil Image for Composition"), type=["jpg", "png", "jpeg", "JPG"],
                                          key="composition")

if soil_image_composition is not None:
    st.image(soil_image_composition, caption=t('Uploaded Soil Image'), use_column_width=True)

    if st.button(t("Predict Composition")):
        # Assuming the predict_soil_composition function returns a list or array with two values
        sand_probability, gravel_probability = predict_soil_composition(soil_image_composition)

        # Store probabilities in session state
        st.session_state.predicted_soil_composition = {'sand': sand_probability, 'gravel': gravel_probability}

        # Display the results
        st.write(t(
            f'The predicted soil composition is: Sand Probability: {sand_probability}, Gravel Probability: {gravel_probability}'))

# Section for Soil Type Prediction
st.header(t("Predict Soil Type📝"))
predict_soil_type = st.checkbox(t("Upload Soil Image [BETA]"))

if predict_soil_type:
    soil_image_type = st.file_uploader(t("Upload a Soil Image for Type"), type=["jpg", "png", "jpeg", "JPG"],
                                       key="type")

    if soil_image_type is not None:
        st.image(soil_image_type, caption=t('Uploaded Soil Image'), use_column_width=True)

        if st.button(t("Predict Soil Type")):
            image = Image.open(BytesIO(soil_image_type.read()))
            input_image = load_and_preprocess_image(image)
            predictions = model.predict(input_image)
            predicted_class = np.argmax(predictions, axis=1)
            st.session_state.predicted_soil_type = class_labels[predicted_class[0]]
            # Display the prediction result
            st.write(t(f'The predicted soil type is: {st.session_state.predicted_soil_type}'))

# Section for State-wise Soil Type
# st.header(t("District-Soil Type Database"))
show_state_soil_type = st.checkbox(t("Using District-Soil Type Database"), key="show_state_soil_type")





if show_state_soil_type:
    # Translate the state names based on the selected language
    translated_states = [
        state_translation.get(state, {}).get(LANGUAGES[selected_language], state)
        for state in data['State Name'].unique()
    ]

    # Select the translated state
    selected_state = st.selectbox(t('Select State Name'), translated_states)

    # Reverse lookup to find the original state name for the selected state
    original_state = next(
        (key for key, value in state_translation.items()
         if value.get(LANGUAGES[selected_language]) == selected_state),
        selected_state  # Fallback to selected state if no translation found
    )

    # Translate the district names based on the selected state and language
    translated_districts = [
        district_translation.get(district, {}).get(LANGUAGES[selected_language], district)
        for district in data[data['State Name'] == original_state]['District'].unique()
    ]

    # Select the translated district
    selected_district = st.selectbox(t('Select District'), translated_districts)

    # Reverse lookup to find the original district name for the selected district
    original_district = next(
        (key for key, value in district_translation.items()
         if value.get(LANGUAGES[selected_language]) == selected_district),
        selected_district  # Fallback to selected district if no translation found
    )

    # Display the Soil Type based on selections
    if original_state and original_district:
        soil_type = get_soil_type(original_state, original_district)
        st.write(t(f"*Soil Type:* {soil_type}"))
        st.session_state.predicted_soil_type = soil_type
    else:
        st.write(t("*Soil Type:* None"))

# Ensure only one section is visible at a time
if predict_soil_type and show_state_soil_type:
    st.warning(t("Please select only one section to display. Uncheck one of the options."))

# Create a session state variable to track if the button has been pressed---------------------------------------
# Weather Prediction Section
st.header(t("Get My Location Weather🌦️"))

# Weather Prediction Button
# Weather Prediction Button
if st.button(t("Predict Weather")):
    st.session_state.weather_button_clicked = True

# Only run the weather prediction if the button has been clicked
if st.session_state.weather_button_clicked:
    location = streamlit_geolocation()

    if location and location['latitude'] and location['longitude']:
        latitude = location['latitude']
        longitude = location['longitude']

        # Dates
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date_last_month = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Last 1 month
        start_date_next_15_days = datetime.now().strftime('%Y-%m-%d')
        end_date_next_15_days = (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d')

        # Get historical weather data for the last 1 month
        historical_weather = get_weather_data(latitude, longitude, start_date_last_month, end_date, api_key)

        # Get weather forecast for the next 15 days
        forecast_weather = get_weather_data(latitude, longitude, start_date_next_15_days, end_date_next_15_days, api_key)

        # Create summaries
        historical_summary = create_summary(historical_weather, t("Last 1 Month"))
        forecast_summary = create_summary(forecast_weather, t("Next 15 Days"))

        # Store the summaries in variables for translation
        weather_summary = f"Historical Weather: {historical_summary}\nForecast Weather: {forecast_summary}"

        # Apply translation function to the combined weather summary
        translated_weather_summary = t(weather_summary)

        # Store the translated summary in session state
        st.session_state.weather_summary = translated_weather_summary

        # Display summaries in Streamlit text boxes
        st.subheader(t("Historical Weather Summary"))
        st.text(t(historical_summary))

        st.subheader(t("Forecast Weather Summary"))
        st.text(t(forecast_summary))
    else:
        st.error(t("Location data is not available."))



# Function to print the generated report

def print_report(response_text):
    st.download_button(
        label="Download Report",
        data=response_text,
        file_name="report.txt",
        mime="text/plain",
    )


# Construct result_string with crop name, disease name, soil composition, soil type, and weather summary
cropd = st.session_state.predicted_crop_disease if st.session_state.predicted_crop_disease is not None else 'No prediction made'
soilc_sand = st.session_state.predicted_soil_composition.get('sand',
                                                             'No prediction made') if st.session_state.predicted_soil_composition else 'No prediction made'
soilc_gravel = st.session_state.predicted_soil_composition.get('gravel',
                                                               'No prediction made') if st.session_state.predicted_soil_composition else 'No prediction made'
soilt = st.session_state.predicted_soil_type if st.session_state.predicted_soil_type is not None else 'No prediction made'
weathersum = st.session_state.weather_summary if st.session_state.weather_summary is not None else 'No weather data available'

st.header(t("RAG Assistance"))

result_string = (f"Crop Name: {st.session_state.crop_name}\n"
                 f"Crop Disease: {st.session_state.disease_name}\n"
                 f"Soil Composition - Sand Probability: {soilc_sand}\n"
                 f"Soil Composition - Gravel Probability: {soilc_gravel}\n"
                 f"Soil Type: {soilt}\n"
                 f"Weather Summary: {weathersum}")


# RAG Farmers Assistant Chat Section

def generate_txt(response_text, file_name="report.txt"):
    # Write the text to a UTF-8 encoded file
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(response_text)

    # Read back the file content to provide it for download
    with open(file_name, 'r', encoding='utf-8') as file:
        txt_output = file.read()

    return txt_output


# Generate the report when "Get Report" button is clicked
if st.button(t("Get Report")):
    query = f"""
You are a agricultural expert that provides valuable insights to por uneducated farmers on the basis of the
conditions they are facing (MANDATORY)and you are expert in :
Crop disease management
Soil health managemnt
Environmental factors affecting agricultural productivity
Your expertise lies in generating detailed reports that:

Analyze crop, soil, and environmental conditions
Provide actionable recommendations for farmers and agricultural stakeholders
Provided Data:

Crop Name: {st.session_state.crop_name}
Crop Disease Name: {st.session_state.disease_name}
Soil Type: {soilt}
Weather Data: {weathersum}
Gravel Probability: {soilc_gravel}
Sand Probability: {soilc_sand}
Task: Prepare a comprehensive report using the provided data:

Crop Name
Crop Disease Name
Soil Type
Weather Data
Soil Composition (Gravel and Sand)
Report Structure: Ensure the report includes organized sub-headings with clearly explained points and a professional format. The content must be easily understandable for uneducated farmers and offer practical, actionable solutions. Every section should be detailed and directly tied to the provided data.

Instructions:

Write 3 paragraphs under each subheading and each paragraph should be of 7 lines each(MANDATORY).
content should not be generalised but based on the current situation the farmer is facing.(MANDATORY)

10 ACTIONABLE SUGGESTION SHOULD BE OF 4 LINES EACH
Strictly Provide ACTIONABLE SUGGESTION based on the given data Crop Name: {st.session_state.crop_name}
Crop Disease Name: {st.session_state.disease_name}
Soil Type: {soilt}
Weather Data: {weathersum}
Gravel Percentage in soil: {soilc_gravel}
Sand Percentage in soil: {soilc_sand}
suggestions should not be generalised but based on the current situation the farmer is facing.(MANDATORY)

Ensure content is easily understandable by uneducated farmers and includes practical, actionable insights.(MANDATORY)
Sections on preventive measures and treatment options should have distinct, non-overlapping answers.
Every answer should be tailored to the specific data provided: Crop Name, Crop Disease Name, Soil Type, Weather Data, and Soil Composition.
Answer only and only in {selected_language}
if telling that a farmer should use this technique the explain how that techniques is used.(MANDATORY)


Sections to Include:

1.Description of Disease:
Provide a detailed description of the disease affecting the crop.
Explain the nature of the disease and its impact on the crop.

2.Symptoms of Disease:
Describe visible and hidden symptoms of the disease.
Explain how to identify these symptoms in the crop.

3.Disease Life Cycle:
Explain how the disease develops and spreads over time.
Describe the stages of the disease life cycle and factors influencing its progression.

4. Disease Solutions
Give all Preventive Measures for Disease and all the Treatment Options for Disease (MANDATORY)
give Organic Control, Chemical Control, Non-Chemical Practises, Alternative Fungicides and Alternative Treatments

Strictly Provide analyses  based on the given data (MANDATORY) like:
Crop Name: {st.session_state.crop_name}
Crop Disease Name: {st.session_state.disease_name}
Soil Type: {soilt}
Weather Data: {weathersum}
Gravel Percentage in soil: {soilc_gravel}
Sand Percentage in soil: {soilc_sand}  
Give answer different to normal ChatGPT answer, give tailored approach after considering all the factors of the given data

Answer should be very long, no limit on answer
Give all the treatment options SPECIALLY FOCUS ON Chemical treatments(tell ingredients) and Organic Treatments And also mentions WHEN TO BE APPLIED  (MANDATORY) 
Also mention giving tailored apporach 
Detail available chemical and organic treatments for the disease.
Suggest effective strategies to prevent future disease outbreaks and soil degradation. 

5.Soil Health:
Strictly Provide analyses on soi health based on the given data (MANDATORY) 
consider soil conditions that enable the disease(MANDATORY)
Crop Name: {st.session_state.crop_name}
Crop Disease Name: {st.session_state.disease_name}
Soil Type: {soilt}
Weather Data: {weathersum}
Gravel Percentage in soil: {soilc_gravel}
Sand Percentage in soil: {soilc_sand}
Also mention giving tailored apporach 
nutrients Profile of soil
Explain the effects of soil composition (Gravel/Sand) on crop growth.
Explain the effect of the weather on soil strictly consider weather data for that
Suggest crops that thrive in the specified soil type.




6.Soil Amendments:
Strictly Provide analyses on Soil Amendments based on the given data (MANDATORY)
remember the given data and give answer on the basis of that only
consider soil conditions that enable the disease(MANDATORY)
consider soil composition conditions that enable the disease
Crop Name: {st.session_state.crop_name}
Crop Disease Name: {st.session_state.disease_name}
Soil Type: {soilt}
Weather Data: {weathersum}
Gravel Percentage in soil: {soilc_gravel}
Sand Percentage in soil: {soilc_sand}
Also mention giving tailored apporach 
Recommend suitable soil amendments (e.g., fertilizers) to address nutrient deficiencies or correct pH imbalances.
Give actionables to improve soil health




7.Effect of Environment:
consider temperature conditions that enable the disease(MANDATORY)
Strictly Provide analyses on Effect of Environment based on the given data (MANDATORY) Crop Name: {st.session_state.crop_name}
Crop Disease Name: {st.session_state.disease_name}
Soil Type: {soilt}
Weather Data: {weathersum}
Gravel Percentage in soil: {soilc_gravel}
Sand Percentage in soil: {soilc_sand}
Also mention giving tailored apporach 
Suggest how environmental factors like temperature and precipitation affect crop health and yield. and how to ptepare for them.

8.Impact on Yield:
Explain how each disease can impact crop yield.
Provide insights based on the given data.
suggest business aspect 


Methods suggested in Treatment Options and Preventive Measures should not overlap each other (MANDATORY)

"""

    if query:
        response = process_query(query)
        st.write(response)

        # Generate TXT file
        txt_data = generate_txt(response)

        # Provide a download button for the TXT file
        st.download_button(label="Download Report", data=txt_data, file_name="report_generated.txt", mime="text/plain")

query = st.text_input(t("RAG Assistance"))
if query:
    query = """You are a agricultural expert that provides valuable insights to poor uneducated farmers on the basis of the you are expert in :
                Crop disease management, Soil health managemnt and Environmental factors affecting agricultural productivity Ensure content is easily understandable by uneducated farmers and includes practical, actionable insights.(MANDATORY), also give expansive answers , extensive minimum 5 paragraphs"""   + query
    response = process_query(query)
    st.write(response)
    st.subheader(t("Print Report"))
    print_report(response)

# Option to update the Chroma vector store
if st.button(t("Update Database")):
    generate_data_store()
    st.success(t("Database updated successfully."))

# Add a footer at the bottom of the page
footer_html = """
    <style>
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgb(0,0,0,0);
        opacity:0.5
        color: blue;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    .footer a {
        color: #0a66c2;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    p{
        font-weight:600;
    }
    strong{
        font-weight:800;
        color:black
    }

    </style>

    <div class="footer">
        <p><strong>Developers:</strong></p>
        <p>👨🏼‍💻PRATYUSH PURI GOSWAMI: <a href="https://www.linkedin.com/in/pratyush-puri-goswami-036285200/" target="_blank">LinkedIn</a> | 
        👨🏼‍💻RISHABH SHARMA: <a href="https://www.linkedin.com/in/rishabh-sharma-5526b21bb/" target="_blank">LinkedIn</a></p>
    </div>
"""

# Inject HTML into the Streamlit app
st.markdown(footer_html, unsafe_allow_html=True)