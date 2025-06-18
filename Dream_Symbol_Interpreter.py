import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import base64
import requests
import os

st.set_page_config(page_title="Dream Interpreter", layout="centered")

MISTRAL_API_KEY = "7JgLxB9RiMIpwMYC4XmdMxYbVsCnYu2d"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions" 

def set_blurred_gradient_background(image_file_path):
    """
    Sets a blurred gradient background for the Streamlit app using a local image.
    """
    if not os.path.exists(image_file_path):
        st.warning(f"Background image '{image_file_path}' not found. Default background will be used.")
        return

    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    ext = image_file_path.split('.')[-1]

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                url("data:image/{ext};base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .block-container {{
            backdrop-filter: blur(8px);
            background-color: rgba(0,0,0,0.5); /* Slightly transparent background for content block */
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow for depth */
        }}
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stSelectbox label, .stTextArea label {{
            color: white !important; /* Ensure text is visible on dark background */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_blurred_gradient_background("bg.jpg")

@st.cache_data 
def load_and_prep_data(file_path):
    """
    Loads the dream symbol dataset, preprocesses it, and trains the model.
    """
    if not os.path.exists(file_path):
        st.error(f"Dataset '{file_path}' not found. Please ensure it's in the same directory.")
        st.stop() 
    df = pd.read_csv(file_path)
    df.dropna(subset=['Word', 'Interpretation'], inplace=True)

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Alphabet'].astype(str))

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['Word'].astype(str))
    y = df['label']

    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    return df, vectorizer, label_encoder, model

df, vectorizer, label_encoder, model = load_and_prep_data("combined_dataset.csv")


def generate_mistral_summary(symbols, interpretations, tone):
    """
    Generates a dream interpretation summary using the Mistral AI API.
    The response is prompted to be in approximately three complete paragraphs.
    """
    tone_instructions = {
        "Mystical": "Craft your interpretation with a spiritual and mystical voice, referencing universal energy, divine forces, or ancient symbolism.",
        "Poetic": "Write the interpretation in a poetic and metaphorical style, as if painting the meaning in words, filled with imagery and emotion.",
        "Psychological": "Write like a Jungian psychologist, focusing on subconscious motives, internal conflict, personal growth, and archetypes."
    }

    prompt = (
        f"You are a highly insightful and deeply intuitive dream analyst. The user had a dream containing:\n\n"
        + "\n".join([f"- {sym}: {interp}" for sym, interp in zip(symbols, interpretations)]) +
        f"\n\n{tone_instructions[tone]}\n"
        "Provide a comprehensive, insightful summary of what this dream could mean, presented in **exactly three complete paragraphs**."
    )

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-tiny", 
        "messages": [
            {"role": "system", "content": "You are a deeply intuitive dream analyst, known for your clear and profound insights."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7, 
        "max_tokens": 400 
    }

    try:
        response = requests.post(MISTRAL_URL, headers=headers, json=payload)
        response.raise_for_status() 
        result = response.json()
        if 'choices' in result and result['choices']:
            return result['choices'][0]['message']['content'].strip()
        else:
            st.error("Mistral AI API Error: No content in response. Please check your prompt or API configuration.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Mistral AI API Request Error: {e}. Please check your internet connection or API key.")
        return None
    except KeyError:
        st.error("Mistral AI API Error: Unexpected response format. Check the API documentation or your key.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None

st.title("💭 Dream Symbol Interpreter")
st.markdown("Unlock the hidden messages of your subconscious mind.")

user_input = st.text_area("Describe what you saw in your dream:", height=150, placeholder="e.g., I saw a black cat crossing my path, and then I was flying over a vast ocean.")
selected_tone = st.selectbox(
    "Choose the tone for your dream analysis:",
    ["Mystical", "Poetic", "Psychological"],
    help="Select a style for your dream's interpretation: Spiritual, expressive, or analytical."
)

if st.button("Interpret My Dream"):
    if not user_input.strip():
        st.warning("Please describe your dream before interpreting.")
    else:
        input_words = set(word.lower() for word in user_input.split())
        matches = df[df['Word'].str.lower().isin(input_words)]

        interpretations = []
        matched_symbols = []

        if not matches.empty:
            for _, row in matches.iterrows():
                interpretations.append(row['Interpretation'])
                matched_symbols.append(row['Word'])

            st.markdown("---")
            st.markdown(f"## 🔮 Dream Analysis ({selected_tone} Style)")
            with st.spinner("Analyzing your dream with Mistral AI..."):
                mistral_summary = generate_mistral_summary(matched_symbols, interpretations, selected_tone)
                if mistral_summary:
                    st.markdown(mistral_summary)
                else:
                    st.error("Could not generate a dream summary. Please try again later or check your API key.")
        else:
            st.warning("No recognizable dream symbols were found in your description from our database.")
            st.info("Try describing your dream with more common objects, animals, or actions.")