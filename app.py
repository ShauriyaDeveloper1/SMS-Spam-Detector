import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="SMS Spam Detection", layout="wide", page_icon="📱")

import os
import re
import joblib
import numpy as np
import pandas as pd
import hashlib
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, f1_score
from scipy.sparse import hstack
from collections import Counter
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Translation libraries
try:
    from langdetect import detect, DetectorFactory
    from deep_translator import GoogleTranslator
    DetectorFactory.seed = 0  # Make language detection deterministic
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False

FEATURES_LIST = [
    "msg_length","word_count","digit_count","uppercase_count",
    "special_char_count","url_count","digit_ratio","uppercase_ratio"
]

# ------------------ Authentication Functions ------------------
USERS_FILE = "users.json"

def validate_password(password: str) -> tuple:
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\":{}|<>)"
    return True, "Password is strong"

def validate_email(email: str) -> bool:
    """Validate email format using regex"""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_pattern, email) is not None

def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username: str, password: str, email: str) -> tuple:
    """Register a new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    # Validate email
    if not validate_email(email):
        return False, "Invalid email format. Please use a valid email address (e.g., user@example.com)"
    
    # Validate password strength
    password_valid, password_msg = validate_password(password)
    if not password_valid:
        return False, password_msg
    
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_users(users)
    return True, "Registration successful"

def verify_user(username: str, password: str) -> bool:
    """Verify user credentials"""
    users = load_users()
    if username not in users:
        return False
    return users[username]["password"] == hash_password(password)

def login_page():
    """Display login page"""
    st.markdown("<h1 style='text-align: center;'>🔐 Login</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Access SMS Spam Detection System</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please fill in all fields")
                elif verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.divider()
        if st.button("Don't have an account? Sign Up", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()

def signup_page():
    """Display signup page"""
    st.markdown("<h1 style='text-align: center;'>📝 Sign Up</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Create your account</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("signup_form"):
            username = st.text_input("Username", placeholder="Choose a username")
            email = st.text_input("Email", placeholder="user@example.com")
            st.caption("📧 Valid email required (e.g., user@example.com)")
            password = st.text_input("Password", type="password", placeholder="Strong password required")
            st.caption("🔒 Must be 8+ chars with uppercase, lowercase, digit & special character")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
            submit = st.form_submit_button("Sign Up", use_container_width=True)
            
            if submit:
                if not username or not email or not password or not confirm_password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(username, password, email)
                    if success:
                        st.success(message)
                        st.info("Please login with your new credentials")
                        st.session_state.show_signup = False
                        st.rerun()
                    else:
                        st.error(message)
        
        st.divider()
        if st.button("Already have an account? Login", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()

# ------------------ NLTK setup ------------------
@st.cache_resource
def ensure_nltk():
    pkgs = ["wordnet", "stopwords"]
    for p in pkgs:
        try:
            nltk.data.find(f"corpora/{p}")
        except LookupError:
            nltk.download(p, quiet=True)
    return True

ensure_nltk()

# ------------------ Helpers ------------------
def clean_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    try:
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in sw or len(w) <= 2]
    return " ".join(cleaned)

def extract_features(message: str):
    msg_length = len(message)
    word_count = len(message.split())
    digit_count = sum(c.isdigit() for c in message)
    uppercase_count = sum(c.isupper() for c in message)
    special_char_count = sum(not c.isalnum() and not c.isspace() for c in message)
    url_count = len(re.findall(r"http[s]?://|www\.", message))
    digit_ratio = digit_count / msg_length if msg_length else 0
    uppercase_ratio = uppercase_count / msg_length if msg_length else 0
    return [msg_length, word_count, digit_count, uppercase_count,
            special_char_count, url_count, digit_ratio, uppercase_ratio]

# ------------------ Translation Functions ------------------
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'zh-cn': 'Chinese (Simplified)', 'zh-tw': 'Chinese (Traditional)',
    'ar': 'Arabic', 'hi': 'Hindi', 'ja': 'Japanese', 'ko': 'Korean',
    'ru': 'Russian', 'pt': 'Portuguese', 'it': 'Italian', 'nl': 'Dutch',
    'tr': 'Turkish', 'pl': 'Polish', 'vi': 'Vietnamese', 'th': 'Thai',
    'id': 'Indonesian', 'bn': 'Bengali'
}

def detect_language(text: str) -> str:
    """Detect the language of the input text"""
    if not TRANSLATION_AVAILABLE:
        return 'en'
    try:
        lang = detect(text)
        return lang
    except Exception:
        return 'en'  # Default to English if detection fails

def translate_to_english(text: str, source_lang: str = None) -> dict:
    """Translate text to English if not already in English"""
    if not TRANSLATION_AVAILABLE:
        return {
            'translated_text': text,
            'original_text': text,
            'source_language': 'unknown',
            'language_name': 'Unknown',
            'was_translated': False,
            'error': 'Translation libraries not installed'
        }
    
    try:
        # Auto-detect if source language not provided
        if source_lang is None or source_lang == 'auto':
            source_lang = detect_language(text)
        
        # Skip translation if already English
        if source_lang == 'en':
            return {
                'translated_text': text,
                'original_text': text,
                'source_language': 'en',
                'language_name': 'English',
                'was_translated': False
            }
        
        # Translate to English using deep-translator
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        
        return {
            'translated_text': translated,
            'original_text': text,
            'source_language': source_lang,
            'language_name': LANGUAGE_NAMES.get(source_lang, source_lang.upper()),
            'was_translated': True
        }
    except Exception as e:
        # Fallback: return original text if translation fails
        return {
            'translated_text': text,
            'original_text': text,
            'source_language': 'unknown',
            'language_name': 'Unknown',
            'was_translated': False,
            'error': str(e)
        }

# ------------------ Data Loading ------------------
@st.cache_data(show_spinner=False)
def load_raw_dataset(path="spam.csv"):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="latin-1")[['v1', 'v2']]
    df.columns = ["label", "message"]
    df.drop_duplicates(inplace=True)
    return df

@st.cache_data(show_spinner=False)
def prepare_dataset(df: pd.DataFrame):
    df = df.copy()
    df["cleaned_message"] = df["message"].apply(clean_text).fillna("")
    df = df[df["cleaned_message"].str.strip().astype(bool)]
    feature_rows = df["message"].apply(extract_features).tolist()
    feature_df = pd.DataFrame(feature_rows, columns=FEATURES_LIST, index=df.index)
    final_df = pd.concat([df[["label", "cleaned_message"]], feature_df], axis=1)
    return final_df

# ------------------ Training ------------------
def train_models(df: pd.DataFrame):
    X_text = df["cleaned_message"]
    y = df["label"].map({"ham": 0, "spam": 1})
    feats = df[FEATURES_LIST]

    X_train_text, X_test_text, y_train, y_test, train_feats, test_feats = train_test_split(
        X_text, y, feats, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    scaler = StandardScaler()
    lr_train_feats = scaler.fit_transform(train_feats)
    lr_test_feats = scaler.transform(test_feats)
    nb_train_feats = train_feats.values
    nb_test_feats = test_feats.values

    X_train_nb = hstack([X_train_tfidf, nb_train_feats])
    X_test_nb = hstack([X_test_tfidf, nb_test_feats])
    X_train_lr = hstack([X_train_tfidf, lr_train_feats])
    X_test_lr = hstack([X_test_tfidf, lr_test_feats])

    nb = MultinomialNB()
    nb.fit(X_train_nb, y_train)
    lr = LogisticRegression(max_iter=400, class_weight='balanced', solver='liblinear')
    lr.fit(X_train_lr, y_train)

    nb_pred = nb.predict(X_test_nb)
    lr_pred = lr.predict(X_test_lr)
    nb_acc = accuracy_score(y_test, nb_pred)
    lr_acc = accuracy_score(y_test, lr_pred)

    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(nb, 'models/naive_bayes_model.pkl')
    joblib.dump(lr, 'models/logistic_regression_model.pkl')

    return {
        "vectorizer": vectorizer,
        "scaler": scaler,
        "nb": nb,
        "lr": lr,
        "nb_acc": nb_acc,
        "lr_acc": lr_acc,
        "y_test": y_test,
        "nb_pred": nb_pred,
        "lr_pred": lr_pred
    }

# ------------------ Model Loading ------------------
def load_models():
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        scaler = joblib.load('models/scaler.pkl')
        nb = joblib.load('models/naive_bayes_model.pkl')
        lr = joblib.load('models/logistic_regression_model.pkl')
        return vectorizer, scaler, nb, lr
    except Exception:
        return None, None, None, None

# ------------------ Prediction ------------------
def predict_message(message: str, vectorizer, scaler, nb, lr):
    clean = clean_text(message)
    feats = np.array(extract_features(message)).reshape(1, -1)
    tfidf = vectorizer.transform([clean])
    scaled_feats = scaler.transform(pd.DataFrame(feats, columns=FEATURES_LIST))
    nb_input = hstack([tfidf, feats])
    lr_input = hstack([tfidf, scaled_feats])
    nb_prob = nb.predict_proba(nb_input)[0]
    lr_prob = lr.predict_proba(lr_input)[0]
    nb_pred = int(nb_prob[1] > nb_prob[0])
    lr_pred = int(lr_prob[1] > lr_prob[0])
    return {
        "features": feats.flatten().tolist(),
        "nb_prob": nb_prob,
        "lr_prob": lr_prob,
        "nb_pred": nb_pred,
        "lr_pred": lr_pred,
        "clean": clean,
    }

# ------------------ Theme Management ------------------
THEMES = {
    "Default": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
        "font": "sans serif"
    },
    "Dark": {
        "primaryColor": "#00d4ff",
        "backgroundColor": "#0e1117",
        "secondaryBackgroundColor": "#262730",
        "textColor": "#fafafa",
        "font": "sans serif"
    },
    "Green": {
        "primaryColor": "#28a745",
        "backgroundColor": "#f8fff8",
        "secondaryBackgroundColor": "#e8f5e9",
        "textColor": "#1b5e20",
        "font": "sans serif"
    },
    "Blue Ocean": {
        "primaryColor": "#0077be",
        "backgroundColor": "#f0f8ff",
        "secondaryBackgroundColor": "#e1f5fe",
        "textColor": "#01579b",
        "font": "sans serif"
    },
    "High Contrast": {
        "primaryColor": "#ffff00",
        "backgroundColor": "#000000",
        "secondaryBackgroundColor": "#1a1a1a",
        "textColor": "#ffffff",
        "font": "monospace"
    },
    "Sunset": {
        "primaryColor": "#ff6b35",
        "backgroundColor": "#fff5f0",
        "secondaryBackgroundColor": "#ffe8e0",
        "textColor": "#4a1a00",
        "font": "sans serif"
    },
    "Purple Dream": {
        "primaryColor": "#9c27b0",
        "backgroundColor": "#faf4ff",
        "secondaryBackgroundColor": "#f3e5f5",
        "textColor": "#4a148c",
        "font": "sans serif"
    }
}

def apply_theme(theme_name: str):
    """Apply selected theme using custom CSS"""
    theme = THEMES.get(theme_name, THEMES["Default"])
    
    css = f"""
    <style>
        /* Main background */
        .stApp {{
            background-color: {theme['backgroundColor']};
            color: {theme['textColor']};
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {theme['secondaryBackgroundColor']};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {theme['primaryColor']} !important;
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: {theme['primaryColor']};
            color: {theme['backgroundColor']};
            border: none;
            border-radius: 5px;
            font-weight: bold;
        }}
        
        .stButton>button:hover {{
            opacity: 0.8;
            transform: translateY(-2px);
            transition: all 0.3s ease;
        }}
        
        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {theme['primaryColor']} !important;
        }}
        
        /* Text inputs */
        .stTextInput>div>div>input {{
            background-color: {theme['secondaryBackgroundColor']};
            color: {theme['textColor']};
            border: 2px solid {theme['primaryColor']};
        }}
        
        /* Text areas */
        .stTextArea>div>div>textarea {{
            background-color: {theme['secondaryBackgroundColor']};
            color: {theme['textColor']};
            border: 2px solid {theme['primaryColor']};
        }}
        
        /* Selectbox */
        .stSelectbox>div>div>div {{
            background-color: {theme['secondaryBackgroundColor']};
            color: {theme['textColor']};
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {theme['secondaryBackgroundColor']};
            color: {theme['primaryColor']} !important;
            border-radius: 5px;
        }}
        
        /* Progress bar */
        .stProgress>div>div>div {{
            background-color: {theme['primaryColor']};
        }}
        
        /* Dataframe */
        [data-testid="stDataFrame"] {{
            background-color: {theme['secondaryBackgroundColor']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme['secondaryBackgroundColor']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {theme['textColor']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme['primaryColor']};
            color: {theme['backgroundColor']};
        }}
        
        /* Success/Error/Warning boxes */
        .stSuccess {{
            background-color: {theme['secondaryBackgroundColor']};
            border-left: 5px solid #28a745;
        }}
        
        .stError {{
            background-color: {theme['secondaryBackgroundColor']};
            border-left: 5px solid #dc3545;
        }}
        
        .stWarning {{
            background-color: {theme['secondaryBackgroundColor']};
            border-left: 5px solid #ffc107;
        }}
        
        .stInfo {{
            background-color: {theme['secondaryBackgroundColor']};
            border-left: 5px solid {theme['primaryColor']};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ------------------ Session State Initialization ------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'theme' not in st.session_state:
    st.session_state.theme = "Default"

# ------------------ Authentication Check ------------------
if not st.session_state.logged_in:
    if st.session_state.show_signup:
        signup_page()
    else:
        login_page()
    st.stop()

# ------------------ Streamlit UI (Protected Area) ------------------
# Apply selected theme
apply_theme(st.session_state.theme)

# Logout button in sidebar
with st.sidebar:
    st.markdown(f"### 👤 User: {st.session_state.username}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()
    st.divider()
    
    # Theme selector
    st.markdown("### 🎨 Theme Settings")
    selected_theme = st.selectbox(
        "Choose Theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        help="Select your preferred color scheme"
    )
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    # Theme preview
    with st.expander("🎨 Theme Preview", expanded=False):
        theme = THEMES[st.session_state.theme]
        st.markdown(f"**Primary Color:** `{theme['primaryColor']}`")
        st.markdown(f"<div style='background-color:{theme['primaryColor']};height:30px;border-radius:5px;'></div>", unsafe_allow_html=True)
        st.markdown(f"**Background:** `{theme['backgroundColor']}`")
        st.markdown(f"<div style='background-color:{theme['backgroundColor']};height:30px;border-radius:5px;border:1px solid #ccc;'></div>", unsafe_allow_html=True)
        st.markdown(f"**Text Color:** `{theme['textColor']}`")
        st.markdown(f"<div style='background-color:{theme['textColor']};height:30px;border-radius:5px;'></div>", unsafe_allow_html=True)
    
    st.divider()

st.title("📱 SMS Spam Detection System")

# Keyboard shortcuts info
with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
    col_ks1, col_ks2 = st.columns(2)
    with col_ks1:
        st.markdown("""
        - **Ctrl+Enter**: Analyze message
        - **Ctrl+L**: Clear message input
        - **Ctrl+S**: Save prediction history
        """)
    with col_ks2:
        st.markdown("""
        - **Ctrl+H**: Toggle history view
        - **Ctrl+M**: Focus message input
        - **Escape**: Close all expanders
        """)

# Add keyboard shortcut handler
if 'trigger_analyze' not in st.session_state:
    st.session_state.trigger_analyze = False
if 'trigger_clear' not in st.session_state:
    st.session_state.trigger_clear = False
if 'show_history_expanded' not in st.session_state:
    st.session_state.show_history_expanded = False

raw_df = load_raw_dataset()
if raw_df is None:
    st.error("Dataset 'spam.csv' not found in project root.")
else:
    with st.expander("📊 Dataset Analytics", expanded=False):
        analytics_tabs = st.tabs(["Overview", "Word Frequency", "Length Distribution"])
        
        with analytics_tabs[0]:
            st.write(raw_df.head())
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Messages", len(raw_df))
            with col_b:
                st.metric("Unique Messages", raw_df['message'].nunique())
            dist_fig, ax = plt.subplots(figsize=(4,3))
            # Use hue + legend removal to satisfy seaborn >=0.14 deprecation warning
            sns.countplot(data=raw_df, x='label', hue='label', palette={'ham': 'green', 'spam': 'red'}, ax=ax)
            if ax.legend_:
                ax.legend_.remove()
            ax.set_title("Class Distribution")
            st.pyplot(dist_fig)
        
        with analytics_tabs[1]:
            st.subheader("Word Frequency Analysis")
            top_n = st.slider("Top N words", 10, 50, 20)
            spam_msgs = raw_df[raw_df['label'] == 'spam']['message'].str.lower().str.split()
            ham_msgs = raw_df[raw_df['label'] == 'ham']['message'].str.lower().str.split()
            spam_words = Counter([w for msg in spam_msgs for w in msg if len(w) > 2])
            ham_words = Counter([w for msg in ham_msgs for w in msg if len(w) > 2])
            
            col_wf1, col_wf2 = st.columns(2)
            with col_wf1:
                spam_top = pd.DataFrame(spam_words.most_common(top_n), columns=['Word', 'Frequency'])
                fig_spam = px.bar(spam_top, x='Frequency', y='Word', orientation='h', 
                                 title=f'Top {top_n} Spam Words', color='Frequency', color_continuous_scale='Reds')
                st.plotly_chart(fig_spam, use_container_width=True)
            with col_wf2:
                ham_top = pd.DataFrame(ham_words.most_common(top_n), columns=['Word', 'Frequency'])
                fig_ham = px.bar(ham_top, x='Frequency', y='Word', orientation='h', 
                                title=f'Top {top_n} Ham Words', color='Frequency', color_continuous_scale='Greens')
                st.plotly_chart(fig_ham, use_container_width=True)
        
        with analytics_tabs[2]:
            st.subheader("Message Length Distribution")
            raw_df['msg_len'] = raw_df['message'].str.len()
            fig_len = px.histogram(raw_df, x='msg_len', color='label', 
                                  color_discrete_map={'spam': 'red', 'ham': 'green'},
                                  title='Message Length Distribution by Class',
                                  labels={'msg_len': 'Message Length (characters)'},
                                  barmode='overlay', opacity=0.7, nbins=50)
            st.plotly_chart(fig_len, use_container_width=True)
            
            col_len1, col_len2 = st.columns(2)
            with col_len1:
                st.metric("Avg Spam Length", f"{raw_df[raw_df['label']=='spam']['msg_len'].mean():.1f} chars")
            with col_len2:
                st.metric("Avg Ham Length", f"{raw_df[raw_df['label']=='ham']['msg_len'].mean():.1f} chars")

prepared_df = prepare_dataset(raw_df) if raw_df is not None else None

vectorizer, scaler, nb_model, lr_model = load_models()
models_loaded = all([vectorizer, scaler, nb_model, lr_model])

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ------------------ Cached evaluation (probabilities) ------------------
@st.cache_data(show_spinner=False)
def evaluate_models_for_curves(prepared_df, _vectorizer, _scaler, _nb_model, _lr_model):
    try:
        if not all([prepared_df is not None, _vectorizer, _scaler, _nb_model, _lr_model]):
            return None
        X_text = prepared_df["cleaned_message"]
        y = prepared_df["label"].map({"ham":0, "spam":1})
        feats = prepared_df[FEATURES_LIST]
        X_train_text, X_test_text, y_train, y_test, train_feats, test_feats = train_test_split(
            X_text, y, feats, test_size=0.2, random_state=42, stratify=y
        )
        X_test_tfidf = _vectorizer.transform(X_test_text)
        lr_test_scaled = _scaler.transform(test_feats)
        nb_test_feats = test_feats.values
        nb_input = hstack([X_test_tfidf, nb_test_feats])
        lr_input = hstack([X_test_tfidf, lr_test_scaled])
        nb_probs = _nb_model.predict_proba(nb_input)[:,1]
        lr_probs = _lr_model.predict_proba(lr_input)[:,1]
        return {
            'y_test': y_test.values,
            'nb_probs': nb_probs,
            'lr_probs': lr_probs
        }
    except Exception:
        return None

with st.sidebar:
    st.header("⚙️ Controls")
    st.write("Model status:" + (" ✅ Loaded" if models_loaded else " ❌ Not found"))
    if st.button("🔄 Train Models"):
        if prepared_df is None:
            st.error("Cannot train: dataset missing or invalid.")
        else:
            with st.spinner("Training models..."):
                results = train_models(prepared_df)
            st.success(f"Training complete. NB Acc: {results['nb_acc']:.3f} | LR Acc: {results['lr_acc']:.3f}")
            vectorizer, scaler, nb_model, lr_model = load_models()
            models_loaded = True
    model_choice = st.radio("Active Model for Single Prediction", ["Naive Bayes", "Logistic Regression", "Compare Both"], index=2)
    st.divider()
    st.markdown("**Batch Classification**")
    batch_file = st.file_uploader("Upload CSV (column 'message')", type=['csv'])
    text_files = st.file_uploader("Upload Text Files (bulk)", type=['txt'], accept_multiple_files=True)
    run_batch = st.button("Run Batch Prediction")

if batch_file and run_batch and models_loaded:
    batch_df = pd.read_csv(batch_file)
    if 'message' not in batch_df.columns:
        st.error("Uploaded file must contain a 'message' column")
    else:
        # Add translation for batch processing
        if TRANSLATION_AVAILABLE:
            with st.spinner("🌍 Translating messages..."):
                batch_df['detected_lang'] = batch_df['message'].apply(detect_language)
                batch_df['translated'] = batch_df['message'].apply(
                    lambda x: translate_to_english(x)['translated_text']
                )
                analysis_messages = batch_df['translated']
        else:
            analysis_messages = batch_df['message']
        
        batch_df['cleaned'] = analysis_messages.apply(clean_text)
        preds = []
        for msg in analysis_messages:
            res = predict_message(msg, vectorizer, scaler, nb_model, lr_model)
            chosen = res['nb_pred'] if model_choice == 'Naive Bayes' else res['lr_pred']
            if model_choice == 'Compare Both':
                chosen = res['nb_pred'] if res['nb_pred'] == res['lr_pred'] else res['lr_pred']
            preds.append('spam' if chosen == 1 else 'ham')
        batch_df['prediction'] = preds
        st.subheader("Batch Results")
        if TRANSLATION_AVAILABLE and 'detected_lang' in batch_df.columns:
            display_cols = ['message', 'detected_lang', 'prediction']
            st.dataframe(batch_df[display_cols])
        else:
            st.dataframe(batch_df[['message','prediction']])
        spam_pct = (batch_df['prediction'] == 'spam').mean() * 100
        st.info(f"Spam detected: {spam_pct:.1f}% of uploaded messages.")
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", csv, "batch_results.csv", "text/csv")

if text_files and run_batch and models_loaded:
    st.subheader("📄 Bulk Text File Analysis")
    all_results = []
    for txt_file in text_files:
        content = txt_file.read().decode('utf-8')
        
        # Translate if enabled
        if TRANSLATION_AVAILABLE:
            trans_result = translate_to_english(content)
            analysis_content = trans_result['translated_text']
            detected_lang = trans_result['language_name']
        else:
            analysis_content = content
            detected_lang = 'N/A'
        
        res = predict_message(analysis_content, vectorizer, scaler, nb_model, lr_model)
        chosen = res['nb_pred'] if model_choice == 'Naive Bayes' else res['lr_pred']
        if model_choice == 'Compare Both':
            chosen = res['nb_pred'] if res['nb_pred'] == res['lr_pred'] else res['lr_pred']
        result_dict = {
            'filename': txt_file.name,
            'prediction': 'spam' if chosen == 1 else 'ham',
            'confidence': max(res['nb_prob']) if model_choice == 'Naive Bayes' else max(res['lr_prob']),
            'length': len(content)
        }
        if TRANSLATION_AVAILABLE:
            result_dict['language'] = detected_lang
        all_results.append(result_dict)
    results_df = pd.DataFrame(all_results)
    st.dataframe(results_df)
    spam_count = (results_df['prediction'] == 'spam').sum()
    st.metric("Files Classified as Spam", f"{spam_count}/{len(text_files)}")
    csv_bulk = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Bulk Results", csv_bulk, "bulk_text_results.csv", "text/csv")

st.subheader("🔍 Message Analysis")

# Multi-language support toggle
if TRANSLATION_AVAILABLE:
    col_lang1, col_lang2 = st.columns([3, 1])
    with col_lang1:
        auto_translate = st.checkbox("🌍 Auto-translate non-English messages to English for analysis", value=True, 
                                     help="Automatically detect and translate messages in other languages")
    with col_lang2:
        if auto_translate:
            manual_lang = st.selectbox("Source Language", 
                                      ['auto', 'es', 'fr', 'de', 'zh-cn', 'ar', 'hi', 'ja', 'ko', 'ru', 'pt', 'it'],
                                      format_func=lambda x: 'Auto-detect' if x == 'auto' else LANGUAGE_NAMES.get(x, x),
                                      help="Override auto-detection if needed")
        else:
            manual_lang = 'auto'
else:
    auto_translate = False
    manual_lang = 'auto'
    # Translation feature is optional - no message shown

# Voice Input Section with audio-recorder-streamlit (more reliable)
col_voice1, col_voice2 = st.columns([3, 1])
with col_voice1:
    st.markdown("**🎤 Voice Input (optional)**")
with col_voice2:
    enable_tts = st.checkbox("🔊 Enable TTS", value=False, help="Text-to-Speech for results")

# Initialize recording timer in session state
if 'voice_duration' not in st.session_state:
    st.session_state.voice_duration = 0
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""

try:
    from audio_recorder_streamlit import audio_recorder
    
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#3498db",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=10.0,
        sample_rate=16000,
        key="audio_recorder"
    )
    
    # Process voice input when audio is received
    if audio_bytes:
        try:
            import speech_recognition as sr
            import wave
            import tempfile
            
            with st.spinner("🎯 Transcribing voice..."):
                # Save audio bytes to temporary WAV file
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_wav.write(audio_bytes)
                temp_wav.close()
                
                # Calculate duration using wave module
                with wave.open(temp_wav.name, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                
                # Speech recognition
                recognizer = sr.Recognizer()
                recognizer.energy_threshold = 300
                recognizer.dynamic_energy_threshold = True
                
                with sr.AudioFile(temp_wav.name) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                    voice_text = recognizer.recognize_google(audio_data)
                
                # Auto-translate voice input if not in English
                if TRANSLATION_AVAILABLE and voice_text:
                    detected_lang = detect_language(voice_text)
                    if detected_lang != 'en':
                        trans_result = translate_to_english(voice_text, detected_lang)
                        if trans_result['was_translated']:
                            st.success(f"✅ Transcribed ({duration:.1f}s) | 🌍 {trans_result['language_name']} → English\n\nOriginal: {voice_text}\nTranslated: {trans_result['translated_text']}")
                            voice_text = trans_result['translated_text']
                        else:
                            st.success(f"✅ Transcribed ({duration:.1f}s): {voice_text}")
                    else:
                        st.success(f"✅ Transcribed ({duration:.1f}s): {voice_text}")
                else:
                    st.success(f"✅ Transcribed ({duration:.1f}s): {voice_text}")
                
                # Store voice text in session state
                st.session_state.voice_text = voice_text
                
                os.unlink(temp_wav.name)
                
        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand audio. Please speak clearly.")
        except sr.RequestError as e:
            st.error(f"❌ Google Speech API error: {e}")
        except Exception as e:
            st.warning(f"⚠️ Voice error: {str(e)}")
            
except ImportError:
    st.info("📝 Voice input requires 'audio-recorder-streamlit' package. Using text input only.")

# Keyboard shortcuts implementation using custom JS
keyboard_js = """
<script>
const doc = window.parent.document;

doc.addEventListener('keydown', function(e) {
    // Ctrl+Enter: Trigger analyze
    if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        const analyzeBtn = doc.querySelector('button[kind="primary"]');
        if (analyzeBtn) analyzeBtn.click();
    }
    
    // Ctrl+L: Clear input
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        const textareas = doc.querySelectorAll('textarea');
        if (textareas.length > 0) textareas[0].value = '';
    }
    
    // Ctrl+M: Focus message input
    if (e.ctrlKey && e.key === 'm') {
        e.preventDefault();
        const textareas = doc.querySelectorAll('textarea');
        if (textareas.length > 0) textareas[0].focus();
    }
    
    // Escape: Collapse all expanders
    if (e.key === 'Escape') {
        const expanders = doc.querySelectorAll('[data-testid="stExpander"] details[open]');
        expanders.forEach(exp => exp.removeAttribute('open'));
    }
});
</script>
"""

import streamlit.components.v1 as components
components.html(keyboard_js, height=0)

# Clear detection
if st.session_state.trigger_clear:
    st.session_state.voice_text = ""
    st.session_state.trigger_clear = False

# Use voice_text directly as the text area value without conflicting key
message_input = st.text_area(
    "Enter an SMS message (or use voice above) ⌨️ Ctrl+Enter to analyze", 
    value=st.session_state.voice_text, 
    height=150
)

# Update session state if user manually edits
if message_input != st.session_state.voice_text:
    st.session_state.voice_text = message_input

# Message playground transformations
with st.expander("Message Playground (optional transforms)"):
    col_pg1, col_pg2, col_pg3 = st.columns(3)
    remove_digits = col_pg1.checkbox("Remove digits")
    remove_punct = col_pg2.checkbox("Remove punctuation")
    lowercase_all = col_pg3.checkbox("Force lowercase", value=True)
    transformed_message = message_input
    if transformed_message:
        if lowercase_all:
            transformed_message = transformed_message.lower()
        if remove_digits:
            transformed_message = re.sub(r"\d+", "", transformed_message)
        if remove_punct:
            transformed_message = re.sub(r"[\W_]", " ", transformed_message)
        st.text_area("Transformed message (used for analysis if Analyze clicked)", transformed_message, height=120)

# Action buttons row
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
with col_btn1:
    analyze = st.button("🔍 Analyze Message (Ctrl+Enter)", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🗑️ Clear (Ctrl+L)", use_container_width=True):
        st.session_state.trigger_clear = True
        st.rerun()
with col_btn3:
    if st.button("⌨️ Shortcuts", use_container_width=True):
        st.info("Press Ctrl+H to toggle prediction history!")

if analyze:
    target_message = transformed_message if transformed_message else message_input
    if not target_message.strip():
        st.warning("Please enter a message.")
    elif not models_loaded:
        st.error("Models not loaded. Train first from the sidebar.")
    else:
        # Language detection and translation
        translation_result = None
        analysis_text = target_message
        
        if auto_translate and TRANSLATION_AVAILABLE:
            with st.spinner("🌍 Detecting language..."):
                translation_result = translate_to_english(target_message, source_lang=manual_lang if manual_lang != 'auto' else None)
            
            if translation_result['was_translated']:
                st.success(f"🌍 Detected: **{translation_result['language_name']}** → Translated to English for analysis")
                col_orig1, col_orig2 = st.columns([1, 3])
                with col_orig1:
                    st.metric("Original Language", translation_result['language_name'])
                with col_orig2:
                    with st.expander("👁️ View Original Message", expanded=False):
                        st.text(translation_result['original_text'])
                        st.caption(f"Translated text: {translation_result['translated_text']}")
                analysis_text = translation_result['translated_text']
            elif 'error' in translation_result and translation_result['error']:
                st.warning(f"⚠️ Translation issue: {translation_result['error']}. Using original text.")
        
        result = predict_message(analysis_text, vectorizer, scaler, nb_model, lr_model)
        nb_prob = result['nb_prob']
        lr_prob = result['lr_prob']
        nb_pred = 'SPAM' if result['nb_pred'] else 'HAM'
        lr_pred = 'SPAM' if result['lr_pred'] else 'HAM'

        if model_choice == 'Naive Bayes':
            final_pred = nb_pred
            final_prob = max(nb_prob)
        elif model_choice == 'Logistic Regression':
            final_pred = lr_pred
            final_prob = max(lr_prob)
        else:
            final_pred = nb_pred if nb_pred == lr_pred else lr_pred
            final_prob = max(nb_prob) if nb_pred == lr_pred else max(lr_prob)

        color = 'red' if final_pred == 'SPAM' else 'green'
        st.markdown(f"### Result: <span style='color:{color}'>{final_pred}</span> (confidence {final_prob*100:.1f}%)", unsafe_allow_html=True)
        
        # Text-to-Speech for result
        if enable_tts:
            try:
                from gtts import gTTS
                import base64
                tts_text = f"This message is classified as {final_pred} with {final_prob*100:.0f} percent confidence"
                tts = gTTS(text=tts_text, lang='en', slow=False)
                tts_file = "temp_tts.mp3"
                tts.save(tts_file)
                with open(tts_file, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    b64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                    st.markdown(audio_html, unsafe_allow_html=True)
                os.remove(tts_file)
            except Exception as e:
                st.info(f"TTS unavailable: {e}")
        
        # Add to prediction history
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'message': target_message[:100] + '...' if len(target_message) > 100 else target_message,
            'prediction': final_pred,
            'confidence': f"{final_prob*100:.1f}%",
            'nb_pred': nb_pred,
            'lr_pred': lr_pred
        })

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Naive Bayes Probabilities**")
            st.progress(nb_prob[1])
            st.caption(f"Spam: {nb_prob[1]*100:.2f}% | Ham: {nb_prob[0]*100:.2f}%")
        with col2:
            st.markdown("**Logistic Regression Probabilities**")
            st.progress(lr_prob[1])
            st.caption(f"Spam: {lr_prob[1]*100:.2f}% | Ham: {lr_prob[0]*100:.2f}%")

        feat_dict = {f: v for f, v in zip(FEATURES_LIST, result['features'])}
        st.markdown("**Extracted Features**")
        st.table(pd.DataFrame(feat_dict, index=['value']).T)
        st.code(result['clean'], language='text')

# ------------------ Prediction History ------------------
if len(st.session_state.prediction_history) > 0:
    history_expanded = st.session_state.show_history_expanded
    with st.expander(f"📜 Prediction History ({len(st.session_state.prediction_history)} predictions) - Press Ctrl+H to toggle", expanded=history_expanded):
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True)
        csv_history = history_df.to_csv(index=False).encode('utf-8')
        col_h1, col_h2 = st.columns([1, 3])
        with col_h1:
            st.download_button("📥 Export History", csv_history, "prediction_history.csv", "text/csv")
        with col_h2:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

# ------------------ Dataset Filter ------------------
if prepared_df is not None:
    with st.expander("Filter Dataset Messages"):
        query = st.text_input("Substring search (case-insensitive)")
        if query:
            filtered = prepared_df[prepared_df['cleaned_message'].str.contains(query, case=False, na=False)].head(50)
            st.write(f"Showing {len(filtered)} messages (limit 50).")
            st.dataframe(filtered[['label','cleaned_message']])

# ------------------ Interactive Model Insights ------------------
if models_loaded:
    st.header("📊 Interactive Model Insights")
    eval_store = evaluate_models_for_curves(prepared_df, vectorizer, scaler, nb_model, lr_model)
    if eval_store:
        y_test = eval_store['y_test']
        nb_probs = eval_store['nb_probs']
        lr_probs = eval_store['lr_probs']
        
        # Model Comparison Dashboard
        st.subheader("⚡ Real-time Model Comparison")
        nb_preds_default = (nb_probs >= 0.5).astype(int)
        lr_preds_default = (lr_probs >= 0.5).astype(int)
        
        comparison_data = {
            'Model': ['Naive Bayes', 'Logistic Regression'],
            'Accuracy': [accuracy_score(y_test, nb_preds_default), accuracy_score(y_test, lr_preds_default)],
            'Precision': [confusion_matrix(y_test, nb_preds_default)[1,1] / (confusion_matrix(y_test, nb_preds_default)[0,1] + confusion_matrix(y_test, nb_preds_default)[1,1]) if (confusion_matrix(y_test, nb_preds_default)[0,1] + confusion_matrix(y_test, nb_preds_default)[1,1]) > 0 else 0,
                         confusion_matrix(y_test, lr_preds_default)[1,1] / (confusion_matrix(y_test, lr_preds_default)[0,1] + confusion_matrix(y_test, lr_preds_default)[1,1]) if (confusion_matrix(y_test, lr_preds_default)[0,1] + confusion_matrix(y_test, lr_preds_default)[1,1]) > 0 else 0],
            'Recall': [confusion_matrix(y_test, nb_preds_default)[1,1] / (confusion_matrix(y_test, nb_preds_default)[1,0] + confusion_matrix(y_test, nb_preds_default)[1,1]) if (confusion_matrix(y_test, nb_preds_default)[1,0] + confusion_matrix(y_test, nb_preds_default)[1,1]) > 0 else 0,
                      confusion_matrix(y_test, lr_preds_default)[1,1] / (confusion_matrix(y_test, lr_preds_default)[1,0] + confusion_matrix(y_test, lr_preds_default)[1,1]) if (confusion_matrix(y_test, lr_preds_default)[1,0] + confusion_matrix(y_test, lr_preds_default)[1,1]) > 0 else 0],
            'F1-Score': [f1_score(y_test, nb_preds_default), f1_score(y_test, lr_preds_default)]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], color='lightgreen'), use_container_width=True)
        
        fig_comparison = px.bar(comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                               x='Metric', y='Score', color='Model', barmode='group',
                               title='Model Performance Comparison', color_discrete_map={'Naive Bayes': 'blue', 'Logistic Regression': 'green'})
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        tabs = st.tabs(["Threshold Tuning", "ROC Curves", "Precision-Recall", "Feature Importance"])

        # Threshold Tuning
        with tabs[0]:
            st.markdown("Adjust threshold to classify as SPAM (prob >= threshold).")
            threshold = st.slider("Spam threshold", 0.1, 0.9, 0.5, 0.01)
            def metrics_for(probs, y_true, thr):
                preds = (probs >= thr).astype(int)
                cm = confusion_matrix(y_true, preds)
                tn, fp, fn, tp = cm.ravel()
                acc = (tp + tn) / cm.sum()
                precision = tp / (tp + fp) if (tp + fp) else 0
                recall = tp / (tp + fn) if (tp + fn) else 0
                return acc, precision, recall, cm
            nb_acc, nb_prec, nb_rec, nb_cm = metrics_for(nb_probs, y_test, threshold)
            lr_acc, lr_prec, lr_rec, lr_cm = metrics_for(lr_probs, y_test, threshold)
            colT1, colT2 = st.columns(2)
            with colT1:
                st.subheader("Naive Bayes")
                st.metric("Accuracy", f"{nb_acc:.3f}")
                st.metric("Precision", f"{nb_prec:.3f}")
                st.metric("Recall", f"{nb_rec:.3f}")
                fig_nb = px.imshow(nb_cm, text_auto=True, color_continuous_scale="Blues", title="NB Confusion Matrix")
                st.plotly_chart(fig_nb, use_container_width=True)
            with colT2:
                st.subheader("Logistic Regression")
                st.metric("Accuracy", f"{lr_acc:.3f}")
                st.metric("Precision", f"{lr_prec:.3f}")
                st.metric("Recall", f"{lr_rec:.3f}")
                fig_lr = px.imshow(lr_cm, text_auto=True, color_continuous_scale="Greens", title="LR Confusion Matrix")
                st.plotly_chart(fig_lr, use_container_width=True)

        # ROC Curves
        with tabs[1]:
            fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probs)
            fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
            auc_nb = auc(fpr_nb, tpr_nb)
            auc_lr = auc(fpr_lr, tpr_lr)
            roc_df_nb = pd.DataFrame({'FPR': fpr_nb, 'TPR': tpr_nb, 'Model': 'Naive Bayes'})
            roc_df_lr = pd.DataFrame({'FPR': fpr_lr, 'TPR': tpr_lr, 'Model': 'Logistic Regression'})
            roc_df = pd.concat([roc_df_nb, roc_df_lr])
            fig_roc = px.line(roc_df, x='FPR', y='TPR', color='Model', title=f"ROC Curves (AUC NB={auc_nb:.3f}, LR={auc_lr:.3f})")
            fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
            st.plotly_chart(fig_roc, use_container_width=True)

        # Precision-Recall
        with tabs[2]:
            pr_nb_p, pr_nb_r, _ = precision_recall_curve(y_test, nb_probs)
            pr_lr_p, pr_lr_r, _ = precision_recall_curve(y_test, lr_probs)
            pr_nb_df = pd.DataFrame({'Recall': pr_nb_r, 'Precision': pr_nb_p, 'Model': 'Naive Bayes'})
            pr_lr_df = pd.DataFrame({'Recall': pr_lr_r, 'Precision': pr_lr_p, 'Model': 'Logistic Regression'})
            pr_df = pd.concat([pr_nb_df, pr_lr_df])
            fig_pr = px.line(pr_df, x='Recall', y='Precision', color='Model', title='Precision-Recall Curves')
            st.plotly_chart(fig_pr, use_container_width=True)

        # Feature Importance (Logistic Regression)
        with tabs[3]:
            try:
                coef = lr_model.coef_[0]
                vocab = list(vectorizer.get_feature_names_out()) + FEATURES_LIST
                if len(vocab) == len(coef):
                    imp_df = pd.DataFrame({'feature': vocab, 'coef': coef})
                    imp_df['abs_coef'] = imp_df['coef'].abs()
                    top_imp = imp_df.sort_values('abs_coef', ascending=False).head(20)
                    fig_imp = px.bar(top_imp.sort_values('abs_coef'), x='abs_coef', y='feature', orientation='h',
                                     color='coef', color_continuous_scale='RdBu', title='Top 20 Features (|Coefficient|)')
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.caption("Positive coefficients push toward SPAM; negative toward HAM.")
                else:
                    st.info("Coefficient length mismatch; cannot map feature importance reliably.")
            except Exception as e:
                st.warning(f"Feature importance unavailable: {e}")
    else:
        st.info("Train models to view interactive insights.")

if models_loaded:
    with st.expander("Model Evaluation (from last training run)"):
        # Recompute evaluation quickly using stored artifacts if possible
        # Load prepared dataset for confusion matrix metrics
        try:
            eval_df = prepared_df
            if eval_df is not None:
                # Quick re-split for metrics (non-cached to reflect underlying logic)
                X_text = eval_df["cleaned_message"]
                y = eval_df["label"].map({"ham":0, "spam":1})
                feats = eval_df[FEATURES_LIST]
                X_train_text, X_test_text, y_train, y_test, train_feats, test_feats = train_test_split(
                    X_text, y, feats, test_size=0.2, random_state=42, stratify=y
                )
                X_test_tfidf = vectorizer.transform(X_test_text)
                lr_test_scaled = scaler.transform(test_feats)
                nb_test_feats = test_feats.values
                nb_input = hstack([X_test_tfidf, nb_test_feats])
                lr_input = hstack([X_test_tfidf, lr_test_scaled])
                nb_eval_pred = nb_model.predict(nb_input)
                lr_eval_pred = lr_model.predict(lr_input)
                cm_nb = confusion_matrix(y_test, nb_eval_pred)
                cm_lr = confusion_matrix(y_test, lr_eval_pred)
                fig, axes = plt.subplots(1,2, figsize=(8,3))
                sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0])
                axes[0].set_title('NB Confusion')
                sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens', ax=axes[1])
                axes[1].set_title('LR Confusion')
                st.pyplot(fig)
                st.text("Naive Bayes Classification Report:\n" + classification_report(y_test, nb_eval_pred))
                st.text("Logistic Regression Classification Report:\n" + classification_report(y_test, lr_eval_pred))
        except Exception as e:
            st.warning(f"Evaluation preview failed: {e}")

st.divider()
