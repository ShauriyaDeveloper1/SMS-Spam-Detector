## 📱 SMS Spam Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shauriyadeveloper1-sms-spam-detector-app-aujbt6.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready Streamlit application for SMS spam detection with advanced ML models, multi-language support, and enterprise features.

### 🌐 Live Demo

**Try it now:** [https://shauriyadeveloper1-sms-spam-detector-app-aujbt6.streamlit.app/](https://shauriyadeveloper1-sms-spam-detector-app-aujbt6.streamlit.app/)

- Create an account and login
- Test with any SMS message
- Try multi-language translation
- Upload batch CSV files
- Explore all 7 custom themes

### 🎯 Core Features

**Machine Learning Models:**
- Multinomial Naive Bayes with TF-IDF vectorization (3000 features)
- Logistic Regression with balanced class weights
- 8 engineered features (length, word count, digit/uppercase ratios, URL detection, special chars)
- Model comparison and ensemble predictions
- Interactive threshold tuning and performance visualization

**Multi-Language Support:**
- Automatic language detection (20+ languages)
- Real-time translation to English using deep-translator
- Voice input with auto-translation (speech recognition)
- Batch processing with language columns
- Supported: Spanish, French, German, Chinese, Arabic, Hindi, Japanese, Korean, Russian, Portuguese, Italian, and more

**User Interface:**
- 🎨 **7 Custom Themes:** Default, Dark, Green, Blue Ocean, High Contrast, Sunset, Purple Dream
- 🔐 **Secure Authentication:** Strong password validation (8+ chars, uppercase, lowercase, digit, special char)
- 📧 **Email Validation:** Regex-based email format checking
- ⌨️ **Keyboard Shortcuts:** Ctrl+Enter (analyze), Ctrl+L (clear), Ctrl+M (focus), Escape (collapse)
- 🎤 **Voice Input:** Record and transcribe messages with auto-translation
- 🔊 **Text-to-Speech:** Audio output for prediction results

**Analysis Features:**
- Single message prediction with confidence scores
- Batch CSV processing with language detection
- Bulk text file analysis
- Message playground (text transformations)
- Prediction history with export capability
- Interactive charts (ROC curves, precision-recall, feature importance)
- Real-time model comparison dashboard

### 📊 Analytics & Visualization

- **Dataset Analytics:** Overview, word frequency, message length distribution
- **Model Insights:** Threshold tuning, ROC curves, precision-recall curves
- **Feature Importance:** Top 20 features visualization
- **Confusion Matrices:** Interactive heatmaps for both models
- **Performance Metrics:** Accuracy, precision, recall, F1-score comparison

### 🚀 Setup

**Prerequisites:**
- Python 3.8+ (tested on Python 3.13)
- Windows PowerShell or equivalent shell

**Install Dependencies:**

```powershell
pip install -r requirements.txt
```

**Required Packages:**
- `streamlit` - Web UI framework
- `scikit-learn` - Machine learning models
- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn`, `plotly` - Visualizations
- `nltk` - Text processing (WordNet lemmatizer, stopwords)
- `langdetect` - Language detection
- `deep-translator` - Translation service
- `audio-recorder-streamlit` - Voice input
- `SpeechRecognition` - Speech-to-text
- `gTTS` - Text-to-speech

**Dataset:**
Ensure `spam.csv` is in the project root with columns:
- `v1`: Label (ham/spam)
- `v2`: Message text

### 🎮 Running the Application

**Start Streamlit App:**

```powershell
streamlit run app.py
```

**Default URL:** `http://localhost:8501`

**First-Time Setup:**
1. Create an account (Sign Up)
2. Password requirements: 8+ chars, uppercase, lowercase, digit, special character
3. Valid email required (user@example.com)
4. Login with credentials

### 🔧 Model Training

1. Navigate to sidebar "⚙️ Controls"
2. Click "🔄 Train Models"
3. Models saved to `models/` directory:
   - `tfidf_vectorizer.pkl`
   - `scaler.pkl`
   - `naive_bayes_model.pkl`
   - `logistic_regression_model.pkl`

### 📤 Batch Processing

**CSV Upload:**
1. Prepare CSV with `message` column
2. Upload via sidebar "Upload CSV"
3. Click "Run Batch Prediction"
4. Download results with language detection

**Bulk Text Files:**
1. Upload multiple `.txt` files
2. Click "Run Batch Prediction"
3. View results with language column
4. Download consolidated CSV

### 🎨 Theme Customization

**Available Themes:**
1. **Default** - Clean blue and white
2. **Dark** - Sleek dark mode with cyan accents
3. **Green** - Fresh green for reduced eye strain
4. **Blue Ocean** - Calm blue tones
5. **High Contrast** - Maximum readability (yellow on black)
6. **Sunset** - Warm orange/coral colors
7. **Purple Dream** - Elegant purple palette

**Change Theme:**
- Sidebar → "🎨 Theme Settings"
- Select from dropdown
- Auto-applies with instant preview

### ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Analyze message |
| `Ctrl+L` | Clear input field |
| `Ctrl+M` | Focus message input |
| `Ctrl+H` | Toggle history view |
| `Escape` | Close all expanders |

### 🌍 Multi-Language Usage

**Text Input:**
```
Spanish: "¡Felicidades! Has ganado un premio"
French: "Félicitations! Vous avez gagné"
German: "Herzlichen Glückwunsch! Sie haben gewonnen"
Arabic: "مبروك! لقد ربحت جائزة"
```

**Voice Input:**
1. Click microphone button
2. Speak in any language
3. Auto-detects and translates
4. Shows original + translated text
5. Analyzes translated English version

### 📁 Project Structure

```
e:\sms\
├── app.py                      # Main Streamlit application (1100+ lines)
├── spam.csv                    # Training dataset (SMS messages)
├── requirements.txt            # Python dependencies
├── users.json                  # User credentials (created on first signup)
├── README.md                   # This file
└── models/                     # Saved ML models (created after training)
    ├── tfidf_vectorizer.pkl
    ├── scaler.pkl
    ├── naive_bayes_model.pkl
    └── logistic_regression_model.pkl
```

### 🔒 Security Features

- **Password Hashing:** SHA-256 encryption
- **Strong Password Policy:** 8+ chars, mixed case, digit, special char
- **Email Validation:** Regex-based format checking
- **Session Management:** Secure login/logout
- **User Isolation:** Per-user session states

### 📈 Performance Metrics

**Model Accuracy (typical):**
- Naive Bayes: ~95-97%
- Logistic Regression: ~96-98%

**Processing Speed:**
- Single message: <100ms
- Batch (100 messages): ~2-5 seconds
- Language detection: ~10-50ms per message
- Translation: ~100-200ms per message

### 🛠️ Technical Architecture

**Machine Learning Pipeline:**
1. Text Cleaning: Lemmatization, stopword removal, URL/email filtering
2. Feature Extraction: TF-IDF (3000 features, 1-2 ngrams) + 8 numeric features
3. Feature Scaling: StandardScaler for Logistic Regression
4. Model Training: 80/20 train-test split, stratified sampling
5. Ensemble Prediction: Compare both models for consensus

**Dependencies:**
- **ML:** scikit-learn, scipy
- **NLP:** nltk, langdetect, deep-translator
- **UI:** streamlit, plotly, matplotlib, seaborn
- **Voice:** audio-recorder-streamlit, SpeechRecognition, gTTS
- **Data:** pandas, numpy

### 🚀 Future Enhancements

**Planned Features:**
- ✅ Multi-language translation (implemented)
- ✅ Custom themes (implemented)
- ✅ Strong authentication (implemented)
- ✅ Voice input with translation (implemented)
- 🔄 REST API with FastAPI
- 🔄 URL safety checking (Google Safe Browsing)
- 🔄 PostgreSQL/MongoDB storage
- 🔄 Slack/Teams webhook integration
- 🔄 Message similarity checker (spam campaign detection)
- 🔄 Admin dashboard with RBAC
- 🔄 A/B testing framework
- 🔄 Real-time learning from user feedback

### 🐛 Troubleshooting

**Import Errors:**
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

**Translation Not Working:**
```powershell
# Verify deep-translator installation
python -c "from deep_translator import GoogleTranslator; print('OK')"

# Test translation
python -c "from deep_translator import GoogleTranslator; print(GoogleTranslator(source='es', target='en').translate('Hola mundo'))"
```

**Voice Input Issues:**
- Ensure microphone permissions granted
- Check audio device in system settings
- Google Speech API requires internet connection

**Model Not Found:**
- Click "🔄 Train Models" in sidebar
- Wait for training to complete
- Check `models/` directory created

### 📝 Usage Examples

**Single Message Analysis:**
```python
# Enter message in text area
"Congratulations! You've won $1000. Click here to claim"

# With translation
"¡Felicidades! Has ganado $1000. Haz clic aquí"
```

**Batch CSV Format:**
```csv
message
"URGENT: Your account has been compromised"
"Hi, are we still meeting at 3pm?"
"FREE gift card! Click now!"
```

### 📄 License

Educational/Internal use. No explicit license - modify as needed for your organization.

### 👨‍💻 Contributing

This is a self-contained project. For enhancements:
1. Test changes locally
2. Update `requirements.txt` if adding dependencies
3. Document new features in README
4. Ensure backward compatibility

### 📧 Support

For issues or questions, refer to inline code documentation in `app.py` (comprehensive docstrings and comments throughout).
