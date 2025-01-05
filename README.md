# Twitter Sentiment Analysis

## Overview
This project aims to analyze and classify sentiments in tweets as part of a sentiment analysis system. The system leverages machine learning models, including Random Forest, LSTM, and BERT, to predict sentiment from text data. The final implementation provides an interactive app for user-friendly predictions.

## Project Structure
```
Twitter-Sentiment-Analysis
├── app
│   └── main.py                      # Streamlit app for sentiment prediction
├── data
│   ├── processed
│   │   ├── clean_test.csv           # Preprocessed test dataset
│   │   ├── clean_train.csv          # Preprocessed train dataset
│   │   ├── test.csv                 # Test dataset
│   │   └── train.csv                # Training dataset
│   └── raw
│       └── tweets.csv               # Original raw dataset
├── database
│   └── tweets.db                    # SQLite database for tweets
├── models
│   ├── bert_lstn.sentiment.pth      # Saved BERT + LSTM model
│   ├── lstm_sentiment_model.h5      # Saved LSTM model
│   ├── random_forest_model.pkl      # Saved Random Forest model
│   ├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│   └── tokenizer.json               # Tokenizer for text preprocessing
├── notebooks
│   ├── Bert_Model.ipynb             # Notebook for BERT model implementation
│   ├── Data_Analysis.ipynb          # Notebook for data exploration and visualization
│   ├── LSTM_Model.ipynb             # Notebook for LSTM model implementation
│   ├── Preprocessing.ipynb          # Notebook for data preprocessing
│   ├── TF_IDF.ipynb                 # Notebook for TF-IDF feature extraction
│   └── Train_Test_Split.ipynb       # Notebook for splitting the dataset
├── requirements.txt                 # Python dependencies
└── Twitter_Sentiment_Synopsis.pdf   # Synopsis document for the project
```

## Features
- **Data Analysis**: Exploratory Data Analysis (EDA) to understand tweet distributions.
- **Data Preprocessing**: Cleaning, tokenization, and feature engineering using TF-IDF.
- **Models**:
  - Random Forest
  - LSTM (Long Short-Term Memory)
  - BERT (Bidirectional Encoder Representations from Transformers)
- **SQLite Database**: Storing and accessing tweet data.
- **Interactive App**: A Streamlit app to allow users to input text and select models for predictions.

## Requirements
To run the project, install the necessary dependencies using the following command:
```
pip install -r requirements.txt
```

## Usage
### Running the App
1. Navigate to the `app` directory.
2. Run the Streamlit app:
   ```
   streamlit run main.py
   ```
3. Open the provided URL in your browser to interact with the app.

### Notebooks
- Each notebook provides modular insights into specific tasks such as data analysis, preprocessing, and model training.
- Open the notebooks in Jupyter or Google Colab for step-by-step walkthroughs.
- **Note**: To ensure compatibility and avoid errors due to changes in embeddings, it is recommended to execute the notebooks in Google Colab to regenerate models and embeddings before running `main.py`.

## Models
### Random Forest
- **Model file**: `models/random_forest_model.pkl`
- **Features**: Trained using TF-IDF vectorized text.

### LSTM
- **Model file**: `models/lstm_sentiment_model.h5`
- **Features**: Processes tokenized text for sequence-based learning.

### BERT
- **Model file**: `models/bert_lstn.sentiment.pth`
- **Features**: Combines BERT embeddings with LSTM for contextual understanding.

## Datasets
- **Raw Data**: Original tweets dataset stored in `data/raw/tweets.csv`.
- **Processed Data**: Preprocessed datasets available in the `data/processed` directory.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
- The dataset was sourced from publicly available Twitter data.
- Libraries used include Scikit-learn, TensorFlow, PyTorch, and Streamlit.

## Contact
For any inquiries or suggestions, please contact [chandra] at [chandrabrucelee31@gmail.com].
