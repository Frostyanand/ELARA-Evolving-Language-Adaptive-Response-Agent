# ELARA-Evolving-Language-Adaptive-Response-Agent
ELARA (Evolving Language & Adaptive Response Agent) is an AI-powered chatbot built from scratch using traditional NLP and machine learning techniques. Designed to understand user queries, adapt dynamically, and improve over time, ELARA combines structured learning, intent recognition, and active learning to enhance conversational flow.

# Chatbot from Scratch

## Overview

This project implements a chatbot **without using any pre-trained models or external APIs**. Instead, it leverages traditional **Natural Language Processing (NLP) techniques** and a **Support Vector Machine (SVM) classifier** trained from scratch. The chatbot continuously learns by logging misclassified queries and integrating user feedback through **active learning.**

## Features

- **Intent Recognition** using **SVM + TF-IDF** trained on a **merged dataset** (MultiWOZ 2.2 + Clinc OOS + Rasa NLU).
- **Rule-based Fallback Mechanism** for handling unknown inputs.
- **Memory System** to retain the last **30 messages** and extract key user information for context-aware responses.
- **Active Learning System** to improve accuracy over time by reviewing misclassified inputs.
- **Structured Storage** of conversation logs in CSV for retraining.
- **Command-Line Interface (CLI)** with planned **GUI integration**.

## Project Structure

```
📂 chatbot_project
 ├── 📂 dataset_raw             # Original datasets (downloaded)
 │   ├── multiwoz_v22/         # MultiWOZ 2.2 raw dataset
 │   ├── clinc_oos/            # Clinc OOS dataset
 │   ├── rasa_nlu/             # Rasa NLU dataset
 │
 ├── 📂 data                   # Merged + Preprocessed dataset
 │   ├── merged_data.json      # Final combined dataset
 │   ├── merged_data.csv       # Same dataset in CSV format
 │   ├── train.json            # Training set
 │   ├── validation.json       # Validation set
 │   ├── test.json             # Test set
 │
 ├── 📂 models                 # Saved models
 │   ├── chatbot_model.pkl     # Trained SVM model
 │   ├── vectorizer.pkl        # TF-IDF vectorizer for text processing
 │
 ├── 📂 src
 │   ├── preprocess.py         # Merges datasets & splits into train/test/val
 │   ├── train.py              # Trains the chatbot using SVM
 │   ├── chatbot.py            # Main chatbot interface
 │   ├── memory.py             # Stores and retrieves important user details
 │   ├── active_learning.py    # Processes misclassified inputs
 │
 ├── README.md
 ├── requirements.txt
 ├── run_chatbot.py            # Entry point to launch chatbot
```

## Setup Instructions

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Prepare Datasets**

```bash
python src/preprocess.py
```

This script:
- **Merges** datasets from `dataset_raw/` (MultiWOZ 2.2, Clinc OOS, Rasa NLU).
- **Cleans** text and formats it properly.
- **Splits** data into `train.json`, `validation.json`, and `test.json`.

### **3. Train the Chatbot**

```bash
python src/train.py
```

This script:
- Loads and preprocesses the dataset from `data/merged_data.json`.
- Trains an **SVM classifier** with **TF-IDF features**.
- Saves the trained model (`chatbot_model.pkl`) and **TF-IDF vectorizer**.

### **4. Run the Chatbot**

```bash
python run_chatbot.py
```

- Stores the last **30 messages** for short-term memory.
- Extracts long-term user details for personalized responses.
- Logs misclassified inputs for future improvements.

## Improving the Chatbot

### **Misclassification Logging**

Whenever the chatbot fails to classify an input correctly, it is logged in `misclassified.csv` with:

- User input
- Predicted intent
- Correct intent (if user provides feedback)

### **Retraining with Active Learning**

Run the following script **periodically** to integrate new data and improve accuracy:

```bash
python src/active_learning.py
```

This script:
- Reads `misclassified.csv`.
- Allows manual correction of misclassified intents.
- Updates `merged_data.json` with new examples.
- Retrains the model with the improved dataset.

## Future Enhancements

✅ GUI with **Tkinter/PyQt** ✅ Context-aware responses using **more advanced memory management** ✅ Improved response diversity using **template-based generation**

## Credits

This chatbot was developed using **NLTK, Scikit-learn, and JSON-based intent storage**, following a structured and scalable approach for continuous learning. 🚀


# 🚀 Chatbot Implementation Workflow

## **Phase 1: Data Preparation & Preprocessing**

### **1.1 Select & Merge Datasets**
✅ Use **Clinc OOS** (primary dataset) + **Rasa NLU** (secondary dataset)  
✅ **Remove duplicate intents & balance classes**  
✅ **Standardize format** (Ensure both datasets follow the same JSON structure)  
✅ Store merged dataset as `intents.json`

### **1.2 Preprocessing Pipeline**
✅ **Tokenization** (split input into words)  
✅ **Lowercasing** (convert all text to lowercase)  
✅ **Stemming** (reduce words to root form)  
✅ **TF-IDF Vectorization** (convert words into numerical features for SVM)  
✅ Use **NLTK** for tokenization & stemming  

---

## **Phase 2: Intent Classification System**

### **2.1 Train SVM for Intent Classification**
✅ Train **SVM classifier** on labeled intent data  
✅ Use **TF-IDF vectorized text features**  
✅ Train-Test Split: **80% training, 20% testing**  
✅ Evaluate accuracy & fine-tune parameters if necessary  
✅ Save trained model as `intent_classifier.pkl`

### **2.2 Implement Misclassification Logging & Active Learning**
✅ If **confidence score < threshold**, log query as `unknown_intents.csv`  
✅ Periodically **review misclassified queries**  
✅ Add relevant ones to dataset & retrain SVM model  

---

## **Phase 3: Response Generation System**

### **3.1 Rule-Based Response System (Initial Implementation)**
✅ Create `responses.json` with predefined answers  
✅ Implement **randomized responses** for each intent  
✅ Fallback: If intent is unknown, respond with **"Could you clarify?"**

### **3.2 Advanced Response Handling (Phase 2 Upgrade)**
✅ Add **dynamic template-based responses** (e.g., `"The weather in {city} is {weather}."`)  
✅ Implement **feedback collection system** (ask "Was this helpful? Y/N")  
✅ Store feedback in `feedback_log.csv`  

---

## **Phase 4: Context & Memory Handling**

### **4.1 Short-Term Memory (Chat History)**
✅ Store last **30 messages** in a rolling buffer  
✅ Use stored context to improve replies  

### **4.2 Long-Term Memory (User Context Storage)**
✅ Detect **key user information** (e.g., "I'm 18 years old.")  
✅ Store key details in `user_profiles.json`  
✅ Retrieve context for future replies  

---

## **Phase 5: User Interface & Deployment**

### **5.1 Build CLI Interface (Initial Deployment)**
✅ Simple text-based chatbot interface in Python  
✅ Load `intent_classifier.pkl` and `responses.json`  
✅ Process user queries & generate responses  

### **5.2 GUI Upgrade (Phase 2 Enhancement)**
✅ Build **Tkinter-based GUI** for a better user experience  
✅ Display **chat history + response feedback buttons**  

---

## **Phase 6: Training & Continuous Improvement**

### **6.1 Model Retraining Process**
✅ Every **X interactions**, review `unknown_intents.csv`  
✅ Add high-quality misclassified queries to dataset  
✅ Retrain `intent_classifier.pkl` with updated data  

### **6.2 Performance Optimization**
✅ Experiment with **hyperparameters** (C value in SVM)  
✅ Optimize **TF-IDF vectorization** (max features, n-grams)  
✅ Speed up inference with caching for frequent queries  

---



