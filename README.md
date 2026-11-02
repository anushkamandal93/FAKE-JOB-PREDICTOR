[README.md](https://github.com/user-attachments/files/34279110/README.1.md)
# 🕵️ Fake Job Post Detector

A machine learning web app that detects fraudulent job postings. Enter a job's title, company profile, description and requirements, and it returns a fraud probability with a clear verdict.

**How it works:** Trained on about 18,000 Kaggle job postings, using CountVectorizer for text and a Logistic Regression classifier with balanced class weights. It reaches about 98% accuracy, with 83% recall on the fraud class.

**Tech stack:** Python, scikit-learn, FastAPI, HTML/CSS/JavaScript, Docker.

---

## 📊 Model Performance

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Genuine (0) | 0.99 | 0.99 | 0.99 |
| Fraudulent (1) | 0.77 | 0.83 | 0.80 |

Overall accuracy: **~98%** on a 20% test split (3,576 samples).

Only about 5% of postings in the dataset are fake, so accuracy alone is misleading. Precision and recall on the fraud class are the metrics that matter.

## 🧠 How It Works

1. The title, company profile, description and requirements are combined into one `text` column.
2. **ColumnTransformer** processes each type of feature:
   - Text → `CountVectorizer`
   - Categorical (employment type, experience, education) → `OneHotEncoder`
   - Binary (remote, company logo, screening questions) → passthrough
3. A **Logistic Regression** classifier with `class_weight="balanced"` handles the class imbalance.
4. Everything is wrapped in a single scikit-learn `Pipeline`, saved with `joblib`, and served through FastAPI.

## 📁 Project Structure

```
├── app.py                              # FastAPI backend
├── index.html                          # Frontend (HTML + CSS + JS)
├── Fake Job Post Detector.ipynb        # Training notebook
├── fake_job_post_detector_model.pkl    # Trained model
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## 🚀 Getting Started

### Option 1: Run locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

### Option 2: Run with Docker

```bash
docker build -t fake-job-detector .
docker run -p 8000:8000 fake-job-detector
```

Open **http://localhost:8000**.

## 🔌 API

### `POST /predict`

**Request body**

```json
{
  "title": "Online Data Entry Operator - Earn 50,000 Per Week",
  "company_profile": "Top international company hiring urgently.",
  "description": "No interview needed. Pay a refundable registration fee to confirm your seat.",
  "requirements": "Smartphone and bank account.",
  "employment_type": "Part-time",
  "required_experience": "Unknown",
  "required_education": "Unknown",
  "telecommuting": 1,
  "has_company_logo": 0,
  "has_questions": 0
}
```

**Response**

```json
{
  "fradulent": 1,
  "probability": 0.97
}
```

`fradulent` is `1` when the fraud probability is above 0.5. The key name is spelled this way in the API.

## 🔁 Retrain the Model

Open `Fake Job Post Detector.ipynb` and run all cells. The notebook downloads the dataset from Kaggle with `kagglehub`, trains the pipeline, prints the evaluation report and saves `fake_job_post_detector_model.pkl`.

> Use the same scikit-learn version for training and serving, since a model saved with one version may not load on another.

## 🔮 Future Improvements

- Try TF-IDF and stronger models (Random Forest, XGBoost, transformers)
- Show which words triggered the fraud flag
- Tune the decision threshold for higher precision or recall
- Deploy online (Render, Railway or AWS)

## ⚠️ Disclaimer

This is a machine learning estimate, not a guarantee. Always verify employers independently before sharing personal information or paying any fees.

## 📚 Dataset

[Real or Fake: Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) by shivamb on Kaggle.

## 📄 License

This project is licensed under the MIT License.
