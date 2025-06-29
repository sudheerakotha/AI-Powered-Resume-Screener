# AI-Powered-Resume-Screener

# 🧠 Resume Screening Assistant

This project is a **Resume Screening Assistant** built using **Natural Language Processing (NLP)** and a **transformer-based model (FLAN-T5)**. It classifies resumes into predefined job categories based on their textual content, supporting both **PDF files** and **CSV files**.

---

## 🚀 Features

- Classifies resumes into categories like:
  - Data Science
  - Human Resources
  - Software Development
- Supports multiple resume formats:
  - PDF resumes
  - CSV files (with a `resume_text` column)
- Utilizes the **FLAN-T5 model** from Google via Hugging Face
- Integrated with **Google Colab** for cloud-based execution
- Outputs downloadable CSV files with predicted categories

---

## 📁 File Structure
├── resume_screening_assistant.ipynb # Main notebook file
└── classified_resumes.csv # Output for CSV inputs
└── pdf_resume_classification.csv # Output for PDF inputs

## 🛠️ Technologies Used

- **Python**
- **Hugging Face Transformers**
- **FLAN-T5 Pretrained Model**
- **pdfplumber** for PDF text extraction
- **pandas** for data processing
- **Google Colab** for execution

## 📌 Prerequisites

- Python (recommended: Google Colab)
- Install required libraries:

```python
!pip install transformers pandas pdfplumber --quiet

🧑‍💻 How It Works
Upload Resumes
Upload one or more PDF resumes or a CSV file containing a column named resume_text.

Text Extraction

For PDFs: Extracts text using pdfplumber.

For CSV: Reads resume text using pandas.

Classification

Constructs a prompt and passes it to the FLAN-T5 model.

Model predicts the category based on resume content.

Download Results

Outputs are stored in a CSV file and downloaded automatically.

📝 Example Prompt to Model
text
Copy
Edit
Classify this resume into one of the following categories:
Data Science, Human Resources, Software Development

Resume:
[Resume text]

Category:
📥 Input Formats
PDF: Plain resumes in .pdf format

CSV: File with a column labeled resume_text

📤 Output
classified_resumes.csv – CSV with predicted categories for each text entry

pdf_resume_classification.csv – CSV with filename and category for PDFs

🧩 Limitations
Very long resumes may be truncated due to token limits.

Scanned PDFs (images) are not supported; OCR integration may be required.

Classification depends on the quality and relevance of text content.

📚 References
FLAN-T5 Model - Hugging Face

Transformers Library

pdfplumber GitHub

pandas Documentation

Google Colab

## ✅ Future Improvements
Add OCR support for scanned PDF resumes

Expand classification categories

Deploy as a web application using Flask or Streamlit
