# STEP 1: Install Required Libraries
!pip install transformers pandas pdfplumber --quiet

# STEP 2: Import Libraries
import pandas as pd
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
from google.colab import files

# STEP 3: Load a Pretrained Model (FLAN-T5)
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# STEP 4: Define Classifier Function
categories = ["Data Science", "Human Resources", "Software Development"]

def classify_resume(text):
    prompt = f"""
    Classify this resume into one of the following categories:
    {', '.join(categories)}

    Resume:
    {text}

    Category:
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=20)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.strip()

# STEP 5: Upload Resume Files or a CSV
print("Please upload a CSV file with a column named 'resume_text' or PDF resumes")
uploaded = files.upload()

pdf_results = []
csv_loaded = False

for filename in uploaded.keys():
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)
        csv_loaded = True
    elif filename.endswith(".pdf"):
        with pdfplumber.open(filename) as pdf:
            full_text = " ".join([page.extract_text() or "" for page in pdf.pages])
        predicted = classify_resume(full_text)
        pdf_results.append({"filename": filename, "category": predicted})

# STEP 6: Process Results
if csv_loaded:
    df['Predicted Category'] = df['resume_text'].apply(classify_resume)
    df.to_csv("classified_resumes.csv", index=False)
    files.download("classified_resumes.csv")
    print("CSV classification complete. File downloaded.")

if pdf_results:
    df_pdf = pd.DataFrame(pdf_results)
    df_pdf.to_csv("pdf_resume_classification.csv", index=False)
    files.download("pdf_resume_classification.csv")
    print("PDF resume classification complete. File downloaded.")
