from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os
import re
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import difflib
import google.generativeai as genai

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Configure the Gemini API
api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key
genai.configure(api_key=api_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Initialize Flask app
app = Flask(__name__)

# Sentence transformer model setup
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Helper functions
def generate_response_gemini(prompt, max_length=8192):
    response = model.generate_content(contents=[prompt])
    generated_text = response.text.strip()
    
    if len(generated_text) > max_length:
        generated_text = generated_text[:max_length]
    
    return generated_text

def extract_images_from_pdf(file_path):
    images = convert_from_path(file_path)
    return images

def perform_ocr(images):
    ocr_text = ""
    for idx, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang='eng')
        ocr_text += f"Text from Image {idx + 1}:\n{text}\n\n"
    return ocr_text

def extract_text_pypdf2(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def extract_text_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(w.lower()) for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
    return set(tokens)

def calculate_similarity_score(tokens_1, tokens_2):
    common_tokens = tokens_1.intersection(tokens_2)
    similarity_score = len(common_tokens) / len(tokens_2)
    return similarity_score

def find_differences(text_1, text_2):
    diff = difflib.ndiff(text_1.splitlines(), text_2.splitlines())
    differences = [(i + 1, line) for i, line in enumerate(diff) if line.startswith('+ ') or line.startswith('- ')]
    return differences

def extract_numerical_values(text):
    pattern = r"\b\d+(\.\d+)?\b"
    return re.findall(pattern, text)

def split_text_into_chunks(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def save_chunks(chunks, file_path):
    with open(file_path, 'w') as file:
        for chunk in chunks:
            file.write(chunk + '\n')

def load_chunks(file_path):
    with open(file_path, 'r') as file:
        chunks = file.readlines()
    return [chunk.strip() for chunk in chunks]

def process_with_model(chunks, model):
    results = []
    for chunk in chunks:
        result = model.process(chunk)  # Replace with actual model processing code
        results.append(result)
    return results

def process_pdf(pdf_path):
    text_pypdf2 = extract_text_pypdf2(pdf_path)
    text_pdfplumber = extract_text_pdfplumber(pdf_path)
    images = extract_images_from_pdf(pdf_path)
    ocr_text = perform_ocr(images)
    
    combined_text = text_pypdf2 + "\n" + text_pdfplumber + "\n" + ocr_text
    
    numerical_values = extract_numerical_values(combined_text)
    chunks = split_text_into_chunks(combined_text)
    document_embeddings = encoder.encode(chunks)
    
    return combined_text, numerical_values, chunks, document_embeddings

def create_faiss_index(embeddings_list):
    all_embeddings = np.vstack(embeddings_list)
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)
    return index

def find_most_similar_chunks(query, df_documents, index, top_k=3):
    query_embedding = encoder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = []
    total_chunks = sum(len(chunks) for chunks in df_documents['text_chunks'])
    for i, idx in enumerate(indices[0]):
        if idx < total_chunks:
            doc_idx = 0
            chunk_idx = idx
            while chunk_idx >= len(df_documents['text_chunks'].iloc[doc_idx]):
                chunk_idx -= len(df_documents['text_chunks'].iloc[doc_idx])
                doc_idx += 1
            results.append({
                'document': df_documents['path'].iloc[doc_idx],
                'chunk': df_documents['text_chunks'].iloc[doc_idx][chunk_idx],
                'distance': distances[0][i]
            })
    return results

def extract_risk_problem_factors(text, numerical_values):
    prompt = (
        f"Based on the following medical report content and numerical values:\n\n"
        f"Medical Report: {text}\n\n"
        f"Numerical Values: {numerical_values}\n\n"
        f"Please extract the following information:\n\n"
        f"1. Risk Domain (such as cardiovascular, kidney, respiratory, etc.)\n"
        f"2. Specific Disease Problem\n"
        f"3. Stage of Risk (e.g., low, medium, high)\n\n"
        f"Provide the output in a structured format, preferably JSON."
    )
    
    response = generate_response_gemini(prompt)
    
    try:
        risk_problem_factors = json.loads(response)
    except json.JSONDecodeError:
        risk_problem_factors = {
            "Risk Domain": "Unknown",
            "Specific Disease Problem": "Unknown",
            "Stage of Risk": "Unknown"
        }
    
    return risk_problem_factors

def generate_prevention_report(risk, disease, age):
    if not risk and not disease:
        return "No significant risks or problems detected. You're safe and healthy. Keep up the good work!"
    
    prompt = f"""
    Provide a detailed wellness report with the following sections:

    1. *Introduction*
       - Purpose of the report
       - Context of general health and wellness

    2. *Risk Description*
       - General description of the identified risk
       - Common factors associated with the risk

    3. *Stage of Risk*
       - General information about the risk stage
       - Any typical considerations

    4. *Risk Assessment*
       - General overview of the risk's impact on health

    5. *Findings*
       - General wellness observations
       - Supporting information

    6. *Recommendations*
       - General wellness tips and lifestyle changes
       - Actions to promote well-being

    7. *Way Forward*
       - Suggested next steps for maintaining health
       - General follow-up actions

    8. *Conclusion*
       - Summary of overall wellness advice
       - General support resources

    Generate the report based on the following details:

    Risk Domain: {risk}
    Disease Problem: {disease}
    User Age: {age}
    """
    response = generate_response_gemini(prompt)
    return response

@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf():
    try:
        # Get PDF file and query from request
        pdf_file = request.files['file']
        query = request.form.get('query', '')
        user_age = int(request.form.get('age', 50))  # Default age is 50 if not provided

        # Save the uploaded file
        pdf_path = os.path.join('uploads', pdf_file.filename)
        pdf_file.save(pdf_path)

        # Process PDF and extract information
        combined_text, numerical_values, chunks, document_embeddings = process_pdf(pdf_path)

        # Create or load FAISS index
        faiss_index_path = 'faiss_index.index'
        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
        else:
            df_documents = pd.DataFrame({
                'path': [pdf_path],
                'text_chunks': [chunks]
            })
            index = create_faiss_index([document_embeddings])
            faiss.write_index(index, faiss_index_path)

        # Find similar chunks
        results = find_most_similar_chunks(query, pd.DataFrame({
            'path': [pdf_path],
            'text_chunks': [chunks]
        }), index)

        # Extract risk factors
        risk_problem_factors = extract_risk_problem_factors(combined_text, numerical_values)
        risk = risk_problem_factors.get('Risk Domain', 'Unknown')
        disease = risk_problem_factors.get('Specific Disease Problem', 'Unknown')

        # Generate prevention report
        prevention_report = generate_prevention_report(risk, disease, user_age)

        return jsonify({
            'results': results,
            'risk_problem_factors': risk_problem_factors,
            'prevention_report': prevention_report
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
