from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import difflib
import fitz  # PyMuPDF
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf','doc','xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix[0, 1]

def find_similar_words(text1, text2):
    text1_words = text1.split()
    text2_words = text2.split()
    matches = difflib.SequenceMatcher(None, text1_words, text2_words).get_matching_blocks()
    similar_words = [text1_words[i] for match in matches for i in range(match.a, match.a + match.size)]
    return similar_words if similar_words else ["None"]

@app.route('/')
def index():
    return render_template('make1.html')

@app.route('/check', methods=['POST'])
def check():
    text1 = ""
    text2 = ""

    # Handling text and PDF input for the first text area/file
    if 'text1' in request.form and request.form['text1']:
        text1 = request.form['text1']
    elif 'file1' in request.files and request.files['file1'].filename != '':
        file1 = request.files['file1']
        if file1 and allowed_file(file1.filename):
            filename = secure_filename(file1.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file1.save(file_path)
            text1 = extract_text_from_pdf(file_path)


    similarity = calculate_similarity(text1, text2)
    similarity_percentage = np.round(similarity * 100, 2)
    similar_words = find_similar_words(text1, text2)

    if similarity_percentage == 0:
        similar_words = ["None"]

    return render_template('make2.html', similarity=similarity_percentage, similar_words=similar_words)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
