import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Load datasets
books_dataset = pd.read_csv("books_dataset.csv", encoding='latin1')
syllabus_dataset = pd.read_csv("SyllabusII_dataset.csv", encoding='latin1')

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = text.replace(",", "").replace(".", "")
    return text

books_dataset['unit_topic'] = books_dataset['unit_topic'].apply(preprocess)
books_dataset['sub_topics'] = books_dataset['sub_topics'].apply(preprocess)
syllabus_dataset['all_topics'] = syllabus_dataset['all_topics'].apply(preprocess)

# Vectorize text data
vectorizer = TfidfVectorizer()
syllabus_matrix = vectorizer.fit_transform(syllabus_dataset['all_topics'])
books_matrix = vectorizer.transform(books_dataset['unit_topic'] + ' ' + books_dataset['sub_topics'])

# Compute similarity scores
similarity_scores = cosine_similarity(syllabus_matrix, books_matrix)

# Train KNN model
X_train, X_test, y_train, y_test = train_test_split(syllabus_matrix, syllabus_dataset['subject'], test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        student_name = request.form.get('student_name')
        student_id = request.form.get('student_id')
        return redirect(url_for('welcome', name=student_name, id=student_id))
    return render_template('index.html')

@app.route('/welcome')
def welcome():
    student_name = request.args.get('name')
    student_id = request.args.get('id')
    return render_template('welcome.html', name=student_name, id=student_id)

@app.route('/select_stream', methods=['POST'])
def select_stream():
    selected_stream = request.form['stream']
    if selected_stream == 'btech':
        return redirect('/btech')
    elif selected_stream == 'diploma':
        return redirect('/diploma')
    else:
        return "Invalid stream selected."

@app.route('/btech', methods=['GET', 'POST'])
def btech_page():
    if request.method == 'POST':
        subject_name = request.form.get('subject_name')
        return redirect(url_for('result', subject_name=subject_name))
    else:
        return render_template('btech.html')

@app.route('/diploma')
def diploma_page():
    return render_template('diploma.html')

# Route to render other HTML pages dynamically
@app.route('/<path:filename>')
def render_page(filename):
    return render_template(filename)

@app.route('/result', methods=['GET', 'POST'])
def result():
    subject_name = request.form.get('subject_name')  # Use request.form for POST method
    
    # Filter books related to the entered subject
    subject_books_indices = np.where(books_dataset['subject'] == subject_name)[0]
    
    if len(subject_books_indices) == 0:
        # Handle the case where there are no books related to the selected subject
        error_message = "No books found for the selected subject."
        return render_template('result.html', error_message=error_message)
    
    # Calculate similarity scores only if there are books related to the selected subject
    subject_books_similarity_scores = similarity_scores[:, subject_books_indices]
    
    max_similarity_index = np.argmax(subject_books_similarity_scores, axis=1)
    best_matching_book_index = subject_books_indices[max_similarity_index]
    best_matching_book = books_dataset.iloc[best_matching_book_index]

    # Display recommended books and details of the best matching book
    recommended_books = books_dataset.iloc[subject_books_indices]['book_title'].drop_duplicates().reset_index(drop=True)
    best_matching_book_details = {
        'book_title': best_matching_book['book_title'].values[0],
        'author': best_matching_book['author'].values[0],
        'published_year': best_matching_book['published_year'].values[0],
        'similarity_score': np.max(subject_books_similarity_scores)
    }

    return render_template('result.html', subject=subject_name, recommended_books=recommended_books, best_matching_book=best_matching_book_details)

if __name__ == '__main__':
    app.run(debug=True)