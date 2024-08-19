import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets (replace with your file paths)
books_dataset = pd.read_csv("books_dataset.csv", encoding='latin1')
syllabus_dataset = pd.read_csv("SyllabusII_dataset.csv", encoding='latin1')

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.replace(",", "").replace(".", "")
    return text

# Preprocess text columns
books_dataset['unit_topic'] = books_dataset['unit_topic'].apply(preprocess)
books_dataset['sub_topics'] = books_dataset['sub_topics'].apply(preprocess)
syllabus_dataset['all_topics'] = syllabus_dataset['all_topics'].apply(preprocess)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit vectorizer on syllabus topics
syllabus_matrix = vectorizer.fit_transform(syllabus_dataset['all_topics'])

# Transform book data (unit topic + sub-topics)
books_matrix = vectorizer.transform(books_dataset['unit_topic'] + ' ' + books_dataset['sub_topics'])

# Compute similarity scores
similarity_scores = cosine_similarity(syllabus_matrix, books_matrix)

def recommend_books(subject_name):
    """Recommends books based on the given subject name.

    Args:
        subject_name (str): The subject name for which to recommend books.

    Returns:
        dict: A dictionary containing recommended books and the best matching book.
    """

    # Filter books related to the subject
    subject_books_indices = np.where(books_dataset['subject'] == subject_name)[0]
    subject_books = books_dataset.iloc[subject_books_indices]

    if len(subject_books_indices) == 0:
        return {"error_message": "No books found for the entered subject."}

    # Compute similarity scores for subject books
    subject_books_similarity_scores = similarity_scores[:, subject_books_indices]

    # Identify the best matching book
    max_similarity_index = np.argmax(subject_books_similarity_scores, axis=1)
    best_matching_book_index = subject_books_indices[max_similarity_index]
    best_matching_book = books_dataset.iloc[best_matching_book_index]

    # Prepare data for recommended books and best matching book
    recommended_books = []
    for idx, row in subject_books.iterrows():
        recommended_books.append({
            "book_title": row['book_title'],
            "author": row['author'],
            "published_year": row['published_year'],
            "similarity_score": subject_books_similarity_scores[idx].max()
        })

    best_matching_book_data = {
        "book_title": best_matching_book['book_title'].values[0],
        "author": best_matching_book['author'].values[0],
        "published_year": best_passing_book['published_year'].values[0],
        "similarity_score": subject_books_similarity_scores.max()
    }

    return {"subject": subject_name, "recommended_books": recommended_books, "best_matching_book": best_matching_book_data}
