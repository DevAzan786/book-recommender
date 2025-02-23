import pandas as pd
import re
import os
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

data = pd.read_csv('books_sentiments.csv')
data['large_thumbnail'] = data['thumbnail'] + '&fife=w800'
data['large_thumbnail'] = np.where(
    data['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    data['large_thumbnail'],
)

raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='\n')
documents = text_splitter.split_documents(raw_documents)
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DB_PATH = "chroma_db"

if os.path.exists(CHROMA_DB_PATH):
    db_books = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=huggingface_embeddings)
else:
    db_books = Chroma.from_documents(documents, embedding=huggingface_embeddings, persist_directory=CHROMA_DB_PATH)


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    docs = db_books.similarity_search(query, k=initial_top_k)
    book_list = []

    for doc in docs:
        match = re.match(r'["]?(\d+)', doc.page_content)
        if match:
            isbn = match.group(1)
            book_list.append(isbn)

    book_recs = data[data['isbn13'].astype(str).isin(book_list)].head(initial_top_k)

    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    tone_mapping = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_mapping:
        book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(data["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {background-image: url('background.jpg'); background-size: cover;}") as dashboard:
    gr.Markdown("<h1 style='color: black; font-family: Arial, sans-serif;'>Semantic Book Recommender</h1>")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness", elem_id="query-box")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All", elem_id="category-dropdown")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All", elem_id="tone-dropdown")
        submit_button = gr.Button("Find recommendations", elem_id="submit-button")

    gr.Markdown("<h2 style='color: #FF5722; font-family: Arial, sans-serif;'>Recommendations</h2>")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2, elem_id="output-gallery")

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()