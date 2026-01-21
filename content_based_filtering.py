import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(data, item_name, top_n=10):
    # Check if item exists
    if item_name not in data['Name'].values:
        print(f"Item '{item_name}' not found in the data.")
        return pd.DataFrame()

    # Reset index to ensure alignment
    data_reset = data.reset_index(drop=True)

    # TF-IDF vectorization on Tags column
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(data_reset['Tags'])

    # Cosine similarity
    cosine_similarity_content = cosine_similarity(
        tfidf_matrix_content, tfidf_matrix_content
    )

    # Get index of the given item in the RESET dataframe
    item_index = data_reset[data_reset['Name'] == item_name].index[0]

    # Similarity scores
    similar_items = list(enumerate(cosine_similarity_content[item_index]))

    # Sort by similarity score (descending)
    similar_prod = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Skip the first item (itself) and get top N similar products
    top_similar_prod = similar_prod[1:top_n+1]

    # Extract indices
    recommended_items_indices = [x[0] for x in top_similar_prod]

    # Fetch recommended item details with all necessary columns
    columns_to_fetch = ['Name', 'ReviewCount', 'Brand', 'Rating', 'ImageURL', 'Description', 'Category']
    available_columns = [col for col in columns_to_fetch if col in data_reset.columns]

    recommended_item_details = data_reset.iloc[recommended_items_indices][available_columns]

    return recommended_item_details

# ==========================
# To test the system
# ==========================
if __name__ == "__main__":
    from preprocess_data import process_data
    raw_data = pd.read_csv("clean_data.csv")
    data = process_data(raw_data)
    item_name = "OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath"
    result = content_based_recommendation(data, item_name, top_n=5)
    print(result)