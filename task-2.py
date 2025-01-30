
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Merge transactions with customers and products
df = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Create a "profile" for each customer based on purchased products
customer_profiles = df.groupby("CustomerID")["ProductName"].apply(lambda x: ' '.join(x))

# Convert text data into numerical form using TF-IDF
vectorizer = TfidfVectorizer()
customer_vectors = vectorizer.fit_transform(customer_profiles)

# Compute similarity matrix
similarity_matrix = cosine_similarity(customer_vectors)

# Create lookalike recommendations
customer_ids = list(customer_profiles.index)
lookalikes = {}

for i, customer in enumerate(customer_ids[:20]):  # First 20 customers
    similar_customers = sorted(
        list(enumerate(similarity_matrix[i])), key=lambda x: x[1], reverse=True
    )[1:4]  # Top 3 excluding self

    lookalikes[customer] = [(customer_ids[idx], round(score, 2)) for idx, score in similar_customers]

# Convert to DataFrame and Save
lookalike_df = pd.DataFrame(lookalikes.items(), columns=["CustomerID", "Lookalikes"])
lookalike_df.to_csv("Lookalike.csv", index=False)
print(lookalike_df.head())

