

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

# Create customer purchase summary
customer_summary = transactions.groupby("CustomerID").agg(
    {"TotalValue": "sum", "TransactionID": "count"}
).rename(columns={"TotalValue": "TotalSpend", "TransactionID": "PurchaseCount"})

# Standardize the data
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_summary)

# Find optimal clusters (DB Index)
db_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(customer_data_scaled)
    db_index = davies_bouldin_score(customer_data_scaled, labels)
    db_scores.append(db_index)

# Choose the number of clusters with the lowest DB Index
optimal_k = range(2, 11)[db_scores.index(min(db_scores))]
print(f"Optimal clusters: {optimal_k}")

# Fit the best KMeans model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_summary["Cluster"] = kmeans.fit_predict(customer_data_scaled)

# Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=customer_summary["TotalSpend"], y=customer_summary["PurchaseCount"],
    hue=customer_summary["Cluster"], palette="viridis", alpha=0.7
)
plt.title("Customer Segmentation Clusters")
plt.xlabel("Total Spend")
plt.ylabel("Purchase Count")
plt.show()
