import numpy as np
from sklearn.cluster import AgglomerativeClustering

print("--- Stage 2: Clinical Microplanning Engine ---")

# 1. MOCK DATA: These are the events that "survived" Stage 1.
# Format: [Event Name, Drug_Pct, Placebo_Pct, Risk_Diff, Relative_Risk, is_SAE, is_Severe]
selected_events = [
    {"name": "Nausea",           "features": [12.5, 4.1,  8.4, 3.04, 0, 0]},
    {"name": "Headache",         "features": [11.0, 3.2,  7.8, 3.43, 0, 0]},
    {"name": "Dizziness",        "features": [10.5, 2.1,  8.4, 5.00, 0, 0]},
    {"name": "Cardiac Arrest",   "features": [0.5,  0.1,  0.4, 5.00, 1, 1]}, # SAE
    {"name": "Ischemic Stroke",  "features": [0.4,  0.0,  0.4, 4.00, 1, 1]}  # SAE
]

# 2. Extract just the mathematical features for the ML algorithm
X = np.array([event["features"] for event in selected_events])

# 3. Apply Agglomerative Clustering
# distance_threshold=4.0 tells the ML: "If events are mathematically close, group them. 
# If they are very different (like a mild headache vs. a fatal stroke), split them."
cluster_engine = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=4.0, 
    linkage='ward'
)

# 4. Predict the clusters
cluster_labels = cluster_engine.fit_predict(X)

# 5. Group the text based on the ML's cluster decisions
grouped_sentences = {}
for i, label in enumerate(cluster_labels):
    if label not in grouped_sentences:
        grouped_sentences[label] = []
    
    event = selected_events[i]
    # Format the data point exactly how it will look in the text
    text_snippet = f"{event['name']} ({event['features'][0]}% vs {event['features'][1]}%)"
    grouped_sentences[label].append(text_snippet)

# 6. Display the Output for the Surface Realizer (Stage 3)
print("\n[ML Clustering Decisions for Sentence Generation]")
for cluster_id, items in grouped_sentences.items():
    # Join the items with 'and' or commas to mimic natural language
    joined_text = " and ".join(items) if len(items) <= 2 else ", ".join(items[:-1]) + ", and " + items[-1]
    print(f"Sentence {cluster_id + 1} Array: [{joined_text}]")