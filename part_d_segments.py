
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


# Step 1: Load the cleaned modelling table
df = pd.read_csv("data/modelling_table.csv")


# Step 2: Create one profile for each hub
hub_profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

# Keep hubs in ascending order
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

print("Hub profile:")
print(hub_profile.head())
print("Number of hubs:", len(hub_profile))

# Save the hub profile
hub_profile.to_csv("outputs/hub_profile.csv", index=False)


# Step 3: Get each hub's city and tier
hub_info = df[
    ["hub_id", "hub_city", "hub_tier"]
].drop_duplicates(subset="hub_id")

hub_info = hub_info.sort_values("hub_id").reset_index(drop=True)


# Step 4: Select the four features for clustering
cluster_features = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

X = hub_profile[cluster_features]


# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:", X_scaled.shape)


# Step 6: Cluster the 24 hubs into three groups
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = kmeans.fit_predict(X_scaled)


# Step 7: Calculate separation ratios for K = 2, 3, 4, 5
# The ratio compares distances between different clusters
# with distances between hubs in the same cluster.

all_distances = pairwise_distances(X_scaled)

separation_results = []

for k in range(2, 6):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    within_distances = []
    across_distances = []

    # Check every unordered pair of hubs
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            distance = all_distances[i, j]

            if labels[i] == labels[j]:
                within_distances.append(distance)
            else:
                across_distances.append(distance)

    mean_within = np.mean(within_distances)
    mean_across = np.mean(across_distances)

    ratio = mean_across / mean_within

    # Sort cluster sizes from largest to smallest
    sizes = pd.Series(labels).value_counts().tolist()
    sizes = sorted(sizes, reverse=True)
    cluster_sizes = "-".join(str(size) for size in sizes)

    separation_results.append({
        "K": k,
        "across_over_within": ratio,
        "cluster_sizes": cluster_sizes
    })

separation_df = pd.DataFrame(separation_results)

print("\nCluster separation:")
print(separation_df.round(3))

separation_df.to_csv(
    "outputs/part_d_separation.csv",
    index=False
)


# Step 8: Name clusters using their mean delay rates
cluster_delay_rates = hub_profile.groupby("cluster")[
    "delay_rate"
].mean().sort_values()

segment_names = ["Standard", "Watch", "Escalate"]

cluster_to_segment = {}

for i, cluster_number in enumerate(cluster_delay_rates.index):
    cluster_to_segment[cluster_number] = segment_names[i]

hub_profile["segment"] = hub_profile["cluster"].map(
    cluster_to_segment
)


# Step 9: Create the hub segments file
hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hub_info,
    on="hub_id",
    how="left",
    validate="one_to_one"
)

hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
].sort_values("hub_id").reset_index(drop=True)

hub_segments.to_csv(
    "outputs/hub_segments.csv",
    index=False
)

print("\nHub segments:")
print(hub_segments.to_string(index=False))


# Step 10: Create the segment profile
segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

# Keep the business-defined segment order
segment_order = {
    "Standard": 0,
    "Watch": 1,
    "Escalate": 2
}

segment_profile["sort_order"] = segment_profile["segment"].map(
    segment_order
)

segment_profile = segment_profile.sort_values(
    "sort_order"
).drop(columns="sort_order").reset_index(drop=True)

segment_profile.to_csv(
    "outputs/segment_profile.csv",
    index=False
)

print("\nSegment profile:")
print(segment_profile.round(4).to_string(index=False))


# Step 11: Identify hubs whose tier and segment do not match
expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

hub_segments["expected_segment"] = hub_segments["hub_tier"].map(
    expected_segment
)

mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

# The mismatch file contains the required hub_segments columns
mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

mismatches.to_csv(
    "outputs/tier_vs_segment_mismatch.csv",
    index=False
)

print("\nTier versus segment mismatches:")
print(mismatches.to_string(index=False))
print("\nNumber of mismatched hubs:", len(mismatches))


# Step 12: Confirm the hub profile requirements
print("\nHub profile checks:")
print("Number of hubs:", len(hub_profile))
print(
    "All hubs have 50 shipments:",
    bool((hub_profile["shipments"] == 50).all())
)

print("\nPart D segmentation completed.")
