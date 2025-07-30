import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE



if os.path.exists("benchmark/featurespaces/embeddings_REF_train.pkl"):
    with open("benchmark/featurespaces/embeddings_REF_train.pkl", "rb") as f:
        train = pickle.load(f)
        print("Loaded embeddings from file")
    with open("benchmark/featurespaces/embeddings_REF_id_test.pkl", "rb") as f:
        idd = pickle.load(f)
        print("Loaded embeddings from file")
    with open("benchmark/featurespaces/embeddings_REF_ood_test.pkl", "rb") as f:
        ood = pickle.load(f)
        print("Loaded embeddings from file")
        print("Loaded embeddings from file")
else:
    raise Exception("Embeddings not found")

train_embeddings = train["embeddings"]
id_embeddings = idd["embeddings"]
ood_embeddings = ood["embeddings"]

embeddings = np.concatenate([train_embeddings, id_embeddings, ood_embeddings])

print(embeddings.shape)
# Initialize t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, early_exaggeration=1)

# Transform embeddings to 2D
embeddings_2d = tsne.fit_transform(embeddings)

embeddings_2d_train = embeddings_2d[:len(train_embeddings)]
embeddings_2d_id = embeddings_2d[len(train_embeddings):(len(train_embeddings)+len(id_embeddings))]
embeddings_2d_ood = embeddings_2d[(len(train_embeddings)+len(id_embeddings)):]

cmap = ListedColormap(plt.cm.tab20.colors[:12])

# Visualize all classes
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
scatter = ax.scatter(embeddings_2d_train[:, 0],
            embeddings_2d_train[:, 1],
            c="#2c3e50",
            #c = [train["meta"][i]["label"] for i in range(len(train["meta"]))],
            #cmap=cmap,
            s=15,
            alpha=0.3,
)
scatter = ax.scatter(embeddings_2d_id[:, 0],
            embeddings_2d_id[:, 1],
            c="#3498db",
            #c = [idd["meta"][i]["label"] for i in range(len(idd["meta"]))],
            #cmap=cmap,
            s=15,
            alpha=0.3,
)
scatter = ax.scatter(embeddings_2d_ood[:, 0],
            embeddings_2d_ood[:, 1],
            c="#c0392b",
            #c = [ood["meta"][i]["label"] for i in range(len(ood["meta"]))],
            #cmap=cmap,
            s=15,
            alpha=0.3,
)
ax.set_axis_off()

#Visualize no atrial activity
plt.savefig("./paper/images/fig5-embeddings_ref.svg")
plt.savefig("./paper/images/fig5-embeddings_ref.png", dpi=300)


# Visualize all classes
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
scatter = ax.scatter(embeddings_2d_id[:, 0],
            embeddings_2d_id[:, 1],
            c="#bdc3c7",
            #c = [idd["meta"][i]["label"] for i in range(len(idd["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
scatter = ax.scatter(embeddings_2d_ood[:, 0],
            embeddings_2d_ood[:, 1],
            c="#bdc3c7",
            #c = [ood["meta"][i]["label"] for i in range(len(ood["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
scatter = ax.scatter(embeddings_2d_train[:, 0],
            embeddings_2d_train[:, 1],
            c="#c0392b",
            #c = [train["meta"][i]["label"] for i in range(len(train["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
ax.set_axis_off()
plt.savefig("./paper/images/fig5-embeddings_ref_train.svg")
plt.savefig("./paper/images/fig5-embeddings_ref_train.png", dpi=300)

# Visualize all classes
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
scatter = ax.scatter(embeddings_2d_train[:, 0],
            embeddings_2d_train[:, 1],
            c="#bdc3c7",
            #c = [train["meta"][i]["label"] for i in range(len(train["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
scatter = ax.scatter(embeddings_2d_ood[:, 0],
            embeddings_2d_ood[:, 1],
            c="#bdc3c7",
            #c = [ood["meta"][i]["label"] for i in range(len(ood["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
scatter = ax.scatter(embeddings_2d_id[:, 0],
            embeddings_2d_id[:, 1],
            c="#c0392b",
            #c = [idd["meta"][i]["label"] for i in range(len(idd["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
ax.set_axis_off()
plt.savefig("./paper/images/fig5-embeddings_ref_id.svg")
plt.savefig("./paper/images/fig5-embeddings_ref_id.png", dpi=300)


# Visualize all classes
fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
scatter = ax.scatter(embeddings_2d_train[:, 0],
            embeddings_2d_train[:, 1],
            c="#bdc3c7",
            #c = [train["meta"][i]["label"] for i in range(len(train["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
scatter = ax.scatter(embeddings_2d_id[:, 0],
            embeddings_2d_id[:, 1],
            c="#bdc3c7",
            #c = [idd["meta"][i]["label"] for i in range(len(idd["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
scatter = ax.scatter(embeddings_2d_ood[:, 0],
            embeddings_2d_ood[:, 1],
            c="#c0392b",
            #c = [ood["meta"][i]["label"] for i in range(len(ood["meta"]))],
            #cmap=cmap,
            s=30,
            alpha=0.5,
)
ax.set_axis_off()
plt.savefig("./paper/images/fig5-embeddings_ref_ood.svg")
plt.savefig("./paper/images/fig5-embeddings_ref_ood.png", dpi=300)