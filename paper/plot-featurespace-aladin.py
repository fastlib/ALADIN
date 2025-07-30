import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class Data(object):
    def __init__(self, name="ALADIN"):

        if os.path.exists(f"benchmark/featurespaces/embeddings_{name}_train.pkl"):
            with open(f"benchmark/featurespaces/embeddings_{name}_train.pkl", "rb") as f:
                train = pickle.load(f)
                print("Loaded embeddings from file")
            with open(f"benchmark/featurespaces/embeddings_{name}_id_test.pkl", "rb") as f:
                idd = pickle.load(f)
                print("Loaded embeddings from file")
            with open(f"benchmark/featurespaces/embeddings_{name}_ood_test.pkl", "rb") as f:
                ood1 = pickle.load(f)
                print("Loaded embeddings from file")
            ood = {"embeddings": ood1["embeddings"], "meta": ood1["meta"]}
        else:
            raise Exception("Embeddings not found")

        self.embeddings = {
            "train": self.cleanup(train["embeddings"]),
            "id": self.cleanup(idd["embeddings"]),
            "ood": self.cleanup(ood["embeddings"])
        }
        self.scaler = StandardScaler()

        self.scale()

        self.allembeddings = np.concatenate([self.embeddings["train"], self.embeddings["id"], self.embeddings["ood"]])

        self.metadata = {
            "train": train["meta"],
            "id": idd["meta"],
            "ood": ood["meta"]
        }
        self.allmetas = self.metadata["train"] + self.metadata["id"] + self.metadata["ood"]

        self.feature_names = np.array(["pc_of_noise", "raw_hr_mean", "raw_hr_std", "raw_hr_min", "raw_hr_max", 
                    "raw_hr_kurtosis", "raw_hr_skewness", "filt_hr_mean", "filt_hr_std", "filt_hr_min",
                    "filt_hr_max", "filt_hr_kurtosis", "filt_hr_skewness", "cosen_filt", "entropy_filt", 
                    "cv_filt", "qrsw_mean", "qrsw_std", "qrs_abnorm", "qrs_snr", "has_vt", "has_bigeminy", 
                    "has_trigeminy", "has_quadrigeminy", "pwave_count", "pwave_power", "pwave_polarity", 
                    "pwave_pr_mean", "pwave_pr_std", "pwave_pr_kurtosis", "pwave_pr_skewness", "pwave_behind", 
                    "dangling_pwaves", "twave_tp_mean", "twave_tp_std"])
    
    def cleanup(self, embeddings):

        for i in range(len(embeddings)):
            for j in range(len(embeddings[i])):
                if np.isnan(embeddings[i][j]):
                    embeddings[i][j] = 0
                if embeddings[i][j] == False:
                    embeddings[i][j] = 0
                if embeddings[i][j] == True:
                    embeddings[i][j] = 1
                if np.isinf(embeddings[i][j]):
                    embeddings[i][j] = 0

        embeddings = np.array(embeddings)

        return embeddings

    def scale(self):

        self.embeddings["train"] = self.scaler.fit_transform(self.embeddings["train"])
        self.embeddings["id"] = self.scaler.transform(self.embeddings["id"])
        self.embeddings["ood"] = self.scaler.transform(self.embeddings["ood"])

    def get_train_and_test(self):

        X = self.embeddings["train"]
        y = np.array([meta["label"][0] for meta in self.metadata["train"]])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(y_train)

        return X_train, X_test, y_train, y_test

class Visualizer(object):
    def __init__(self, data):
        self.data = data
        self.model = mlp = MLPClassifier(hidden_layer_sizes=(256,),  # Single hidden layer with 64 neurons
                    activation='relu',  # ReLU activation
                    solver='adam',  # Adam optimizer
                    max_iter=500,  # Train for 500 epochs
                    random_state=42)

        self.tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, early_exaggeration=1)

    def fit_model(self):
        X_train, X_test, y_train, y_test = self.data.get_train_and_test()

        # Train Model
        self.model.fit(X_train, y_train)

        # Evaluate Model
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Compute F1-scores
        f1_train = f1_score(y_train, y_train_pred, average="macro")  # Use "macro" or "weighted"
        f1_test = f1_score(y_test, y_test_pred, average="macro")

        print(f"Train F1-Score: {f1_train:.2f}")
        print(f"Test F1-Score: {f1_test:.2f}")

    def get_hidden_layer_activations(self, X_data):
        """ Extract activations from the first hidden layer """
        hidden_layer_weights = self.model.coefs_[0]  # Weights from input to hidden layer
        hidden_layer_bias = self.model.intercepts_[0]  # Bias for the hidden layer
        hidden_activations = np.maximum(0, X_data @ hidden_layer_weights + hidden_layer_bias)  # Apply ReLU
        return hidden_activations

    def ecg_to_base64(self, ecg_signal):
        """ Converts ECG waveform to a base64-encoded image. """
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(ecg_signal, color='black')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.close(fig)

        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches='tight')
        return base64.b64encode(buf.getvalue()).decode()

    def plot_embeddings(self):
        X_train, X_id, X_ood = self.data.embeddings["train"], self.data.embeddings["id"], self.data.embeddings["ood"]

        # Get hidden layer activations
        hidden_activations = self.get_hidden_layer_activations(np.concatenate([X_train, X_id, X_ood]))

        # Transform embeddings to 2D
        embeddings_2d = self.tsne.fit_transform(hidden_activations)
        #embeddings_2d = self.umap.fit_transform(hidden_activations)
        #embeddings_2d = self.pca.fit_transform(np.concatenate([X_train, X_id, X_ood]))

        embeddings_2d_train = embeddings_2d[:len(X_train)]
        embeddings_2d_id = embeddings_2d[len(X_train):(len(X_train)+len(X_id))]
        embeddings_2d_ood = embeddings_2d[(len(X_train)+len(X_id)):]

        print(len(embeddings_2d_train), len(embeddings_2d_id), len(embeddings_2d_ood))

        # Visualize all classes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        scatter = ax.scatter(embeddings_2d_train[:, 0],
                    embeddings_2d_train[:, 1],
                    c="#2c3e50",
                    s=15,
                    alpha=0.3,
        )
        scatter = ax.scatter(embeddings_2d_id[:, 0],
                    embeddings_2d_id[:, 1],
                    c="#3498db",
                    s=15,
                    alpha=0.3,
        )
        scatter = ax.scatter(embeddings_2d_ood[:, 0],
                    embeddings_2d_ood[:, 1],
                    c="#c0392b",
                    s=15,
                    alpha=0.3,
        )
        ax.set_axis_off()
        
        plt.savefig("./paper/images/fig5-embeddings_ALADIN.svg")
        plt.savefig("./paper/images/fig5-embeddings_ALADIN.png", dpi=300)  

        # Visualize all classes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        scatter = ax.scatter(embeddings_2d_ood[:, 0],
                    embeddings_2d_ood[:, 1],
                    c="#bdc3c7",
                    s=30,
                    alpha=0.5,
        )
        scatter = ax.scatter(embeddings_2d_id[:, 0],
                    embeddings_2d_id[:, 1],
                    c="#bdc3c7",
                    s=30,
                    alpha=0.5,
        )
        scatter = ax.scatter(embeddings_2d_train[:, 0],
                    embeddings_2d_train[:, 1],
                    c="#c0392b",
                    s=30,
                    alpha=0.5,
        )
        ax.set_axis_off()
        plt.savefig("./paper/images/fig5-embeddings_ALADIN_train.svg")
        plt.savefig("./paper/images/fig5-embeddings_ALADIN_train.png", dpi=300)      

        # Visualize all classes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        scatter = ax.scatter(embeddings_2d_train[:, 0],
                    embeddings_2d_train[:, 1],
                    c="#bdc3c7",
                    s=30,
                    alpha=0.5,
        )
        scatter = ax.scatter(embeddings_2d_ood[:, 0],
                    embeddings_2d_ood[:, 1],
                    c="#bdc3c7",
                    s=30,
                    alpha=0.5,
        )
        scatter = ax.scatter(embeddings_2d_id[:, 0],
                    embeddings_2d_id[:, 1],
                    c="#c0392b",
                    s=30,
                    alpha=0.5,
        )
        ax.set_axis_off()
        plt.savefig("./paper/images/fig5-embeddings_ALADIN_id.svg")
        plt.savefig("./paper/images/fig5-embeddings_ALADIN_id.png", dpi=300)       

        # Visualize all classes
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
        scatter = ax.scatter(embeddings_2d_train[:, 0],
                    embeddings_2d_train[:, 1],
                    c="#bdc3c7",
                    s=30,
                    alpha=0.5,
        )
        scatter = ax.scatter(embeddings_2d_id[:, 0],
                    embeddings_2d_id[:, 1],
                    c="#bdc3c7",
                    s=30,
                    alpha=0.5,
        )
        scatter = ax.scatter(embeddings_2d_ood[:, 0],
                    embeddings_2d_ood[:, 1],
                    c="#c0392b",
                    s=30,
                    alpha=0.5,
        )
        ax.set_axis_off()
        plt.savefig("./paper/images/fig5-embeddings_ALADIN_ood.svg")
        plt.savefig("./paper/images/fig5-embeddings_ALADIN_ood.png", dpi=300)       


if __name__ == "__main__":
    data = Data(name="ALADIN")
    visualizer = Visualizer(data)
    visualizer.fit_model()
    visualizer.plot_embeddings()