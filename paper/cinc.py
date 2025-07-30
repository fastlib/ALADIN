import numpy as np
import os
import glob
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import matplotlib.pyplot as plt

def rule_based_classes(x):
    mostnoise = x[0]
    mostafib = x[1]
    pacpvcrhythm = x[2]
    onlynsr = x[3]

    if mostnoise:
        return "~"
    elif pacpvcrhythm:
        return "O"
    elif onlynsr:
        return "N"
    elif mostafib:
        return "A"
    else:
        return "O"

    # if pacpvcrhythm:
    #     return "O"
    # elif onlynsr:
    #     if mostnoise:
    #         return "~"
    #     else:
    #         return "N"
    # elif mostafib:
    #     return "A"
    # elif mostnoise:
    #     return "~"
    # else:
    #     return "O"


    # if mostnoise:
    #     return "~"
    # elif mostafib:
    #     return "A"
    # elif onlynsr and not pacpvcrhythm:
    #     return "N"
    # else:
    #     return "O"

def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None

basefolder = os.environ.get('benchmark_results')
aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_CINC")

data = json.load(open(aladin_file, 'r'))
results = data["results"][0]["results"]

X = [r["raw"] for r in results]
Y = [r["true"][0] for r in results]
mapper = {y: i for i, y in enumerate(np.unique(Y))}
mapper = {'N': 0, 'A': 1, 'O': 2, '~': 3}

feats = []
for x in X:
    hasnoise = any([episode["type"] == "NOISE" for episode in x if type(episode) is dict])
    hasnoise50 = any([episode["type"] == "NOISE" and episode["duration"] > 0.5 for episode in x if type(episode) is dict])
    mainrhythms = [episode for episode in x if type(episode) is dict]
    mainrhythms = sorted(mainrhythms, key=lambda x: x["duration"], reverse=True)
    print(mainrhythms)
    mostnoise = mainrhythms[0]["type"] == "NOISE" and mainrhythms[0]["duration"] > 0.5 if len(mainrhythms) > 0 else False
    mostnsr = mainrhythms[0]["type"] == "NSR" if len(mainrhythms) > 0 else False
    mostafib = mainrhythms[0]["type"] == "AFIB" if len(mainrhythms) > 0 else False
    haspac = any([episode == "PAC" for episode in x if type(episode) is not dict])
    haspvc = any([episode == "PVC" for episode in x if type(episode) is not dict])
    hasivb = any([episode == "IVB" for episode in x if type(episode) is not dict])
    hastachycardia = any([episode == "TACHYCARDIA" for episode in x if type(episode) is not dict])
    hasbradycardia = any([episode == "BRADYCARDIA" for episode in x if type(episode) is not dict])
    onlynsr = all([episode["type"] == "NSR" or episode["type"] == "NOISE" for episode in x if type(episode) is dict])
    pacpvcrhythm = haspac or haspvc or hasivb or hastachycardia or hasbradycardia

    feats.append([mostnoise, mostafib, pacpvcrhythm, onlynsr])
    #feats.append([mostnoise, mostafib, onlynsr, haspac, haspvc, hasivb, hastachycardia, hasbradycardia])

X = np.array(feats)
Y = np.array([mapper[y] for y in Y])

clf = DecisionTreeClassifier(random_state=0, min_samples_leaf=5)
clf.fit(X, Y)

# Print the tree structure
print("Decision Tree Structure:")
fig, ax = plt.subplots(figsize=(50, 50))
plot_tree(clf, filled=True, feature_names=[
    "Most Noise > 50%", "Most AFIB", "Has PAC or PVC or IVB", "Only NSR or Noise"
], class_names=list(mapper.keys()), ax=ax)
plt.savefig("paper/decision_tree_structure.png", bbox_inches='tight')
predictions = clf.predict(X)


predictions = [rule_based_classes(x) for x in X]

# for i, pred in enumerate(predictions):
#     data["results"][0]["results"][i]["predicted"] = [pred]

# with open(aladin_file, 'w') as f:
#     json.dump(data, f, indent=4)

predictions = np.array([mapper[p] for p in predictions])

confusion = confusion_matrix(Y, np.round(predictions).astype(int), labels=list(mapper.values()))
print(confusion)

f1_scores = []

for i in range(len(confusion)):
    TP = confusion[i, i]
    FP = confusion[:, i].sum() - TP
    FN = confusion[i, :].sum() - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    f1_scores.append(f1)
    print(f1)

print("F1 Scores:", f1_scores)
print("Average F1 Score:", np.mean(f1_scores[:3]))


