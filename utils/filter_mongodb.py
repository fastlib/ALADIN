import pymongo
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np


client = MongoClient('mongodb://localhost:27017/', username='root', password='j230sdncjsdf234')
db = client['orion']
datadb = db['data']

#get labeled record where annotations are empty
record = list(datadb.find({"status": "labelled"}, {"_id": 0, "recordname": 1, "ecg": 1}))
#record = list(datadb.find({"status": "labelled"}, {"_id": 0, "recordname": 1}))

print(len(record), "records found")

for i in range(0, len(record), 20):
    fig, ax = plt.subplots(5, 4, figsize=(20, 10), dpi=200)
    ax = ax.flatten()
    for j in range(20):
        if i+j >= len(record):
            break
        r = record[i+j]
        ecg = np.array(r["ecg"])
        ax[j].plot(ecg[:,0], label=r["recordname"])
        ax[j].set_title(r["recordname"])
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].set_xticklabels([])
        ax[j].set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig(f"record_{i//20}.png")


recordnames = [r["recordname"] for r in record]