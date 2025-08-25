import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob 
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from typing import Tuple, Dict, Literal, Optional

from aladin.utils.benchmark_utils import Data, Model, DiagnosticBenchmark, CINCData
mpl.rcParams['hatch.linewidth'] = 0.5

class ALADINVirtual():
    def __init__(self):
        self.name = "ALADIN"
        self.save_output = False

class HumanCardiologist():
    def __init__(self):
        self.name = f"Human Cardiologist"
        self.save_output = False

    def predict(self, sig, fs, meta=None, preprocess=False):

        case = meta["record"]
        predicted_episodes = []
        basefolder = os.environ.get('benchmark_data')
        annfile = f"{basefolder}/CINC/training/REFERENCE-v0.csv"
        annotations = pd.read_csv(annfile, header=None, names=["record", "ann"])

        row = annotations[annotations["record"] == case]
        if len(row) == 0:
            return None, {}
        
        ann = row["ann"].values[0]
        predicted_episodes = [{"type": ann, "start": 0, "end": len(sig)/fs}]

        return predicted_episodes, {}

        # for i in range(6):
        #     predpath = f"{basefolder}/CINC/training/{case}_rev{i}*.episodes.json"
        #     annotations = json.load(open(glob.glob(predpath)[0]))
        #     rev_id = annotations["reviewer_id"]
        #     if rev_id == self.id+1:
        #         predicted_episodes = annotations["episodes"]
        #         break

        # #change key name in each item
        # if len(predicted_episodes) > 0:
        #     predicted_episodes = [{k.replace("rhythm_name", "type"): v for k, v in episode.items()} for episode in predicted_episodes]
        #     return predicted_episodes
        # else:
        #     return None


def get_random_kappa_lower_bound():
    x = np.random.uniform(0.0, 8528.0)
    kappa = min(1.0,max(0.25,-0.0001067*x + 1.053))
    return kappa

def get_random_kappa_upper_bound():
    x = np.random.uniform(0.0, 8528.0)
    kappa = 1.0 if x < (8528-1129) else 0.2
    return kappa

def sample_uncertain_label(cls, kappa):
    annotation_frequency_bad = {
        "N": [1203,136,353,367],
        "A": [134,283,203,98],
        "O": [1539,236,685,376],
        "~": [81, 23, 51, 306]
    }
    annotation_frequency_bad = {k: np.array(v) / np.sum(v) for k, v in annotation_frequency_bad.items()}
    annotation_frequency_good = {
        "N": [1, 0, 0, 0],
        "A": [0, 1, 0, 0],
        "O": [0, 0, 1, 0],
        "~": [0, 0, 0, 1]
    }
    factor = (kappa - 0.25) / 0.75
    annotation_frequency = {k: (1-factor) * np.array(v) + factor * np.array(annotation_frequency_good[k]) for k, v in annotation_frequency_bad.items()}
    annotation_frequency = {k: v / np.sum(v) for k, v in annotation_frequency.items()}

    newlabel = np.random.choice(["N", "A", "O", "~"], p=annotation_frequency[cls])
    return newlabel


def get_annotations_per_class(annotations, cls):
    cls_to_idx = { "N": 0, "A": 1, "O": 2, "~": 3 }
    uncertainty_n = [386, 131, 525, 87]
    n_per_class = [7513, 1044, 3098, 531]
    uncertainty_frequency = [
        uncertainty_n[0] / n_per_class[0],
        uncertainty_n[1] / n_per_class[1],
        uncertainty_n[2] / n_per_class[2],
        uncertainty_n[3] / n_per_class[3]
    ]
    train_per_class = annotations[annotations["class"] == cls].reset_index()

    #train_per_class.loc[uncertain_samples_idx, "class"] = train_per_class.loc[uncertain_samples_idx, "class"].apply(lambda x: sample_uncertain_label(cls, get_random_kappa()))
    train_per_class["class_lower"] = train_per_class["class"].apply(lambda x: sample_uncertain_label(cls, get_random_kappa_lower_bound()))
    train_per_class["class_upper"] = train_per_class["class"].apply(lambda x: sample_uncertain_label(cls, get_random_kappa_upper_bound()))

    return train_per_class

def get_cardiologist_metrics(triage=False):

    basefolder = os.environ.get('benchmark_data')
    annfile = f"{basefolder}/CINC/training/REFERENCE-v3.csv"
    annotations = pd.read_csv(annfile, index_col=0, header=None, names=["record", "class"])


    #sort predictions by record name
    annotations.sort_values(by="record", inplace=True)

    annlabels = annotations["class"].values

    if triage:
        annlabels = np.where(annlabels == "N", "NORMAL", annlabels)  # Triage: A -> N
        annlabels = np.where(annlabels == "A", "NOTACUTE", annlabels)  # Triage: A -> N
        annlabels = np.where(annlabels == "O", "NOTACUTE", annlabels)  # Triage: A -> N
        annlabels = np.where(annlabels == "~", "NORMAL", annlabels)  # Triage: A -> N


    arrhythmia = ["NOTACUTE"] if triage else ["N", "A", "O", "~"]
    distributions = {}

    for i in tqdm(range(5)):

        N = get_annotations_per_class(annotations, "N")
        A = get_annotations_per_class(annotations, "A")
        O = get_annotations_per_class(annotations, "O")
        P = get_annotations_per_class(annotations, "~")

        predictions = pd.concat([N, A, O, P], ignore_index=True)
        predictions.sort_values(by="record", inplace=True)
        predlabels_bad = predictions["class_lower"].values
        predlabels_good = predictions["class_upper"].values

        if triage:
            predlabels_bad = np.where((predlabels_bad == "N") | (predlabels_bad == "~"), "NORMAL", predlabels_bad)  # Triage: A -> N
            predlabels_bad = np.where((predlabels_bad == "A") | (predlabels_bad == "O"), "NOTACUTE", predlabels_bad)  # Triage: A -> N
            predlabels_good = np.where((predlabels_good == "N") | (predlabels_good == "~"), "NORMAL", predlabels_good)  # Triage: A -> N
            predlabels_good = np.where((predlabels_good == "A") | (predlabels_good == "O"), "NOTACUTE", predlabels_good)  # Triage: A -> N


        for arr in arrhythmia:
            if arr not in distributions:
                distributions[arr] = {
                    "f1_lower": [], "f1_upper": [],
                    "se_lower": [], "se_upper": [],
                    "sp_lower": [], "sp_upper": [],
                    "tp_lower": [], "tp_upper": [],
                    "fp_lower": [], "fp_upper": [],
                    "fn_lower": [], "fn_upper": [],
                    "acc_lower": [], "acc_upper": [],
                    "ppv_lower": [], "ppv_upper": [],
                    "npv_lower": [], "npv_upper": []
                }
            ann_binary = np.where(annlabels == arr, 1, 0)
            pred_lower_binary = np.where(predlabels_bad == arr, 1, 0)
            tp_lower = np.sum((ann_binary == 1) & (pred_lower_binary == 1))
            fp_lower = np.sum((ann_binary == 0) & (pred_lower_binary == 1))
            fn_lower = np.sum((ann_binary == 1) & (pred_lower_binary == 0))
            tn_lower = np.sum((ann_binary == 0) & (pred_lower_binary == 0))
            acc_lower = (tp_lower + tn_lower) / (tp_lower + fp_lower + fn_lower + tn_lower) if (tp_lower + fp_lower + fn_lower + tn_lower) != 0 else np.nan
            se_lower = tp_lower / (tp_lower + fn_lower) if (tp_lower + fn_lower) != 0 else np.nan
            sp_lower = tn_lower / (tn_lower + fp_lower) if (tn_lower + fp_lower) != 0 and tp_lower > 0 else np.nan
            ppv_lower = tp_lower / (tp_lower + fp_lower) if (tp_lower + fp_lower) != 0 else np.nan
            npv_lower = tn_lower / (tn_lower + fn_lower) if (tn_lower + fn_lower) != 0 else np.nan
            f1_lower = 2*tp_lower / (2*tp_lower + fp_lower + fn_lower) if (2*tp_lower + fp_lower + fn_lower) != 0 else np.nan

            pred_upper_binary = np.where(predlabels_good == arr, 1, 0)
            tp_upper = np.sum((ann_binary == 1) & (pred_upper_binary == 1))
            fp_upper = np.sum((ann_binary == 0) & (pred_upper_binary == 1))
            fn_upper = np.sum((ann_binary == 1) & (pred_upper_binary == 0))
            tn_upper = np.sum((ann_binary == 0) & (pred_upper_binary == 0))
            acc_upper = (tp_upper + tn_upper) / (tp_upper + fp_upper + fn_upper + tn_upper) if (tp_upper + fp_upper + fn_upper + tn_upper) != 0 else np.nan
            se_upper = tp_upper / (tp_upper + fn_upper) if (tp_upper + fn_upper) != 0 else np.nan
            sp_upper = tn_upper / (tn_upper + fp_upper) if (tn_upper + fp_upper) != 0 and tp_upper > 0 else np.nan
            ppv_upper = tp_upper / (tp_upper + fp_upper) if (tp_upper + fp_upper) != 0 else np.nan
            npv_upper = tn_upper / (tn_upper + fn_upper) if (tn_upper + fn_upper) != 0 else np.nan
            f1_upper = 2*tp_upper / (2*tp_upper + fp_upper + fn_upper) if (2*tp_upper + fp_upper + fn_upper) != 0 else np.nan

            distributions[arr]["acc_upper"].append(acc_upper)
            distributions[arr]["acc_lower"].append(acc_lower)
            distributions[arr]["ppv_upper"].append(ppv_upper)
            distributions[arr]["ppv_lower"].append(ppv_lower)
            distributions[arr]["npv_upper"].append(npv_upper)
            distributions[arr]["npv_lower"].append(npv_lower)
            distributions[arr]["f1_upper"].append(f1_upper)
            distributions[arr]["f1_lower"].append(f1_lower)
            distributions[arr]["se_upper"].append(se_upper)
            distributions[arr]["se_lower"].append(se_lower)
            distributions[arr]["sp_upper"].append(sp_upper)
            distributions[arr]["sp_lower"].append(sp_lower)
            distributions[arr]["tp_upper"].append(tp_upper)
            distributions[arr]["tp_lower"].append(tp_lower)
            distributions[arr]["fp_upper"].append(fp_upper)
            distributions[arr]["fp_lower"].append(fp_lower)
            distributions[arr]["fn_upper"].append(fn_upper)
            distributions[arr]["fn_lower"].append(fn_lower)

            #print(f"F1 Score for {arr}: {f1:.4f}")

    return distributions



def lighter(col_str):
    #get rgb from hex string
    rgb = tuple(int(col_str.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    #add 50 to each value
    rgb = tuple([x + 100 if x + 100 < 255 else 255 for x in rgb])
    #convert back to hex string

    return '#%02x%02x%02x' % rgb

def boxplot(ax, data):
    #get 7 colors of the pastel palette.815
    colors = {
        'ECGFounder': "#2968A4",
        'ResNet': "#5C8FC6",
        'ALADIN': "#B4413D",
    }

    for i, model in enumerate(models):
        if model == "Cardiologist":
            sns.boxplot(x='Model', y='Score_lower', data=data[data['Model'] == model], ax=ax, color=colors[model], width=0.5, linecolor=colors[model], fliersize=0.5, linewidth=0.5)
        else:
            sns.boxplot(x='Model', y='Score', data=data[data['Model'] == model], ax=ax, color=colors[model], width=0.5, linecolor=colors[model], fliersize=0.5, linewidth=0.5)

def barplot(ax, data):

    #get 7 colors of the pastel palette
    models = data['Model'].unique()

    colors = {
        'ECGFounder': "#E2E2EA",
        'ResNet': "#C2C7D4", 
        'ALADIN': "#D63F3D", 
        "Cardiologist": "#E0DBC9",
        'Card. 1': "#9099AA",
        'Card. 2': "#9099AA",
        'Card. 3': "#9099AA",
        'Card. 4': "#9099AA",
        'Card. 5': "#9099AA",
        'Card. 6': "#9099AA",
        'Card. 7': "#9099AA",
        'Card. 8': "#9099AA",
        'Card. 9': "#9099AA"
    }
    edgecolors = {
        'ECGFounder': "#C2C7D4",
        'ResNet': "#9099AA",
        'ALADIN': "#B4413D",
        "Cardiologist": "#E0DBC9"
    }
    for i, model in enumerate(models):
        if model == "Cardiologist":
            ax.bar(model, data[data['Model'] == model]['Score_lower'].mean(), color=colors[model], edgecolor=edgecolors[model] if model in edgecolors else colors[model], linewidth=0.5, width=0.8, zorder=1)
            ax.bar(model, 
                data[data['Model'] == model]['Score_upper'].mean()-data[data['Model'] == model]['Score_lower'].mean(), 
                bottom=data[data['Model'] == model]['Score_lower'].mean(), 
                color=lighter(colors[model]), 
                edgecolor=edgecolors[model] if model in edgecolors else colors[model], 
                linewidth=0.5, 
                width=0.8, 
                zorder=1,
                alpha=.99,
                hatch='/////'
            )
        else:
            ax.bar(model, data[data['Model'] == model]['Score'].mean(), color=colors[model], edgecolor=edgecolors[model] if model in edgecolors else colors[model], linewidth=0.5, width=0.8, zorder=1)

        ax.set_xlim(-1.5, len(models) + 0.5)
        ci_upper = np.percentile(data[data['Model'] == model]['Score'], 97.5)
        ci_lower = np.percentile(data[data['Model'] == model]['Score'], 2.5)
        mean_score = data[data['Model'] == model]['Score'].mean()

        # Draw error bars manually
        ax.errorbar(
            x=[i],
            y=[mean_score],
            yerr=[[mean_score - ci_lower], [ci_upper - mean_score]],
            fmt='none',
            color="black",
            elinewidth=0.5,
            capsize=2,
            capthick=0.5,
            zorder=3
        )


def make_piechart():

    data = {
        "Normal": 5076,
        "AFib.": 758,
        "Other": 2415,
        "Noise": 279
    }

    for key in data.keys():
        data[key] /= 8528
        data[key] *= 100

    #ax.pie(data.values(), labels=data.keys(), autopct='%1.0f%%', colors=colors, startangle=90, textprops={'fontsize': 5})

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.6), dpi=300)
    #use colormap blues

    ax.bar(data.keys(), data.values(), color="#E0DBC9", width=0.5, linewidth=0.5, edgecolor="#BFBA9F", zorder=3)
    ax.set_ylabel("Percentage", fontsize=6)
    ax.tick_params(axis='x', width=0.5, labelsize=6, rotation=0, color='#95a5a6')
    ax.tick_params(axis='y', width=0.5, labelsize=6, color='#95a5a6')
    ax.set_yticks([0,20,40,60,80,100])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
        spine.set_color('#1C1C1C')


    plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.9, bottom=0.2, left=0.12, right=0.99)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig3-class_distribution_cinc.svg")

def make_ranking_plot(df):

    basefolder = os.environ.get('benchmark_data')
    other_models = pd.read_csv(basefolder+"/results_all_F1_scores_for_each_classification_type.csv", skiprows=2)
    other_models.columns = ['Rank', 'F1n_test', 'F1a_test', 'F1o_test', 'F1p_test', 'F1tot_test', 'F1n_train','F1a_train','F1o_train','F1p_train','F1tot_train','Entry','Closed','Authors']
    other_models["F1_test"] = other_models.apply(lambda x: (x["F1n_test"] + x["F1a_test"] + x["F1o_test"])/3, axis=1)
    print(other_models)

    other_models = other_models.sort_values(by='F1_test', ascending=False)
    other_f1s = other_models['F1_test'].values

    own_f1_dist = []
    class_a = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "f1")]['Score'].values
    class_n = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "f1")]['Score'].values
    class_o = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "f1")]['Score'].values
    class_p = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == '~') & (df["Metric"] == "f1")]['Score'].values
    for i in range(len(class_a)):
        own_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)

    own_f1 = np.mean(own_f1_dist)
    print("N:", np.mean(class_n), "A:", np.mean(class_a), "O:", np.mean(class_o))
    print("Best competitor:", np.max(other_f1s))
    own_f1_low, own_f1_high = np.percentile(own_f1_dist, 2.5), np.percentile(own_f1_dist, 97.5)
    print(f"ALADIN: {own_f1:.3f}% (95% CI={own_f1_low:.3f}-{own_f1_high:.3f}%)")

    ecgf_f1_dist = []
    class_a = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "f1")]['Score'].values
    class_n = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "f1")]['Score'].values
    class_o = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "f1")]['Score'].values
    class_p = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == '~') & (df["Metric"] == "f1")]['Score'].values
    for i in range(len(class_a)):
        ecgf_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)
        
    ecgf_f1 = np.mean(ecgf_f1_dist)
    ecgf_f1_low, ecgf_f1_high = np.percentile(ecgf_f1_dist, 2.5), np.percentile(ecgf_f1_dist, 97.5)
    print(f"ECGFounder: {ecgf_f1:.3f}% (95% CI={ecgf_f1_low:.3f}-{ecgf_f1_high:.3f}%)")

    resnet_f1_dist = []
    class_a = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "f1")]['Score'].values
    class_n = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "f1")]['Score'].values
    class_o = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "f1")]['Score'].values
    for i in range(len(class_a)):
        resnet_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)
        
    resnet_f1 = np.mean(resnet_f1_dist)
    resnet_f1_low, resnet_f1_high = np.percentile(resnet_f1_dist, 2.5), np.percentile(resnet_f1_dist, 97.5)
    print(f"ResNet: {resnet_f1:.3f}% (95% CI={resnet_f1_low:.3f}-{resnet_f1_high:.3f}%)")

    all_f1s = []
    for i in range(len(other_f1s)):
        all_f1s.append((other_f1s[i],"white","#E2E2EA"))
    all_f1s.append((own_f1,"#D63F3D", "#B4413D"))
    all_f1s.append((resnet_f1,"#C2C7D4", "#9099AA"))
    all_f1s.append((ecgf_f1,"#E2E2EA", "#C2C7D4"))

    all_f1s = sorted(all_f1s, key=lambda x: x[0], reverse=True)
    ranking = np.arange(1, len(all_f1s)+1)

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.5), dpi=300)
    ax.bar(ranking, [x[0] for x in all_f1s], color=[x[1] for x in all_f1s], edgecolor=[x[2] for x in all_f1s], linewidth=0.5, width=0.7)
    ax.set_xticks([1] + list(np.arange(5, len(all_f1s), 5)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)], fontsize=5)
    ax.set_ylabel("Average F1 score (N, A, O)", fontsize=6)
    ax.set_xlabel("Pseudo CinC competition ranking", fontsize=6)
    ax.set_xlim(0.5, len(all_f1s)+0.5)
    ax.spines['top'].set_vicincsible(False)
    ax.spines['right'].set_visible(False)

    own_pos = [x[0] for x in all_f1s].index(own_f1)+1
    resnet_pos = [x[0] for x in all_f1s].index(resnet_f1) +1
    ecgf_pos = [x[0] for x in all_f1s].index(ecgf_f1)+1
    print(own_pos, resnet_pos, ecgf_pos)

    #draw confidence intervals
    ax.plot([own_pos,own_pos], [own_f1_low, own_f1_high], color="black", linewidth=0.5)
    ax.plot([own_pos-0.2, own_pos+0.2], [own_f1_low, own_f1_low], color="black", linewidth=0.5)
    ax.plot([own_pos-0.2, own_pos+0.2], [own_f1_high, own_f1_high], color="black", linewidth=0.5)

    ax.plot([resnet_pos,resnet_pos], [resnet_f1_low, resnet_f1_high], color="black", linewidth=0.5)
    ax.plot([resnet_pos-0.2, resnet_pos+0.2], [resnet_f1_low, resnet_f1_low], color="black", linewidth=0.5)
    ax.plot([resnet_pos-0.2, resnet_pos+0.2], [resnet_f1_high, resnet_f1_high], color="black", linewidth=0.5)

    ax.plot([ecgf_pos,ecgf_pos], [ecgf_f1_low, ecgf_f1_high], color="black", linewidth=0.5)
    ax.plot([ecgf_pos-0.2, ecgf_pos+0.2], [ecgf_f1_low, ecgf_f1_low], color="black", linewidth=0.5)
    ax.plot([ecgf_pos-0.2, ecgf_pos+0.2], [ecgf_f1_high, ecgf_f1_high], color="black", linewidth=0.5)

    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#9099AA')

    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig3-ranking_cinc.svg")
    plt.savefig("paper/images/fig3-ranking_cinc.png", dpi=300)

def make_boxplots(df):

    # Create a figure with 2 rows and 6 columns of subplots
    fig, axs = plt.subplots(1, 4, figsize=(7.08, 1.5), dpi=300)
    axs = axs.flatten()

    titlemap = {
        "N": "Normal",
        "A": "Atrial fibrillation",
        "O": "Other",
        "~": "Noise"
    }

    arrhythmias = ["N", "A", "O", "~"]
    arrhythmia_formatted = ["Normal", "Atrial fibrillation", "Other", "Noise"]

    for i, arr in enumerate(arrhythmias):
        ax = axs[i]
        subset = df[(df['Arrhythmia'] == arr) & (df["Metric"] == "f1")]
        barplot(ax, subset)
        ax.set_xlabel(arrhythmia_formatted[i], fontsize=6)
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', width=0.25, labelsize=3, color='#1C1C1C')
        if i == 0:
            ax.set_ylabel('F1 Score', fontsize=6)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
            spine.set_color('#1C1C1C')
            
        ax.tick_params(axis='y', width=0.25, labelsize=5, color='#1C1C1C')
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=5)
        ax.set_ylim(0, 1)


    plt.subplots_adjust(wspace=0.5, hspace=0.4, top=0.95, bottom=0.2, left=0.12, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig3-boxplot-cinc.svg")
    plt.savefig("./paper/images/fig3-boxplot-cinc.png", dpi=300)
 
def horizontal_barplot(ax, data):
    #get 7 colors of the pastel palette
    #data = data[data["Metric"] == "f1"]
    models = data['Model'].unique()

    colors = {
        'ECGFounder': "#E2E2EA",
        'ResNet': "#C2C7D4", 
        'ALADIN': "#D63F3D", 
        "Cardiologist": "#E0DBC9",
        'Card. 1': "#9099AA",
        'Card. 2': "#9099AA",
        'Card. 3': "#9099AA",
        'Card. 4': "#9099AA",
        'Card. 5': "#9099AA",
        'Card. 6': "#9099AA",
        'Card. 7': "#9099AA",
        'Card. 8': "#9099AA",
        'Card. 9': "#9099AA"
    }
    edgecolors = {
        'ECGFounder': "#C2C7D4",
        'ResNet': "#9099AA",
        'ALADIN': "#B4413D",
        "Cardiologist": "#E0DBC9"
    }
    models = reversed(models)

    for i, model in enumerate(models):

        if model == "Cardiologist":
            ax.barh(model, (100*data[data['Model'] == model]['Score_lower'].mean()), color=colors[model], edgecolor=edgecolors[model] if model in edgecolors else colors[model], linewidth=0.5, height=0.8, zorder=1)
            ax.barh(model, 
                (100*data[data['Model'] == model]['Score_upper'].mean())-(100*data[data['Model'] == model]['Score_lower'].mean()), 
                left=(100*data[data['Model'] == model]['Score_lower'].mean()), 
                color=lighter(colors[model]), 
                edgecolor=edgecolors[model] if model in edgecolors else colors[model], 
                linewidth=0.5, 
                height=0.8, 
                zorder=1,
                alpha=.99,
                hatch='/////'
            )
        else:
            ax.barh(model, (100*data[data['Model'] == model]['Score'].mean()), color=colors[model], edgecolor=edgecolors[model] if model in edgecolors else colors[model], linewidth=0.5, height=0.8, zorder=1)


        ci_upper = (100*np.percentile(data[data['Model'] == model]['Score'], 97.5))
        ci_lower = (100*np.percentile(data[data['Model'] == model]['Score'], 2.5))
        mean_score = (100*data[data['Model'] == model]['Score'].mean())

        # Draw error bars manually
        ax.errorbar(
            x=[mean_score],
            y=[i],
            xerr=[[mean_score - ci_lower], [ci_upper - mean_score]],
            fmt='none',
            color="black",
            elinewidth=0.5,
            capsize=2,
            capthick=0.5,
            zorder=3
        )

def make_horizontal_barplots(df):

    fig, axs = plt.subplots(1, 3, figsize=(3.54, 1), dpi=300)
    axs = axs.flatten()

    metriclabels = ["Correct (%)", "False alarm (%)", "Missed (%)"]

    # Plot a boxplot for each arrhythmia in its own subplot
    for i, arr in enumerate(arrhythmias):
        for j, metric in enumerate(["TP", "FP", "FN"]):
            ax = axs[i*3+j]
            subset = df[df['Arrhythmia'] == arr]
            # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
            horizontal_barplot(ax, subset[subset['Metric'] == metric])

            #ax.set_title(arr, fontsize=6)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks(np.arange(0, 51, 10))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='x', width=0, labelsize=3, color='#1C1C1C')
            if i == 0:
                ax.spines['bottom'].set_visible(True)
                ax.tick_params(axis='x', width=0.25, labelsize=3, color='#1C1C1C')
                ax.set_xlabel(metriclabels[j], fontsize=6)
                ax.set_xticklabels(np.arange(0, 51, 10), fontsize=5)

            if j == 0:
                ax.set_ylabel(arr, fontsize=5)

            # make spines linewidth 0.5
            for spine in ax.spines.values():
                spine.set_linewidth(0.25)
                spine.set_color('#1C1C1C')
            # rotate x labels 45 degrees and draw tick lines
            ax.tick_params(axis='y', width=0.25, labelsize=5, color='#1C1C1C')
            ax.set_xlim(0, 61)
            ax.grid(axis='x', linestyle='--', linewidth=0.25, color='#bdc3c7')

    
    plt.subplots_adjust(wspace=0.25, hspace=0.4, top=0.95, bottom=0.4, left=0.2, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig3-boxplot-cinc-triage.svg")
    plt.savefig("./paper/images/fig3-boxplot-cinc-triage.png", dpi=300, facecolor="white", edgecolor="white", transparent=False)

def make_confusion_matrix_aladin(file):

    with open(file, 'r') as f:
        data = json.load(f)

    trues = []
    preds = []
    mapping = {
        "N": 0,
        "A": 1,
        "O": 2,
        "~": 3
    }

    results = data["results"][0]["results"]

    for result in results:
        trues.append(mapping[result["true"][0]])
        preds.append(mapping[result["predicted"][0]])

    trues = np.array(trues)
    preds = np.array(preds)

    cm = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            cm[i, j] = int(np.sum((trues == i) & (preds == j)))

    #cm = cm / np.sum(cm, axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6), dpi=200)
    sns.heatmap(cm, annot=True, cmap="Reds", cbar=False, ax=ax, fmt='g', annot_kws={"size": 5})
    ax.set_xticklabels(["Normal", "AFib.", "Other", "Noise"], fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5)
    ax.set_yticklabels(["Normal", "AFib.", "Other", "Noise"], fontsize=5)
    ax.tick_params(axis='y', width=0.5, labelsize=5)
    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("True", fontsize=6)

    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.2, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig3-confusion_matrix_aladin.svg")
    plt.savefig("./paper/images/fig3-confusion_matrix_aladin.png", dpi=300)

def make_confusion_matrix_ecgfounder(file):

    with open(file, 'r') as f:
        data = json.load(f)

    trues = []
    preds = []
    mapping = {
        "N": 0,
        "A": 1,
        "O": 2,
        "~": 3
    }

    results = data["results"][0]["results"]

    for result in results:
        trues.append(mapping[result["true"][0]])
        preds.append(mapping[result["predicted"][0]])

    trues = np.array(trues)
    preds = np.array(preds)

    cm = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            cm[i, j] = int(np.sum((trues == i) & (preds == j)))

    #cm = cm / np.sum(cm, axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(1, 1, figsize=(1.6, 1.6), dpi=200)
    sns.heatmap(cm, annot=True, cmap="Blues", cbar=False, ax=ax, fmt='g', annot_kws={"size": 5})
    ax.set_xticklabels(["Normal", "AFib.", "Other", "Noise"], fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5)
    ax.set_yticklabels(["Normal", "AFib.", "Other", "Noise"], fontsize=5)
    ax.tick_params(axis='y', width=0.5, labelsize=5)
    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("True", fontsize=6)

    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.2, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig3-confusion_matrix_ecgfounder.svg")
    plt.savefig("./paper/images/fig3-confusion_matrix_ecgfounder.png", dpi=300)

def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None


def best_non_overlapping(intervals, lowerbetter=False):
    """
    Return the non-overlapping interval with the highest mean value.
    If no such interval exists, return None.
    """
    n = len(intervals)
    non_overlapping = []

    for i in range(n):
        lo1, hi1 = intervals[i]
        overlap = False
        for j in range(n):
            if i == j:
                continue
            lo2, hi2 = intervals[j]
            # Check overlap (endpoints count as overlap)
            if not (hi1 < lo2 or hi2 < lo1):
                overlap = True
                break
        if not overlap:
            non_overlapping.append(intervals[i])

    if not non_overlapping:
        return None

    if not lowerbetter:
        for i, (lo, hi) in enumerate(non_overlapping):
            highest = True
            for j in range(n):
                if i == j:
                    continue
                lo2, hi2 = intervals[j]
                v = (lo2 + hi2) / 2

                # Check if the current interval is higher than any other interval
                if v > (lo + hi) / 2:
                    highest = False
                    break

            if highest:
                return i
    else:
        for i, (lo, hi) in enumerate(non_overlapping):
            lowest = True
            for j in range(n):
                if i == j:
                    continue
                lo2, hi2 = intervals[j]
                v = (lo2 + hi2) / 2

                # Check if the current interval is higher than any other interval
                if v < (lo + hi) / 2:
                    lowest = False
                    break

            if lowest:
                return i

    # Pick interval with highest mean
    return -1

def generate_metrics_table(df, percentage=True):

    maxcol = 12

    def format_ci(ci, percentage=True, only_ci=False):
        if percentage:
            if only_ci:
                return "(" + str(np.round(ci[1]*1000)/10) + "-" + str(np.round(ci[2]*1000)/10) + ")"
            else:
                return str(np.round(ci[0]*1000)/10)
        else:
            if only_ci:
                return  "(" + str(np.round(ci[1], 2)) + "-" + str(np.round(ci[2], 2)) + ")"
            else:
                return str(np.round(ci[0], 2))

    print(df)
    metrics = df["Metric"].unique()
    models = df["Model"].unique()
    arrhythmias = df["Arrhythmia"].unique()
    print(models)

    metrics_formatted = {
        "acc": "Accuracy (95\% CI), \%",
        "se": "Sensitivity (95\% CI), \%",
        "sp": "Specificity (95\% CI), \%",
        "npv": "NPV (95\% CI), \%",
        "ppv": "PPV (95\% CI), \%",
        "f1": "F1 Score (95\% CI), \%",
        "TP": "Correct diagnoses (95\% CI)",
        "FP": "False alarms (95\% CI)",
        "FN": "Missed cases (95\% CI)",
    }
    arrhythmias_formatted = {
        "A": "AFIB",
        "O": "Other",
        "~": "Noise",
        "N": "Normal",
        "NOTACUTE": "Flagged"
    }

    maxmetrics = int(maxcol/len(models))

    latex = "\\begin{tabular}{" + "l|"
    for metric in metrics[:maxmetrics]:
        for model in models:
            latex += "r"
        latex += "|" if metric != metrics[-1] else "} \n"

    latex += "Arrhythmia & "
    for metric in metrics[:maxmetrics]:
        latex += "\\multicolumn{"+str(len(models))+"}{c|}{\\textbf{" + metrics_formatted[metric] + "} } & "
    latex = latex[:-2] + " \\\\ \\hline\n"

    latex += " & "
    for metric in metrics[:maxmetrics]:
        for model in models:
            latex += "\\textbf{" + model + "} & "
    latex = latex[:-2] + " \\\\ \\hline\n"

    for arrhythmia in arrhythmias:
        latex += arrhythmias_formatted[arrhythmia] + " & "

        for metric in metrics[:maxmetrics]:
            intervals = []
            for model in models:
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                #print(xs)
                if not pd.isna(xs["Score_lower"]).all():
                    ci_lower = xs["Score_lower"].mean()
                    ci_upper = xs["Score_upper"].mean()
                    intervals.append((ci_lower, ci_upper))
                else:
                    xs = xs['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

            lowerbetter = True if metric in ["FP","FN"] else False
            best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

            for i, model in enumerate(models):
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                if not pd.isna(xs["Score_lower"]).all():
                    ci_lower = xs["Score_lower"].mean()
                    ci_upper = xs["Score_upper"].mean()
                    mean = (ci_upper+ci_lower)/2
                else:
                    xs = xs['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)

                print(mean, ci_lower, ci_upper)

                if best_interval == i:
                    latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage) + "} & "
                else:
                    latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage) + " & "

        latex = latex[:-2] + " \\\\\n"

        latex +=  " & "
        for metric in metrics[:maxmetrics]:
            intervals = []
            for model in models:
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                if not pd.isna(xs["Score_lower"]).all():
                    ci_lower = xs["Score_lower"].mean()
                    ci_upper = xs["Score_upper"].mean()
                    intervals.append((ci_lower, ci_upper))
                    continue
                xs = xs['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)
                intervals.append((ci_lower, ci_upper))

            lowerbetter = True if metric in ["FP","FN"] else False
            best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

            for i, model in enumerate(models):
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                if not pd.isna(xs["Score_lower"]).all():
                    ci_lower = xs["Score_lower"].mean()
                    ci_upper = xs["Score_upper"].mean()
                    mean = (ci_upper+ci_lower)/2
                else:
                    xs = xs['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)

                if best_interval == i:
                    latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + "} & "
                else:
                    latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + " & "

        latex = latex[:-2] + " \\\\ \\hline \n"

    if len(metrics) > maxmetrics:
        latex = latex[:-2] + "\\hline \n"
        latex += " & "
        for metric in metrics[maxmetrics:]:
            latex += "\\multicolumn{"+str(len(models))+"}{c|}{\\textbf{" + metrics_formatted[metric] + "} } & "
        latex = latex[:-2] + " \\\\ \\hline\n"

        latex += " & "
        for metric in metrics[maxmetrics:]:
            for model in models:
                latex += "\\textbf{" + model + "} & "
        latex = latex[:-2] + " \\\\ \\hline\n"

        for arrhythmia in arrhythmias:
            latex += arrhythmias_formatted[arrhythmia] + " & "
            for metric in metrics[maxmetrics:]:
                intervals = []
                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                    if not pd.isna(xs["Score_lower"]).all():
                        ci_lower = xs["Score_lower"].mean()
                        ci_upper = xs["Score_upper"].mean()
                        intervals.append((ci_lower, ci_upper))
                        continue
                    xs = xs['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

                lowerbetter = True if metric in ["FP","FN"] else False
                best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

                for i, model in enumerate(models):
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                    if not pd.isna(xs["Score_lower"]).all():
                        ci_lower = xs["Score_lower"].mean()
                        ci_upper = xs["Score_upper"].mean()
                        mean = (ci_upper+ci_lower)/2
                    else:
                        xs = xs['Score']
                        mean = xs.mean()
                        ci_lower = np.percentile(xs, 2.5)
                        ci_upper = np.percentile(xs, 97.5)

                    if best_interval == i:
                        latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage) + "} & "
                    else:
                        latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage) + " & "

            latex = latex[:-2] + " \\\\\n"

            latex +=  " & "
            for metric in metrics[maxmetrics:]:
                intervals = []
                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                    if not pd.isna(xs["Score_lower"]).all():
                        ci_lower = xs["Score_lower"].mean()
                        ci_upper = xs["Score_upper"].mean()
                        intervals.append((ci_lower, ci_upper))
                        continue
                    xs = xs['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

                lowerbetter = True if metric in ["FP","FN"] else False
                best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

                for i, model in enumerate(models):
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]
                    if not pd.isna(xs["Score_lower"]).all():
                        ci_lower = xs["Score_lower"].mean()
                        ci_upper = xs["Score_upper"].mean()
                        mean = (ci_upper+ci_lower)/2
                    else:
                        xs = xs['Score']
                        mean = xs.mean()
                        ci_lower = np.percentile(xs, 2.5)
                        ci_upper = np.percentile(xs, 97.5)

                    if best_interval == i:
                        latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + "} & "
                    else:
                        latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage, only_ci=True) + " & "

            latex = latex[:-2] + " \\\\ \\hline \n"

    latex = latex[:-1] + " \\hline \n"
    latex += "\\end{tabular}"

    return latex

if __name__ == "__main__":
    
    # Load the data
    cinc = CINCData("CINC", asynchronous=True)
    aladinvirt = ALADINVirtual()
    aladin_experiment = DiagnosticBenchmark(cinc, aladinvirt)

    basefolder = os.environ.get('benchmark_results')
    aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_CINC")
    resnet_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_Hannun_CINC")
    ecgfounder_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ECGFounder_CINC")

    aladin_metrics, aladin_distributions = aladin_experiment.aggregate(aladin_file, bootstrap=True)
    aladin_metrics_triage, aladin_distributions_triage = aladin_experiment.aggregate(aladin_file, bootstrap=True, triage=True)

    resnet_metrics, resnet_distributions = aladin_experiment.aggregate(resnet_file, bootstrap=True)
    resnet_metrics_triage, resnet_distributions_triage = aladin_experiment.aggregate(resnet_file, bootstrap=True, triage=True)
    ecgfounder_metrics, ecgfounder_distributions = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True)
    ecgfounder_metrics_triage, ecgfounder_distributions_triage = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True, triage=True)

    human_distributions = get_cardiologist_metrics()
    human_distributions_triage = get_cardiologist_metrics(triage=True)

    data = {
        "ECGFounder": ecgfounder_distributions,
        "ResNet": resnet_distributions,
        "ALADIN": aladin_distributions,
        "Cardiologist": human_distributions
    }
    
    data_triage = {
        "ECGFounder": ecgfounder_distributions_triage,
        "ResNet": resnet_distributions_triage,
        "ALADIN": aladin_distributions_triage,
        "Cardiologist": human_distributions_triage
    }

    arrhythmias = list(data["ECGFounder"].keys())
    models = list(data.keys())
    print(models)

    rowdata = []
    metrics = ["acc", "se", "sp", "ppv", "npv", "f1"]
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            for metric in metrics:
                if model == "Cardiologist":
                    metric_scores = (data[model][arr][metric+"_lower"], data[model][arr][metric+"_upper"])
                    for i in range(len(metric_scores[0])):
                        rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score_lower': metric_scores[0][i], 'Score_upper': metric_scores[1][i], "Iteration": i, "Metric": metric})
                else:
                    scores = data[model][arr][metric]
                    for i, score in enumerate(scores):
                        rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": metric})

    
    df = pd.DataFrame(rowdata)
    print(generate_metrics_table(df))

    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)

    make_boxplots(df)
    make_confusion_matrix_aladin(aladin_file)
    make_confusion_matrix_ecgfounder(ecgfounder_file)
    make_ranking_plot(df)
    
    arrhythmias = ["NOTACUTE"]
    models = list(data_triage.keys())
    
    rowdata_triage = []
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            if "tp" not in data_triage[model][arr]:
                tp_scores = (data_triage[model][arr]["tp_lower"], data_triage[model][arr]["tp_upper"])
                fp_scores = (data_triage[model][arr]["fp_lower"], data_triage[model][arr]["fp_upper"])
                fn_scores = (data_triage[model][arr]["fn_lower"], data_triage[model][arr]["fn_upper"])
                
                for i, score in enumerate(tp_scores[0]):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score_lower': score/8528, "Iteration": i, "Metric": "TP"})
                for i, score in enumerate(tp_scores[1]):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score_upper': score/8528, "Iteration": i, "Metric": "TP"})
                for i, score in enumerate(fp_scores[0]):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score_lower': score/8528, "Iteration": i, "Metric": "FP"})
                for i, score in enumerate(fp_scores[1]):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score_upper': score/8528, "Iteration": i, "Metric": "FP"})
                for i, score in enumerate(fn_scores[0]):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score_lower': score/8528, "Iteration": i, "Metric": "FN"})
                for i, score in enumerate(fn_scores[1]):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score_upper': score/8528, "Iteration": i, "Metric": "FN"})
            else:
                tp_scores = data_triage[model][arr]['tp']
                fp_scores = data_triage[model][arr]['fp']
                fn_scores = data_triage[model][arr]['fn']

                for i, score in enumerate(tp_scores):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score': score/8528, "Iteration": i, "Metric": "TP"})
                for i, score in enumerate(fp_scores):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score': score/8528, "Iteration": i, "Metric": "FP"})
                for i, score in enumerate(fn_scores):
                    rowdata_triage.append({'Arrhythmia': arr, 'Model': model, 'Score': score/8528, "Iteration": i, "Metric": "FN"})

    df = pd.DataFrame(rowdata_triage)

    print(generate_metrics_table(df, percentage=True))
    make_horizontal_barplots(df)

