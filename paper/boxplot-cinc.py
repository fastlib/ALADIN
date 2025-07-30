import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob 
import json
import matplotlib.pyplot as plt

from aladin.utils.benchmark_utils import Data, Model, DiagnosticBenchmark

class ALADINVirtual():
    def __init__(self):
        self.name = "ALADIN"
        self.save_output = False

def lighter(col_str):
    #get rgb from hex string
    rgb = tuple(int(col_str.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    #add 50 to each value
    rgb = tuple([x + 100 if x + 100 < 255 else 255 for x in rgb])
    #convert back to hex string

    return '#%02x%02x%02x' % rgb

def boxplot(ax, data):
    #get 7 colors of the pastel palette
    colors = sns.color_palette('pastel', 7)
    models = ["ECGFounder", "ResNet", "ALADIN"]

    colors = {
        'ECGFounder': "#2968A4",
        'ResNet': "#5C8FC6",
        'ALADIN': "#B4413D",
    }

    for i, model in enumerate(models):
        sns.boxplot(x='Model', y='Score', data=data[data['Model'] == model], ax=ax, color=lighter(colors[model]), width=0.5, linecolor=colors[model], fliersize=0.5, linewidth=0.5)

def make_piechart():

    data = {
        "Normal": 5076,
        "AFib.": 758,
        "Other": 2415,
        "Noise": 279
    }

    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.8), dpi=300)
    #use colormap blues
    cmap = plt.get_cmap("Blues")
    colors = cmap(np.linspace(0, 0.5, 4))

    ax.pie(data.values(), labels=data.keys(), autopct='%1.0f%%', colors=colors, startangle=90, textprops={'fontsize': 5})

    plt.savefig("paper/images/piechart.png")

def make_ranking_plot(df):

    basefolder = os.environ.get('benchmark_data')
    other_models = pd.read_csv(basefolder+"/results_all_F1_scores_for_each_classification_type.csv", skiprows=2)
    other_models.columns = ['Rank', 'F1n_test', 'F1a_test', 'F1o_test', 'F1p_test', 'F1tot_test', 'F1n_train','F1a_train','F1o_train','F1p_train','F1tot_train','Entry','Closed','Authors']
    other_models["F1_test"] = other_models.apply(lambda x: (x["F1n_test"] + x["F1a_test"] + x["F1o_test"])/3, axis=1)
    print(other_models)

    other_models = other_models.sort_values(by='F1_test', ascending=False)
    other_f1s = other_models['F1_test'].values

    own_f1_dist = []
    class_a = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "F1")]['Score'].values
    class_n = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "F1")]['Score'].values
    class_o = df[(df['Model'] == 'ALADIN') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "F1")]['Score'].values
    for i in range(len(class_a)):
        own_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)

    own_f1 = np.mean(own_f1_dist)
    print("ALADIN:", own_f1)
    print("N:", np.mean(class_n), "A:", np.mean(class_a), "O:", np.mean(class_o))
    print("Best competitor:", np.max(other_f1s))
    own_f1_low, own_f1_high = np.percentile(own_f1_dist, 2.5), np.percentile(own_f1_dist, 97.5)

    ecgf_f1_dist = []
    class_a = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "F1")]['Score'].values
    class_n = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "F1")]['Score'].values
    class_o = df[(df['Model'] == 'ECGFounder') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "F1")]['Score'].values
    for i in range(len(class_a)):
        ecgf_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)
        
    ecgf_f1 = np.mean(ecgf_f1_dist)
    print("ECGFounder:", ecgf_f1)
    ecgf_f1_low, ecgf_f1_high = np.percentile(ecgf_f1_dist, 2.5), np.percentile(ecgf_f1_dist, 97.5)

    hannun_f1_dist = []
    class_a = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'A') & (df["Metric"] == "F1")]['Score'].values
    class_n = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'N') & (df["Metric"] == "F1")]['Score'].values
    class_o = df[(df['Model'] == 'ResNet') & (df['Arrhythmia'] == 'O') & (df["Metric"] == "F1")]['Score'].values
    for i in range(len(class_a)):
        hannun_f1_dist.append((class_a[i] + class_n[i] + class_o[i])/3)
        
    hannun_f1 = np.mean(hannun_f1_dist)
    print("ResNet:", hannun_f1)
    hannun_f1_low, hannun_f1_high = np.percentile(hannun_f1_dist, 2.5), np.percentile(hannun_f1_dist, 97.5)

    all_f1s = []
    for i in range(len(other_f1s)):
        all_f1s.append((other_f1s[i],"#E2E2EA"))
    all_f1s.append((own_f1,"#B4413D"))
    all_f1s.append((hannun_f1,"#5C8FC6"))
    all_f1s.append((ecgf_f1,"#2968A4"))

    all_f1s = sorted(all_f1s, key=lambda x: x[0], reverse=True)
    ranking = np.arange(1, len(all_f1s)+1)

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.5), dpi=300)
    ax.bar(ranking, [x[0] for x in all_f1s], color=[x[1] for x in all_f1s], width=0.7)
    ax.set_xticks([1] + list(np.arange(5, len(all_f1s), 5)))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)], fontsize=5)
    ax.set_ylabel("F1 score", fontsize=6)
    ax.set_xlabel("Final competition ranking", fontsize=6)
    ax.set_xlim(0.5, len(all_f1s)+0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    own_pos = [x[0] for x in all_f1s].index(own_f1)+1
    hannun_pos = [x[0] for x in all_f1s].index(hannun_f1) +1
    ecgf_pos = [x[0] for x in all_f1s].index(ecgf_f1)+1
    print(own_pos, hannun_pos, ecgf_pos)

    #draw confidence intervals
    ax.plot([own_pos,own_pos], [own_f1_low, own_f1_high], color="black", linewidth=0.5)
    ax.plot([own_pos-0.2, own_pos+0.2], [own_f1_low, own_f1_low], color="black", linewidth=0.5)
    ax.plot([own_pos-0.2, own_pos+0.2], [own_f1_high, own_f1_high], color="black", linewidth=0.5)

    ax.plot([hannun_pos,hannun_pos], [hannun_f1_low, hannun_f1_high], color="black", linewidth=0.5)
    ax.plot([hannun_pos-0.2, hannun_pos+0.2], [hannun_f1_low, hannun_f1_low], color="black", linewidth=0.5)
    ax.plot([hannun_pos-0.2, hannun_pos+0.2], [hannun_f1_high, hannun_f1_high], color="black", linewidth=0.5)

    ax.plot([ecgf_pos,ecgf_pos], [ecgf_f1_low, ecgf_f1_high], color="black", linewidth=0.5)
    ax.plot([ecgf_pos-0.2, ecgf_pos+0.2], [ecgf_f1_low, ecgf_f1_low], color="black", linewidth=0.5)
    ax.plot([ecgf_pos-0.2, ecgf_pos+0.2], [ecgf_f1_high, ecgf_f1_high], color="black", linewidth=0.5)

    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#9099AA')

    plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig4-ranking_plot.svg")
    plt.savefig("paper/images/fig4-ranking_plot.png", dpi=300)

def make_boxplots(df):

    # Create a figure with 2 rows and 6 columns of subplots
    fig, axs = plt.subplots(3, 4, figsize=(3.54, 2.5), sharey=True, dpi=200)
    axs = axs.flatten()

    titlemap = {
        "N": "Normal",
        "A": "Atrial fibrillation",
        "O": "Other",
        "~": "Noise"
    }

    #se_means = df.groupby(['Arrhythmia', 'Model'])['SE'].mean().reset_index()
    #sp_means = df.groupby(['Arrhythmia', 'Model'])['SP'].mean().reset_index()
    #boxplot(axs[0], se_means)
    arrhythmias = ["N", "A", "O", "~"]

    # Plot a boxplot for each arrhythmia in its own subplot
    for i, arr in enumerate(arrhythmias):
        ax = axs[i]
        subset = df[(df['Arrhythmia'] == arr) & (df["Metric"] == "F1")]
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        boxplot(ax, subset)
        ax.set_title(titlemap[arr], fontsize=6)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([], fontsize=5)
        if i > 0:
            ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
        else:
            ax.tick_params(axis='x', width=0.5, rotation=0, labelsize=5, color='#9099AA')
        if i == 0:
            ax.set_ylabel('F1 score', fontsize=6)
        #else:
            #ax.spines['left'].set_visible(False)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9099AA')
        # rotate x labels 45 degrees and draw tick lines
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#9099AA')
        ax.set_ylim(0, 1)

    for i, arr in enumerate(arrhythmias):
        ax = axs[i+4]
        subset = df[(df['Arrhythmia'] == arr) & (df["Metric"] == "SE")]
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        boxplot(ax, subset)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([], fontsize=5)
        if i > 0:
            ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
        else:
            ax.tick_params(axis='x', width=0.5, rotation=0, labelsize=5, color='#9099AA')
        if i == 0:
            ax.set_ylabel('Sensitivity', fontsize=6)
        #else:
            #ax.spines['left'].set_visible(False)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9099AA')
        # rotate x labels 45 degrees and draw tick lines
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#9099AA')
        ax.set_ylim(0, 1)

    for i, arr in enumerate(arrhythmias):
        ax = axs[i+8]
        subset = df[(df['Arrhythmia'] == arr) & (df["Metric"] == "SP")]
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        boxplot(ax, subset)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i > 0:
            xlabels = ["ECGF", "Res.", "ALADIN"]
            ax.set_xticklabels(xlabels, fontsize=5)
            ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
        else:
            ax.tick_params(axis='x', width=0.5, rotation=45, labelsize=5, color='#9099AA')
        if i == 0:
            ax.set_ylabel('Specificity', fontsize=6)
        #else:
            #ax.spines['left'].set_visible(False)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9099AA')
        # rotate x labels 45 degrees and draw tick lines
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#9099AA')
        ax.set_ylim(0, 1)
    
    
    plt.subplots_adjust(wspace=0.15, hspace=0.4, top=0.93, bottom=0.2, left=0.12, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig4-boxplot-cinc.svg")
    plt.savefig("./paper/images/fig4-boxplot-cinc.png", dpi=300)
 
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
    plt.savefig("./paper/images/fig4-confusion_matrix_aladin.svg")
    plt.savefig("./paper/images/fig4-confusion_matrix_aladin.png", dpi=300)

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
    plt.savefig("./paper/images/fig4-confusion_matrix_ecgfounder.svg")
    plt.savefig("./paper/images/fig4-confusion_matrix_ecgfounder.png", dpi=300)

def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None

if __name__ == "__main__":
    
    # Load the data
    cinc = Data("CINC", "")
    aladinvirt = ALADINVirtual()
    aladin_experiment = DiagnosticBenchmark(cinc, aladinvirt)

    basefolder = os.environ.get('benchmark_results')
    aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_CINC")
    hannun_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_Hannun_CINC")
    ecgfounder_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ECGFounder_CINC")

    aladin_metrics, aladin_distributions = aladin_experiment.aggregate(aladin_file, bootstrap=True)
    hannun_metrics, hannun_distributions = aladin_experiment.aggregate(hannun_file, bootstrap=True)
    ecgfounder_metrics, ecgfounder_distributions = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True)

    data = {
        "ECGFounder": ecgfounder_distributions,
        "ResNet": hannun_distributions,
        "ALADIN": aladin_distributions
    }

    arrhythmias = list(data["ECGFounder"].keys())
    models = list(data.keys())

    # Simulate bootstrapped F1 score distributions: 1000 scores per model per arrhythmia.
    rowdata = []
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            f1_scores = data[model][arr]['f1']
            se_scores = data[model][arr]['se']
            sp_scores = data[model][arr]['sp']

            for i, score in enumerate(f1_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "F1"})
            for i, score in enumerate(se_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "SE"})
            for i, score in enumerate(sp_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "SP"})

    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)

    # make_boxplots(df)
    make_confusion_matrix_aladin(aladin_file)
    # make_confusion_matrix_ecgfounder(ecgfounder_file)
    make_ranking_plot(df)
