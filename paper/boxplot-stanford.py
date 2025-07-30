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

class HumanCardiologist():
    def __init__(self, id=0):
        self.name = f"Human Cardiologist {id}"
        self.id = id
        self.save_output = False

    def predict(self, sig, fs, meta=None, preprocess=False):

        case = meta["record"]
        predicted_episodes = []
        basefolder = os.environ.get('benchmark_data')
        for i in range(6):
            predpath = f"{basefolder}/STANFORD/{case}_rev{i}*.episodes.json"
            annotations = json.load(open(glob.glob(predpath)[0]))
            rev_id = annotations["reviewer_id"]
            if rev_id == self.id+1:
                predicted_episodes = annotations["episodes"]
                break

        #change key name in each item
        if len(predicted_episodes) > 0:
            predicted_episodes = [{k.replace("rhythm_name", "type"): v for k, v in episode.items()} for episode in predicted_episodes]
            return predicted_episodes
        else:
            return None
    
def get_average_cardiologist_metrics(triage=False):

    average_cardiologist = {}

    human1 = HumanCardiologist(0)
    human2 = HumanCardiologist(1)
    human3 = HumanCardiologist(2)
    human4 = HumanCardiologist(3)
    human5 = HumanCardiologist(4)
    human6 = HumanCardiologist(5)
    human7 = HumanCardiologist(6)
    human8 = HumanCardiologist(7)
    human9 = HumanCardiologist(8)

    stanford = Data("STANFORD", "")
    experiment = DiagnosticBenchmark(stanford, [human1, human2, human3, human4, human5, human6, human7, human8, human9])
    experiment.run()
    metrics, distributions = experiment.aggregate(bootstrap=True, triage=triage)

    return distributions

def get_cardiologist_metrics(i):

    cardiologist = {}

    human = HumanCardiologist(i)
    stanford = Data("STANFORD", "")
    experiment = DiagnosticBenchmark(stanford, human)
    experiment.run()
    metrics, distributions = experiment.aggregate(bootstrap=True)

    return distributions

def lighter(col_str):
    #get rgb from hex string
    rgb = tuple(int(col_str.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    #add 50 to each value
    rgb = tuple([x + 100 if x + 100 < 255 else 255 for x in rgb])
    #convert back to hex string

    return '#%02x%02x%02x' % rgb

def boxplot(ax, data):
    #get 7 colors of the pastel palette
    models = data['Model'].unique()

    colors = {
        'ECGFounder': "#2968A4",
        'Hannun': "#5C8FC6",
        'ALADIN': "#B4413D",
        "Avg. Card.": "#924D85",
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

    for i, model in enumerate(models):
        sns.boxplot(x='Model', y='Score', data=data[data['Model'] == model], ax=ax, color=lighter(colors[model]), width=0.5, linecolor=colors[model], fliersize=0.5, linewidth=0.5)

def make_piechart():

    data = {
        "AFIB/FL": 59,
        "AVB": 48,
        "BIG": 22,
        "EAR": 22,
        "IVR": 34,
        "JUNC": 36,
        "NOISE": 40,
        "NSR": 213,
        "SVT": 34,
        "TRI": 20,
        "VT": 17,
        "WENCK": 29
    }

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 0.8), dpi=300)
    #use colormap blues
    cmap = plt.get_cmap("Blues")
    colors = cmap(np.linspace(0, 0.75, 12))

    ax.bar(data.keys(), data.values(), color=lighter("#9099AA"), width=0.5, linewidth=0.5, edgecolor="#9099AA", zorder=3)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#bdc3c7')
    ax.set_ylabel("Count", fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5, rotation=0, color='#95a5a6')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#95a5a6')
    ax.set_yticks([0,50,100,150,200,250])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#9099AA')


    plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.9, bottom=0.2, left=0.12, right=0.99)
    

    #ax.pie(data.values(), labels=data.keys(), autopct='%1.0f%%', colors=colors, startangle=90, textprops={'fontsize': 5})

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/piechart_stanford.svg")

def make_boxplots(df):

    # Create a figure with 2 rows and 6 columns of subplots
    fig, axs = plt.subplots(2, 6, figsize=(7.08, 3), sharey=True, dpi=300)
    axs = axs.flatten()

    # Plot a boxplot for each arrhythmia in its own subplot
    for i, arr in enumerate(arrhythmias):
        ax = axs[i]
        subset = df[df['Arrhythmia'] == arr]
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        boxplot(ax, subset)
        ax.set_title(arr, fontsize=6)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i > 0:
            xlabels = ["F", "H", "A", "C", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            ax.set_xticklabels(xlabels, fontsize=5)
            ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
        else:
            ax.tick_params(axis='x', width=0.5, rotation=90, labelsize=5, color='#9099AA')
        if i % 6 == 0:
            ax.set_ylabel('F1 Score', fontsize=6)
        #else:
            #ax.spines['left'].set_visible(False)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9099AA')
        # rotate x labels 45 degrees and draw tick lines
        ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#bdc3c7')

    
    plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.95, bottom=0.05, left=0.05, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig3-boxplot-stanford.svg")
    plt.savefig("./paper/images/fig3-boxplot-stanford.png", dpi=300)

def make_boxplots_triage(df):

    # Create a figure with 2 rows and 6 columns of subplots
    fig, axs = plt.subplots(2, 4, figsize=(3.4, 2), sharey=True, dpi=300)
    axs = axs.flatten()

    arrhythmias = ["NORMAL", "NOTACUTE", "SUBACUTE", "ACUTE", "NORMAL", "NOTACUTE", "SUBACUTE", "ACUTE"]
    metrics = ["SE"] * 4 + ["SP"] * 4

    # Plot a boxplot for each arrhythmia in its own subplot
    for i, arr in enumerate(arrhythmias):
        ax = axs[i]
        subset = df[(df['Arrhythmia'] == arr ) & (df['Metric'] == metrics[i])]
        #replace nans scores with 0
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        boxplot(ax, subset)
        #if i < 4:
            #ax.set_title(arr, fontsize=6)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        xlabels = ["F", "R", "A", "C", "1", "2", "3", "4", "5", "6"]
        ax.set_xticklabels(xlabels, fontsize=5)
        ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
        if i % 4 == 0:
            if metrics[i] == "SE":
                ax.set_ylabel('Sensitivity', fontsize=5)
            elif metrics[i] == "SP":
                ax.set_ylabel('Specificity', fontsize=5)
        #else:
            #ax.spines['left'].set_visible(False)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#9099AA')
        # rotate x labels 45 degrees and draw tick lines
        ax.tick_params(axis='y', width=0.5, labelsize=5, color='#9099AA')
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='#9099AA')
        ax.set_ylim(0, 1)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.5, top=0.95, bottom=0.1, left=0.12, right=0.99)
    #plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.95, bottom=0.05, left=0.05, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig3-boxplot-stanford-triage.svg")
    plt.savefig("./paper/images/fig3-boxplot-stanford-triage.png", dpi=300)

def make_barcharts(df):

    fig, ax = plt.subplots(1, 1, figsize=(3.54, 2.2), dpi=300)

    
    colors = {
        'ECGFounder': "#2968A4",
        'Hannun': "#5C8FC6",
        'ALADIN': "#B4413D",
        "Avg. Card.": "#924D85",
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
    numbers = {
        'ECGFounder': "ECGF.",
        'Hannun': "Han.",
        'ALADIN': "ALADIN",
        "Avg. Card.": "AVG",
        'Card. 1': "1",
        'Card. 2': "2",
        'Card. 3': "3",
        'Card. 4': "4",
        'Card. 5': "5",
        'Card. 6': "6",
        'Card. 7': "7",
        'Card. 8': "8",
        'Card. 9': "9"
    }

    arrhythmias = df['Arrhythmia'].unique()
    #reverse the order of arrhythmias
    arrhythmias = arrhythmias[::-1]

    modelscores = {}

    for i, arr in enumerate(arrhythmias):

        dat = df[df['Arrhythmia'] == arr][['Model', 'Score']]

        dat = dat.groupby('Model').mean().reset_index()
        dat = dat.sort_values('Score', ascending=False)

        for j, model in enumerate(dat['Model']):
            ax.barh(i+1.5, 1/len(numbers.keys()), color=lighter(colors[model]), height=0.8, label=model, edgecolor=colors[model], linewidth=0.5, left=j/len(numbers.keys()), zorder=2)
            if model not in ['ECGFounder', 'Hannun', 'ALADIN', 'Avg. Card.']:
                ax.text(j/len(numbers.keys()) + 1/(len(numbers.keys())*2), i+1.5, f"{numbers[dat['Model'].iloc[j]]}", ha='center', va='center', fontsize=5, color='black')
            if model not in modelscores:
                modelscores[model] = j
            else:
                modelscores[model] += j

    #sort modelscores
    modelscores = {k: v for k, v in sorted(modelscores.items(), key=lambda item: item[1])}

    for j, model in enumerate(modelscores.keys()):
        ax.barh(0, 1/len(numbers.keys()), color=lighter(colors[model]), height=0.8, label=model, edgecolor="black", linewidth=0.5, left=j/len(numbers.keys()), zorder=2)
        if model not in ['ECGFounder', 'Hannun', 'ALADIN', 'Avg. Card.']:
            ax.text(j/len(numbers.keys()) + 1/(len(numbers.keys())*2), 0, f"{numbers[model]}", ha='center', va='center', fontsize=5, color='black')

    arrhythmias = np.concatenate([['Average'],arrhythmias])
    ax.set_yticks([0] + list(np.arange(1.5, len(arrhythmias)+0.5, 1)))
    ax.set_yticklabels(arrhythmias, fontsize=5)
    ax.set_xticks(np.arange(1/(len(numbers.keys())*2), 1 + 1/(len(numbers.keys())*2), 1/len(numbers.keys())))
    ax.set_xticklabels(["1st.","2nd.","3rd.","4th.","5th.","6th.","7th.","8th.","9th.", "10th.", "11th.", "12th.", "13th"], fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', width=0.5, labelsize=5, color='#9099AA')
    ax.tick_params(axis='y', width=0, labelsize=5, color='#9099AA')

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('#9099AA')

    plt.subplots_adjust(wspace=0, hspace=0, top=0.95, bottom=0.1, left=0.2, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig3-ranking-stanford.svg")
    plt.savefig("paper/images/fig3-ranking-stanford.png", dpi=300)

def get_most_recent_file(folder, prefix):
    files = glob.glob(os.path.join(folder, f"{prefix}*.json"))
    files.sort(key=os.path.getmtime)
    return files[-1] if files else None

if __name__ == "__main__":

    stanford = Data("STANFORD", "")

    aladinvirt = ALADINVirtual()
    aladin_experiment = DiagnosticBenchmark(stanford, aladinvirt)
    
    basefolder = os.environ.get('benchmark_results')
    aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_STANFORD") 
    hannun_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_Hannun_STANFORD") 
    ecgfounder_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ECGFounder_STANFORD") 

    aladin_metrics, aladin_distributions = aladin_experiment.aggregate(aladin_file, bootstrap=True)
    aladin_metrics_triage, aladin_distributions_triage = aladin_experiment.aggregate(aladin_file, bootstrap=True, triage=True)

    hannun_metrics, hannun_distributions = aladin_experiment.aggregate(hannun_file, bootstrap=True)
    hannun_metrics_triage, hannun_distributions_triage = aladin_experiment.aggregate(hannun_file, bootstrap=True, triage=True)
    ecgfounder_metrics, ecgfounder_distributions = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True)
    ecgfounder_metrics_triage, ecgfounder_distributions_triage = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True, triage=True)

    data = {
        "ECGFounder": ecgfounder_distributions,
        "Hannun": hannun_distributions,
        "ALADIN": aladin_distributions
    }
    data_triage = {
        "ECGFounder": ecgfounder_distributions_triage,
        "Hannun": hannun_distributions_triage,
        "ALADIN": aladin_distributions_triage
    }

    #data["Avg. Card."] = get_average_cardiologist_metrics(triage=False)
    #data_triage["Avg. Card."] = get_average_cardiologist_metrics(triage=True)
    for i in range(9):
        data[f"Card. {i+1}"] = get_cardiologist_metrics(i)

    if not os.path.exists("./paper/images"):
        os.makedirs("./paper/images")

    #diagnosis application
    arrhythmias = list(data["ECGFounder"].keys())
    models = list(data.keys())

    # Simulate bootstrapped F1 score distributions: 1000 scores per model per arrhythmia.
    rowdata = []
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            f1_scores = data[model][arr]['f1']
            for i, score in enumerate(f1_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "F1"})
                

    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)
    make_boxplots(df)
    make_barcharts(df)


    #triage application
    arrhythmias = list(data_triage["ECGFounder"].keys())
    models = list(data_triage.keys())

    # Simulate bootstrapped F1 score distributions: 1000 scores per model per arrhythmia.
    rowdata = []
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            f1_scores = data_triage[model][arr]['f1']
            for i, score in enumerate(f1_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "F1"})
            se_scores = data_triage[model][arr]['se']
            for i, score in enumerate(se_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "SE"})
            sp_scores = data_triage[model][arr]['sp']
            for i, score in enumerate(sp_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": "SP"})


    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)
    make_boxplots_triage(df)
