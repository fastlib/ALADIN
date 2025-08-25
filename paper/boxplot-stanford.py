import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob 
import json
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, Literal, Optional

from aladin.utils.benchmark_utils import Data, Model, DiagnosticBenchmark, StanfordData, CINCData, ICENTIAData

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
            return predicted_episodes, {}
        else:
            return None, {}
    
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

    data = StanfordData("STANFORD")
    experiment = DiagnosticBenchmark(data, [human1, human2, human3, human4, human5, human6, human7, human8, human9])
    experiment.run()
    metrics, distributions = experiment.aggregate(bootstrap=True, triage=triage)

    return distributions

def get_cardiologist_metrics(i):

    cardiologist = {}

    human = HumanCardiologist(i)
    data = StanfordData("STANFORD")
    experiment = DiagnosticBenchmark(data, human)
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


def barplot(ax, data):
    #get 7 colors of the pastel palette
    models = data['Model'].unique()
    models = [m for m in models if m[:4] != "Card"]  # Exclude these models from the bar plot
    withoutaverage = [m for m in models if m != "Avg. Card."]

    colors = {
        'ECGFounder': "#E2E2EA", 
        'ResNet': "#C2C7D4", 
        'ALADIN': "#D63F3D", 
        "Avg. Card.": "#E0DBC9",
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
        "Avg. Card.": "#E0DBC9"
    }

    for i, model in enumerate(models):
        print(f"Model: {model}", data[data['Model'] == model]['Score'].mean())

        ax.bar(i, data[data['Model'] == model]['Score'].mean(), 
            color=colors[model],
            edgecolor=edgecolors[model] if model in edgecolors else colors[model],
            linewidth=0.5,
            width=0.8,  # Width of the bars
            zorder=1
        )

        ax.set_xlim(-1.5, len(models) + 0.5)
        ci_upper = np.percentile(data[data['Model'] == model]['Score'], 97.5)
        ci_lower = np.percentile(data[data['Model'] == model]['Score'], 2.5)
        mean_score = data[data['Model'] == model]['Score'].mean()
        print(f"Model: {model}, Mean: {mean_score:.2f}, CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
        print(data[data['Model'] == model])

        if not np.all(data[data['Model'] == model]['Score'] == 0):
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

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 1.6), dpi=300)
    #use colormap blues

    ax.bar(data.keys(), data.values(), color="#E0DBC9", width=0.5, linewidth=0.5, edgecolor="#BFBA9F", zorder=3)
    ax.set_ylabel("Count", fontsize=5)
    ax.tick_params(axis='x', width=0.5, labelsize=5, rotation=0, color='#95a5a6')
    ax.tick_params(axis='y', width=0.5, labelsize=5, color='#95a5a6')
    ax.set_yticks([0,50,100,150,200,250])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
        spine.set_color('#1C1C1C')


    plt.subplots_adjust(wspace=0.15, hspace=0.5, top=0.9, bottom=0.2, left=0.12, right=0.99)
    

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("paper/images/fig2-class_distribution_stanford.svg")

def make_boxplots(df):

    # Create a figure with 2 rows and 6 columns of subplots
    fig, axs = plt.subplots(2, 6, figsize=(7.08, 3), dpi=300)
    axs = axs.flatten()


    arrhythmia_formatted = {
        "AFIB/AFL": "AFIB/AFL",
        "AVB": "AV Block",
        "AVB_TYPE2": "Second-degree AVB",
        "BIGEMINY": "Bigeminy",
        "TRIGEMINY": "Trigeminy",
        "IVR": "IVR",
        "NOISE": "Noise",
        "NSR": "Normal Sinus Rhythm",
        "SUDDEN_BRADY": "Third-degree AVB",
        "SVT": "SVT",
        "VT": "VT",
        "WENCKEBACH": "Wenckebach",
        "EAR": "Ectopic Atrial Rhythm",
        "JUNCTIONAL": "Junctional Rhythm"
    }

    # Plot a boxplot for each arrhythmia in its own subplot
    for i, arr in enumerate(arrhythmias):
        ax = axs[i]
        subset = df[(df['Arrhythmia'] == arr) & (df["Metric"] == "f1")]
        # The 'width' parameter is reduced to 0.6 to leave small gaps between the boxes.
        barplot(ax, subset)
        ax.set_xlabel(arrhythmia_formatted[arr], fontsize=6)
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', width=0.25, labelsize=3, color='#1C1C1C')
        if i % 6 == 0:
            ax.set_ylabel('F1 Score', fontsize=6)

        # make spines linewidth 0.5
        for spine in ax.spines.values():
            spine.set_linewidth(0.25)
            spine.set_color('#1C1C1C')
        # rotate x labels 45 degrees and draw tick lines
        ax.tick_params(axis='y', width=0.25, labelsize=5, color='#1C1C1C')
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=5)
        ax.set_ylim(0, 1)
    
    plt.subplots_adjust(wspace=0.2, hspace=0.4, top=0.95, bottom=0.1, left=0.05, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig2-boxplot-stanford.svg")
    plt.savefig("./paper/images/fig2-boxplot-stanford.png", dpi=300, facecolor="white", edgecolor="white", transparent=False)

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
        barplot(ax, subset)
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
    plt.savefig("./paper/images/fig2-boxplot-stanford-triage.svg")
    plt.savefig("./paper/images/fig2-boxplot-stanford-triage.png", dpi=300)

def make_barcharts(df):

    fig, ax = plt.subplots(1, 1, figsize=(3.54, 1.8), dpi=300)

    colors = {
        'ECGFounder': "#E2E2EA",
        'ResNet': "#C2C7D4",
        'ALADIN': "#D63F3D", 
        "Avg. Card.": "#BFBA9F", 
        'Card. 1': "#E0DBC9",
        'Card. 2': "#E0DBC9",
        'Card. 3': "#E0DBC9",
        'Card. 4': "#E0DBC9",
        'Card. 5': "#E0DBC9",
        'Card. 6': "#E0DBC9",
        'Card. 7': "#E0DBC9",
        'Card. 8': "#E0DBC9",
        'Card. 9': "#E0DBC9"
    }
    numbers = {
        'ECGFounder': "ECGF.",
        'ResNet': "Han.",
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
    modeln = {}

    for i, arr in enumerate(arrhythmias):

        dat = df[(df['Arrhythmia'] == arr) & (df['Metric'] == "f1")][['Model', 'Score']]

        dat = dat.groupby('Model').mean().reset_index()
        dat = dat.sort_values('Score', ascending=False)
        dat = dat[dat['Score'] > 0]

        for j, model in enumerate(dat['Model']):
            if model in ['ECGFounder', 'ResNet', 'ALADIN', 'Avg. Card.']:
                ax.barh(i+1.5, 0.9/len(numbers.keys()), color=colors[model], height=0.4, label=model, left=(j+0.1)/len(numbers.keys()), zorder=2)
            else:
                ax.barh(i+1.5, 0.9/len(numbers.keys()), color="white", edgecolor=colors[model], linewidth=0.5, height=0.6, label=model, left=(j+0.1)/len(numbers.keys()), zorder=2)
            
            if model not in modelscores:
                modelscores[model] = j
                modeln[model] = 1
            else:
                modelscores[model] += j
                modeln[model] += 1

    #sort modelscores
    modelscores = {k: v for k, v in sorted(modelscores.items(), key=lambda item: item[1])}
    modelscores = {k: v/modeln[k] for k, v in modelscores.items()}

    for j, model in enumerate(modelscores.keys()):
        if model in ['ECGFounder', 'ResNet', 'ALADIN', 'Avg. Card.']:
            ax.barh(0, 0.9/len(numbers.keys()), color=colors[model], height=0.4, label=model, left=(j+0.1)/len(numbers.keys()), zorder=2)
        else:
            ax.barh(0, 0.9/len(numbers.keys()), color="white", edgecolor=colors[model], linewidth=0.5, height=0.6, label=model, left=(j+0.1)/len(numbers.keys()), zorder=2)

    arrhythmias = np.concatenate([['Average'],arrhythmias])
    ax.set_yticks([0] + list(np.arange(1.5, len(arrhythmias)+0.5, 1)))
    ax.set_yticklabels(arrhythmias, fontsize=6)
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
    plt.savefig("paper/images/fig2-ranking_stanford.svg")
    plt.savefig("paper/images/fig2-ranking_stanford.png", dpi=300)

def horizontal_barplot(ax, data):
    #get 7 colors of the pastel palette
    models = data['Model'].unique()
    models = [m for m in models if m[:4] != "Card"]  # Exclude individual cardiologists

    colors = {
        'ECGFounder': "#E2E2EA", 
        'ResNet': "#C2C7D4", 
        'ALADIN': "#D63F3D", 
        "Avg. Card.": "#E0DBC9",
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
        "Avg. Card.": "#E0DBC9"
    }
    for i, model in enumerate(models):
        sns.barplot(
            x='Score', 
            y='Model', 
            data=data[data['Model'] == model], 
            ax=ax, 
            orient='h',
            color=colors[model], 
            edgecolor=edgecolors[model] if model in edgecolors else colors[model],
            linewidth=0.5,
            width=0.8,  
            errorbar=None,
            zorder=2
        )
        #ax.set_ylim(-1.5, len(models) + 0.5)
        ci_upper = np.percentile(data[data['Model'] == model]['Score'], 97.5)
        ci_lower = np.percentile(data[data['Model'] == model]['Score'], 2.5)
        mean_score = data[data['Model'] == model]['Score'].mean()

        # Draw error bars manually
        ax.errorbar(
            x=[mean_score],
            y=[i],
            xerr=[[mean_score - ci_lower], [ci_upper - mean_score]],
            fmt='none',
            color="black",
            elinewidth=0.5,
            capsize=1.5,
            capthick=0.5,
            zorder=3
        )

def make_horizontal_barplots(df):

    fig, axs = plt.subplots(3, 3, figsize=(3.54, 1.8), dpi=300)
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
            if i == 2:
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
            ax.set_xlim(0, 51)
            ax.grid(axis='x', linestyle='--', linewidth=0.25, color='#bdc3c7')

    
    plt.subplots_adjust(wspace=0.2, hspace=0.4, top=0.95, bottom=0.2, left=0.2, right=0.99)
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica Neue'
    plt.savefig("./paper/images/fig2-boxplot-stanford_triage.svg")
    plt.savefig("./paper/images/fig2-boxplot-stanford_triage.png", dpi=300, facecolor="white", edgecolor="white", transparent=False)

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

    models = [m for m in models if m[:4] != "Card"]  # Exclude these models from the bar plot

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

    metrics = df["Metric"].unique()
    models = df["Model"].unique()
    models = [m for m in models if m[:4] != "Card"]  # Exclude individual cardiologists
    arrhythmias = df["Arrhythmia"].unique()

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
        "AFIB/AFL": "AFIB/AFL",
        "AVB_TYPE2": "Second-degree AVB",
        "AVB": "AV Block",
        "BIGEMINY": "Bigeminy",
        "TRIGEMINY": "Trigeminy",
        "SUDDEN_BRADY": "CHB",
        "IVR": "IVR",
        "NOISE": "Noise",
        "NSR": "NSR",
        "JUNCTIONAL": "Junctional Rhythm",
        "SVT": "SVT",
        "VT": "VT",
        "EAR": "EAR",
        "WENCKEBACH": "Wenckebach",
        "NORMAL": "Normal",
        "NOTACUTE": "Not Acute",
        "SUBACUTE": "Sub-acute",
        "ACUTE": "Acute"
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
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)
                intervals.append((ci_lower, ci_upper))

            lowerbetter = True if metric in ["FP","FN"] else False
            best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

            for i, model in enumerate(models):
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)

                if best_interval == i:
                    latex += "\\textbf{" + format_ci((mean, ci_lower, ci_upper), percentage=percentage) + "} & "
                else:
                    latex += format_ci((mean, ci_lower, ci_upper), percentage=percentage) + " & "

        latex = latex[:-2] + " \\\\\n"

        latex +=  " & "
        for metric in metrics[:maxmetrics]:
            for model in models:
                intervals = []
            for model in models:
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                mean = xs.mean()
                ci_lower = np.percentile(xs, 2.5)
                ci_upper = np.percentile(xs, 97.5)
                intervals.append((ci_lower, ci_upper))

            lowerbetter = True if metric in ["FP","FN"] else False
            best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

            for i, model in enumerate(models):
                xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
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
                for model in models:
                    intervals = []
                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

                lowerbetter = True if metric in ["FP","FN"] else False
                best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

                for i, model in enumerate(models):
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
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
                for model in models:
                    intervals = []
                for model in models:
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
                    mean = xs.mean()
                    ci_lower = np.percentile(xs, 2.5)
                    ci_upper = np.percentile(xs, 97.5)
                    intervals.append((ci_lower, ci_upper))

                lowerbetter = True if metric in ["FP","FN"] else False
                best_interval = best_non_overlapping(intervals, lowerbetter=lowerbetter)

                for i, model in enumerate(models):
                    xs = df[(df['Model'] == model) & (df['Metric'] == metric) & (df['Arrhythmia'] == arrhythmia)]['Score']
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

    stanford = Data("STANFORD", "")

    aladinvirt = ALADINVirtual()
    aladin_experiment = DiagnosticBenchmark(stanford, aladinvirt)
    
    basefolder = os.environ.get('benchmark_results')
    aladin_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ALADIN_STANFORD") 
    resnet_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_Hannun_STANFORD") 
    ecgfounder_file = get_most_recent_file(basefolder+"/diagnosis", "set_level_diagnosis_ECGFounder_STANFORD") 

    aladin_metrics, aladin_distributions = aladin_experiment.aggregate(aladin_file, bootstrap=True)
    aladin_metrics_triage, aladin_distributions_triage = aladin_experiment.aggregate(aladin_file, bootstrap=True, triage=True)

    resnet_metrics, resnet_distributions = aladin_experiment.aggregate(resnet_file, bootstrap=True)
    resnet_metrics_triage, resnet_distributions_triage = aladin_experiment.aggregate(resnet_file, bootstrap=True, triage=True)
    ecgfounder_metrics, ecgfounder_distributions = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True)
    ecgfounder_metrics_triage, ecgfounder_distributions_triage = aladin_experiment.aggregate(ecgfounder_file, bootstrap=True, triage=True)

    data = {
        "ECGFounder": ecgfounder_distributions,
        "ResNet": resnet_distributions,
        "ALADIN": aladin_distributions
    }
    data_triage = {
        "ECGFounder": ecgfounder_distributions_triage,
        "ResNet": resnet_distributions_triage,
        "ALADIN": aladin_distributions_triage
    }

    data["Avg. Card."] = get_average_cardiologist_metrics(triage=False)
    data_triage["Avg. Card."] = get_average_cardiologist_metrics(triage=True)
    for i in range(9):
        data[f"Card. {i+1}"] = get_cardiologist_metrics(i)
        

    if not os.path.exists("./paper/images"):
        os.makedirs("./paper/images")

    #diagnosis application
    arrhythmias = list(data["ALADIN"].keys())
    models = list(data.keys())

    # Simulate bootstrapped F1 score distributions: 1000 scores per model per arrhythmia.
    rowdata = []
    metrics = ["acc", "se", "sp", "ppv", "npv", "f1"]
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            for metric in metrics:
                # if arr in data[model]:
                scores = data[model][arr][metric]
                for i, score in enumerate(scores):
                    rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': score, "Iteration": i, "Metric": metric})
                # else:
                #     rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': 0, "Iteration": 0, "Metric": metric})

    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)
    #print(generate_metrics_table(df))
    make_boxplots(df)
    make_barcharts(df)
    make_piechart()


    #triage application
    arrhythmias = ["NOTACUTE", "SUBACUTE", "ACUTE"]
    models = list(data_triage.keys())

    # Simulate bootstrapped F1 score distributions: 1000 scores per model per arrhythmia.
    rowdata = []
    for arr in arrhythmias:
        for model in models:
            # Simulate F1 scores using a beta distribution (values between 0 and 1).
            f1_scores = data_triage[model][arr]['tp']
            for i, score in enumerate(f1_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': (100*score)/328, "Iteration": i, "Metric": "TP"})
            se_scores = data_triage[model][arr]['fp']
            for i, score in enumerate(se_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': (100*score)/328, "Iteration": i, "Metric": "FP"})
            sp_scores = data_triage[model][arr]['fn']
            for i, score in enumerate(sp_scores):
                rowdata.append({'Arrhythmia': arr, 'Model': model, 'Score': (100*score)/328, "Iteration": i, "Metric": "FN"})


    # Convert data into a long-format DataFrame
    df = pd.DataFrame(rowdata)
    #print(generate_metrics_table(df, percentage=False))
    make_horizontal_barplots(df)
