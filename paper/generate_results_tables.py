import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
import argparse



def blend_with_white(rgba):
    """
    Blend the RGBA color with a white background to simulate transparency.
    """
    r, g, b, a = rgba
    # White background has RGB (1, 1, 1)
    r_blended = a * r + (1 - a) * 1
    g_blended = a * g + (1 - a) * 1
    b_blended = a * b + (1 - a) * 1
    return (r_blended, g_blended, b_blended)

def value_to_color(val, map='RdYlGn', alpha=0.5):
    # Normalize value between 0 and 1 for red to green transition with alpha = 0.5
    cmap = plt.get_cmap(map)  # Red to green colormap
    rgba = cmap(val)
    rgba_with_alpha = (rgba[0], rgba[1], rgba[2], alpha)  # Set alpha to 0.5
    return blend_with_white(rgba_with_alpha)



def rgb_to_latex_color(rgb):
    # Convert RGB values to LaTeX color format (assumes RGB are in range [0, 1])
    return f'{rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f}'

def condition_formatter(condition):
    condition = condition.replace("_", " ")
    return condition

def metric_formatter(metric):
    metric = metric.replace("BDice", "BDSC")
    metric = metric.replace("Dice", "DSC")
    metric = metric.replace("Pixel SE", "Se")
    metric = metric.replace("Pixel SP", "Sp")
    metric = metric.replace("Error Mean", "M")
    metric = metric.replace("Error SD", "SD")
    metric = metric.replace("F1", "$F_1 (\\%)$")
    metric = metric.replace("SE", "$SE (\\%)$")
    metric = metric.replace("PP", "$PPV (\\%)$")
    metric = metric.replace("Error", "$\mu\pm\sigma (ms)$")
    return metric

def normalize_maxpercent(val):
    return max(0,((val/100)*1.5)-0.5)

def normalize_minms(val):
    return 1-min(25,abs(val))/25

def generate_macro_latex_table(experiment, models, metrics, waves=["P", "QRS", "T"], fiducials=["onset", "offset"], use_abbr=False):

    algorithms = [model["algorithm"] for model in models]
    # Step 1: Load the data
    data_per_algorithm = []
    for i, algorithm in enumerate(algorithms):
        data = pd.read_csv(f'./results/delineation/macro_{algorithm}_{experiment}.csv')
        data_per_algorithm.append(data)


    latex = "\\begin{threeparttable} \n"
    latex += "\\begin{tabular}{l|l" + ("|"+(("r"*len(fiducials)+"|")*len(waves))[:-1]) + "} \n"

    #Header with model names
    latex += "\\toprule Method & Metric & " + " & ".join(["$"+wave+"_{"+fiducial[:-3]+"}$" for wave in waves for fiducial in fiducials ]) + "\\\\ \\midrule\n"

    #Find the minimum and maximum values for each metric
    mins = {}
    maxs = {}

    for metric in metrics:
        for wave in waves:
            for fiducial in fiducials:
                if metric != "Error":
                    subset = pd.concat([data_per_algorithm[i][(data_per_algorithm[i]["Condition"] == wave.lower()+"_"+fiducial)][metric] for i in range(len(models))], axis=0)
                    print(subset, subset.abs().min())
                    mins[f"{wave}_{fiducial}_{metric}"] = subset.abs().min()
                    maxs[f"{wave}_{fiducial}_{metric}"] = subset.abs().max()

    print(mins)
    
    for i, algorithm in enumerate(algorithms):
        latex += models[i]["header"] + " & "
        for metric in metrics:
            latex += metric_formatter(metric) + " & "
            for wave in waves:
                for fiducial in fiducials:
                    if metric == "Error":
                        value = str(data_per_algorithm[i][(data_per_algorithm[i]["Condition"] == wave.lower()+"_"+fiducial)]["Error Mean"].values[0]) + "$\pm$" + str(data_per_algorithm[i][(data_per_algorithm[i]["Condition"] == wave.lower()+"_"+fiducial)]["Error SD"].values[0])
                        latex += str(value) + " & "
                    else:
                        value = data_per_algorithm[i][(data_per_algorithm[i]["Condition"] == wave.lower()+"_"+fiducial)][metric].values[0]
                        alpha = 0.6
                        if metric in ["SE", "PP", "F1"]:
                            normval = normalize_maxpercent(value)
                        else:
                            normval = normalize_minms(value)

                        rgb_color = value_to_color(normval, 'RdYlGn', alpha)
                        latex_color = rgb_to_latex_color(rgb_color[:3])  # Convert to LaTeX format

                        if value == maxs[f"{wave}_{fiducial}_{metric}"] and value > 0:
                            latex += "\\cellcolor[rgb]{" + latex_color +"} \\textbf{" + str(value) + "} & "
                        else:
                            latex += "\\cellcolor[rgb]{" + latex_color +"} " + str(value) + " & "
            latex = latex[:-2] + " \\\\ \n & "
        latex = latex[:-2] + " \\midrule \n"

    latex = latex[:-10] + " \\bottomrule \n \\end{tabular} \n \\end{threeparttable}"

    print(latex)


def generate_latex_table(experiment, models, metrics, waves=["P", "QRS", "T"], use_abbr=False, best="maxpercent"):

    basefolder = os.environ.get('benchmark_results')
    algorithms = [model["algorithm"] for model in models]
    # Step 1: Load the data
    data_per_algorithm = []
    for i, algorithm in enumerate(algorithms):
        data = pd.read_csv(f'{basefolder}/delineation/micros_{algorithm}_{experiment}.csv')
        data_per_algorithm.append(data)

    arrhythmia_table = json.load(open(f'{basefolder}/delineation/arrhythmia_table_{experiment}.json'))  

    conditions = data_per_algorithm[0]["Condition"].unique()
    conditions.sort()
    obj = {}

    nrecords = {
        "VAL": 797,
        "STANFORD": 324,
        "LUDB": 200,
        "RDB":2400
    }


    for condition in conditions:
        if condition == "LongP":
            continue
        conditionform = condition_formatter(condition)
        if "Condition" not in obj: obj["Condition"] = []
        if "Prevalence" not in obj: obj["Prevalence"] = []
        if "PrevalenceN" not in obj: obj["PrevalenceN"] = []
        if "Count" not in obj: obj["Count"] = []
        if "Support" not in obj: obj["Support"] = []

        obj["Condition"].append(arrhythmia_table[conditionform]["abbr"] if use_abbr else arrhythmia_table[conditionform]["name"])
        obj["Prevalence"].append(np.round((arrhythmia_table[conditionform]["count"]/nrecords[experiment])*1000)/10)
        obj["PrevalenceN"].append(arrhythmia_table[conditionform]["count"])
        obj["Count"].append(arrhythmia_table[conditionform]["count"])
        obj["Support"].append(data_per_algorithm[0][(data_per_algorithm[0]["Condition"] == condition) & (data_per_algorithm[0]["Fiducial"] == "p_center")]["Support"].values[0])

        for i, algorithm in enumerate(algorithms):
            for wave in waves:
                for metric in metrics:
                    colname = f"{wave}_{metric}_{algorithm}"
                    if colname not in obj: obj[colname] = []
                    dat = data_per_algorithm[i]
                    obj[colname].append(dat[(dat["Condition"] == condition) & (dat["Fiducial"] == f"{wave.lower()}_center")][metric].values[0])

    df = pd.DataFrame(obj)

    #Table begin and column definition
    latex = "\\begin{threeparttable} \n"
    latex += "\\begin{tabular}{l|r|r" + ("||"+(("r"*len(metrics)+"|")*len(waves))[:-1])*len(algorithms) + "} \n"

    #Header with model names
    latex += "\\toprule & & &" + " & ".join(["\multicolumn{"+str(len(waves)*len(metrics))+"}{c||}{"+model["header"]+"}" for model in models]) + "\\\\ \\midrule\n"

    #Find the minimum and maximum values for each metric
    sinusrhythm = "SR" if use_abbr else "Sinus Rhythm"
    supportSR = df[df["Condition"] == sinusrhythm]["Support"].values[0]
    df = df[df["Condition"] != "NOISE"]
    counts = df["Count"]
    df = df.drop(columns=["Count"])

    mins = {"Prevalence": df["Prevalence"]}
    maxs = {"Prevalence": df["Prevalence"]}

    for metric in metrics:
        for wave in waves:
            subset = df[[f"{wave}_{metric}_{algorithm['algorithm']}" for algorithm in models]]
            mins[f"{wave}_{metric}"] = subset.abs().min(axis=1)
            maxs[f"{wave}_{metric}"] = subset.abs().max(axis=1)

    latex += "\\textit{Wave} & Pr.(\\%) & Pr.(n) & " + ((" & ".join(["\multicolumn{"+str(len(metrics))+"}{c|}{"+wave+"}" for wave in waves]) + " & ")*len(algorithms))[:-2] + "\\vspace{1mm}\\\\ \n"
    latex += "\\textit{Metric} & & &" + ((" & ".join([metric_formatter(metric) for metric in metrics]) + " & ")*len(waves)*len(algorithms))[:-2] + "\\\\ \\midrule\n"
    latex += "\\textit{Condition} & & &" + (" & "*len(metrics)*len(waves)*len(algorithms))[:-2] + "\\\\ \n"

    for i in range(len(df)):
        row = df.iloc[i]

        latex += row["Condition"] + " & "

        prevalence = (row["Prevalence"] - mins["Prevalence"].min()) / (maxs["Prevalence"].max() - mins["Prevalence"].min())
        rgb_color = value_to_color(1-prevalence,'coolwarm')
        latex_color = rgb_to_latex_color(rgb_color[:3])
        latex += "\\cellcolor[rgb]{" + latex_color +"} " + str(row["Prevalence"]) + " & \\cellcolor[rgb]{" + latex_color +"} " + str(row["PrevalenceN"]) + " & "

        for j, algorithm in enumerate(algorithms):
            for wave in waves:
                for metric in metrics:
                    alpha = 0.6
                    value = row[f"{wave}_{metric}_{algorithm}"]
                    support = row["Support"]
                    if wave == "P":
                        alpha = max(0.1,min(0.75,support/supportSR))
                    
                    if best == "maxpercent":
                        value = normalize_maxpercent(value)
                    elif best == "minms":
                        value = normalize_minms(value)

                    rgb_color = value_to_color(value, 'RdYlGn', alpha)
                    latex_color = rgb_to_latex_color(rgb_color[:3])  # Convert to LaTeX format
                    if np.isnan(row[f"{wave}_{metric}_{algorithm}"]):
                        latex += "\\cellcolor[rgb]{1,1,1} & "
                    # elif wave == "P" and metric != "Pixel SP" and support < 0.05:
                    #     if row[f"{wave}_{metric}_{algorithm}"] == maxs[f"{wave}_{metric}"].values[i] and row[f"{wave}_{metric}_{algorithm}"] > 0:
                    #         latex += "\\cellcolor[rgb]{1,1,1} \color{gray} \\textbf{" + str(row[f"{wave}_{metric}_{algorithm}"]) + "} & "
                    #     else:
                    #         latex += "\\cellcolor[rgb]{1,1,1} \color{gray}" + str(row[f"{wave}_{metric}_{algorithm}"]) + " & "
                    # elif wave == "P" and metric == "Pixel SP" and support > 0.05:
                    #     if row[f"{wave}_{metric}_{algorithm}"] == maxs[f"{wave}_{metric}"].values[i] and row[f"{wave}_{metric}_{algorithm}"] > 0:
                    #         latex += "\\cellcolor[rgb]{1,1,1} \color{gray} \\textbf{" + str(row[f"{wave}_{metric}_{algorithm}"]) + "} & "
                    #     else:
                    #         latex += "\\cellcolor[rgb]{1,1,1} \color{gray}" + str(row[f"{wave}_{metric}_{algorithm}"]) + " & "
                    else:
                        bestval = maxs[f"{wave}_{metric}"].values[i] if best == "maxpercent" else mins[f"{wave}_{metric}"].values[i]
                        if abs(row[f"{wave}_{metric}_{algorithm}"]) == bestval and row[f"{wave}_{metric}_{algorithm}"] != 0:
                            latex += "\\cellcolor[rgb]{" + latex_color +"} \\textbf{" + str(row[f"{wave}_{metric}_{algorithm}"]) + "} & "
                        else:
                            latex += "\\cellcolor[rgb]{" + latex_color +"} " + str(row[f"{wave}_{metric}_{algorithm}"]) + " & "
        
        latex = latex[:-2] + " \\\\ \n"
    
    latex = latex[:-2] + " \\midrule \n"

    # df.loc[df["Support"] < 0.05, [f"P_{metric}_{algorithm}" for metric in metrics if metric != "Pixel SP" for algorithm in algorithms]] = pd.NA
    # df.loc[df["Support"] > 0.05, [f"P_{metric}_{algorithm}" for metric in metrics if metric == "Pixel SP" for algorithm in algorithms]] = pd.NA

    latex += "Average & & &"

    for i, algorithm in enumerate(algorithms):
        for wave in waves:
            for metric in metrics:
                avg = df[f"{wave}_{metric}_{algorithm}"].mean()

                if best == "maxpercent":
                    value = normalize_maxpercent(avg)
                elif best == "minms":
                    value = normalize_minms(avg)

                rgb_color = value_to_color(value)  # Get color
                latex_color = rgb_to_latex_color(rgb_color[:3])  # Convert to LaTeX format
                latex += "\\cellcolor[rgb]{" + latex_color +"} " + str(np.round(avg*10)/10) + " & "

    latex = latex[:-2] + "\\\\ \\bottomrule \n"
    latex += "\\end{tabular}"
    if use_abbr:
        latex += "\\begin{tablenotes} \n"
        latex += "\\item Pr: Prevalence. " + " \\item ".join([arrhythmia_table[condition_formatter(condition)]["abbr"]+": "+arrhythmia_table[condition_formatter(condition)]["name"] for condition in conditions]) + " \n"
        latex += "\\end{tablenotes} \n"
    latex += "\\end{threeparttable}"
        
    print(latex)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run benchmark')
    parser.add_argument('--dataset', type=str, help='Dataset used to benchmark (VAL, RDB)', required=True)
    args = parser.parse_args()

    experiment = args.dataset
    if experiment not in ["VAL", "RDB"]:
        print("No experiment selected, choose VAL or RDB")

    algorithms = [
        {
            "algorithm":"martinez",
            "header":"Martinez 2004 \cite{martinez2004wavelet}",
        },
        {
            "algorithm":"DelineatorSwitchAndCompose", 
            "header":"Jimenez 2024 \cite{jimenez2024delineation}"
        }, 
        {
            "algorithm":"ALADIN",
            "header":"ALADIN"
        }]
    
    #Generate supplementary tables 1-6 of the manuscript
    #metrics = ["Pixel SE", "Pixel SP"]
    #metrics = ["Dice", "IoU"]
    metrics = ["BDice"]
    generate_latex_table(experiment, algorithms, metrics, use_abbr=True)

    #Generate the macro tables
    #metrics = ["SE","PP","F1","Error"]
    #generate_macro_latex_table(experiment, algorithms, metrics, use_abbr=True)
    