import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas
from pathlib import Path
from statistics import mean
import seaborn  # Heatmap (Confusion-Matrix) plotten


def load_log(logfile: Path, average=True):
    """Lädt eine Log-Datei `logfile` als Pandas Dataframe.
    Falls `average`=True (Standard) werden Acc und Loss für Train und Test über alle Folds gemittelt"""
    # Spaltennamen der Log-Datei
    header_names = ["gpu", "dauer", "batchsize", "learningrate", "datensatzname", "imgsize", "klassifikation", "netz",
                    "gewichte", "trainpercent", "epochen", "fold", "extra_infos", "train_loss", "train_acc", "val_loss",
                    "val_acc"]
    # Lade Log mit entsprechenden Spaltennamen
    log = pandas.read_csv(logfile, header=None, names=header_names)
    # Falls gemittelt werden soll
    if average:
        # Spaltennamen nach denen gruppiert werden soll (GPU-Name geht dabei verloren)
        group_header_names = ["batchsize", "learningrate", "datensatzname", "imgsize", "klassifikation", "netz",
                              "gewichte",
                              "trainpercent", "epochen", "extra_infos"]
        # Gruppiere und bilde Mittelwert und Standardabweichung
        log = log.groupby(group_header_names, as_index=False).agg(["mean", "std"]).reset_index()
    return log


def build_filter_func(filter_func, log_col, values, typ=str):
    """Erstellt eine Filter-Funktion für Pandas Dataframes.
    `filter_func` ist eine vorher existierende Filter-Funktion (oder True) mit der weiter verknüpft werden soll,
    `log_col` ist die Spalte des Logs für den aktuell zu erstellenden Filter-Ausdruck,
    `values` ist der erlaubte Wert oder eine Liste mit erlaubten Werten für den Filter-Ausdruck,
    `typ` gibt an, ob `values` eine Liste oder ein einzelner Wert ist."""
    # Einzelner Wert für Filter-Ausdruck
    if type(values) is typ:
        # UND-Verknüpfung der alten Filter-Funktion mit dem neuen Ausdruck
        filter_func &= (log_col == values)
    # eine Liste von erlaubten Werten wurde übergeben
    elif type(values) is list:
        # Baue eine Sub-Filter-Funktion
        sub_filter_func = False
        # Fülle Sub-Filter-Funktion mit einem Ausdruck je Eintrag in der Liste (ODER-Verknüpfung!)
        for value in values:
            sub_filter_func |= (log_col == value)
        # Verknüpfe Sub-Filter-Funktion mit alter Filter-Funktion (UND)
        filter_func &= sub_filter_func
    else:
        exit(1)
    return filter_func


def filter_log(log: pandas.DataFrame, dataset_name, net_type, weights, extra_infos):
    """Filtert einen Log gemäß der übergebenen Parameter für `dataset_name`, `net_type`, `weights` und `extra_infos`"""
    # Filter-Funktion aufbauen (&-Verknüpfung von verschiedenen Bedingungen)
    filter_func = True
    filter_func = build_filter_func(filter_func, log.datensatzname, dataset_name)
    filter_func = build_filter_func(filter_func, log.netz, net_type)
    filter_func = build_filter_func(filter_func, log.gewichte, weights)
    filter_func = build_filter_func(filter_func, log.extra_infos, extra_infos)
    # Log entsprechend der Filter-Funktion filtern
    filtered = log[filter_func]
    # gefilterten Log zurückgeben
    return filtered


def create_array(f_log: pandas.DataFrame):
    """Erstellt ein Array (numpy-Matrix) aus dem übergebenen, gefilterten Log
    Der Log muss 10 Einträge haben (2 Klassifikationsschemata (OvO und OvA), 5 Trainsizes (10,20,50,80 und 100 %))
    Das erstellte Array hat die selbe Form wie die Tabellen von Pawara et al.:
    Train-size      OvO           OvA
    10%             X +- Y       X +- Y
    20%             X +- Y       X +- Y
    50%             X +- Y       X +- Y
    80%             X +- Y       X +- Y
    100%            X +- Y       X +- Y

    Also Zeile 0 = 10% Train-size, Zeile 1 = 20%, ...
    Spalte 0 = OvO Mean, Spalte 1 = OvO Stddev, Spalte 2 = OvA Mean, Spalte 3 = OvA Stddev
    """
    # Name der Spalte, dessen Werte in die Tabelle eingefügt werden sollen
    column_name = "val_acc"
    # column_name = "dauer"
    table = np.zeros(shape=(5, 4), dtype=float)  # Leere Matrix erzeugen
    if len(f_log) != 10:  # Log MUSS genau 10 Einträge enthalten (2*5)
        print("Fehler! Log mit gemittelten Werten ist nicht 10 Einträge lang!")
        print(f_log)
    # Werte an passende Stelle in Matrix schreiben

    # OvO
    table[0][0] = float(f_log[(f_log.trainpercent == 10) & (f_log.klassifikation == "OvO")][column_name]["mean"])
    table[0][1] = float(f_log[(f_log.trainpercent == 10) & (f_log.klassifikation == "OvO")][column_name]["std"])

    table[1][0] = float(f_log[(f_log.trainpercent == 20) & (f_log.klassifikation == "OvO")][column_name]["mean"])
    table[1][1] = float(f_log[(f_log.trainpercent == 20) & (f_log.klassifikation == "OvO")][column_name]["std"])

    table[2][0] = float(f_log[(f_log.trainpercent == 50) & (f_log.klassifikation == "OvO")][column_name]["mean"])
    table[2][1] = float(f_log[(f_log.trainpercent == 50) & (f_log.klassifikation == "OvO")][column_name]["std"])

    table[3][0] = float(f_log[(f_log.trainpercent == 80) & (f_log.klassifikation == "OvO")][column_name]["mean"])
    table[3][1] = float(f_log[(f_log.trainpercent == 80) & (f_log.klassifikation == "OvO")][column_name]["std"])

    table[4][0] = float(f_log[(f_log.trainpercent == 100) & (f_log.klassifikation == "OvO")][column_name]["mean"])
    table[4][1] = float(f_log[(f_log.trainpercent == 100) & (f_log.klassifikation == "OvO")][column_name]["std"])

    # OvA
    table[0][2] = float(f_log[(f_log.trainpercent == 10) & (f_log.klassifikation == "OvA")][column_name]["mean"])
    table[0][3] = float(f_log[(f_log.trainpercent == 10) & (f_log.klassifikation == "OvA")][column_name]["std"])

    table[1][2] = float(f_log[(f_log.trainpercent == 20) & (f_log.klassifikation == "OvA")][column_name]["mean"])
    table[1][3] = float(f_log[(f_log.trainpercent == 20) & (f_log.klassifikation == "OvA")][column_name]["std"])

    table[2][2] = float(f_log[(f_log.trainpercent == 50) & (f_log.klassifikation == "OvA")][column_name]["mean"])
    table[2][3] = float(f_log[(f_log.trainpercent == 50) & (f_log.klassifikation == "OvA")][column_name]["std"])

    table[3][2] = float(f_log[(f_log.trainpercent == 80) & (f_log.klassifikation == "OvA")][column_name]["mean"])
    table[3][3] = float(f_log[(f_log.trainpercent == 80) & (f_log.klassifikation == "OvA")][column_name]["std"])

    table[4][2] = float(f_log[(f_log.trainpercent == 100) & (f_log.klassifikation == "OvA")][column_name]["mean"])
    table[4][3] = float(f_log[(f_log.trainpercent == 100) & (f_log.klassifikation == "OvA")][column_name]["std"])

    return table


def create_latex_table(table: np.array, table_name: str, train_percents=None, show_train_percents=False):
    """Erstellt aus einem Numpy-Array (5 Zeilen, 4 Spalten; s. `create:array(...)`) eine LaTeX-Tabelle"""
    # anzuzeigende Train-sizes setzen
    if train_percents is None:
        train_percents = [10, 20, 50, 80, 100]
    # Wenn die linke Spalte der Tabelle(
    # Train-size ...
    # 10% ...
    # 20%
    # 50%
    # 80%
    # 100%
    # ) zusätzlich erstellt werden soll
    if show_train_percents:
        latex = "\\begin{tabular}{|c|c|c|}\n\\multicolumn{3}{c}{%s} \\\\\n\\hline\n\\hline\n\\scriptsize{Train Prozent} & OvO & OvA \\\\\n\hline " % table_name
    else:
        latex = "\\begin{tabular}{|c|c|}\n\\multicolumn{2}{c}{%s} \\\\\n\\hline\n\\hline\nOvO & OvA \\\\\n\hline " % table_name
    # Fülle die LaTeX-Tabelle mit Werten
    for row_index in range(0, table.__len__()):
        # Füge die aktuelle Train-size in die ganz linke Spalte ein
        if show_train_percents:
            latex += str(train_percents[row_index]) + " & "
        # Füge OvO Mean und Stddev ein
        latex += "%.2f \\plm %.2f" % (table[row_index][0], table[row_index][1]) + " & "
        # Füge OvA Mean und Stddev ein
        latex += "%.2f \\plm %.2f" % (table[row_index][2], table[row_index][3])
        # \\ und Zeilenumbruch einfügen (muss alles escaped werden)
        latex += "\\\\\n"

    latex += "\\hline \n\\end{tabular}"
    return latex


def replicate_original_table(net_type: str, weights: str, extra_info: str, save_path: Path):
    latex = ""
    close_resizebox = False
    # Lade Log
    full_log = load_log(save_path.parent / "allModelsLog.txt")
    # Zu erstellende Sub-Tabellen für Datensätze (True gibt an, dass ein Zeilenumbruch erfolgen soll)
    datasets_percent = [("agrilplant3", True), ("agrilplant5", False), ("agrilplant10", False),
                        ("tropic3", True), ("tropic5", False), ("tropic10", False), ("tropic20", False),
                        ("swedishLeaves3folds3", True), ("swedishLeaves3folds5", False),
                        ("swedishLeaves3folds10", False), ("swedishLeaves3folds15", False),
                        ("pawara-tropic10", True), ("pawara-monkey10", False), ("pawara-umonkey10", False)
                        ]
    # Cifar10 existiert nur für Resnet Scratch
    if net_type == "R" and weights == "S":
        datasets_percent.append(("cifar10", False))
    # Gehe alle Datensätze durch, show_train_percents gibt dabei Zeilenumbruch an und fügt
    # die ganz linke Spalte mit Train-sizes (10, 20, 50, 80, 100%) als Beschriftung ein
    for dataset_name, show_train_percents in datasets_percent:
        print(dataset_name)
        # Beende aktuell geöffnete resizebox und füge einen vertikalen Abstand ein
        if show_train_percents and close_resizebox:
            close_resizebox = False
            latex += "}%\n\\vspace{2mm}\n"
        # Starte neue resizebox
        if show_train_percents:
            latex += "\\resizebox{\\textwidth}{!}{"
            close_resizebox = True  # muss beim nächsten Mal wieder geschlossen werden
        # Erstelle sub-Tabelle (5 Zeilen, 2 Spalten: Trainsizes x {OvO, OvA})
        # Zuerst aus dem Log die entsprechende Matrix extrahieren
        # (5 Zeilen, 4 Spalten: Trainsizes x {OvO_mean, OvO_stddev, OvA_mean, OvA_stddev})}
        sub_table = create_array(filter_log(full_log, dataset_name, net_type, weights, extra_info))
        # Erstelle dann aus der Matrix LaTeX-Code für eine Tabelle
        latex += "\n%s\n" % create_latex_table(sub_table, dataset_name, show_train_percents=show_train_percents)
    latex += "}%\n"  # schließende Klammer von resizebox
    # Speichere gesamten LaTeX-Code in übergebene Datei
    with open(save_path, "w+") as latex_file:
        latex_file.write(latex)


def normalize_time_gpu(log):
    """Normalisiert die Trainingsdauer auf NVIDIA 2080"""
    # Gigaflops pro GPU (Single Precision)
    flops_2080 = 11750
    flops_v100 = 14899
    flops_titanrtx = 12441
    flops_titanxp = 10790

    # Normalisiere auf GPU 2080
    normalize_to = flops_2080

    # Normalisiere alle Einträge in "Dauer" abhängig vom verwendeten Grafikkartentyp

    # 2080
    log.loc[(log.gpu == "GeForce RTX 2080 Ti") | (
            log.gpu == "NVIDIA GeForce RTX 2080 Ti"), "dauer"] *= flops_2080 / normalize_to
    # TITAN XP
    log.loc[(log.gpu == "TITAN Xp") | (log.gpu == "NVIDIA TITAN Xp"), "dauer"] *= flops_titanxp / normalize_to
    # TITAN RTX
    log.loc[(log.gpu == "TITAN RTX") | (log.gpu == "NVIDIA TITAN RTX"), "dauer"] *= flops_titanrtx / normalize_to
    # V100
    log.loc[(log.gpu == "Tesla V100-SXM2-16GB") | (
            log.gpu == "NVIDIA Tesla V100-SXM2-16GB"), "dauer"] *= flops_v100 / normalize_to
    # Gebe normalisierten Log zurück
    return log


def boxplot(save_path: Path, netz: str, weights: str, key="val_acc", key_label="Validation Accuracy"):
    """Erstellt einen Boxplot für übergebenes `netz`, dessen Gewichte `weights` für die Werte `key` und
    beschriftet die Y-Achse mit `kay_label`.
    `save_path` gibt Ordner an, in dem allModelsLog.txt liegt"""
    # Lade kompletten Log (OHNE Mittelwerte über Folds zu bilden!)
    full_log = load_log(save_path / "allModelsLog.txt", False)
    # Filtere den Log nach Netz und Gewicht für den aktuellen Boxplot
    filter_func = True
    filter_func = build_filter_func(filter_func, full_log.netz, netz)
    filter_func = build_filter_func(filter_func, full_log.gewichte, weights)
    full_log = full_log[filter_func]
    # Wenn die dauer geplottet werden soll, normalisiere auf eine Grafikkarte (NVIDIA 2080)
    if key == "dauer":
        full_log = normalize_time_gpu(full_log)

    # Zu betrachtende Datensätze
    datasets = ["agrilplant3", "agrilplant5", "agrilplant10",
                "tropic3", "tropic5", "tropic10", "tropic20",
                "swedishLeaves3folds3", "swedishLeaves3folds5", "swedishLeaves3folds10", "swedishLeaves3folds15",
                "pawara-tropic10", "pawara-monkey10", "pawara-umonkey10"]
    # Indizes, an die im 8x2 Raster geplottet werden soll (gleiche Reihenfolge wie `datasets`)
    plot_indices = [(0, 0), (1, 0), (1, 1), (2, 0), (3, 0), (4, 0), (5, 0), (2, 1), (3, 1), (4, 1), (5, 1), (6, 0),
                    (7, 0), (7, 1)]
    # Frameworks, für die geplottet werden soll (torch kommt später dazu, wenn `netz` != "IP")
    # Tupel besteht aus (Framework, Marker-Typ, Marker-Size) für PyPlot
    frameworks = [("TF1-13-1-detTS", "v", 20), ("TF2-4-1-detTS", "^", 20)]
    # Erstelle 8x2 Subplot Gitter
    fig, axs = plt.subplots(nrows=8, ncols=2)
    # Färbe Tropic  Graphen leicht grau
    axs[2][0].set_facecolor("#e6e6e6")
    axs[3][0].set_facecolor("#e6e6e6")
    axs[4][0].set_facecolor("#e6e6e6")
    axs[5][0].set_facecolor("#e6e6e6")
    axs[6][0].set_facecolor("#e6e6e6")
    # Färbe swedishLeaves Graphen leicht grün
    axs[2][1].set_facecolor("#ccffd9")
    axs[3][1].set_facecolor("#ccffd9")
    axs[4][1].set_facecolor("#ccffd9")
    axs[5][1].set_facecolor("#ccffd9")
    # Färbe Monkey Graphen leicht violett
    axs[7][1].set_facecolor("#e6ccff")
    axs[7][0].set_facecolor("#e6ccff")

    # Wenn mit Resnet Scratch geplottet wird existiert zusätzlich der Datensatz "Cifar10" und er soll unten rechts
    # geplottet werden
    if netz == "R" and weights == "S":
        # Cifar10 hinzufügen
        plot_indices.append((6, 1))
        datasets.append("cifar10")
        # Färbe Cifar10 leicht orange
        axs[6][1].set_facecolor("#ffe6cc")
    else:
        # Ansonsten: plot für Cifar10 unsichtbar machen
        axs[6][1].axis("off")

    # Wenn das Netz nicht "Inception-Pawara" ist, existieren Ergebnisse mit dem Framework "torch"
    if netz != "IP":
        # Tupel besteht aus (Framework, Marker-Typ, Marker-Size) für PyPlot
        frameworks.append(("torch", "3", 40))
    # Passe Layout an (weniger Abstand zwischen Subplots)
    plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.1)

    # Gehe jeden Subplot-Index und zugehörigen Datensatz durch
    for (plot_row, plot_col), dataset in zip(plot_indices, datasets):
        # Starte den Subplot bei x=0
        x_pos = 0
        # Hole aktuellen Subplot aus dem axs-Array
        current_plot = axs[plot_row][plot_col]
        # Filtere den Log nach diesem konkreten Datensatz
        log_per_dataset = full_log[full_log.datensatzname == dataset]
        # Ermittle Minimum und Maximum für diesen Subplot (wichtig für Trennlinien vertikal durch den ganzen Subplot)
        min_value = min(log_per_dataset[key])
        max_value = max(log_per_dataset[key])
        # Gehe alle Train-sizes durch
        for train_size in [10, 20, 50, 80, 100]:
            # NACH der ersten Train-size (10) soll immer eine Trennlinie gezogen werden
            if train_size != 10:
                current_plot.vlines(x=x_pos - 2.5, ymin=min_value, ymax=max_value, colors="k", linewidth=1, alpha=0.7)
            # Filtere den Log nach der aktuellen Train-size
            log_per_trainsize = log_per_dataset[log_per_dataset.trainpercent == train_size]
            # Plotte für diese Train-size für alle Frameworks
            for framework, marker_style, marker_size in frameworks:
                # Filtere nach Framework
                log_per_framework = log_per_trainsize[log_per_trainsize.extra_infos == framework]
                if framework != "TF1-13-1-detTS":  # ersten Trennbalken überspringen
                    current_plot.vlines(x=x_pos - 1, ymin=min_value, ymax=max_value, colors="k", linewidth=1, alpha=0.1)
                # Beide Klassifikationsschemata durchgehen
                for ovo_ova in ["OvO", "OvA"]:
                    # Filtere nach aktuellem Klassifikationsschema
                    current_row = log_per_framework[log_per_framework.klassifikation == ovo_ova]
                    # Jetzt ist in current_row der aktuelle Eintrag, zeichne Boxplot dafür
                    values = current_row[key]  # Nehme Werte aus Spalte `key`
                    color = "orangered" if ovo_ova == "OvO" else "royalblue"  # OvO orange-rot, OvA royalblau
                    # Scattere die Einzelnen Ergebnisse (ein Ergebnis pro Fold) im Pixel-Stil
                    current_plot.scatter(x=len(values) * [x_pos], y=list(values), marker=",", s=1, color=color,
                                         alpha=0.2)
                    # Zeichne Mittelwert ein mit `marker_style` als Markierung und `marker_size` als Größe der
                    # Markierung (s. Tupel in der Liste der Frameworks)
                    current_plot.scatter(x=x_pos, y=mean(list(values)), marker=marker_style, s=marker_size, color=color)
                    x_pos += 1  # Zähle nach einem OvO/OvA Durchgang x um 1 hoch
                # Zähle nach einem Framework x ERNEUT um 1 hoch (zwischen zwei Frameworks ist also eine 2-breite Lücke)
                x_pos += 1
            # Zähle nach einer Train-Size x um weitere 3 hoch, (zwischen zwei Train-Sizes ist also eine 5-breite Lücke)
            x_pos += 3

        axs[plot_row][plot_col].xaxis.set_label_coords(0.5, -0.15)
        # Eine Trainsize hat eine Breite von 7, zwischen ihnen ist 5 Platz
        # Also erstes Label mittig in die 1. Trainsize setzen (7/2 = 3.5)
        # Zweites Label wieder (Anfang bei 7+5=12, Mitte bei 12 + 3.5 = 15.5) und so weiter
        # Anfänge der Train-Sizes: 0, 12, 24, 36, 48
        ticks = [3.5, 15.5, 27.5, 39.5, 51.5]

        if netz == "IP":  # Es gibt keine Daten für "torch", die Abstände verschieben sich also
            # 4er Breite für eine Trainsize mit 5 Platz dazwischen
            # Anfänge bei 0, 9, 18, 27, 36 -> immer +2 um die Mitte zu treffen
            ticks = [2, 11, 20, 29, 38]
        # X-Achse im Abstand gemäß `ticks` mit den Train-sizes beschriften
        current_plot.set_xticks(ticks)
        current_plot.set_xticklabels(["10%", "20%", "50%", "80%", "100%"])
        # Wenn der aktuelle Subplot ganz links ist (plot_col==0), dann beschrifte die Y-Achse
        # andernfalls nicht wegen platzgründen
        if plot_col == 0:
            current_plot.set_ylabel(key_label)
        # Setze Titel für Subplot auf den Namen des dort geplotteten Datensatzes
        current_plot.set_title(dataset, fontsize=12)
    # Setze Haupt-Titel (Abkürzungen für Netz und Gewichte werden erst aufgelöst)
    netz_str = ""
    if netz == "R":
        netz_str = "Resnet"
    elif netz == "IP":
        netz_str = "Inception-Pawara"
    elif netz == "I":
        netz_str = "Inception"
    weights_str = "Finetune" if weights == "F" else "Scratch"
    plt.suptitle(netz_str + " " + weights_str, fontsize=20)

    # Legende basteln
    # Plot oben rechts zweckentfremden um Label-Symbole für Legende zu generieren
    legend_row = 0
    legend_col = 1
    # Oben rechts werden die Achsen ausgeschaltet
    axs[legend_row][legend_col].axis("off")
    axs[legend_row][legend_col].scatter(0, 0, marker="x", s=2, color="orangered", alpha=0.2, label="einzelne Folds OvO")
    axs[legend_row][legend_col].scatter(0, 0, marker="x", s=2, color="royalblue", alpha=0.2, label="einzelne Folds OvA")

    axs[legend_row][legend_col].scatter(x=0, y=0, marker="v", s=20, color="orangered", label="Mittelwert OvO TF 1.13.1")
    axs[legend_row][legend_col].scatter(x=0, y=0, marker="v", s=20, color="royalblue", label="Mittelwert OvA TF 1.13.1")

    axs[legend_row][legend_col].scatter(x=0, y=0, marker="^", s=20, color="orangered", label="Mittelwert OvO TF 2.4.1")
    axs[legend_row][legend_col].scatter(x=0, y=0, marker="^", s=20, color="royalblue", label="Mittelwert OvA TF 2.4.1")

    axs[legend_row][legend_col].scatter(x=0, y=0, marker="1", s=40, color="orangered", label="Mittelwert OvO Torch")
    axs[legend_row][legend_col].scatter(x=0, y=0, marker="1", s=40, color="royalblue", label="Mittelwert OvA Torch")
    # Grenzen vom Plot (x-Achse) so legen, dass die Dummmy-Einträge nicht sichtbar sind
    axs[legend_row][legend_col].set_xlim([1, 2])
    # Legende (für alle Subplots) an der Stelle des (unsichtbaren) Plots anzeigen
    axs[legend_row][legend_col].legend(bbox_to_anchor=(0.7, 1.6))

    # Speichere Plot ab unter 3_<Netztyp>-<Gewichte>.svg bzw. 3_<Netztyp>-<Gewichte>-<Key>.svg falls key!="val_acc"
    fig.set_size_inches(12, 15)
    filename = "3_" + netz + "-" + weights
    if key != "val_acc":
        filename += "-" + key
    filename += ".svg"
    plt.savefig(filename)


def tables():
    """Erstellt LaTeX-Tabellen zu einer Log-Datei"""
    # Grundverzeichnis (allModelsLog.txt wird in diesem Verzeichnis erwartet, Tabellen werden ebenfalls dort abgelegt)
    _BASE_PATH = Path("G://Bachelorarbeit/Tabellen/")

    # TensorFlow 1.13.1
    replicate_original_table("R", "S", "TF1-13-1-detTS", _BASE_PATH / "resnet_scratch_TF1-13-1.tex")
    replicate_original_table("R", "F", "TF1-13-1-detTS", _BASE_PATH / "resnet_finetune_TF1-13-1.tex")
    replicate_original_table("I", "S", "TF1-13-1-detTS", _BASE_PATH / "inception_scratch_TF1-13-1.tex")
    replicate_original_table("I", "F", "TF1-13-1-detTS", _BASE_PATH / "inception_finetune_TF1-13-1.tex")
    replicate_original_table("IP", "S", "TF1-13-1-detTS", _BASE_PATH / "inception-pawara_scratch_TF1-13-1.tex")
    replicate_original_table("IP", "F", "TF1-13-1-detTS", _BASE_PATH / "inception-pawara_finetune_TF1-13-1.tex")

    # TensorFlow 2.4.1
    replicate_original_table("R", "S", "TF2-4-1-detTS", _BASE_PATH / "resnet_scratch_TF2-4-1.tex")
    replicate_original_table("R", "F", "TF2-4-1-detTS", _BASE_PATH / "resnet_finetune_TF2-4-1.tex")
    replicate_original_table("I", "S", "TF2-4-1-detTS", _BASE_PATH / "inception_scratch_TF2-4-1.tex")
    replicate_original_table("I", "F", "TF2-4-1-detTS", _BASE_PATH / "inception_finetune_TF2-4-1.tex")
    replicate_original_table("IP", "S", "TF2-4-1-detTS", _BASE_PATH / "inception-pawara_scratch_TF2-4-1.tex")
    replicate_original_table("IP", "F", "TF2-4-1-detTS", _BASE_PATH / "inception-pawara_finetune_TF2-4-1.tex")

    # PyTorch (kein IP-Netz)
    replicate_original_table("R", "S", "torch", _BASE_PATH / "resnet_scratch_torch.tex")
    replicate_original_table("R", "F", "torch", _BASE_PATH / "resnet_finetune_torch.tex")
    replicate_original_table("I", "S", "torch", _BASE_PATH / "inception_scratch_torch.tex")
    replicate_original_table("I", "F", "torch", _BASE_PATH / "inception_finetune_torch.tex")


def boxplots():
    """Erstelle Boxplots für alle Netztypen und Gewichte"""
    # Als Pfad wird ein Ordner erwartet, in dem "allModelsLog.txt" liegt
    # Key: val_acc Standardmäßig
    boxplot(Path("G://Bachelorarbeit/Code"), "R", "S")
    boxplot(Path("G://Bachelorarbeit/Code"), "R", "F")
    boxplot(Path("G://Bachelorarbeit/Code"), "I", "S")
    boxplot(Path("G://Bachelorarbeit/Code"), "I", "F")
    boxplot(Path("G://Bachelorarbeit/Code"), "IP", "S")
    boxplot(Path("G://Bachelorarbeit/Code"), "IP", "F")

    # Key: dauer (normalisiert automatisch auf GPU-Modell Nvidia 2080)
    boxplot(Path("G://Bachelorarbeit/Code"), "R", "S", key="dauer", key_label="Trainingsdauer (Min.)")
    boxplot(Path("G://Bachelorarbeit/Code"), "R", "F", key="dauer", key_label="Trainingsdauer (Min.)")
    boxplot(Path("G://Bachelorarbeit/Code"), "I", "S", key="dauer", key_label="Trainingsdauer (Min.)")
    boxplot(Path("G://Bachelorarbeit/Code"), "I", "F", key="dauer", key_label="Trainingsdauer (Min.)")
    boxplot(Path("G://Bachelorarbeit/Code"), "IP", "S", key="dauer", key_label="Trainingsdauer (Min.)")
    boxplot(Path("G://Bachelorarbeit/Code"), "IP", "F", key="dauer", key_label="Trainingsdauer (Min.)")


def plot_loss_graphs():
    """Erstellt Plots zu den in Kapitel 2.1.2 OvO Kodierung genannten drei Fällen
    bei der Loss-Funktion"""
    # Wertebereich (0, 1)
    xs = np.arange(0.001, 0.999, 0.001)
    # Ergebnis-Liste für die 3 Fälle (y = 1, y=0 und y=0.5)
    ys_1 = - np.log(xs)
    ys_0 = - np.log(1 - xs)
    ys_05 = - (0.5 * np.log(xs) + 0.5 * np.log(1 - xs))
    # Erstelle Plots
    for ys in [ys_1, ys_0, ys_05]:
        plt.plot(xs, ys)
        plt.xlabel(r"$\hat y_{l}^{'}$", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.show()


def plot_comparison():
    """Plottet einen Graphen pro Framework, der die Mittelwert-Differenzen zwischen OvO und OvA visualisiert"""
    # Lade Log-Datei (ohne Mittelwert-Bildung)
    log = load_log(Path("G://Bachelorarbeit/Code/allModelsLog.txt"), average=False)

    # verfügbare Frameworks (3 Subplots)
    frameworks = ["TF1-13-1-detTS", "TF2-4-1-detTS", "torch"]
    # Starte in subplot 0
    subplot_index = 0
    # Erstelle Subplot-Gitter mit 1 Zeile und 3 Spalten
    fig, axs = plt.subplots(nrows=1, ncols=3)
    # min und max Werte für die y-Achse (damit später ALLE 3 Subplots gleich skaliert werden können),
    # initialisiert mit Werten die auf jeden Fall später überschrieben werden
    min_ = 100  # ein sehr großer Wert für min_
    max_ = -100  # ein sehr kleiner Wert max_

    # Gehe alle Frameworks (und damit Subplots) durch
    for framework in frameworks:
        # Starte bei x=0
        x = 0
        # Hole aktuellen Subplot entsprechend des subplot_index aus axs
        current_plot = axs[subplot_index]
        # Wenn Subplot ganz links, beschrifte die Y-Achse
        if subplot_index == 0:
            current_plot.set_ylabel("Differenz (OvO - OvA)", fontsize=12)
        # Zähle subplot_index hoch für nächsten Durchlauf
        subplot_index += 1
        # Verfügbare Netztypen
        networks = ["R", "I"]
        # Falls nicht mit torch trainiert wurde
        if framework != "torch":
            # existiert zusätzlich "Inception-Pawara" als Netztyp
            networks.append("IP")
        # Gehe alle Netztypen durch
        for network in networks:
            # Gehe beide Gewicht-Initialisierungen durch (Scratch und Finetune)
            for gewichte in ["S", "F"]:
                # Gehe alle Train-Sizes durch
                for train_size_num, train_size in enumerate([10, 20, 50, 80, 100]):
                    # verfügbare Datensätze
                    datasets = ["agrilplant3", "agrilplant5", "agrilplant10",
                                "tropic3", "tropic5", "tropic10", "tropic20",
                                "swedishLeaves3folds3", "swedishLeaves3folds5", "swedishLeaves3folds10",
                                "swedishLeaves3folds15",
                                "pawara-tropic10", "pawara-monkey10", "pawara-umonkey10"]
                    # Für Resnet Scratch existiert zusätzlich der Datensatz Cifar10
                    if network == "R" and gewichte == "S":
                        datasets.append("cifar10")
                    # Gehe alle Datensätze durch
                    for dataset in datasets:
                        # Filtere Log-Datei
                        rows = filter_log(log, dataset, network, gewichte, framework)
                        rows = rows[rows.trainpercent == train_size]
                        # Ziehe Werte für OvA und OvO raus
                        rows_ova = rows[rows.klassifikation == "OvA"]
                        rows_ovo = rows[rows.klassifikation == "OvO"]
                        # Bilde Mittelwert
                        mean_ova = rows_ova["val_acc"].mean()
                        mean_ovo = rows_ovo["val_acc"].mean()
                        # Bilde Differenz (OvO-OvA)
                        diff = mean_ovo - mean_ova
                        # Passe ggf. min_ und max_ Werte an
                        if diff < min_:
                            min_ = diff
                        if diff > max_:
                            max_ = diff
                        # Setze Farbe zum Plotten abhängig vom Netztypen
                        if network == "R":
                            color = "red"
                        elif network == "I":
                            color = "blue"
                        else:
                            color = "purple"
                        # Plotte Differenz zwischen OvO und OvA als einen Datenpunkt mit oben gesetzter Farbe
                        # und einer steigenden Durchsichtigkeit pro Train-Size (10% ist fett, 100% sehr durchsichtig)
                        current_plot.scatter(x, diff, marker="x", s=8, color=color, alpha=-0.2 * (train_size_num - 5))
                        # Zähle x ein bisschen hoch (damit nicht alles übereinander liegt)
                        x += 0.3
                    # Zwischen Train-Sizes ist eine 15er Lücke
                    x += 15
                # Zwischen Gewicht-Typen (S / F) ist zusätzlich eine 20er Lücke
                x += 20
        # Ziehe Linie bei y=0 um Ergebnisse nahe bei y=0 besser zu erkennen
        current_plot.hlines(0, -20, x - 15, color="k")
        # Baue Subplot-Titel (und erstelle Überschriften für Netztypen, grob mit Leertasten ausgerichtet)
        if framework == "TF1-13-1-detTS":
            framework_str = "TF 1.13.1\n\n" + 9 * " " + "ResNet" + 17 * " " + "Inception" + 10 * " " + "Inception-Pawara "
        elif framework == "TF2-4-1-detTS":
            framework_str = "TF 2.4.1\n\n" + 9 * " " + "ResNet" + 17 * " " + "Inception" + 10 * " " + "Inception-Pawara "
        else:
            framework_str = framework + "\n\n" + 10 * " " + "ResNet" + 32 * " " + "Inception" + 5 * " "
        current_plot.set_title(framework_str, fontsize=12)
        # Beschränke X-Achse (sonst ist automatisch viel Rand vorhanden)
        current_plot.set_xlim(-20, x - 15)
    # Y-Achsen-Limitierung wird um +- 1.5 erweitert, damit man alles gut sieht
    min_ -= 1.5
    max_ += 1.5
    axs[0].set_ylim(min_, max_)
    axs[1].set_ylim(min_, max_)
    axs[2].set_ylim(min_, max_)

    for ax_id, ax in enumerate(axs):
        # Ziehe Trennstriche zwischen Scratch und Finetune
        # Jede 2. Linie ist dicker (S->F ist dünn, F->S ist dick)
        ax.vlines(106, min_, max_, color="k", alpha=0.1)
        ax.vlines(222, min_, max_, color="k", alpha=0.7)
        ax.vlines(338, min_, max_, color="k", alpha=0.1)
        if ax_id != 2:  # Nur ausführen, wenn Netztyp Inception-Pawara vorhanden (nicht beim letzen Subplot für Torch)
            ax.vlines(454, min_, max_, color="k", alpha=0.7)
            ax.vlines(550, min_, max_, color="k", alpha=0.1)
            # X-Achsen Ticks und Label festlegen
            ticks = [53, 106 + 53, 222 + 53, 338 + 53, 454 + 53, 570 + 53]
            labels = ["S", "F", "S", "F", "S", "F"]
        else:  # Nur für letzen Subplot (torch) ausführen
            # X-Achsen Ticks und Label festlegen
            ticks = [53, 106 + 53, 222 + 53, 338 + 53]
            labels = ["S", "F", "S", "F"]
        # Ticks und Label setzen
        ax.set_xticks(ticks=ticks)
        ax.set_xticklabels(labels=labels, fontsize=12)
        # Tick-Größe anpassen (man erkennt sonst kaum was)
        ax.tick_params(axis="both", labelsize=12)

    # Gesamttitel für Plot setzen
    plt.suptitle("Differenz zwischen Accuracy-Mittelwerten von OvO und OvA", fontsize=20)
    # Plot anzeigen
    plt.show()

def load_pickle(file: Path):
    """Lädt eine Pickle Datei `file`"""
    with open(file, "rb") as picklefile:
        res = pickle.load(picklefile)
    return res

def create_confusion_matrix(pred_labels, true_labels):
    num_classes = max(max(pred_labels), max(true_labels)) + 1  # Label Starten bei 0
    confusion_matrix = np.zeros((num_classes, num_classes), float)

    assert len(pred_labels) == len(true_labels)
    for i in range(len(pred_labels)):
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        confusion_matrix[true_label][pred_label] += 1
    # Confusion-Matrix zeilenweise normalisieren
    for row_id in range(len(confusion_matrix)):
        summe = sum(confusion_matrix[row_id , :])
        for col_id in range(len(confusion_matrix[row_id, :])):
            confusion_matrix[row_id, col_id] = round(confusion_matrix[row_id, col_id] / summe, 2)

    return confusion_matrix

def plot_train_history():
    """Erstellt Plots für ein paar konkrete Beispiele für den Trainingsverlauf (Loss und Accuracy
    über die Epochen hinweg) und einer Confusion-Matrix"""
    # Verzeichnisse setzen
    BASE_PATH = Path ("G://Bachelorarbeit/Ergebnisse/Beispiele")
    torch_path = BASE_PATH / "Torch"
    tfAlt_path = BASE_PATH / "TF1-13-1"
    tfNeu_path = BASE_PATH / "TF2-4-1"
    # Mehrere Beispiele durchgehen
    for beispiel in ["Inception-Scratch-Tropic10-10-Fold1", "Resnet-Scratch-swedishLeaves3Folds5-10-Fold1", "Resnet-Finetune-Agrilplant10-100-Fold1"]:
        ### Trainingsverlauf plotten ###
        fig, axs = plt.subplots(2, 3)
        # Pfad zu gespeicherter History
        torch_history_ovo = load_pickle(torch_path / beispiel / "OvO" / "historySave.dat")
        torch_history_ova = load_pickle(torch_path / beispiel / "OvA" / "historySave.dat")
        # In Torch stehen Prozentwerte in Accuracy
        torch_history_ovo["val_acc"] = [acc/100 for acc in torch_history_ovo["val_acc"]]
        torch_history_ova["val_acc"] = [acc / 100 for acc in torch_history_ova["val_acc"]]
        axs[0,0].plot(range(len(torch_history_ovo["val_loss"])), torch_history_ovo["val_loss"], color="orangered", label="OvO Loss")
        axs[0,0].plot(range(len(torch_history_ova["val_loss"])), torch_history_ova["val_loss"], color="royalblue", label="OvA Loss")
        axs[1,0].plot(range(len(torch_history_ovo["val_acc"])), torch_history_ovo["val_acc"], color="orangered", label="OvO Accuracy")
        axs[1,0].plot(range(len(torch_history_ova["val_acc"])), torch_history_ova["val_acc"], color="royalblue", label="OvA Accuracy")

        tfAlt_history_ovo = load_pickle(tfAlt_path / beispiel / "OvO" / "historySave.dat")
        tfAlt_history_ova = load_pickle(tfAlt_path / beispiel / "OvA" / "historySave.dat")
        axs[0,1].plot(range(len(tfAlt_history_ovo["val_loss"])), tfAlt_history_ovo["val_loss"], color="orangered", label="OvO Loss")
        axs[0,1].plot(range(len(tfAlt_history_ova["val_loss"])), tfAlt_history_ova["val_loss"], color="royalblue", label="OvA Loss")
        axs[1,1].plot(range(len(tfAlt_history_ovo["val_ovo_accuracy_metric"])), tfAlt_history_ovo["val_ovo_accuracy_metric"], color="orangered", label="OvO Accuracy")
        axs[1,1].plot(range(len(tfAlt_history_ova["val_acc"])), tfAlt_history_ova["val_acc"], color="royalblue", label="OvA Accuracy")

        tfNeu_history_ovo = load_pickle(tfNeu_path / beispiel / "OvO" / "historySave.dat")
        tfNeu_history_ova = load_pickle(tfNeu_path / beispiel / "OvA" / "historySave.dat")
        axs[0, 2].plot(range(len(tfNeu_history_ovo["val_loss"])), tfNeu_history_ovo["val_loss"], color="orangered", label="OvO Loss")
        axs[0, 2].plot(range(len(tfNeu_history_ova["val_loss"])), tfNeu_history_ova["val_loss"], color="royalblue", label="OvA Loss")
        axs[1, 2].plot(range(len(tfNeu_history_ovo["val_ovo_accuracy_metric"])), tfNeu_history_ovo["val_ovo_accuracy_metric"], color="orangered", label="OvO Accuracy")
        axs[1, 2].plot(range(len(tfNeu_history_ova["val_acc"])), tfNeu_history_ova["val_acc"], color="royalblue", label="OvA Accuracy")

        axs[0,0].set_title("PyTorch 1.9.0 Loss", fontsize=12)
        axs[1, 0].set_title("PyTorch 1.9.0 Accuracy", fontsize=12)
        axs[0, 1].set_title("TensorFlow 1.13.1 Loss", fontsize=12)
        axs[1, 1].set_title("TensorFlow 1.13.1 Accuracy", fontsize=12)
        axs[0, 2].set_title("TensorFlow 2.4.1 Loss", fontsize=12)
        axs[1, 2].set_title("TensorFlow 2.4.1 Accuracy", fontsize=12)
        # Tick-Größe anpassen (man erkennt sonst kaum was) und Achsen beschriften
        for row in range(2):
            for col in range(3):
                if row == 0:
                    axs[row, col].set_ylabel("Validation Loss")
                else:
                    axs[row, col].set_ylabel("Validation Accuracy")
                axs[row, col].set_xlabel("Epochen")
                axs[row, col].tick_params(axis="both", labelsize=12)
                axs[row, col].legend()
        plt.suptitle("Trainingsverlauf \n" + beispiel, fontsize=20)
        plt.show()

        ### Confusion-Matrix plotten ###
        # Vorhergesagte und korrekte Label laden
        torch_ova_pred_true = []
        torch_ovo_pred_true = []
        tfAlt_ova_pred_true = []
        tfAlt_ovo_pred_true = []
        tfNeu_ova_pred_true = []
        tfNeu_ovo_pred_true = []

        torch_ova_pred_true.append(np.load(torch_path / beispiel / "OvA" / "predicted_classes_test.npy"))
        torch_ova_pred_true.append(np.load(torch_path / beispiel / "OvA" / "true_classes_test.npy"))
        torch_ovo_pred_true.append(np.load(torch_path / beispiel / "OvO" / "predicted_classes_test.npy"))
        torch_ovo_pred_true.append(np.load(torch_path / beispiel / "OvO" / "true_classes_test.npy"))

        tfAlt_ova_pred_true.append(np.load(tfAlt_path / beispiel / "OvA" / "predicted_classes_test.npy"))
        tfAlt_ova_pred_true.append(np.load(tfAlt_path / beispiel / "OvA" / "true_classes_test.npy"))
        tfAlt_ovo_pred_true.append(np.load(tfAlt_path / beispiel / "OvO" / "predicted_classes_test.npy"))
        tfAlt_ovo_pred_true.append(np.load(tfAlt_path / beispiel / "OvO" / "true_classes_test.npy"))

        tfNeu_ova_pred_true.append(np.load(tfNeu_path / beispiel / "OvA" / "predicted_classes_test.npy"))
        tfNeu_ova_pred_true.append(np.load(tfNeu_path / beispiel / "OvA" / "true_classes_test.npy"))
        tfNeu_ovo_pred_true.append(np.load(tfNeu_path / beispiel / "OvO" / "predicted_classes_test.npy"))
        tfNeu_ovo_pred_true.append(np.load(tfNeu_path / beispiel / "OvO" / "true_classes_test.npy"))


        # Daraus jeweils eine Confusion-Matrix erstellen (NumPy Matrix, zeilenweise normalisiert)
        torch_cm_ova = create_confusion_matrix(torch_ova_pred_true[0], torch_ova_pred_true[1])
        torch_cm_ovo = create_confusion_matrix(torch_ovo_pred_true[0], torch_ovo_pred_true[1])

        tfAlt_cm_ova = create_confusion_matrix(tfAlt_ova_pred_true[0], tfAlt_ova_pred_true[1])
        tfAlt_cm_ovo = create_confusion_matrix(tfAlt_ovo_pred_true[0], tfAlt_ovo_pred_true[1])

        tfNeu_cm_ova = create_confusion_matrix(tfNeu_ova_pred_true[0], tfNeu_ova_pred_true[1])
        tfNeu_cm_ovo = create_confusion_matrix(tfNeu_ovo_pred_true[0], tfNeu_ovo_pred_true[1])


        # Numpy-Matrizen mit seaborn als Heatmap plotten
        fig, axs = plt.subplots(2,3)
        cmap = "flare"
        seaborn.heatmap(torch_cm_ova, annot=True, cmap=cmap, ax=axs[0, 0])
        seaborn.heatmap(torch_cm_ovo, annot=True, cmap=cmap, ax=axs[1, 0])

        seaborn.heatmap(tfAlt_cm_ova, annot=True, cmap=cmap, ax=axs[0, 1])
        seaborn.heatmap(tfAlt_cm_ovo, annot=True, cmap=cmap, ax=axs[1, 1])

        seaborn.heatmap(tfNeu_cm_ova, annot=True, cmap=cmap, ax=axs[0, 2])
        seaborn.heatmap(tfNeu_cm_ovo, annot=True, cmap=cmap, ax=axs[1, 2])

        # Beschriftungen hinzufügen
        for row in range(len(axs)):
            for col in range(len(axs[row])):
                ax = axs[row, col]
                ax.set_ylabel("True Class", fontsize=8)
                ax.set_xlabel("Predicted Class", fontsize=8)
        axs[0, 0].set_title("Torch OvA", fontsize=12)
        axs[1, 0].set_title("Torch OvO", fontsize=12)

        axs[0, 1].set_title("TF 1.13.1 OvA", fontsize=12)
        axs[1, 1].set_title("TF 1.13.1 OvO", fontsize=12)

        axs[0, 2].set_title("TF 2.4.1 OvA", fontsize=12)
        axs[1, 2].set_title("TF 2.4.1 OvO", fontsize=12)
        plt.suptitle("Confusion Matrizen \n" + beispiel, fontsize=20)
        plt.show()




# Erstellt LaTeX-Tabellen aus der Log-Datei
# tables()

# Erstellt BoxPlots zu einer Log Datei
# boxplots()

# Für die Loss-Graphen aus Kapitel 2.1.2
# plot_loss_graphs()

# Plottet die Differenzen zwischen OvO und OvA Mittelwerten sortiert nach Framework, Netztyp und Trainsizes
#plot_comparison()

# Plottet ein konkretes Beispiel (Trainingsverlauf und Confusion Matrix)
plot_train_history()
