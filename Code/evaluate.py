import matplotlib.pyplot as plt
import numpy as np
import pandas
from pathlib import Path
from statistics import mean


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
    # Indizes, an die im 4x4 Raster geplottet werden soll (gleiche Reihenfolge wie `datasets`)
    plot_indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 0),
                    (3, 1), (3, 2)]
    # Frameworks, für die geplottet werden soll (torch kommt später dazu, wenn `netz` != "IP")
    # Tupel besteht aus (Framework, Marker-Typ, Marker-Size) für PyPlot
    frameworks = [("TF1-13-1-detTS", "v", 20), ("TF2-4-1-detTS", "^", 20)]
    # Erstelle 4x4 Subplot Gitter
    fig, axs = plt.subplots(nrows=4, ncols=4)
    # Oben rechts werden die Achsen ausgeschaltet
    # dieser Plot an Stelle (0,3) oben rechts wird später für die Erstellung der Legende zweckentfremdet
    axs[0][3].axis("off")
    # Wenn mit Resnet Scratch geplottet wird existiert zusätzlich der Datensatz "Cifar10" und er soll unten rechts
    # geplottet werden
    if netz == "R" and weights == "S":
        plot_indices.append((3, 3))
        datasets.append("cifar10")
    else:  # Ansonsten mache unten rechts auch die Achsen aus (unsichtbarer Subplot)
        axs[3][3].axis("off")

    # Wenn das Netz nicht "Inception-Pawara" ist, existieren Ergebnisse mit dem Framework "torch"
    if netz != "IP":
        # Tupel besteht aus (Framework, Marker-Typ, Marker-Size) für PyPlot
        frameworks.append(("torch", "1", 40))
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
                    current_plot.scatter(x=len(values) * [x_pos], y=list(values), marker=",", s=2, color=color,
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
    axs[0][3].scatter(0, 0, marker="x", s=2, color="orangered", alpha=0.2, label="einzelne Folds OvO")
    axs[0][3].scatter(0, 0, marker="x", s=2, color="royalblue", alpha=0.2, label="einzelne Folds OvA")

    axs[0][3].scatter(x=0, y=0, marker="v", s=20, color="orangered", label="Mittelwert OvO TF 1.13.1")
    axs[0][3].scatter(x=0, y=0, marker="v", s=20, color="royalblue", label="Mittelwert OvA TF 1.13.1")

    axs[0][3].scatter(x=0, y=0, marker="^", s=20, color="orangered", label="Mittelwert OvO TF 2.4.1")
    axs[0][3].scatter(x=0, y=0, marker="^", s=20, color="royalblue", label="Mittelwert OvA TF 2.4.1")

    axs[0][3].scatter(x=0, y=0, marker="1", s=40, color="orangered", label="Mittelwert OvO Torch")
    axs[0][3].scatter(x=0, y=0, marker="1", s=40, color="royalblue", label="Mittelwert OvA Torch")
    # Grenzen vom Plot (x-Achse) so legen, dass die Dummmy-Einträge nicht sichtbar sind
    axs[0][3].set_xlim([1, 2])
    # Legende (für alle Subplots) an der Stelle des (unsichtbaren) Plots anzeigen
    axs[0][3].legend()

    plt.show()


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


# Erstellt LaTeX-Tabellen aus der Log-Datei
# tables()

# Erstellt BoxPlots zu einer Log Datei
boxplots()

# Für die Loss-Graphen aus Kapitel 2.1.2
# plot_loss_graphs()
