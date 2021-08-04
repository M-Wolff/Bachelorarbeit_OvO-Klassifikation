import shutil
from pathlib import Path
import os
from operator import add
import random
from matplotlib import pyplot as plt
from distutils.dir_util import copy_tree

_VERBOSE = True  # gibt detaillierte Infos aus

# _WORK_DIR = Path("G://Bachelorarbeit")
# _WORK_DIR = Path("D://Bachelorarbeit/")
_WORK_DIR = Path("/scratch/tmp/m_wolf37/Bachelorarbeit")


def check_fold_file(file: Path, folds: list):
    """Prüft, wie oft eine Datei (mit gleichem Dateinamen wie 'file') in dem train- oder testsplit der Folds (aus der
    Liste 'folds') vorkommt und gibt ein Tupel zurück bestehend aus
    (Liste der gefundenen Dateipfade, Anzahl Vorkommnisse im Trainsplit, Anzahl Vorkommnisse im Testsplit)"""

    num_test = 0  # Anzahl an gefundenen Dateien mit gleichem Namen in einem Testsplit
    num_train = 0  # Anzahl an gefundenen Dateien mit gleichem Namen in einem Trainsplit
    found_paths = []  # Liste mit absoluten Pfaden auf gefundene Dateien mit gleichem Namen

    # Gehe alle Folds durch
    for fold in folds:
        # Gehe alle Splits durch (sollte "test" und "train" sein)
        for split in fold.iterdir():
            # Baue (möglichen) Dateipfad in aktuellem Fold und Split mit ursprünglichem Ordnernamen für Klassen (parent)
            p = Path(split / file.parent.name / file.name)
            if p.exists():  # (möglicher) Dateipfad existiert tatsächlich
                found_paths.append(p)  # Füge ihn zur Rückgabeliste hinzu
                if split.name == "test":  # Datei kommt im testsplit vor
                    num_test += 1
                if split.name == "train":  # Datei kommt im trainsplit vor
                    num_train += 1
    return found_paths, num_train, num_test


def verify(directory: Path, k=5, full_scan=True):
    """Prüft, ob alle Splits im Ordner 'directory' gültig sind und gibt ggf. gefundene Fehler aus
    'k' gibt die Anzahl an erwarteten Folds an
    'full_scan' prüft jeden Fold vollständig mit allen anderen (sinnvoll um zu prüfen, ob in einem Fold Bilder fehlen)
    """

    # Liste mit Pfaden für alle Folds in 'directory'
    fold_dirs = [fold for fold in directory.iterdir() if fold.is_dir()]
    print("Es existieren %x Folds:" % fold_dirs.__len__())
    for fold in fold_dirs:
        print(fold)
    print(20 * "-")
    if fold_dirs.__len__() != k:
        print("Fehler! Es wurden nur %x Folds erwartet" % k)
    print(
        "Prüfe, ob jedes Bild nur einmal im Test-Split und %x-mal im Train-Split vorkommt" % (fold_dirs.__len__() - 1))

    errors = 0  # Gefundene Fehler
    for fold in fold_dirs:  # Gehe alle Folds durch
        for split in fold.iterdir():  # Gehe jeweils alle Splits durch
            print("Prüfe Fold %s, Split %s" % (fold.name, split.name))
            for class_folder in split.iterdir():  # Gehe jeweils alle Klassen durch
                for file in class_folder.iterdir():  # Gehe jeweils alle Dateien durch
                    # Suche, wie oft eine Datei insgesamt in allen Folds im Train- oder Testsplit vorkommt
                    (found_paths, is_train, is_test) = check_fold_file(file, fold_dirs)
                    # Wenn Datei nicht k-1 Mal im Train und 1 Mal im Testsplit vorkommt
                    if is_train != k - 1 and is_test != 1:
                        # Erhöhe die gefundenen Fehler und gebe ggf. eine Übersicht aus, wo die Datei vorkommt
                        if _VERBOSE:
                            print("Datei %s kommt %x Mal im Train- und %x Mal im Testsplit vor!" % (
                                file.name, is_train, is_test))
                            for p in found_paths:
                                print(p)
                            print(20 * "-")
                        errors += 1
        # Wenn kein vollständiger Scan durchgeführt werden soll, breche nach erstem Schleifendurchlauf für 'fold' ab
        # Alle Dateien, die nicht in irgendeinem Split in Fold 1 vorkommen wurden nicht überprüft
        if not full_scan:
            break
    print("gefundene Fehler: " + str(errors))


def visualize_folds(directory: Path):
    """Visualisiert die Aufteilung in Folds
    'directory' ist der Ordner, in dem alle Folds liegen"""
    fold_dirs = [fold for fold in directory.iterdir() if fold.is_dir()]  # Liste mit allen Unterordnern (Folds)
    # Dictionary: Klassenname -> Liste mit Datei-Anzahlen (erst alle Trainsplits, dann alle Testsplits aufsteigend)
    class_file_count = {}
    groups = []  # Liste mit Gruppierungen (bestehend aus Fold-Name und Split-Name)
    fig, ax = plt.subplots()
    for split in ["train_100", "test"]:  # Gehe erst alle Trainsplits durch, dann Testsplits (sieht schöner aus im Graph)
        for fold in fold_dirs:  # Gehe alle Folds durch
            groups.append(str(fold.name) + " " + str(split))  # Füge den Gruppennamen zu 'groups' hinzu
            for class_folder in (fold / split).iterdir():  # Gehe alle Klassen durch
                num_files_in_class = len(os.listdir(class_folder))  # Zähle Dateien
                # Füge die Dateianzahl an das Ende der Liste im zugehörigen Dictionary Eintrag hinzu
                if class_folder.name in class_file_count:
                    class_file_count[class_folder.name].append(num_files_in_class)
                else:
                    class_file_count[class_folder.name] = [num_files_in_class]

    # Je Gruppe die aktuelle Oberkante des zugehörigen Balkens (mehrere Balken werden übereinander gestackt)
    for c in class_file_count:
        bottom_y = [0 for _ in range(0, class_file_count[c].__len__())]
        break

    for c in class_file_count:  # Gehe alle Einträge im Dictionary durch
        # Füge Balken hinzu (über alle vorherigen) und labele ihn entsprechend seiner Klasse
        ax.bar(groups, class_file_count[c], bottom=bottom_y, label=str(c))
        # Erhöhe bisherige Oberkante der Balken (elementweise Addition)
        bottom_y = list(map(add, bottom_y, class_file_count[c]))

        for i in range(groups.__len__()):  # Füge Beschriftung hinzu (Dateianzahl mittig in Balken)
            ax.text(i - 0.05, bottom_y[i] - class_file_count[c][i] / 2, class_file_count[c][i])
    for i in range(groups.__len__()):  # Füge Beschriftung hinzu (gesamte Dateianzahl oberhalb von Balken je Gruppe)
        ax.text(i - 0.05, bottom_y[i] * 1.02, bottom_y[i], weight='bold')
    ax.legend()
    plt.title(directory.name)
    plt.show()


def create_k_folds(src: Path, dst: Path, k=5):
    """Erstellt aus Datensatz 'src' einen 'k'-Fold Versuchsaufbau in 'dst'"""
    # Erstelle Ziel-Verzeichnis, falls es nicht existiert
    dst.mkdir(parents=True, exist_ok=True)
    # Liste mit Pfaden in alle Klassenordner
    classes = [c for c in src.iterdir()]
    num_per_class = []  # Anzahl an Bildern je Klasse
    images = []  # 2D-Array mit Liste aller Bild-Pfade je Klasse
    for class_ in classes:
        images_per_class = [i for i in class_.iterdir()]
        num_per_class.append(images_per_class.__len__())
        images.append(images_per_class)
    print("Anzahl Bilddateien je Klasse insgesamt:")
    print(num_per_class)
    print(20 * "-")
    num_per_fold = []  # Anzahl an Bildern je Fold (durchschnittlich, gerundet)
    num_leftover = []  # übrig gebliebene Bilder-Anzahlen je Klasse, die nicht ganz auf Folds aufgeteilt werden können
    for ci in range(0, classes.__len__()):
        num_per_fold.append(num_per_class[ci] // k)  # Integer-Division (rundet ab)
        num_leftover.append(num_per_class[ci] % k)  # Übrig gebliebene, nicht glatt aufteilbare Bilderanzahl je Klasse

    # 'num_leftover' über alle Folds gleichmäßig aufteilen, sodass alle Folds am Ende gleich groß sind
    folds = []  # "Plan" für das Einteilen in Folds. Enthält je Fold und Klasse die geplante Bildanzahl (noch ungenau)
    for i in range(0, k):
        folds.append(num_per_fold.copy())  # Füge durchschnittliche Anzahl je Fold in den "Plan" ein

    fill_into_fold = 0  # Fold in den das nächste überschüssige Bild eingefügt werden soll
    for i in range(0, classes.__len__()):  # Gehe durch alle Klassen
        while num_leftover[i] != 0:  # So lange wie es überschüssige Bilder gibt
            folds[fill_into_fold][i] += 1  # Füge es in 'fill_into_fold' ein (im Plan)
            num_leftover[i] -= 1
            # Gehe zum nächsten Fold (zyklisch), in den dann das nächste überschüssige Bild eingefügt wird
            fill_into_fold = (fill_into_fold + 1) % k
    # Jetzt wurden alle überzähligen Bilder berücksichtigt und möglichst gleichmäßig über alle Folds verteilt
    print("Anzahl Bilddateien je Fold:")
    for fi in range(folds.__len__()):
        print("Fold " + str(fi + 1) + "\t" + str(folds[fi]))

    # Bilder je Klasse zufällig durchmischen
    for i in range(0, classes.__len__()):
        random.shuffle(images[i])
    # Bilder in "single_folds" in einzelne Folds abspeichern
    if (dst / "single_folds").exists():
        print("Verzeichnis mit Folds existiert schon... Breche ab!")
        exit(1)
    print("Erstelle einzelne Folds unter" + str(dst / "single_folds"))
    for class_index in range(0, classes.__len__()):  # Gehe alle Klassen durch
        print("Klasse " + str(class_index + 1) + "...")
        start_copying_from = 0  # Untergrenze zum Kopieren (je Klasse, startet bei 0)
        for fold_index in range(0, k):  # Gehe alle Folds durch
            # Gehe (entsprechend dem Plan) viele Bilder in 'images' in der passenden Klasse durch
            for src_img_path in images[class_index][
                                start_copying_from: start_copying_from + folds[fold_index][class_index]]:
                # und speichere sie im aktuellen Fold
                dst_path = dst / "single_folds" / ("fold" + str(fold_index + 1)) / classes[class_index].name
                dst_path.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_img_path, dst_path / src_img_path.name)
            # untere Grenze für Index für nächsten Fold hochzählen
            start_copying_from += folds[fold_index][class_index]
    # Mische einzelne Folds (jeweils 1/k des Datensatzes) zu richtiger k-Fold-CrossValidation zusammen
    mix_single_folds_to_exps(dst / "single_folds", dst / "exps", k)


def mix_single_folds_to_exps(src_single_folds: Path, dst_exps: Path, k):
    """Mischt einzelne Folds (1/k des Datensatzes) richtige k-Fold-Cross-Validation zusammen"""
    # Jetzt existieren die 5 Folds in 'dst'/single_folds alle einzeln und müssen noch jeweils (k-1):1 zu k verschiedenen
    # Train- und Testsplits zusammengesetzt werden
    for number_of_fold_for_test in range(1, k + 1):  # Jeder Fold wird einmal Testsplit
        dst_path = dst_exps / ("exp" + str(number_of_fold_for_test))
        print("Erstelle " + str(dst_path))
        dst_path_train = dst_path / "train"
        # Kopiere den kompletten Fold in den Testsplit
        shutil.copytree(src_single_folds / ("fold" + str(number_of_fold_for_test)), dst_path / "test")
        for number_of_fold_for_train in range(1, k + 1):  # Alle anderen Folds werden Trainsplit
            if number_of_fold_for_train == number_of_fold_for_test:
                continue
            # copy_tree aus "distutils", weil "shutil" nicht in bereits existierende Ordner (mit Dateien) kopieren kann
            copy_tree(str(src_single_folds / ("fold" + str(number_of_fold_for_train))), str(dst_path_train))


def draw_random_trainsize(src: Path, dst: Path, train_size: int):
    """Erstellt im aktuellen Fold einen Ordner train_X, der X Prozent der Trainingsdaten von 'src' enthält."""
    print(src.parent.name + " mit " + str(train_size) + "Prozent...")
    for exp in src.iterdir():  # Alle Folds durchgehen
        print(exp.name + "...")
        copy_from = exp / "train_100"  # Kopiere aus 100 prozentigem Trainsplit
        for klasse in copy_from.iterdir():
            imgs_per_class = [img for img in klasse.iterdir()]
            # Kopiere JE KLASSE nur 'train_size' Prozent der Bilder
            random.shuffle(imgs_per_class)
            imgs_to_copy = imgs_per_class[0: round(imgs_per_class.__len__() * train_size / 100) + 1]
            dst_path = exp / ("train_" + str(train_size)) / klasse.name
            dst_path.mkdir(exist_ok=True, parents=True)
            for img_to_copy in imgs_to_copy:
                shutil.copyfile(img_to_copy, dst_path / img_to_copy.name)

# zufällige Subsets mit 10, 20, 50 oder 80 Prozent der Trainingsdaten je Datensatz erstellen

# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-tropic10/exps", _WORK_DIR / "datasets_exps/pawara-tropic10/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-tropic10/exps", _WORK_DIR / "datasets_exps/pawara-tropic10/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-tropic10/exps", _WORK_DIR / "datasets_exps/pawara-tropic10/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-tropic10/exps", _WORK_DIR / "datasets_exps/pawara-tropic10/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-monkey10/exps", _WORK_DIR / "datasets_exps/pawara-monkey10/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-monkey10/exps", _WORK_DIR / "datasets_exps/pawara-monkey10/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-monkey10/exps", _WORK_DIR / "datasets_exps/pawara-monkey10/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/pawara-monkey10/exps", _WORK_DIR / "datasets_exps/pawara-monkey10/", 80)

#draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds3/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds3/", 10)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds3/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds3/", 20)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds3/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds3/", 50)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds3/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds3/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds5/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds5/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds5/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds5/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds5/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds5/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds5/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds5/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds10/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds10/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds10/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds10/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds10/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds10/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds10/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds10/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds15/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds15/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds15/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds15/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds15/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds15/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/swedishLeaves3folds15/exps", _WORK_DIR / "datasets_exps/swedishLeaves3folds15/", 80)


#draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic3/exps", _WORK_DIR / "datasets_exps/tropic3/", 10)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic3/exps", _WORK_DIR / "datasets_exps/tropic3/", 20)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic3/exps", _WORK_DIR / "datasets_exps/tropic3/", 50)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic3/exps", _WORK_DIR / "datasets_exps/tropic3/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic5/exps", _WORK_DIR / "datasets_exps/tropic5/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic5/exps", _WORK_DIR / "datasets_exps/tropic5/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic5/exps", _WORK_DIR / "datasets_exps/tropic5/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic5/exps", _WORK_DIR / "datasets_exps/tropic5/", 80)


# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic10/exps", _WORK_DIR / "datasets_exps/tropic10/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic10/exps", _WORK_DIR / "datasets_exps/tropic10/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic10/exps", _WORK_DIR / "datasets_exps/tropic10/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic10/exps", _WORK_DIR / "datasets_exps/tropic10/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic20/exps", _WORK_DIR / "datasets_exps/tropic20/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic20/exps", _WORK_DIR / "datasets_exps/tropic20/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic20/exps", _WORK_DIR / "datasets_exps/tropic20/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/tropic20/exps", _WORK_DIR / "datasets_exps/tropic20/", 80)

#draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant3/exps", _WORK_DIR / "datasets_exps/agrilplant3/", 10)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant3/exps", _WORK_DIR / "datasets_exps/agrilplant3/", 20)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant3/exps", _WORK_DIR / "datasets_exps/agrilplant3/", 50)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant3/exps", _WORK_DIR / "datasets_exps/agrilplant3/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant5/exps", _WORK_DIR / "datasets_exps/agrilplant5/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant5/exps", _WORK_DIR / "datasets_exps/agrilplant5/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant5/exps", _WORK_DIR / "datasets_exps/agrilplant5/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant5/exps", _WORK_DIR / "datasets_exps/agrilplant5/", 80)

# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant10/exps", _WORK_DIR / "datasets_exps/agrilplant10/", 10)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant10/exps", _WORK_DIR / "datasets_exps/agrilplant10/", 20)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant10/exps", _WORK_DIR / "datasets_exps/agrilplant10/", 50)
# draw_random_trainsize(_WORK_DIR / "datasets_exps/agrilplant10/exps", _WORK_DIR / "datasets_exps/agrilplant10/", 80)


#draw_random_trainsize(_WORK_DIR / "datasets_exps/cifar10/exps", _WORK_DIR / "datasets_exps/cifar10/", 10)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/cifar10/exps", _WORK_DIR / "datasets_exps/cifar10/", 20)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/cifar10/exps", _WORK_DIR / "datasets_exps/cifar10/", 50)
#draw_random_trainsize(_WORK_DIR / "datasets_exps/cifar10/exps", _WORK_DIR / "datasets_exps/cifar10/", 80)

# ---------------------------------------------------------------------------------------------------------------------
# Aus "rohem" Datensatz den Versuchsaufbau für k-Fold-Cross-Validation erstellen

# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "tropic20", dst=_WORK_DIR / "datasets_exps" / "tropic20", k=5)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "tropic10", dst=_WORK_DIR / "datasets_exps" / "tropic10", k=5)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "tropic5", dst=_WORK_DIR / "datasets_exps" / "tropic5", k=5)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "tropic3", dst=_WORK_DIR / "datasets_exps" / "tropic3", k=5)

# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "agrilplant10", dst=_WORK_DIR / "datasets_exps" / "agrilplant10", k=5)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "agrilplant5", dst=_WORK_DIR / "datasets_exps" / "agrilplant5", k=5)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "agrilplant3", dst=_WORK_DIR / "datasets_exps" / "agrilplant3", k=5)


# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "cifar10", dst=_WORK_DIR / "datasets_exps" / "cifar10", k=5)

# 3-Fold-Cross-Validation für swedishLeaves, dann händisch jeweils train und test tauschen (Ziel: 1/3 Train, 2/3 Test)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "swedishLeaves15", dst=_WORK_DIR / "datasets_exps" / "swedishLeaves15", k=3)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "swedishLeaves10", dst=_WORK_DIR / "datasets_exps" / "swedishLeaves10", k=3)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "swedishLeaves5", dst=_WORK_DIR / "datasets_exps" / "swedishLeaves5", k=3)
# create_k_folds(src=_WORK_DIR / "datasets" / "raw" / "swedishLeaves3", dst=_WORK_DIR / "datasets_exps" / "swedishLeaves3folds3", k=3)


# ---------------------------------------------------------------------------------------------------------------------
# Aufteilung in Folds und Train/Test visualisieren

# visualize_folds(_WORK_DIR / "datasets_exps/pawara-tropic10")
# visualize_folds(_WORK_DIR / "datasets_exps/pawara-monkey10")
# visualize_folds(_WORK_DIR / "datasets_exps/pawara-umonkey10" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "tropic20" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "tropic10" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "tropic5" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "tropic3" / "exps")

# visualize_folds(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds15" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds10" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds5" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds3" / "exps")

# visualize_folds(_WORK_DIR / "datasets_exps" / "agrilPlant10" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "agrilPlant5" / "exps")
# visualize_folds(_WORK_DIR / "datasets_exps" / "agrilPlant3" / "exps")

# ---------------------------------------------------------------------------------------------------------------------
# erstellte Folds auf Gültigkeit überprüfen

# verify(_WORK_DIR / "datasets_exps/pawara-tropic10", full_scan=True)
# verify(_WORK_DIR / "datasets_exps/pawara-monkey10", full_scan=True)
# verify(_WORK_DIR / "datasets_exps" / "pawara-umonkey10" / "exps", full_scan=True)

# verify(_WORK_DIR / "datasets_exps" / "tropic20" / "exps", full_scan=True)
# verify(_WORK_DIR / "datasets_exps" / "tropic10" / "exps", full_scan=True)
# verify(_WORK_DIR / "datasets_exps" / "tropic5" / "exps", full_scan=True)
# verify(_WORK_DIR / "datasets_exps" / "tropic3" / "exps", full_scan=True)

# verify(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds15" / "exps", full_scan=True, k=3)
# verify(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds10" / "exps", full_scan=True, k=3)
# verify(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds5" / "exps", full_scan=True, k=3)
# verify(_WORK_DIR / "datasets_exps" / "swedishLeaves3folds3" / "exps", full_scan=True, k=3)

# verify(_WORK_DIR / "datasets_exps" / "agrilplant10" / "exps", full_scan=True)
# verify(_WORK_DIR / "datasets_exps" / "agrilplant5" / "exps", full_scan=True)
# verify(_WORK_DIR / "datasets_exps" / "agrilplant3" / "exps", full_scan=True)

# verify(_WORK_DIR / "datasets_exps" / "cifar10" / "exps", full_scan=True)
