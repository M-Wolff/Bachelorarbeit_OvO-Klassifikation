# Torch-Implementierung auf Grundlage der TensorFlow Implementierung von Pornntiwa Pawara (https://www.ai.rug.nl/~p.pawara/dataset.php  -> Tropic Dataset -> source code -> main.py)

import numpy as np
import argparse
from datetime import datetime
import pickle
import sys
from pathlib import Path

import torch
import torchvision
import torchsummary

# _WORK_DIR = Path("G://Bachelorarbeit")
_WORK_DIR = Path("/scratch/tmp/m_wolf37/Bachelorarbeit/")
_DATASET_DIR = _WORK_DIR / "datasets_exps"
# _DATASET_DIR = Path("/scratch/tmp/m_wolf37/Bachelorarbeit/datasets_exps")

init_learning_rate = 0.001  # initiale Learning Rate (wird fuer Finetune passend überschrieben)
_BATCH_SIZE = 16
_OVO_MATRIX_TRANSPOSED = None  # OvO-Matrix (transponiert)
_DATA_AUGMENTATION = True
_NUM_CLASSES = None
# _DEVICE = torch.device("cpu")
# Verwende Cuda (GPU)
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Converter:
    """Konvertiert ein Label (integer) zu OvO-Kodierung
    Als Klasse umgesetzt, da der DataLoader mit mehrere Threads arbeitet und so die OvO-Matrix leichter als
    Klassenvariable gespeichert werden kann"""

    def __init__(self, ovo_matrix_transposed):
        self._OVO_MATRIX_TRANSPOSED = ovo_matrix_transposed

    def convert_label_to_ovo(self, label):
        ovo_encoded_label = self._OVO_MATRIX_TRANSPOSED[:, label]
        return ovo_encoded_label


def ovo_crossentropy_loss(y_true, y_pred):
    """    Berechnet die OvO Crossentropy nach der Formel aus dem Paper von Pawara et al."""
    # Bei OvO wird als Aktivierungsfunktion 'tanh' verwendet. Diese produziert Werte aus (-1, 1)
    # Auf Wertebereich [0,1] hochskalieren (eigentlich möchte man (0,1) erreichen um später im Logarithmus
    # keine undefinierten Werte zu erhalten, aber wegen numerischen Problemen sind auch 0 und 1 denkbare Werte)
    y_true_scaled = (y_true + 1.0) / 2.0
    y_pred_scaled = (y_pred + 1.0) / 2.0

    # Wertebereich von y_pred_scaled von [0,1] auf [0.00001, 0.99999] einschränken wegen Logarithmen. Näherung an (0,1)

    zeroes = torch.zeros_like(y_pred_scaled)  # Tensor mit gleicher Dimension wie 'y_pred_scaled' bestehend aus nur 0en
    # Alle kleineren Werte als 0.00001 in 'y_pred_scaled' auf 0.00001 setzen (untere Schranke für Wertebereich)
    y_pred_scaled = torch.where(y_pred_scaled < 0.00001, zeroes + 0.00001, y_pred_scaled)
    # Alle größeren Werte als 0.99999 in 'y_pred_scaled' auf 0.99999 setzen (obere Schranke für Wertebereich)
    y_pred_scaled = torch.where(y_pred_scaled > 0.99999, zeroes + 0.99999, y_pred_scaled)

    # J_{OvO} aus Pawara et al. anwenden
    loss = - torch.mean(y_true_scaled * torch.log(y_pred_scaled) + (1 - y_true_scaled) * torch.log(1 - y_pred_scaled))
    return loss


def ovo_encoding_to_label(encoded_prediction):
    """Berechne aus OvO-Kodierung wieder ein Label (Integer)"""
    # Tensor aus OvO-Matrix machen
    ovo_matrix_tensor = torch.from_numpy(_OVO_MATRIX_TRANSPOSED)
    # Auf das Device laden
    ovo_matrix_tensor = ovo_matrix_tensor.to(_DEVICE)
    ovo_matrix_tensor = ovo_matrix_tensor.float()
    # Durch Multiplikation bekommt man aus der OvO-Kodierung ein Klassenlabel (s. Paper von Pawara)
    # Matrix und kodierter Vektor sind vertauscht, da Matrix transponiert
    y_pred_one_hot = torch.tensordot(encoded_prediction.float(), ovo_matrix_tensor, dims=1)
    # Ziehe aus dem One-Hot kodierten Wahrscheinlichkeitsvektor das argmax heraus
    pred_class = torch.argmax(y_pred_one_hot, dim=1).long()
    return pred_class


def get_ovo_matrix():
    """Berechnet die OvO-Kodierungsmatrix passend zu globaler Variable _NUM_CLASSES"""
    global _OVO_MATRIX_TRANSPOSED
    np.set_printoptions(threshold=sys.maxsize)
    # Liste mit allen Klassifikatoren, gespeichert als Tupel (a,b) -> Dieser Klassifikator unterscheidet
    # Klasse a vs Klasse b
    classifier_pair = []
    # Baue Liste mit Klassifikatoren
    for lower_limit in range(2, _NUM_CLASSES + 1):
        for i in range(0, _NUM_CLASSES - lower_limit + 1):
            classifier_pair.append((lower_limit - 1, lower_limit + i))
    print("Paare von Klassifikatoren für die Kodierungs-Matrix:")
    print(classifier_pair)
    # Anzahl an Klassifikatoren sollte mit dem Ergebnis der Formel aus Pawara et al. übereinstimmen
    assert classifier_pair.__len__() == _NUM_CLASSES * (_NUM_CLASSES - 1) // 2

    # Erstelle leere Matrix [_NUM_CLASSES  X  Anzahl Klassifikatoren]
    matrix = np.zeros((_NUM_CLASSES, _NUM_CLASSES * (_NUM_CLASSES - 1) // 2), dtype=float)
    # Fülle Matrix abhängig von aktueller Zeilennummer (True Class)
    for row in range(matrix.__len__()):
        for col in range(matrix[row].__len__()):
            # Hole Klassifikator (Paar von zu trennenden Klassen) aus Klassifikator Liste
            classifier_one, classifier_two = classifier_pair[col]
            # (Paare von zu trennenden Klassen fangen bei 1 an, row und col bei 0)
            # Wenn True-Class nicht vom aktuellen Klassifikator (Spalte) getrennt wird, lasse 0 stehen
            if classifier_one != row + 1 and classifier_two != row + 1:
                continue
            # Wenn 1. Klasse von aktuellem Klassifikator der True-Class entspricht, fülle Zelle mit 1
            elif classifier_one == row + 1 and classifier_two != row + 1:
                matrix[row][col] = 1
            # Wenn 2. Klasse von aktuellem Klassifikator der True-Class entspricht, fülle Zelle mit -1
            elif classifier_one != row + 1 and classifier_two == row + 1:
                matrix[row][col] = -1
            else:
                # Sollte nie passieren
                print("Fehler! Kodierungs-Matrix falsch berechnet")
                exit(12)
    # Transponiere die Matrix (macht später die Berechnungen einfacher)
    _OVO_MATRIX_TRANSPOSED = matrix.transpose()
    print("Kodierungs-Matrix für OvO:")
    print(_OVO_MATRIX_TRANSPOSED)
    print(20 * "-")
    return _OVO_MATRIX_TRANSPOSED


def get_dataloader(dataset_name: str, train_percent: int, fold_name: str, img_size: int, is_ovo: bool,
                   data_augmentaion: bool):
    """Erstelle Data-Loader für Train und Testdaten des geforderten Datensatzes mit entsprechenden Parametern"""
    num_workers = 6  # Mehrere Threads für Data-Loader (evtl. etwas mehr Speicherverbrauch aber bessere Performance)

    # Nach dem Original-Code von Pawara et al. soll von den Trainings- UND Testdaten
    # der Mittelwert der Trainingsdaten abgezogen werden
    # Transformiere Bild auf passende Größe und mache einen Tensor daraus
    mean_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((img_size, img_size)),  # Bildgröße anpassen (224x224 bzw. 299x299)
         torchvision.transforms.ToTensor()])  # daraus einen Tensor machen
    # Lade Trainings-Daten
    mean_train_data = torchvision.datasets.ImageFolder(
        _DATASET_DIR / dataset_name / "exps" / fold_name / ("train_" + str(train_percent)), transform=mean_transforms,
        target_transform=None)
    # Mittelwerte der Trainingsbilder (jeweils Tupel: (R, G, B))
    means = []
    # Gehe alle Bilder durch
    for X, y in mean_train_data:
        # Füge in means den Durchschnitt je Farbkanal hinzu
        means.append((X.mean(dim=(1, 2)) / X.shape[0]).numpy())
    # Mache aus der Liste an Tupeln ein Array
    means = np.asarray(means)
    # Dann einen Tensor
    means = torch.Tensor(means)
    # Berechne Mittelwert der Pixel über alle Bilder hinweg, Ergebnis ist ein einziges Tupel (R, G, B)
    mean = torch.mean(means, dim=0)
    # std Abweichung soll (1.0, 1.0, 1.0) sein, da dann die gleiche Berechnung wie bei Pawara et al. reproduziert wird
    std = torch.ones(3)

    # Jetzt steht in mean der gesuchte Mittelwert der Pixel der Trainingsdaten je Farbkanal (Durch ToTensor() schon
    # auf [0...1] normalisiert)

    # Wenn Data-Augmentation betrieben werden soll
    if data_augmentaion:
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((img_size, img_size)),  # Bildgröße anpassen (224x224 bzw. 299x299)
             torchvision.transforms.RandomHorizontalFlip(),  # zufällige horizontale Spiegelung
             # Bild bis zu 10% horizontal und vertikal shiften
             torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
             torchvision.transforms.ToTensor(),  # daraus einen Tensor machen (skaliert zu 0...1)
             # Normalisieren wie bei Pawara (Mittelwert der Train-Daten von Train- UND Testdaten abziehen)
             # dabei ist mean schon auf 0...1 skaliert, also muss ToTensor zuerst kommen
             torchvision.transforms.Normalize(mean, std)])
    else:  # Ohne Data-Augmentation
        transforms = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((img_size, img_size)),  # Bildgröße anpassen (224x224 bzw. 299x299)
             torchvision.transforms.ToTensor(),  # daraus einen Tensor machen
             # Normalisieren wie bei Pawara (Mittelwert der Train-Daten von Train- UND Testdaten abziehen)
             # dabei ist mean schon auf 0...1 skaliert, also muss ToTensor zuerst kommen
             torchvision.transforms.Normalize(mean, std)])
    if is_ovo:  # Falls OvO Kodierung angewendet werden soll
        # Erstelle Transformations-Funktion für Label (Label -> OvO Kodierung)
        label_converter = Converter(_OVO_MATRIX_TRANSPOSED)
        transforms_label = label_converter.convert_label_to_ovo
    else:  # ansonsten Label nicht um-kodieren
        transforms_label = None
    # Lade Daten mit entsprechender Batch-Size und oben definierten Transformationen
    train_data = torchvision.datasets.ImageFolder(
        _DATASET_DIR / dataset_name / "exps" / fold_name / ("train_" + str(train_percent)), transform=transforms,
        target_transform=transforms_label)
    test_data = torchvision.datasets.ImageFolder(_DATASET_DIR / dataset_name / "exps" / fold_name / "test",
                                                 transform=transforms, target_transform=transforms_label)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=_BATCH_SIZE, shuffle=True,
                                               num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=_BATCH_SIZE, shuffle=True,
                                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def train(dataset: str, fold: str, img_size: int, is_ovo: bool, net_type: str, epochs: int, is_finetune: bool,
          train_percent: int, learning_rate: int, extra_info=""):
    global init_learning_rate, _OVO_MATRIX_TRANSPOSED, _NUM_CLASSES

    start = datetime.now()  # Startzeit
    # übergebene Parameter auflisten
    print(20 * "-" + "Parameter für das Training" + 20 * "-")
    print("Datensatz: %s" % dataset)
    print("Fold: %s" % fold)
    print("Bildgröße: %s" % img_size)
    print("Kodierung: %s" % ("OvO" if is_ovo else "OvA"))
    print("Netz: %s" % net_type)
    print("Epochen: %s" % epochs)
    print("Gewichte: %s" % ("Finetune" if is_finetune else "Scratch"))
    print("Prozentsatz des Trainingssplits: %s" % train_percent)
    print("Initiale Learning-Rate: %f" % learning_rate)
    print(66 * "-")

    init_learning_rate = learning_rate
    # Klassenanzahl aus Datensatz-Name ableiten (Zahl am Ende des Datensatz-Namens ist Klassenanzahl)
    last_digits = 0
    for c in dataset[::-1]:
        if c.isdigit():
            last_digits += 1
        else:
            break
    _NUM_CLASSES = int(dataset[dataset.__len__() - last_digits:])
    print("Anzahl an Klassen: %s" % _NUM_CLASSES)

    # OvO-Matrix erstellen
    if _OVO_MATRIX_TRANSPOSED is None:
        get_ovo_matrix()

    # Anzahl an Klassifikatoren abhängig von Klassenanzahl und Kodierung (OvO oder OvA) berechnen
    num_classificators = _NUM_CLASSES if not is_ovo else (_NUM_CLASSES * (_NUM_CLASSES - 1)) // 2

    # Netz laden
    print("Bereite Netz vor")
    if net_type.lower() in ["resnet", "resnet50", "r"]:  # Resnet50
        net_type = "R"  # Für Logging der Ergebnisse
        model = torchvision.models.resnet50(pretrained=is_finetune)  # Lade Netz
        model.fc = torch.nn.Linear(2048, num_classificators)  # Passe Dimension der letzten Schicht an
        if is_ovo:  # Falls OvO
            model = torch.nn.Sequential(model, torch.nn.Tanh())  # Füge Tanh() Funktion an das Netz an
            # WICHTIG: bei OvA darf KEINE Softmax() Funktion angefügt werden, da die CCE-Loss Funktion selbst schon
            # Softmax anwendet. Das Netz gibt im Falle von OvO die OvO-kodierte Prediction aus, im Falle
            # von OvA werden die "rohen" logits ausgegeben. Um Wahrscheinlichkeitsvektor zu bekommen, muss
            # Softmax() dann explizit von Hand angewendet werden
    elif net_type.lower() in ["inception", "inceptionv3", "i"]:  # InceptionV3
        net_type = "I"  # Für Logging der Ergebnisse
        # Lade Netz (ohne die auxilary Logits, werden eh nicht verwendet)
        model = torchvision.models.inception_v3(pretrained=is_finetune, aux_logits=False)
        model.fc = torch.nn.Linear(2048, num_classificators)  # Passe Dimension der letzten Schicht an
        if is_ovo:  # Falls OvO
            model = torch.nn.Sequential(model, torch.nn.Tanh())  # Füge Tanh() Funktion an das Netz an
            # WICHTIG: bei OvA darf KEINE Softmax() Funktion angefügt werden, da die CCE-Loss Funktion selbst schon
            # Softmax anwendet. Das Netz gibt im Falle von OvO die OvO-kodierte Prediction aus, im Falle
            # von OvA werden die "rohen" logits ausgegeben. Um Wahrscheinlichkeitsvektor zu bekommen, muss
            # Softmax() dann explizit von Hand angewendet werden
    # Bereite Modell für Berechnung auf GPU vor (evtl. Datentypen anpassen, generell das Modell auf die GPU laden)
    model.cuda()
    print("Netz vorbereitet")
    print("Netz:")
    print(model)
    print(50 * "-")
    print(torchsummary.summary(model, input_size=(3, img_size, img_size)))  # Printe schöne Übersicht des Modells
    # Adam-Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    # Schrittweiser LR-Scheduler (alle 50 Epochen wird lr *= 0.1 gerechnet, wie bei Pawara et al.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    print("Lade Daten")
    # Daten laden
    train_loader, test_loader = get_dataloader(dataset, train_percent, fold, img_size, is_ovo, _DATA_AUGMENTATION)

    # Logging des Trainingsverlaufs (wie bei TF das erzeugte history dict)
    history = {}
    history["acc"] = []
    history["val_acc"] = []
    history["loss"] = []
    history["val_loss"] = []
    history["lr"] = []

    print("Starte Training")
    # Loss-Funktion-Instanz für OvA (CCE)
    loss_fn_ova = torch.nn.CrossEntropyLoss()
    # Softmax um aus Logits einen Wahrscheinlichkeitsvektor zu machen (OvA)
    softmax_fn = torch.nn.Softmax(dim=1)
    # Schleife über alle Epochen
    for epoch_number in range(epochs):
        # LR loggen
        history["lr"].append(lr_scheduler.get_last_lr()[0])
        # Für aktuelle Epoche die Statistiken zurücksetzen
        correct_train = 0  # korrekt klassifizierte Daten im Train-Split
        correct_test = 0  # korrekt klassifizierte Daten im Test-Split
        num_train = 0  # Daten insgesamt im Train-Split
        num_test = 0  # Daten insgesamt im Test-Split
        train_losses = []  # Loss-Werte im Train-Split (je Batch einer)
        test_losses = []  # Loss-Werte im Test-Split (je Batch einer)
        # Train-Loop
        model.train()  # Modell in train-Modus setzen
        # Alle Batches im train_loder durchgehen
        for batch, (X, y) in enumerate(train_loader):
            # Daten auf GPU kopieren
            X = X.to(_DEVICE)
            y = y.to(_DEVICE)
            # Netzausgabe erzeugen
            pred = model(X)
            if is_ovo:
                # Loss berechnen (für OvO)
                loss = ovo_crossentropy_loss(y_true=y, y_pred=pred)
                # Label und Predictions von OvO-Kodierung zu Integer konvertieren
                pred = ovo_encoding_to_label(pred)
                y_label = ovo_encoding_to_label(y)
            else:
                # Loss berechnen (für OvA)
                loss = loss_fn_ova(pred, y)
                # Softmax auf Netzausgabe (rohe Logits) anwenden, dann Argmax über Dimension 1
                # (Dimension 0 sind die verschiedenen Bilder, _BATCH_SIZE-viele)
                pred = torch.argmax(softmax_fn(pred), dim=1)
                y_label = y
            # Gradienten aus letzter Epoche löschen (aktuelle würden sonst dazu addiert werden)
            optimizer.zero_grad()
            # Gradienten für diese Epoche setzen
            loss.backward()
            # Gewichte im Netz entsprechend einen Schritt optimieren
            optimizer.step()
            # Loss für aktuellen Batch merken
            train_losses.append(float(loss))
            # Für Accuracy-Berechnung
            correct_train += int((pred == y_label).float().sum())
            num_train += len(y)
        # Mittelwert aller Batches loggen
        history["loss"].append(np.array(train_losses).mean())
        # Evaluation-Loop
        model.eval()  # Netz in eval() Modus
        with torch.no_grad():
            for batch, (X, y) in enumerate(test_loader):
                # Daten auf GPU kopieren
                X = X.to(_DEVICE)
                y = y.to(_DEVICE)
                # Netzausgabe erzeugen
                pred = model(X)
                if is_ovo:
                    # Loss berechnen (für OvO)
                    loss = ovo_crossentropy_loss(y_true=y, y_pred=pred)
                    # Label und Predictions von OvO-Kodierung zu Integer konvertieren
                    pred = ovo_encoding_to_label(pred)
                    y_label = ovo_encoding_to_label(y)
                else:
                    # Loss berechnen (für OvA)
                    loss = loss_fn_ova(pred, y)
                    # Softmax auf Netzausgabe (rohe Logits) anwenden, dann Argmax über Dimension 1
                    # (Dimension 0 sind die verschiedenen Bilder, _BATCH_SIZE-viele)
                    pred = torch.argmax(softmax_fn(pred), dim=1)
                    y_label = y
                # Loss für aktuellen Batch merken
                test_losses.append(float(loss))
                # Für Accuracy-Berechnung
                correct_test += int((pred == y_label).float().sum())
                num_test += len(y)
        # Learning-Rate Scheduler eine Schritt weiter setzen (am Ende jeder Epoche)
        lr_scheduler.step()
        # Metriken loggen
        history["val_loss"].append(np.array(test_losses).mean())
        history["val_acc"].append(correct_test / num_test * 100)
        history["acc"].append(correct_train / num_train * 100)
        print("Epoche {} / {}, Loss: {}, Acc: {} ({} von {}), Val_Loss: {}, Val_Acc: {} ({} von {}), Lr: {}".format(
            epoch_number, epochs, history["loss"][-1], history["acc"][-1], correct_train, num_train,
            history["val_loss"][-1], history["val_acc"][-1], correct_test, num_test, history["lr"][-1]))

    # Training fertig, erzeuge und speichere finale Predictions und Label ab, Logge Ergebnisse des Trainings
    end = datetime.now()  # Endzeit
    elapsed = (end - start).total_seconds() / 60  # vergangene Zeit (für Trainings inkl. Laden des Datensatzes)
    # Verzeichnis um alles zu diesem Modell zu speichern
    current_model_string = dataset + "," + str(img_size) + "," + (
        "OvO" if is_ovo else "OvA") + "," + net_type + "," + ("F" if is_finetune else "S") + "," + str(
        train_percent) + "," + str(epochs) + "," + str(fold) + "," + str(extra_info)

    # mehrere Folds zum gleichen Netz zusammenfassen in Unterordner
    current_model_folder_name = extra_info + "," + dataset + "," + str(img_size) + "," + (
        "OvO" if is_ovo else "OvA") + "," + net_type + "," + ("F" if is_finetune else "S") + "," + str(
        train_percent) + "," + str(epochs)
    save_dir = _WORK_DIR / "saved_results" / current_model_folder_name.replace(",", "_").replace(".", ",") / str(fold)
    if save_dir.exists():
        print("Der Ordner für die aktuelle Konfiguration existiert bereits!")
        print(str(save_dir))
        exit(13)
    save_dir.mkdir(parents=True)

    # Durchlaufe Trainingsdaten (ohne die Data Augmentation -> FALSE)
    train_loader, test_loader = get_dataloader(dataset, train_percent, fold, img_size, is_ovo, False)
    # zu speichernde Numpy-Arrays (Netzausgaben, Predictions, Ground-truth)
    predictions_test = None
    predictions_train = None
    raw_net_output_test = None
    raw_net_output_train = None
    true_classes_test = None
    true_classes_train = None
    with torch.no_grad():  # Performance-Gewinn da keine Gradienten benötigt werden
        model.eval()  # Modell im eval() Modus
        # Schleifen wie im Training oben, Füge alle Ausgaben je Batch zu einem großen Array zusammen
        for batch, (X, y) in enumerate(test_loader):
            X = X.to(_DEVICE)
            y = y.to(_DEVICE)
            pred = model(X)

            raw_net_output_test = concat_array(raw_net_output_test, pred)

            if is_ovo:
                pred_label = ovo_encoding_to_label(pred)
                y_label = ovo_encoding_to_label(y)
            else:
                pred_label = torch.argmax(pred, dim=1)
                y_label = y
            predictions_test = concat_array(predictions_test, pred_label)
            true_classes_test = concat_array(true_classes_test, y_label)

        for batch, (X, y) in enumerate(train_loader):
            X = X.to(_DEVICE)
            y = y.to(_DEVICE)
            pred = model(X)

            raw_net_output_train = concat_array(raw_net_output_train, pred)

            if is_ovo:
                pred_label = ovo_encoding_to_label(pred)
                y_label = ovo_encoding_to_label(y)
            else:
                pred_label = torch.argmax(pred, dim=1)
                y_label = y
            predictions_train = concat_array(predictions_train, pred_label)
            true_classes_train = concat_array(true_classes_train, y_label)
    # Speichere die 6 Numpy Arrays in save_dir

    np.save(save_dir / "predicted_classes_train.npy", predictions_train)
    np.save(save_dir / "predicted_classes_test.npy", predictions_test)

    np.save(save_dir / "raw_net_output_train.npy", raw_net_output_train)
    np.save(save_dir / "raw_net_output_test.npy", raw_net_output_test)

    np.save(save_dir / "true_classes_train.npy", true_classes_train)
    np.save(save_dir / "true_classes_test.npy", true_classes_test)

    np.set_printoptions(threshold=sys.maxsize)
    print(50 * "-")
    print("Train predictions")
    print(predictions_train)
    print("True Classes Train")
    print(true_classes_train)
    print(50 * "-")
    print("Test Predictions")
    print(predictions_test)
    print("True Classes Test")
    print(true_classes_test)
    # Speichere die history als pickle-Datei
    with open(save_dir / "historySave.dat", 'wb') as pickle_file:
        pickle.dump(history, pickle_file)

    # Schreibe in Log

    with open(save_dir.parent.parent / "allModelsLog.txt", "a+") as log_file:
        log_string = "%s,%.2f,%s,%s," % (
            torch.cuda.get_device_name(0), elapsed, _BATCH_SIZE, learning_rate) + current_model_string + "," + str(
            history["loss"][-1]) + "," + str(history["acc"][-1]) + "," + str(history["val_loss"][-1]) + "," + str(
            history["val_acc"][-1])
        log_file.write(log_string + "\n")
        print(log_string)
    print("Finale Accuracy (Train): " + str(history["acc"][-1]))
    print("Finaler Loss (Train): " + str(history["loss"][-1]))
    print("Finale Accuracy (Test): " + str(history["val_acc"][-1]))
    print("Finaler Loss (Test): " + str(history["val_loss"][-1]))


def concat_array(old, to_add):
    """Hängt ein Array 'to_add' an ein bestehendes Array 'old' an
    Falls 'old' nicht existiert, wird es erstellt"""
    # Hole anzufügendes Array zur CPU (war evtl. auf GPU) und trenne es aus dem Berechnungsgraphen von Torch
    # (Gradientenberechnung, ...)
    to_add_det = to_add.cpu().detach().numpy()
    if old is None:
        old = np.array(to_add_det)
    else:
        old = np.concatenate((old, to_add_det))
    return old


def str2bool(s: str):
    """Konvertiert einen String in einen Boolean"""

    if s.lower() in ["true", "yes", "1"]:
        return True
    elif s.lower() in ["false", "no", "0"]:
        return False
    else:
        print("Fehler: Boolean erwartet! %s ist nicht als Boolean interpretierbar" % s)
        exit(1)


def parse_arguments():
    p = argparse.ArgumentParser(description="Training mit übergebenen Parametern")
    p.add_argument("--dataset", type=str, help="Name des Datensatzes in " + str(_DATASET_DIR))
    p.add_argument("--fold", type=str, help="Name des Foldes (z.B. \"exp1\")")
    p.add_argument("--img_size", type=int, help="Größe des Bildes in Pixeln")
    p.add_argument("--is_ovo", type=str2bool, help="True für OvO Ansatz")
    p.add_argument("--net_type", type=str, help="Name des Netzes (resnet, inception-pawara oder inception)")
    p.add_argument("--epochs", type=int, help="Anzahl an zu trainierenden Epochen")
    p.add_argument("--is_finetune", type=str2bool,
                   help="True für finetuning des Netzes, False für scratch-training")
    p.add_argument("--train_percent", type=int, help="Prozentsatz des zu verwendenden Train-Splits")
    p.add_argument("--learning_rate", type=float,
                   help="Initiale Learning-Rate (z.B. 0.001 oder 0.0001)")
    p.add_argument("--extra_info", type=str, help="Kommentar / Markierung für Ergebnisse im"
                                                  "CSV-Log (z.B. verwendete TF Version)")
    args = p.parse_args()

    # Prüfe ob alle Argumente angegeben wurden
    if args.dataset is None:
        print("Parameter --dataset wird benötigt!")
        exit(2)
    if args.fold is None:
        print("Parameter --fold wird benötigt!")
        exit(3)
    if args.img_size is None:
        print("Parameter --img_size wird benötigt!")
        exit(4)
    if args.is_ovo is None:
        print("Parameter --is_ovo wird benötigt!")
        exit(5)
    if args.net_type is None:
        print("Parameter --net_type wird benötigt!")
        exit(6)
    if args.epochs is None:
        print("Parameter --epochs wird benötigt!")
        exit(7)
    if args.is_finetune is None:
        print("Parameter --is_finetune wird benötigt!")
        exit(8)
    if args.train_percent is None:
        print("Parameter --train_percent wird benötigt!")
        exit(9)
    if args.learning_rate is None:
        print("Parameter --learning_rate wird benötigt!")
        exit(10)
    if args.extra_info is None:
        extra_info = ""
    else:
        extra_info = args.extra_info

    # Trainiere mit angegebenen Parametern
    train(dataset=args.dataset, fold=args.fold, img_size=args.img_size, is_ovo=args.is_ovo, net_type=args.net_type,
          epochs=args.epochs, is_finetune=args.is_finetune, train_percent=args.train_percent,
          learning_rate=args.learning_rate, extra_info=extra_info)


if __name__ == "__main__":
    print("CUDA: ")
    print(torch.cuda.is_available())
    parse_arguments()
    # train("tropic3", "exp1", 224, False, "resnet", 10, False, 100, 0.001)
