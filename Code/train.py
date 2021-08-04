import argparse
from datetime import datetime
import pickle
import sys
from pathlib import Path

import tensorflow as tf
import keras
import numpy as np
import tflearn.data_utils
# Workaround für Modulfehler: tensorflow.python kann später aus irgendwelchen Gründen nicht mehr
# direkt unter diesem Namen aufgerufen werden
# Daher from ... import ... as tfclient um tensorflow.python.client später (als tfclient) noch verwenden zu können
from tensorflow.python import client as tfclient  # Nur wichtig um GPU-Name zu ermitteln

# _WORK_DIR = Path("G://Bachelorarbeit")
_WORK_DIR = Path("/scratch/tmp/m_wolf37/Bachelorarbeit/")
_DATASET_DIR = Path("/scratch/tmp/m_wolf37/Bachelorarbeit/datasets_exps")

init_learning_rate = 0.001
_BATCH_SIZE = 16
_OVO_MATRIX_TRANSPOSED = None
_VERBOSE = True
_DATA_AUGMENTATION = True


def get_learning_rate(epoch):
    """Gibt Learning-Rate abhängig von aktueller Epoche zurück (alle 50 Epochen um 0.1 verringern)"""
    lr = init_learning_rate

    if epoch > 150:
        lr = 0.001 * init_learning_rate
    elif epoch > 100:
        lr = 0.01 * init_learning_rate
    elif epoch > 50:
        lr = 0.1 * init_learning_rate
    print("Epoche %s -> Learning-Rate: %s" % (epoch, lr))
    return lr


def ovo_crossentropy_loss(y_true, y_pred):
    """Berechnet die OvO Crossentropy nach der Formel aus dem Paper von Pawara et al."""
    # Bei OvO wird als Aktivierungsfunktion 'tanh' verwendet. Diese produziert Werte aus (-1, 1)
    # Auf Wertebereich [0,1] hochskalieren (eigentlich möchte man (0,1) erreichen um später im Logarithmus
    # keine undefinierten Werte zu erhalten, aber wegen numerischen Problemen sind auch 0 und 1 denkbare Werte)
    y_true_scaled = (y_true + 1.0) / 2.0
    y_pred_scaled = (y_pred + 1.0) / 2.0

    # Wertebereich von y_pred_scaled von [0,1] auf [0.00001, 0.99999] einschränken wegen Logarithmen. Näherung an (0,1)

    zeroes = tf.zeros_like(y_pred_scaled)  # Tensor mit gleicher Dimension wie 'y_pred_scaled' bestehend aus nur 0en
    # Alle kleineren Werte als 0.00001 in 'y_pred_scaled' auf 0.00001 setzen (untere Schranke für Wertebereich)
    y_pred_scaled = tf.where(y_pred_scaled < 0.00001, zeroes + 0.00001, y_pred_scaled)
    # Alle größeren Werte als 0.99999 in 'y_pred_scaled' auf 0.99999 setzen (obere Schranke für Wertebereich)
    y_pred_scaled = tf.where(y_pred_scaled > 0.99999, zeroes + 0.99999, y_pred_scaled)

    # J_{OvO} aus Pawara et al. anwenden
    log_function = tf.log if tf.__version__ == "1.13.1" else tf.math.log  # flexibel für neue / alte Version
    loss = - tf.reduce_mean(
        y_true_scaled * log_function(y_pred_scaled) + (1 - y_true_scaled) * log_function(1 - y_pred_scaled))
    return loss


def ovo_accuracy_metric(y_true, y_pred):
    """Errechnet die vorhergesagte Klasse aus der OvO-kodierten Netzausgabe (y_pred) und berechnet mit Hilfe der
    erwarteten Klasse (y_true, ebenfalls OvO-kodiert) die Accuracy"""
    # OvO Matrix als Single-Precision float
    single_prec_matrix = _OVO_MATRIX_TRANSPOSED.astype(np.single)
    # One-Hot kodierten Wahrscheinlichkeitsvektor aus OvO-Kodierung berechnen
    y_true_one_hot = tf.tensordot(y_true, single_prec_matrix, axes=1)
    y_pred_one_hot = tf.tensordot(y_pred, single_prec_matrix, axes=1)
    # Klassennummern berechnen (argmax des One-Hot kodierten Wahrscheinlichkeitsvektors)
    true_class = keras.backend.argmax(y_true_one_hot, axis=-1)
    pred_class = keras.backend.argmax(y_pred_one_hot, axis=-1)
    # Zählen, wie oft erwartete und vorhergesagte Klasse übereinstimmen
    correct_pred = keras.backend.equal(true_class, pred_class)
    return keras.backend.mean(correct_pred)


def load_dataset(dataset_name: str, fold_name: str, train_percent: int, is_ovo: bool, img_size: int):
    """Lädt einen Datensatz entsprechend der übergebenen Parameter"""
    # Zu ladendes Verzeichnis
    dir_to_load = _DATASET_DIR / dataset_name / "exps" / fold_name
    # train und test Unterordner
    train_dir = dir_to_load / ("train_" + str(train_percent))
    test_dir = dir_to_load / "test"

    print("Lade Datensatz aus %s" % str(dir_to_load))
    print("Train-Bilder aus %s" % str(train_dir))
    print("Test-Bilder aus %s" % str(test_dir))
    # categorical_labels=True sorgt dafür, dass die Label als One-Hot (bzw. als Zielvektor) kodiert geladen werden
    #                   =False lädt einfach nur die Klassennummer
    x_train, y_train = tflearn.data_utils.image_preloader(train_dir, image_shape=(img_size, img_size), grayscale=False,
                                                          mode="folder", categorical_labels=not is_ovo, normalize=True)
    x_test, y_test = tflearn.data_utils.image_preloader(test_dir, image_shape=(img_size, img_size), grayscale=False,
                                                        mode="folder", categorical_labels=not is_ovo, normalize=True)

    print("Lade Train-Bilder...")
    x_train = np.asarray(x_train)
    print("Lade Test-Bilder...")
    x_test = np.asarray(x_test)
    print("Lade Train-Label...")
    y_train = np.asarray(y_train)
    print("Lade Test-Label...")
    y_test = np.asarray(y_test)

    assert x_train.__len__() == y_train.__len__()
    assert x_test.__len__() == y_test.__len__()

    print("Bilder im Train-Split %s" % x_train.__len__())
    print("Bilder im Test-Split %s" % x_test.__len__())

    # Im kompletten Train-Split (ohne train_size Prozent auszuwählen) liegen eigentlich so viele Dateien:
    complete_train_split_path = train_dir.parent / "train_100"  # voller Trainsplit liegt im train_100 Ordner (100%)
    # Zähle, wie groß der komplette Trainsplit (100%) ist
    orig_number_train = 0
    for klasse in complete_train_split_path.iterdir():
        orig_number_train += [f for f in klasse.iterdir()].__len__()

    print("Trainiere auf %s Prozent des Train-Splits. %s / %s Bildern im Train-Split" % (
        train_percent, x_train.__len__(), orig_number_train))
    print("Testsplit enthält %s Bilder" % x_test.__len__())

    # wie bei Pawara et al. wird der Mittelwert der Pixel im Trainsplit vom Train- und Testsplit abgezogen
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    return x_train, y_train, x_test, y_test


def get_ovo_matrix(num_classes: int):
    """Berechnet die OvO-Kodierungsmatrix passend zu num_classes"""
    global _OVO_MATRIX_TRANSPOSED
    np.set_printoptions(threshold=sys.maxsize)
    # Liste mit allen Klassifikatoren, gespeichert als Tupel (a,b) -> Dieser Klassifikator unterscheidet
    # Klasse a vs Klasse b
    classifier_pair = []
    # Baue Liste mit Klassifikatoren
    for lower_limit in range(2, num_classes + 1):
        for i in range(0, num_classes - lower_limit + 1):
            classifier_pair.append((lower_limit - 1, lower_limit + i))
    print("Paare von Klassifikatoren für die Kodierungs-Matrix:")
    print(classifier_pair)
    # Anzahl an Klassifikatoren sollte mit dem Ergebnis der Formel aus Pawara et al. übereinstimmen
    assert classifier_pair.__len__() == num_classes * (num_classes - 1) // 2

    # Erstelle leere Matrix [_num_classes  X  Anzahl Klassifikatoren]
    matrix = np.zeros((num_classes, num_classes * (num_classes - 1) // 2), dtype=float)
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


def convert_labels_to_ovo(labels: np.array, num_classes: int):
    """Label zu OvO-Kodierung konvertieren"""
    print("Mappe Klassennummer zu OvO-Vektor...")
    ovo_encoded_labels = np.zeros((labels.__len__(), num_classes * (num_classes - 1) // 2))
    for label_index in range(0, labels.__len__()):
        # OvO-Matrix ist transposed. Spalten und Zeilen vertauscht, hole komplette Spalte zu Klassennummer
        ovo_encoded_labels[label_index] = _OVO_MATRIX_TRANSPOSED[:, labels[label_index]]
        if _VERBOSE:
            print("%s gemappt zu " % (labels[label_index] + 1))
            print(ovo_encoded_labels[label_index])
    print(20 * "-")
    return ovo_encoded_labels


def evaluate_model(model, x, y_true, is_ovo, save_dir: Path, train_test: str):
    """Wertet ein übergebenes Modell auf übergebenen Daten aus und gibt Metriken dazu zurück.
    'save_dir' gibt an, wo die Netzausgabe, die erwartete und vorhergesagte Klassennummer als Numpy-Array abgespeichert werden soll
    'train_test' ist lediglich ein String, um abgespeicherte Numpy-Arrays für Train und Test (im gleichen Ordner)
    voneinander zu unterscheiden"""
    np.set_printoptions(threshold=sys.maxsize)
    if is_ovo:  # OvO
        # vorhergesagte Klassennummer zu den Eingabedaten bestimmen
        output_prediction = model.predict(x)
        one_hot_pred = np.matmul(output_prediction, _OVO_MATRIX_TRANSPOSED)
        predicted_classes = np.argmax(one_hot_pred, axis=1)

        # erwartete Klassennummer aus OvO-Kodierung bestimmen
        y_true_one_hot = np.matmul(y_true, _OVO_MATRIX_TRANSPOSED)
        y_true_classes = np.argmax(y_true_one_hot, axis=1)
        # Accuracy berechnen
        correct_predictions = np.equal(predicted_classes, y_true_classes)
        acc = correct_predictions.mean() * 100
        # Loss berechnen (mit OvO-kodierten y_true und y_pred)
        loss = ovo_crossentropy_loss(y_true=y_true, y_pred=output_prediction).eval(session=tf.compat.v1.Session())
    else:  # OvA
        # Loss und Accuracy bestimmen
        loss_acc = model.evaluate(x, y_true)
        acc = loss_acc[1] * 100  # Accuracy an Stelle 1
        loss = loss_acc[0]
        # Zum Abspeichern Netzausgabe, vorhergesagte und erwartete Klassennummer berechnen
        output_prediction = model.predict(x)
        predicted_classes = np.argmax(output_prediction, axis=1)
        y_test_classes = np.argmax(y_true, axis=1)

    # Speichere 'output_prediction', 'predicted_classes' und 'y_test_classes' in 'save_dir' einzeln als Datei ab
    np.save(save_dir / ("raw_net_output_" + train_test + ".npy"), output_prediction)
    np.save(save_dir / ("predicted_classes_" + train_test + ".npy"), predicted_classes)
    np.save(save_dir / ("true_classes_" + train_test + ".npy"), y_test_classes)
    return acc, loss


def train(dataset: str, fold: str, img_size: int, is_ovo: bool, net_type: str, epochs: int, is_finetune: bool,
          train_percent: int, learning_rate: int, extra_info=""):
    """Trainiert ein Netz mit den angegebenen Parametern, wertet es aus und schreibt die Ergebnisse als Numpy-Array
    in einen Ordner bzw. in die Logdatei"""
    global init_learning_rate, _OVO_MATRIX_TRANSPOSED
    start = datetime.now()
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

    # Learning-Rate setzen
    init_learning_rate = learning_rate

    # weights setzen (Scratch oder Pretrained mit Imagenet)
    weights = None
    if is_finetune:
        weights = "imagenet"
    # Klassenanzahl aus Datensatz-Name ableiten (Zahl am Ende des Datensatz-Namens ist Klassenanzahl)
    last_digits = 0
    for c in dataset[::-1]:
        if c.isdigit():
            last_digits += 1
        else:
            break

    num_classes = int(dataset[dataset.__len__() - last_digits:])
    print("Anzahl an Klassen: %s" % num_classes)

    # Verschiedene Netz-Varianten

    if net_type.lower() in ["resnet", "resnet50", "r"]:
        net_type = "R"
        # Erste und letzte Schicht weglassen (include_top=False) und eigene Input-Shape
        model = keras.applications.resnet50.ResNet50(weights=weights, include_top=False,
                                                     input_shape=(img_size, img_size, 3))
        out = model.output
        # vorletzte Schicht wieder herstellen (so wie sie im Original Netz auch wäre)
        out = keras.layers.GlobalAveragePooling2D()(out)
    elif net_type.lower() in ["inception-pawara", "inceptionv3-pawara", "ip"]:
        net_type = "IP"
        # Erste und letzte Schicht weglassen (include_top=False) und eigene Input-Shape
        model = keras.applications.inception_v3.InceptionV3(weights=weights, include_top=False,
                                                            input_shape=(img_size, img_size, 3))

        # Letzte Schichten ändern wie im Code von Pawara et al.
        x = model.output
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.AveragePooling2D(pool_size=(8, 8))(x)
        x = keras.layers.Dropout(0.4)(x)
        out = keras.layers.Flatten()(x)
    elif net_type.lower() in ["inception", "inceptionv3", "i"]:
        net_type = "I"

        # Erste und letzte Schicht weglassen (include_top=False) und eigene Input-Shape
        model = keras.applications.inception_v3.InceptionV3(weights=weights, include_top=False,
                                                            input_shape=(img_size, img_size, 3))
        out = model.output
        # vorletzte Schicht wieder herstellen (so wie sie im Original Netz auch wäre)
        out = keras.layers.GlobalAveragePooling2D()(out)

    else:
        print("Netz %s wird nicht unterstützt" % net_type)
        exit(11)
    # Verzeichnis um alles zu diesem Modell zu speichern
    current_model_string = dataset + "," + str(img_size) + "," + (
        "OvO" if is_ovo else "OvA") + "," + net_type + "," + ("F" if is_finetune else "S") + "," + str(
        train_percent) + "," + str(epochs) + "," + str(fold) + "," + str(extra_info)

    # mehrere Folds zum gleichen Netz zusammenfassen in Unterordner
    current_model_folder_name = extra_info + "," + dataset + "," + str(img_size) + "," + (
        "OvO" if is_ovo else "OvA") + "," + net_type + "," + ("F" if is_finetune else "S") + "," + str(
        train_percent) + "," + str(epochs)
    save_dir = _WORK_DIR / "saved_results" / current_model_folder_name.replace(",", "_").replace(".", ",") / str(fold)
    save_dir_cp = _WORK_DIR / "saved_checkpoints"
    cp_name = str(extra_info) + "," + current_model_string + ".cp"

    if save_dir.exists():
        print("Der Ordner für die aktuelle Konfiguration existiert bereits!")
        print(str(save_dir))
        exit(13)
    save_dir.mkdir(parents=True)
    save_dir_cp.mkdir(parents=True, exist_ok=True)
    optimizer = keras.optimizers.Adam(lr=get_learning_rate(0))

    # Datensatz laden
    x_train, y_train, x_test, y_test = load_dataset(dataset, fold, train_percent, is_ovo, img_size)

    steps_per_epoch = x_train.__len__() // _BATCH_SIZE if x_train.__len__() // _BATCH_SIZE > 0 else 1

    # Data Augmentation (bis zu 10% shiften vertikal und horizontal, horizontal spiegeln)
    if _DATA_AUGMENTATION:
        data_augmentation = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)
        data_augmentation.fit(x_train)

    if is_ovo:
        # Y-Label müssen von Klassennummer (z.B. 5) zu OvO-Vektor kodiert werden
        get_ovo_matrix(num_classes)  # speichert OvO-Matrix für passende Klassenanzahl in globale Variable _OVO_MATRIX
        y_train = convert_labels_to_ovo(y_train, num_classes)
        y_test = convert_labels_to_ovo(y_test, num_classes)

        output_layer_size = (num_classes * (num_classes - 1)) // 2
        # Modell für OvO vorbereiten (tanh() als letzte Schicht im Netz einfügen)
        output_layer = keras.layers.Dense(output_layer_size, kernel_initializer="he_normal", activation="tanh")(out)
        model = keras.models.Model(inputs=model.inputs, outputs=output_layer)
        model.compile(loss=ovo_crossentropy_loss, optimizer=optimizer,
                      metrics=[ovo_crossentropy_loss, ovo_accuracy_metric])
    else:  # OvA
        output_layer_size = num_classes
        # Softmax Schicht am Ende des Netzes einfügen für OvA
        output_layer = keras.layers.Dense(output_layer_size, kernel_initializer="he_normal", activation="softmax")(
            out)
        model = keras.models.Model(inputs=model.inputs, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy', "categorical_crossentropy"])

    checkpoint = keras.callbacks.ModelCheckpoint(filepath=str(save_dir_cp / cp_name), monitor="val_loss",
                                                 verbose=1,
                                                 save_best_only=True)
    callbacks = [checkpoint, keras.callbacks.LearningRateScheduler(get_learning_rate)]

    model.summary()
    # Trainiere Netz (mit oder ohne Data-Augmentation)
    if _DATA_AUGMENTATION:
        history = model.fit_generator(data_augmentation.flow(x_train, y_train, batch_size=_BATCH_SIZE),
                                      validation_data=(x_test, y_test),
                                      epochs=epochs, shuffle=True, workers=1, verbose=1,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=callbacks)  # TODO workers=4 in Pawara, thread safe warning
    else:
        history = model.fit(x=x_train, y=y_train, batch_size=_BATCH_SIZE,
                            validation_data=(x_test, y_test),
                            epochs=epochs, shuffle=True, workers=1, verbose=1,
                            steps_per_epoch=steps_per_epoch,
                            callbacks=callbacks)  # TODO workers=4 in Pawara, thread safe warning
    end = datetime.now()
    elapsed = (end - start).total_seconds() / 60  # benötigte Zeit für das Training (und Laden des Datensatzes)

    # Speichere die history als pickle-Datei
    with open(save_dir / "historySave.dat", 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)

    # Acc und Loss für Test und Train ausrechnen
    acc_test, loss_test = evaluate_model(model, x_test, y_test, is_ovo, save_dir, "test")
    acc_train, loss_train = evaluate_model(model, x_train, y_train, is_ovo, save_dir, "train")
    # Ergebnis in Logdatei schreiben
    with open(save_dir.parent.parent / "allModelsLog.txt", "a+") as log_file:
        log_string = "%s,%.2f,%s,%s," % (
            get_gpu_name(), elapsed, _BATCH_SIZE, learning_rate) + current_model_string + "," + str(
            loss_train) + "," + str(acc_train) + "," + str(loss_test) + "," + str(acc_test)
        log_file.write(log_string + "\n")
        print(log_string)
    print("Finale Accuracy (Train): " + str(acc_train))
    print("Finaler Loss (Train): " + str(loss_train))
    print("Finale Accuracy (Test): " + str(acc_test))
    print("Finaler Loss (Test): " + str(loss_test))


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


def get_gpu_name():
    # Workaround für Modulfehler, s. Imports
    devices = tfclient.device_lib.list_local_devices()
    for device in devices:
        if device.device_type == "GPU":
            device_string = device.physical_device_desc.split(",")[1].replace("name:", "").strip()
            return device_string


if __name__ == "__main__":
    parse_arguments()
