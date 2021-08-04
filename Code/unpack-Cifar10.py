# Entpackt den als Pickle-Datei abgespeicherten Cifar10-Datensatz (https://www.cs.toronto.edu/~kriz/cifar.html)
# und speichert den Inhalt als einzelne Bilder ab
import pickle
from pathlib import Path
import numpy as np
from PIL import Image


_BASE_PATH = Path("G://Bachelorarbeit/datasets/raw/cifar10/cifar-10-batches-py")
_OUTPUT_PATH = _BASE_PATH.parent / "output"
_OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

def unpickle(file):
    """Lade Inhalt aus Pickle-Datei"""
    with open(str(file), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


meta = unpickle(_BASE_PATH / "batches.meta")  # Lese Meta-Daten
_LABEL_NAMES = [str(label, "UTF-8") for label in meta[b"label_names"]]  # Label-Namen bestimmen (Klassennamen)
# Ausgabepfad für jede Klasse anlegen
for l in _LABEL_NAMES:
    p = _OUTPUT_PATH / l
    p.mkdir(exist_ok=True, parents=True)


def save_images(pickle_dict):
    """Speichert die im Dictionary gegebenen Bilder als echte Bilddateien ab"""
    for img_index in range(len(pickle_dict[b"data"])):  # Gehe alle Bilder durch
        img = pickle_dict[b"data"][img_index]  # extrahiere Bilddaten-Array
        filename = str(pickle_dict[b"filenames"][img_index], "UTF-8")  # Hole Dateinamen (inkl. Dateiendung)
        label = pickle_dict[b"labels"][img_index]  # Hole Klassennummer
        # Einzelne Kanäle (R,G,B) waren direkt hintereinander gespeichert, hole einzelne Kanäle einzeln heraus
        c1 = img[0:1024]  # R
        c2 = img[1024:2048]  # G
        c3 = img[2048:3072]  # B
        # Arrays (1x1024) in 32x32 reshapen
        c1 = np.reshape(c1, (32,32))
        c2 = np.reshape(c2, (32,32))
        c3 = np.reshape(c3, (32,32))
        # Farbkanäle aufeinander stacken (3D-Array entsteht)
        rgb_img_arr = np.dstack((c1,c2,c3))
        # Bild aus 3D-Array erstellen
        rgb_img = Image.fromarray(rgb_img_arr)
        # und im Ordner passend zur Klasse abspeichern
        rgb_img.save(_OUTPUT_PATH / _LABEL_NAMES[label] / filename)

# Daten sind in 6 Pickle-Dateien aufgeteilt (je 10000 Bilder)
b1 = unpickle(_BASE_PATH / "data_batch_1")
b2 = unpickle(_BASE_PATH / "data_batch_2")
b3 = unpickle(_BASE_PATH / "data_batch_3")
b4 = unpickle(_BASE_PATH / "data_batch_4")
b5 = unpickle(_BASE_PATH / "data_batch_5")
b6 = unpickle(_BASE_PATH / "test_batch")

save_images(b1)
save_images(b2)
save_images(b3)
save_images(b4)
save_images(b5)
save_images(b6)