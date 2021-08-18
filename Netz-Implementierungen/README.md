# Netz-Implementierungen

In diesem Ordner befindet sich die Ausgabe von `model.summary()` für TensorFlow bzw. 

`print(model)` und `torchsummary.summary(model, input_size=(3, img_size, img_size))` für PyTorch.

Dabei wurde das Python-Paket [`torchsummary`](https://github.com/sksq96/pytorch-summary) von sksq96 verwendet.

Es wird immer eine Ausgabeschicht der Größe 10 verwendet, damit die Anzahlen der Parameter miteinander gut verglichen werden können.
Die Textdateien tragen den Namen `<Framework>_<Netztyp>`
