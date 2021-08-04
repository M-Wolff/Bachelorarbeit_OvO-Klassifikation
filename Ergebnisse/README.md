# Ergebnisse
## Logdatei für alle Durchläufe
[allModelsLog.txt](allModelsLog.txt) beinhaltet eine Zeile pro trainiertem Netz.

Die Spaltennamen lauten in dieser Reihenfolge:

- **GPU-Name**: Name des zum Training verwendeten GPU-Modells
- **Dauer**: Dauer des Trainings (inkl. Laden des Datensatzes) in Minuten
- **Batch-Size**: Größe eines Batches (Standard: 16)
- **Learning-Rate**: initiale Learning-Rate zum Trainieren
    - 0.001 für *Scratch*
    - 0.0001 für *Finetune*
- **Datensatz**: Name des Datensatzes (z.B. *tropic10*)
- **Bildgröße**: Größe der Eingabebilder in Pixeln
    - 224 für *Resnet*
    - 299 für *Inception V3*
- **Klassifikationsschema**: Name des verwendeten Klassifikationsschemas
    - OvO für *One-vs-One*
    - OvA für *One-vs-All*
- **Netz**: Name des Netzes 
    - R für "Resnet"
    - I für "Inception V3"
    - IP für "Inception V3 - Pawara" mit Pawara's Änderungen an den letzten Schichten
- **Gewichte**: 
    - S für "Scratch" (zufällig initialisiert)
    - F für "Finetune" (vortrainierte Gewichte auf ImageNet)
- **Prozentsatz Train**: Prozentsatz der Trainingsdaten, die tatsächlich zum Training verwendet werden sollen.
Subsets wurden ein einziges Mal zufällig erstellt und für jeden Trainingsdurchlauf wiederverwendet, damit alle Durchläufe auf den gleichen Daten
trainiert werden und somit besser verglichen werden können
- **Epochen**: zu trainierende Epochen
    - 100 bei *Finetune*
    - 200 bei *Scratch*
- **Fold**: Name des Foldes (z.B. exp1 für den 1. Fold einer 3 oder 5-Fold-Crossvalidation)
- **Extra Informationen**: extra Informationen zum Training
    - "TF1-13-1-detTS": TensorFlow 1.13.1
    - "TF2-4-1-detTS": TensorFlow 2.4.1
    - "torch": Torch
- **Loss Train**: Loss-Wert auf den Trainingsdaten (verwendete Loss-Funktion ist abhängig vom *Klassifikationsschema*)
- **Accuracy Train**: Accuracy auf den Trainingsdaten in Prozent
- **Loss Test**: Loss-Wert auf den Testdaten (verwendete Loss-Funktion ist abhängig vom *Klassifikationsschema*)
- **Accuracy Test**: Accuracy auf den Testdaten in Prozent

## Detaillierte Ergebnisse je Durchlauf
[Hier](https://uni-muenster.sciebo.de/s/YApuzVCRHb5JRZR) sind alle detaillierten Daten als Zip-Archiv abrufbar.
In dem Zip-Archiv befindet sich für jede Kombination von den oben aufgelisteten Parametern ein Ordner mit Namen

`<Extra Informationen>_<Datensatz>_<Bildgröße>_<Klassifikationsschema>_<Netz>_<Gewichte>_<Prozentsatz Train>_<Epochen>`

Innerhalb eines solchen Ordners befindet sich je ein Ordner für jeden der 5 bzw. 3 Folds (z.B. `exp1` für den 1. Fold).
Darin befinden sich jeweils 7 Dateien:
- **raw_net_output**: rohe Ausgabewerte aus dem Netz nach der letzten Epoche (standardmäßig Softmax-Wahrscheinlichkeitsvektor bzw. OvO-kodierte Ausgabe. Bei mit Torch trainierten Netzen besteht die Ausgabe aus den Logits-Rohwerten direkt bevor Softmax angewendet wird)
    - raw_net_output_train.npy: abgespeichertes Numpy-Array der rohen Netzausgabe aller verwendeter Trainingsdaten
    - raw_net_output_test.npy: abgespeichertes Numpy-Array der rohen Netzausgabe aller verwendeter Testdaten
- **predicted_classes**: vorhergesagte Klassennummer je Sample
    - predicted_classes_train.npy: abgespeichertes Numpy-Array der vorhergesagten Klassennummer je Sample der Trainingsdaten
    - predicted_classes_test.npy: abgespeichertes Numpy-Array der vorhergesagten Klassennummer je Sample der Testdaten
- **true_classes**: tatsächliche Klassennummer je Sample
    - true_classes_train.npy: abgespeichertes Numpy-Array der tatsächlichen Klassennummer je Sample der Trainingsdaten
    - true_classes_test.npy: abgespeichertes Numpy-Array der tatsächlichen Klassennummer je Sample der Testdaten
- **historySave.dat**: Beim Training nach jeder einzelnen Epoche erzielte Metriken / Werte. Mit Pickle abgespeichertes Dictionary mit Listen für
    - Learningrate
    - Loss (Train & Test)
    - Accuracy (Train & Test)