# Code
- **[evaluate.py](evaluate.py)**: Methoden zur Auswertung und Visualisierung der Ergebnisse
- **[k_fold.py](k_fold.py)**: Methoden zur Erstellung, Überprüfung und Visualisierung von k-Fold Cross Validations
- **[start_training.sh](start_training.sh)**: Startskript zum Starten von Jobs in SLURM auf Palma II
- **[submit_jobs.py](submit_jobs.py)**: Erstellt Kombinationen von Trainingsparametern und startet für jede einzelne Kombination einen Job, indem die Parameter in der richtigen Reihenfolge an das Startskript übergeben werden
- **[train.py](train.py)**: Training der Netze mit TensorFlow 1.13.1 oder 2.4.1. Parameter für das Training müssen übergeben werden (s. `python3 train.py --help`)
- **[train_torch.py](train_torch.py)**: Training der Netze mit PyTorch 1.9.0. Parameter für das Training müssen übergeben werden (s. `python3 train_torch.py --help`)
- **[unpack-Cifar10.py](unpack-Cifar10.py)**: Entpackt Cifar10 (mit Pickle gespeichertes Dictionary mit den Bildern als 1D-Array, [hier](https://www.cs.toronto.edu/~kriz/cifar.html) verfügbar) in einzelne Farb-Bilder der Auflösung 32x32 Pixel
- **[requirements.txt](requirements.txt)**: Python Requirements zum Ausführen der Python-Programme.