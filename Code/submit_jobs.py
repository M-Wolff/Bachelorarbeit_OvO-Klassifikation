import os

_OUTPUT = "./outputs/"
_BATCH_FILE_NAME = "start_training.sh"

def run_all_dataset(dataset_name: str):
    """Startet einen Trainingsdurchlauf für jede Kombination an Parametern für einen bestimmten Datensatz"""
    os.system("mkdir " + _OUTPUT)  # Output-Ordner erstellen (SLURM-Ausgaben werden dort gespeichert)
    # Alle Netzvarianten (inception-pawara existiert nur für Tensorflow)
    for net_type in ["resnet", "inception", "inception-pawara"]:
        for is_ovo in [False, True]:  # OvA und OvO
            for is_finetune in [False, True]:  # Scratch und Finetune
                for fold in ["exp1", "exp2", "exp3", "exp4", "exp5"]:  # Alle Folds durchgehen
                    for train_percent in [10, 20, 50, 80, 100]:  # Alle Train-Prozentsätze durchgehen
                        # abgeleitete Parameter entsprechend setzen
                        learning_rate = 0.0001 if is_finetune else 0.001  # LR abhängig von Scratch / Finetune
                        epochs = 100 if is_finetune else 200  # Epochen abhängig von Scratch / Finetune
                        img_size = 224 if net_type == "resnet" else 299  # Bildgröße wie im Code von Pawara

                        # Baue Model-String
                        if net_type == "resnet":
                            net_type_s = "R"
                        elif net_type == "inception-pawara":
                            net_type_s = "IP"
                        elif net_type == "inception":
                            net_type_s = "I"
                        current_model_string = dataset_name + "," + str(img_size) + "," + (
                            "OvO" if is_ovo else "OvA") + "," + net_type_s + "," + (
                                                   "F" if is_finetune else "S") + "," + str(
                            train_percent) + "," + str(epochs) + "," + str(fold)
                        current_model_string = current_model_string.replace(",", "_")

                        # Baue Befehl
                        # Starte start_training.sh mit entsprechenden Job-Namen und Output-Datei und übergebe
                        # die vorher erstellten Parameter an die Bash-Datei
                        submit_job_command = "sbatch --job-name %s --output %s " % (
                            current_model_string, _OUTPUT + current_model_string + ".txt")
                        submit_job_command += _BATCH_FILE_NAME + " "
                        submit_job_command += str(dataset_name) + " "
                        submit_job_command += str(fold) + " "
                        submit_job_command += str(is_ovo) + " "
                        submit_job_command += str(net_type) + " "
                        submit_job_command += str(is_finetune) + " "
                        submit_job_command += str(train_percent) + " "
                        submit_job_command += str(epochs) + " "
                        submit_job_command += str(learning_rate) + " "
                        submit_job_command += str(img_size)


                        print("starte Training von: %s mit dem Befehl:" % current_model_string)
                        print(submit_job_command)
                        print(100 * "-")
                        os.system(submit_job_command)


#run_all_dataset("pawara-tropic10")
#run_all_dataset("pawara-monkey10")
#run_all_dataset("pawara-umonkey10")

#run_all_dataset("tropic3")
#run_all_dataset("tropic5")
#run_all_dataset("tropic10")
#run_all_dataset("tropic20")

#run_all_dataset("agrilplant3")
#run_all_dataset("agrilplant5")
#run_all_dataset("agrilplant10")

#run_all_dataset("swedishLeaves3folds3")
#run_all_dataset("swedishLeaves3folds5")
#run_all_dataset("swedishLeaves3folds10")
#run_all_dataset("swedishLeaves3folds15")

#run_all_dataset("cifar10")
