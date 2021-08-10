# Datensätze
Hier sind alle zum Training verwendeten Zusammensetzungen der Datensätze aufgelistet.
Die hier hochgeladenen Textdateien beinhalten die Ausgabe von `tree <Datensatzname>`.

Die Datensätze tropic, agrilplant und (u)monkey sind auf der [Homepage von Pornntiwa Pawara](https://www.ai.rug.nl/~p.pawara/dataset.php#) zu finden.

Die anderen Datensätze sind hier verlinkt: [Cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html), [swedishLeaves](https://www.cvl.isy.liu.se/en/research/datasets/swedish-leaf/)


Für jeden der 15 Datensätze gibt es einen Wurzel-Ordner:
- agrilplant3/
- agrilplant5/
- agrilplant10/
- cifar10/
- pawara-monkey10/
- pawara-tropic10/
- pawara-umonkey10/
- swedishLeaves3folds3/
- swedishLeaves3folds5/
- swedishLeaves3folds10/
- swedishLeaves3folds15/
- tropic3/
- tropic5/
- tropic10/
- tropic20/

In jedem dieser Wurzelverzeichnisse sind 2 Unterordner:
- exps/
- single_folds/

In `single_folds/` befinden sich die 3 bzw. 5 **einzelnen** Folds, also jeweils ein Drittel bzw. ein Fünftel des gesamten Datensatzes (fold1, fold2, ...).

In `exps/` befinden sich die 3 bzw. 5 aus den einzelnen Folds **fertig zusammengemischten** Experimente (exp1, exp2, ...), also die Grundlage für einen Durchlauf der 3- bzw. 5-Fold Cross Validation.

Als Unterordner befinden sich in `exps/` die einzelnen Test- und Trainsplits:
- test/     
- train_10/ 
- train_20/ 
- train_50/ 
- train_80/ 
- train_100/

Dabei bezeichnet die Zahl am Ende der `train_*` Ordner die Train-Size in Prozent (Anteil der Teilmenge vom ganzen Datensatz in Prozent).


Die gesamte Ordnerstruktur für z.B. den agrilplant3 Datensatz sieht also wie folgt aus:
- agrilplant3/
    - exps/
        - exp1/
            - test/
                - apple/
                - banana/
                - grape/
            - train_10/
                - apple/
                - banana/
                - grape/
            - train_20/
                - apple/
                - banana/
                - grape/
            - train_50/
                - apple/
                - banana/
                - grape/
            - train_80/
                - apple/
                - banana/
                - grape/
            - train_100/
                - apple/
                - banana/
                - grape/
        - exp2/
            - ... analog zu exp1/
        - exp3/
            - ... analog zu exp1/
        - exp4/
            - ... analog zu exp1/
        - exp5/
            - ... analog zu exp1/
    - single_folds/
        - fold1/
            - apple/
            - banana/
            - grape/
        - fold2/
            - apple/
            - banana/
            - grape/
        - fold3/
            - apple/
            - banana/
            - grape/
        - fold4/
            - apple/
            - banana/
            - grape/
        - fold5/
            - apple/
            - banana/
            - grape/
