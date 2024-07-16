# Progetto DL24: YesWeKAN - Istruzioni

In questo progetto abbiamo fatto uso di librerie custom, ovvero di classi implementate da noi in file `.py` nella directory del progetto. Questo preclude l'utilizzo di Colab per eseguire i notebook in maniera automatica. È possibile eseguire i notebook su Colab dopo essersi assicurati di aver introdotto nella directory di Colab tutti i file necessari all'esecuzione e dopo aver installato tutte le librerie utilizzate nel progetto. Abbiamo inoltre notato che la riproducibilità può essere dipendente dalla versione di Python utilizzata: per questo ci teniamo a precisare che abbiamo svolto tutti gli esperimenti su Python `3.12.4`

## Installazione delle librerie

Per installare le librerie è possibile eseguire il comando `pip install -r requirements.txt` dopo essersi assicurati di aver importato come working directory la cartella del progetto.

## Ordine dei notebook

Il progetto è suddiviso in tre notebook, che consigliamo di visionare nel seguente ordine:
1) Introduzione KAN - In questo notebook introdurremo il progetto, i concetti matematici alla base delle reti KAN, e ne mostreremo il funzionamento
2) Analisi, confronto e interpretabilità - In questa parte descriveremo il dataset scelto, addestreremo i modelli, li confronteremo e ne valueteremo i risultati, successivamente ci occuperemo della trasparenza algoritmica
3) Tuning degli iperparametri - In quest'ultima parte presentiamo il codice che abbiamo usato per aiutarci a trovare le reti ottimali
