# Folder `data`

This folder contains lists of text samples in CSV files from different sources.
Each source has its own subdirectory and `README.md`.

Each record in a CSV file should hold two fields:
1) `labels`: language label like 'en', 'es', 'de'
2) `text`: text sample containing some words to a few sentences

### Naming conventions
* The training set should be named `train.csv`. It should hold most of the records. Like 70% of the samples.
* The validation set should be named `valid.csv`. It should hold about 20% of the samples.
* The test set  should be named `test.csv`. It should hold about 10% of the samples.
