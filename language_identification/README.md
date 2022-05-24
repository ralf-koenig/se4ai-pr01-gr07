# Language identification of German, English, Spanish.

This module allows language identification of short text samples (around 50 words).
It comes with training data and models for three languages, but can be extended
and adapted to more languages.

## Package ``language_identification``

This folder contains Python code for the machine learning part of the task.

The code is heavily based on code from the Jupyter notebook `language_identification_tf.ipynb`
but reorganizes the code to a Python module structure with defined functions.

## Training pipeline in `training.py`

This module encapsulates the model building functions, this means
applying supervised learning on a classification problem:
start with labeled text samples. `label` in the `train.csv` files is the true language 
of the sample, expressed as an ISO code such as `en`, `es`, `de`.

### Model building 

The model building functions will take care of the steps needed to
process the text samples and to train a proper KERAS model for
this classifier.

Use with:
```bash
cd language_idenfication

python training.py --train_model huggingface

python training.py --train_model wikipedia
```

Directories are found automatically in `se4ai-pr01-gr07/data/<subdir>` (input csv for training, validation and test) 
and `se4ai-pr01-gr07/model` (model output files of the KERAS layers). 
These directories are configured in `constants.py`. Input files must be named: `train.csv`, `valid.csv` and `test.csv`
for training data, validation data and test data, respectively.

For the target model files, overwrite protection (needs explicit confirmation by user) is enabled. 
Model file output names are hard-coded.  Please rename files in the `model/<subdir>` folders before 
if you want to keep old model files _and_ agree to overwrite a model file.

### Get feedback records from database

In the web app `app.py`, users can provide feedback and suggest the correct language for a text sample.
This feedback is written to the Postgres database.
`training.py` contains a function to get this user feedback from a DB table and write to a local CSV file
in the same format suitable as training input at a later point in time.

Use with:
```bash
cd language_idenfication

export DATABASE_URL=postgres://user:password@host.com/dbname

python training.py --feedback <filename.csv>
```

Output is written to `data/feedback/<filename.csv>`.

## Inference pipeline in `inference.py`

This module encapsulates the inference functions, this means:
input a text sample in one of the three languages
and the inference functions will take care of the steps needed to
process the text sample and then to apply the pre-trained model.
The classifier responds with the most probable language (out of the three) 
and its probability.

This module is imported by `app.py` to apply the inference engine in the GUI `app.py`.

### Common constants in `constants.py`

This module encapsulates common constants for model training and inference pipeline, 
such as directories and training parameters.

## Scraping component in ``scraping.py``

This module is the scraping component for wikipedia articles.
It has a built-in list of URLS of articles that exists on each of
the wikipedia sites: de.wikipedia.org, en.wikipedia.org, es.wikipedia.org.
These are all downloaded, cleaned mildly, separated to three sets for
training, validation, test and saved to CSV files in the current folder.

Use with:
```bash
cd language_idenfication

python scraping.ppy
```

The URLs are hard-coded into the Python file. Articles were selected that have all three language versions:
* a Spanish version from https://es.wikipedia.org
* an English version from https://es.wikipedia.org
* a German version from https://es.wikipedia.org

In our experiments, this approach improved classification results over the dataset from Huggingface despite
fewer training records.