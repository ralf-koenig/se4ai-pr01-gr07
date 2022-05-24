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
start with labeled text samples. "label" is the true language of the sample,
expressed as ISO code like `en`, `es`, `de`.

### Model building

The model building functions will take care of the steps needed to
process the text samples and to train a proper KERAS model for
this classifier.

Use with:
``$ python training.py --train_model huggingface``
``$ python training.py --train_model wikipedia``

It also contains a component to get the user feedback from a DB to a local CSV file.

### Get feedback records from database

Use with:
``$ python training.py --feedback <filename.csv>``

Output is written to `data/feedback/<filename.csv>`.

## Inference pipeline in `inference.py`

This module encapsulates the inference functions, this means:
input a text sample in one of the three languages
and the inference functions will take care of the steps needed to
process the text sample and then to apply the pre-trained model.
The classifier responds with the most probable language (out of the three) 
and its probability.

This module is imported by `app.py` to use in the GUI.

### Helper in `constants.py`
This module encapsulates common constants for model training and inference pipeline.


## Scraping component in ``scraping.py``

This module is the scraping component for wikipedia articles.
It has a built-in list of URLS of articles that exists on each of
the wikipedia sites: de.wikipedia.org, en.wikipedia.org, es.wikipedia.org.
These are all downloaded, cleaned mildly, separated to three sets for
training, validation, test and saved to CSV files in the current folder.