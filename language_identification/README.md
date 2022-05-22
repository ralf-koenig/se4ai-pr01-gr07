# Folder `language_identification`

### Use Case

Language identification of short text samples (around 50 words) for the three languages: 
German, English, Spanish.

### Contents
This folder contains Python code for the machine learning part of the task.

The code is heavily based on code from the Jupyter notebook `language_identification_tf.ipynb`
but reorganizes the code to a Python module structure with defined functions.

The code is split into training.py and inference.py. Common aspects needed by both of them are in 
constants.py.

### constants.py
This module encapsulates common constants for model training and inference pipeline.

### training.py

This module encapsulates the model building functions, this means
applying supervised learning on a classification problem:
start with labeled text samples. "label" is the true language of the sample,
expressed as ISO code like en, es, de.

The model building functions will take care of the steps needed to
process the text samples and to train a proper KERAS model for
this classifier.

### inference.py

This module encapsulates the inference functions, this means:
input a text sample in one of the three languages
and the inference functions will take care of the steps needed to
process the text sample and then to apply the pre-trained model.
The classifier responds with the most probable language (out of the three) 
and its probability.