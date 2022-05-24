"""
Language identification for the three languages: German, English, Spanish.

This module scraping.py is the scraping component for wikipedia articles.
It has a built-in list of URLS of articles that exists on each of
the wikipedia sites: de.wikipedia.org, en.wikipedia.org, es.wikipedia.org.
These are all downloaded, cleaned mildly, separated to three sets for
training, validation, test and saved to CSV files in the current folder.
"""

import random
import requests
from bs4 import BeautifulSoup
import re
import csv
import nltk.tokenize
import math
import os.path
import sys

# append the "language_identification" folder to PYTHONPATH
# where Python searches for packages
sys.path.append('..')
# this way you can start scraping.py with
# python scraping.py inside the language_identification folder

# common constants that must match between training and inference
import language_identification.constants as constants

nltk.download('punkt')


# link for extracting html data
def get_data(url):
    print(f"Downloading {url} ...")
    r = requests.get(url)
    return r.text


def get_text_from_html(url):
    html_data = get_data(url)
    soup = BeautifulSoup(html_data, 'html.parser')
    clean_text = ""
    for data in soup.find_all("p"):
        clean_text = clean_text + data.get_text()
    return clean_text


def remove_spanish_citation(spanish_sentence):
    return spanish_sentence.replace("\u200b\u200b", " ").replace("\u200b", " ").replace("«", " ").replace("»", " ")


def remove_short_sentence(split_sentence, language_prefix):
    # Append short cut sentence to prior sentence and remove it from dataset
    i = 1
    remove_elements = []
    while i < len(split_sentence):
        if len(split_sentence[i].split()) < 7:
            split_sentence[i - 1] += split_sentence[i]
            remove_elements.append(i)
        split_sentence[i] = language_prefix + split_sentence[i]
        i += 1

    for element in sorted(remove_elements, reverse=True):
        del split_sentence[element]

    return split_sentence


def generate_sentences_from_wikipedia(url, language):
    clean_text = get_text_from_html(url)
    # Remove citation remarks
    pattern = r'\[.*?\]'
    clean_text = re.sub(pattern, '', clean_text)
    # Remove weird spanish citation remarks
    if language == "es":
        clean_text = remove_spanish_citation(clean_text)
    # Split text into sentences
    split_sentence = nltk.tokenize.sent_tokenize(clean_text)
    # Create prefix for training data
    language_prefix = language + ","

    return remove_short_sentence(split_sentence, language_prefix)


def get_language_of_article(url):
    result = re.search("https://(.*).wikipedia.org", url)
    return result.group(1)


def generate_sentence_from_list_of_articles(urls):
    sentences = []
    for url in urls:
        language = get_language_of_article(url)
        for sentence in generate_sentences_from_wikipedia(url, language):
            sentences.append(sentence)
    return sentences


def specify_language_of_wikipedia_article(urls, language):
    collected_urls = []
    for url in urls:
        collected_urls.append("https://" + language + url)
    return collected_urls


def generate_csv(sentences, filename):
    filename = os.path.join(constants.DATA_DIRECTORY, "wikipedia", filename)
    with open(filename, 'w', encoding='utf-8', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        file_writer.writerow(['labels', 'text'])
        for sentence in sentences:
            file_writer.writerow([sentence[:2], sentence[3:]])
    print(f"{filename}: {len(sentences)} text samples written.")


def remove_long_rows(sentences):
    for sentence in sentences:
        if '\n' in sentence:
            sentences.remove(sentence)
    return sentences


def remove_weird_rows(sentences):
    for sentence in sentences:
        if not sentence.startswith("de") or sentence.startswith("en") or sentence.startswith("es"):
            sentences.remove(sentence)
    return sentences


def split_data(sentences):
    train_test_ratio = 0.8
    train_size = int(math.ceil(len(sentences)) * train_test_ratio)
    test_size = int(train_size + (len(sentences) - train_size) / 2)
    training_list = sentences[0:train_size]
    test_list = sentences[train_size + 1:test_size]
    validation_list = sentences[test_size + 1:len(sentences)]
    return training_list, test_list, validation_list


if __name__ == "__main__":

    partial_urls = [".wikipedia.org/wiki/Linus_Torvalds",
                    ".wikipedia.org/wiki/Leipzig",
                    ".wikipedia.org/wiki/Madrid",
                    ".wikipedia.org/wiki/Sigmund_Freud",
                    ".wikipedia.org/wiki/Pablo_Picasso",
                    ".wikipedia.org/wiki/Wolfgang_Amadeus_Mozart",
                    ".wikipedia.org/wiki/Marie_Curie",
                    ".wikipedia.org/wiki/Stephen_Hawking",
                    ".wikipedia.org/wiki/Galileo_Galilei",
                    ".wikipedia.org/wiki/Immanuel_Kant",
                    ".wikipedia.org/wiki/James_Clerk_Maxwell",
                    ".wikipedia.org/wiki/Angela_Merkel"]

    all_urls = []
    languages = list(constants.lang_labels.keys())
    for language_id in languages:
        all_articles = specify_language_of_wikipedia_article(partial_urls, language_id)
        for article in all_articles:
            all_urls.append(article)

    all_sentences = generate_sentence_from_list_of_articles(all_urls)
    all_sentences = remove_long_rows(all_sentences)
    all_sentences = remove_weird_rows(all_sentences)
    random.shuffle(all_sentences)
    train, test, valid = split_data(all_sentences)
    generate_csv(train, "train.csv")
    generate_csv(test, "test.csv")
    generate_csv(valid, "valid.csv")
