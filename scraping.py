import random
import requests
from bs4 import BeautifulSoup
import re
import csv
import nltk
nltk.download('punkt')
from nltk import tokenize

# link for extracting html data
def getdata(url):
    r = requests.get(url)
    return r.text


def generate_sentences_from_wikipedia(url, language):
    html_data = getdata(url)
    soup = BeautifulSoup(html_data, 'html.parser')
    clean_text = ""
    for data in soup.find_all("p"):
        clean_text = clean_text + data.get_text()

    # Remove citation remarks
    pattern = r'\[.*?\]'
    clean_text = re.sub(pattern, '', clean_text)
    # Split text into sentences
    split_sentence = tokenize.sent_tokenize(clean_text)
    # Create prefix for training data
    language_id = language + ","

    # Append short cut sentence to prior sentence and remove it from dataset
    i = 1
    remove_elements = []
    while i < len(split_sentence):
        if len(split_sentence[i].split()) < 7:
            split_sentence[i - 1] += split_sentence[i]
            remove_elements.append(i)
        split_sentence[i] = language_id + split_sentence[i]
        i += 1

    for element in sorted(remove_elements, reverse=True):
        del split_sentence[element]

    return split_sentence


def get_language_of_article(url):
    result = re.search("https://(.*).wikipedia.org", url)
    return result.group(1)


def generate_sentence_from_list_of_articles(urls):
    all_sentences = []
    for url in urls:
        language_id = get_language_of_article(url)
        for sentence in generate_sentences_from_wikipedia(url, language_id):
            all_sentences.append(sentence)
    return all_sentences


def specify_language_of_wikipedia_article(urls, language_id):
    all_urls = []
    for url in urls:
        all_urls.append("https://" + language_id + url)
    return all_urls


def generate_csv(all_sentences):
    with open('train.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', )
        filewriter.writerow(['labels', 'text'])
        for sentence in all_sentences:
            filewriter.writerow([sentence[:2], sentence[3:]])

if __name__=="__main__":

    urls = [".wikipedia.org/wiki/Linus_Torvalds",
                ".wikipedia.org/wiki/Leipzig",
                ".wikipedia.org/wiki/Madrid",
                ".wikipedia.org/wiki/Sigmund_Freud",
                ".wikipedia.org/wiki/Pablo_Picasso"]

    all_urls = []
    languages = ["de", "en", "es"]
    for language_id in languages:
        all_articles = specify_language_of_wikipedia_article(urls, language_id)
        for article in all_articles:
            all_urls.append(article)

    all_sentences = generate_sentence_from_list_of_articles(all_urls)
    random.shuffle(all_sentences)
    generate_csv(all_sentences)












