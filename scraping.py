import requests
from bs4 import BeautifulSoup
import re
import string

if __name__=="__main__":
    # link for extract html data
    def getdata(url):
        r = requests.get(url)
        return r.text

    htmldata = getdata("https://de.wikipedia.org/wiki/Tiger")
    soup = BeautifulSoup(htmldata, 'html.parser')
    clean_text = ""
    for data in soup.find_all("p"):
        clean_text = clean_text + data.get_text()

    clean_text = clean_text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers and new lines
    pattern = r'[0-9]'
    clean_text = re.sub(pattern, '', clean_text).replace("\n", " ")
    print(clean_text)

