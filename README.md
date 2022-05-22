# SE4AI Project 1, Language Identification, Group 07, Summer Term 2022 

## Group Members
* Jonas Greim, student in Master Computer Science,  term
* Cuong Vo Ta, student in Master  Computer Science, term
* Ralf König, student in Master Data Science, 1st term

## Architecture Overview
* [**Web App server**](https://se4ai-pr01-gr07.herokuapp.com/) at Heroku
* **Database server:** Postgres DB server at Heroku
* **Code Repository:** public [Github repository](https://github.com/ralf-koenig/se4ai-pr01-gr07)
* **CI/CD pipelines:** Github workflow `languague-pipeline-cicd` to build docker container and deploy to Heroku

## Run web app locally
```bash
pip install -r requirements.txt

# if streamlit is on $PATH
streamlit run app.py  

# else
python -m streamlit run app.py
```
