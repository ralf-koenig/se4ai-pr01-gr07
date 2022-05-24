# SE4AI Project 1, Language Identification, Group 07, Summer Term 2022 

## Group Members
* Jonas Greim, student in Master Computer Science,  term
* Cuong Vo Ta, student in Master  Computer Science, term
* Ralf König, student in Master Data Science, 1st term

## Architecture Overview
* **Web App server** at Heroku: https://se4ai-pr01-gr07.herokuapp.com/
  * Web server needs environment variable `DATABASE_URL` to connect to the database to send feedback to.
  * Is configured for the Heroku servers at Heroku => Settings => Config Vars.
* **Database server:** Postgres DB server at Heroku. Login via https://dashboard.heroku.com/apps/se4ai-pr01-gr07
* **Code Repository:** public GitHub repository: https://github.com/ralf-koenig/se4ai-pr01-gr07
* **CI/CD pipelines:** GitHub workflow `languague-pipeline-cicd` to:
  * run tests on the inference engine and classification quality before deployment
  * build docker container
  * deploy build docker to Heroku using secrets in GitHub => Settings => Secrets => Actions secrets

## Run web app locally
```bash
pip install -r requirements.txt

# set connection URI for the Postgres DB including user, pw, host, database
export DATABASE_URL=postgres://user:password@host.com/dbname

cd se4ai-pr01-gr07

# if streamlit is on $PATH
streamlit run app.py  

# else
python -m streamlit run app.py
```

## Run tests locally (also part of CI/CD pipeline at GitHub server)

```bash
cd se4ai-pr01-gr07
python -m pytest -vvv test_app.py -s
```

### Workflows resulting from GitHub Actions

see [Github Project - All workflows](https://github.com/ralf-koenig/se4ai-pr01-gr07/actions)