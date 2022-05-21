import os
from dotenv import load_dotenv
# https://www.psycopg.org/docs/usage.html
import psycopg2

load_dotenv()
conn = psycopg2.connect(
    database=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST'),
)

# Open cursor to perform database operation
cur = conn.cursor()

# create table
# cur.execute('''CREATE TABLE feedback(sentence VARCHAR(511), probabilityRight SMALLINT, guessedLanguage VARCHAR(63), userFeedback BOOLEAN, id INT GENERATED ALWAYS AS IDENTITY);''')
# conn.commit()

# get data
# cur.execute('''SELECT * FROM test3;''')
# print(cur.fetchall())

# set data
# cur.execute("INSERT INTO test3 (name, age) VALUES ('Jonas', 24)")
# conn.commit()

# Close communications with database
cur.close()
conn.close()
