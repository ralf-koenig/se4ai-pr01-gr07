# https://www.psycopg.org/docs/usage.html
import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    password="postgres",
    host="localhost"
)

# Open cursor to perform database operation
cur = conn.cursor()

cur.execute('''CREATE TABLE test3(name VARCHAR(255), age VARCHAR(255), id INT GENERATED ALWAYS AS IDENTITY);''')
# open the csv file into the table from line 2
with open('test.csv', 'r') as f:
    next(f)  # Skip the header row.
    cur.copy_from(f, 'test3', sep=',', columns=('name', 'age'))  # f=csv , <database name>, Comma-Seperated
    conn.commit()
    conn.close()
f.close()

# Close communications with database
cur.close()
conn.close()
