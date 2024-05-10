import pymysql
import os
import re

class cities_db:
    def __init__(self, database_dir, db_password='password', user='root', host='localhost'):
        self.database_dir = database_dir
        self.connection = pymysql.connect(
            host=host,
            user=user,
            password=db_password,
            database='cities',
            cursorclass=pymysql.cursors.DictCursor,
            port=3306  # Change port if necessary
        )
        self.cursor = self.connection.cursor()
        # self.cursor.execute(f"SHOW DATABASES LIKE {database_dir}")
        self.cursor.execute(f"SHOW DATABASES LIKE 'worldcities'")
        # self.cursor.execute(f"SHOW DATABASES")
        result = self.cursor.fetchone()
        print(f'if result: {result is True}')

        if not result:
            # Database does not exist, create it
            with open(os.path.join('data', database_dir + ".sql"), 'r') as file:
                sql_script = file.read()

            # Execute each SQL command
            for idx, sql_command in enumerate(sql_script.split(';')):
                if sql_command.strip():
                    self.cursor.execute(sql_command)

            # rows = self.cursor.execute("use cities")
            # rows = self.cursor.execute("select * from worldcities  where country = 'ALB")
            # print(f'rows: {rows}')


            print(f"Database '{database_dir}' created successfully.")
        else:
            print(f"Database '{database_dir}' already exists.")

# =============================================
def filter_countries(cities_db, countries_list, max_city_per_country):

    condition_str = ''.join([f"'{c}', " for c in countries_list])
    condition_str = '(' + condition_str[:-2] + ')'

    # cities_db.cursor = cities_db.connection.cursor()
    query_str = "SELECT * from (select city, lat, lng, country, population, DENSE_RANK() " \
                "OVER (partition by country ORDER BY population DESC) AS pop_ranking " \
                f"FROM worldcities where country in {condition_str}) as ranked_cities where pop_ranking <= {max_city_per_country};"
    print(f'query_str: {query_str}')
    cities_db.cursor.execute(query_str)
    # self.cursor.execute(f"SHOW DATABASES")
    result = cities_db.cursor.fetchall()

    return result

# =============================================
def filter_cities(cities_db, max_cities_per_country=10):
    pass

