import psycopg2
import pandas as pd


class Database:
    def __init__(self):
        self.host = "dpg-chitsbd269v2e2btbgh0-a.frankfurt-postgres.render.com"# change
        self.db = "db_arhc" #change
        self.user = "aimspot" #change
        self.password = "9V4WaTb2DTfkDk0HKuDEbgZUhfAps5xv" # change

        self.connection = self.connect_server()
        self.cur = self.connection.cursor()

    def create_table(self, df):
        table_name = 'dataset'
        table_name = 'your_table_name'
        create_table_query = f'CREATE TABLE {table_name} ('
        for column in df.columns:
            # Determine the data type based on the DataFrame column dtype
            if df[column].dtype == 'int64':
                data_type = 'integer'
            elif df[column].dtype == 'float64':
                data_type = 'numeric'
            elif df[column].dtype == 'bool':
                data_type = 'boolean'
            else:
                data_type = 'text'
    
        create_table_query += f'{column} {data_type}, '
        create_table_query = create_table_query[:-2]
        create_table_query += ')'

        # Execute the CREATE TABLE query
        self.cur.execute(create_table_query)


        self.connection.commit()
        print("create table works")

    def insert_data():
        print("CSV added to db")

    def insert_model():
        print("Model added to db")

    def connect_server(self):
        return psycopg2.connect(host=self.host, database=self.db, user=self.user, password=self.password)
