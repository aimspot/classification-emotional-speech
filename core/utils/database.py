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

    def drop_table_data(self):
        drop_query = f"DROP TABLE IF EXISTS dataset"
        self.cur.execute(drop_query)
        self.connection.commit()
        

    def create_table_data(self, df):
        df = df.add_prefix('s_')
        create_table_query = f'CREATE TABLE dataset ('
        for column in df.columns:
            if df[column].dtype == 'int64':
                data_type = 'integer'
            elif df[column].dtype == 'float64':
                data_type = 'numeric'
            elif df[column].dtype == 'bool':
                data_type = 'boolean'
            else:
                data_type = 'text'
            create_table_query += f'{str(column)} {data_type}, '
        create_table_query = create_table_query[:-2]
        create_table_query += ')'
        self.cur.execute(create_table_query)
        self.connection.commit()


    def insert_data(self, df):
        df = df.add_prefix('s_')
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_query = f"INSERT INTO dataset ({columns}) VALUES ({placeholders})"
        values = [tuple(row) for row in df.values]
        self.cur.executemany(insert_query, values)
        self.connection.commit()
        print("CSV added to db")


    def getting_data(self):
        query = f"SELECT * FROM dataset"
        df = pd.read_sql(query, self.connection)
        return df


    def insert_model():
        print("Model added to db")

    def connect_server(self):
        return psycopg2.connect(host=self.host, database=self.db, user=self.user, password=self.password)
