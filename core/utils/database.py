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

    def delete_table_data(self):
        query = f"DELETE FROM dataset"
        self.cur.execute(query)
        self.connection.commit()
        

    def create_table_data(self, df):
        self.drop_table_data()
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
        self.delete_table_data()
        df = df.add_prefix('s_')
        columns = ', '.join(df.columns)
        placeholders = ', '.join(['%s'] * len(df.columns))
        insert_query = f"INSERT INTO dataset ({columns}) VALUES ({placeholders})"
        values = [tuple(row) for row in df.values]
        self.cur.executemany(insert_query, values)
        self.connection.commit()
        print("CSV added to db")
        

    def insert_metrics(self, name, name_model, precision, recall, accuracy, f1):
        query = '''INSERT INTO "models" ("name", "name_model", "precision", "recall", "accuracy", "f1")
                VALUES('{0}', '{1}', '{2}', '{3}', '{4}', '{5}')'''.format(name, name_model, precision, recall, accuracy, f1)
        self.cur.execute(query)
        self.connection.commit()


    def delete_best_model(self):
        query = '''DELETE FROM best_model'''
        self.cur.execute(query)
        self.connection.commit()


    def insert_best_model(self, name_model):
        query = '''INSERT INTO "best_model" ("name_model")
                VALUES('{0}')'''.format(name_model)
        self.cur.execute(query)
        self.connection.commit()


    
    def insert_model_name(self, name, name_model):
        query = '''INSERT INTO "models" ("name", "name_model")
                VALUES('{0}', '{1}')'''.format(name, name_model)
        self.cur.execute(query)
        self.connection.commit()


    def insert_eval(self, name, name_model, precision, recall, accuracy, f1):
        query = '''INSERT INTO "models" ("name", "name_model", "precision", "recall", "accuracy", "f1")
                SELECT '{0}', '{1}', '{2}', '{3}', '{4}', '{5}'
                WHERE NOT EXISTS (
                    SELECT 1 FROM "models" WHERE "name"='{0}' AND "name_model"='{1}'
                )'''.format(name, name_model, precision, recall, accuracy, f1)
        self.cur.execute(query)
        self.connection.commit()

    def delete_null_metrics(self, name, name_model):
        query = '''DELETE FROM models WHERE name = %s AND name_model = %s'''
        values = (name, name_model)
        self.cur.execute(query, values)
        self.connection.commit()


    
    def get_empty_metrics(self):
        query = '''SELECT "name", "name_model"
                FROM "models"
                WHERE "precision" IS NULL AND "recall" IS NULL AND "accuracy" IS NULL AND "f1" IS NULL'''
        self.cur.execute(query)
        results = self.cur.fetchall()
        names = [result[0] for result in results]
        name_models = [result[1] for result in results]
        return names, name_models
    

    def get_model_metrics(self):
        query = '''SELECT name_model, accuracy, f1 FROM models'''
        self.cur.execute(query)
        results = self.cur.fetchall()
        name_model_list = []
        accuracy_list = []
        f1_list = []

        for row in results:
            name_model_list.append(row[0])
            accuracy_list.append(row[1])
            f1_list.append(row[2])

        return name_model_list, accuracy_list, f1_list

    def getting_data(self):
        query = f"SELECT * FROM dataset"
        df = pd.read_sql(query, self.connection)
        return df


    def insert_model():
        print("Model added to db")

    def connect_server(self):
        return psycopg2.connect(host=self.host, database=self.db, user=self.user, password=self.password)
