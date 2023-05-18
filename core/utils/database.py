import psycopg2
import argparse

class Database:
    def __init__(self):
        self.connection = self.connect_server()

    def connect_server(self):
        return psycopg2.connect(host="dpg-chitsbd269v2e2btbgh0-a.compute.amazonaws.com", database="db_arhc", user="aimspot", password="9V4WaTb2DTfkDk0HKuDEbgZUhfAps5xv")

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_csv', type=str, default='final_csv_actor.csv', help='csv file with data')
    return parser.parse_args()

def main(opt):
    db = Database()
    print("Hello")


if __name__ == "__main__":
    opt = opt()
    main(opt)
