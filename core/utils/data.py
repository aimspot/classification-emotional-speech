import argparse
import pandas as pd
from database import Database


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_csv', type=str, default='final_csv_actor.csv', help='csv file with data')
    return parser.parse_args()

def main(opt):
    df = pd.read_csv(opt.path_csv)
    db = Database()
    #db.create_table_data(df)
    db.insert_data(df)
    #df = db.getting_data()
    #print(df.shape)
    print("Data uploaded")
    #print("Hello")


if __name__ == "__main__":
    opt = opt()
    main(opt)