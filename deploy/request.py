import os 
import argparse


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='03-01-01-01-01-01-05.wav', help='name wav')
    return parser.parse_args()

def main():
    os.system(f"curl -X POST -H 'Content-Type: application/json' -d '{'audio_path': {opt.name}}' http://localhost:5000/predict")

if __name__ == "__main__":
    opt = opt()
    main(opt)