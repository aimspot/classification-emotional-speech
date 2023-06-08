import os 
import argparse


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='03-01-01-01-01-01-05.wav', help='name wav')
    return parser.parse_args()

def main(opt):
    audio_path = opt.name
    command = f"curl -X POST -H 'Content-Type: application/json' -d '{{\"audio_path\": \"{audio_path}\"}}' http://localhost:5000/predict"
    os.system(command)

if __name__ == "__main__":
    opt = opt()
    main(opt)