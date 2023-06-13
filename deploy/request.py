import os 
import argparse


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='03-01-01-01-01-01-05.wav', help='name wav or model')
    parser.add_argument('--change', type=bool, default=False)
    return parser.parse_args()

def main(opt):
    if opt.change:
        name_model = opt.name
        command = f"curl -X POST -H 'Content-Type: application/json' -d '{{\"name_model\": \"{name_model}\"}}' http://localhost:5000/model"
        os.system(command)
    else:
        audio_path = opt.name
        command = f"curl -X POST -H 'Content-Type: application/json' -d '{{\"audio_path\": \"{audio_path}\"}}' http://localhost:5000/predict"
        os.system(command)

if __name__ == "__main__":
    opt = opt()
    main(opt)