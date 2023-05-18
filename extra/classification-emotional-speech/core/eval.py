import argparse
from tensorflow.keras.models import load_model
from train import get_split_dataset

def main(opt):
    x_train, x_test, y_train, y_test = get_split_dataset(opt.data)
    model = load_model(opt.path_model)
    evaluation = model.evaluate(x_test, y_test)
    loss = evaluation[0]
    accuracy = evaluation[1]
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    
    

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CNN', help='initial model')
    parser.add_argument('--path_model', type=str, default='CNN', help='initial model')
    return parser.parse_args()

if __name__ == "__main__":
    opt = opt()
    main(opt)