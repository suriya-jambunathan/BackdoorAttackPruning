import keras
import h5py
import argparse
import numpy as np
from GoodNet import GoodNet

parser = argparse.ArgumentParser()
parser.add_argument("--clean_set", help="clean data filename", default='./data/cl/test.h5', type=str)
parser.add_argument("--bad_set", help="poisoned data filename", default='./data/bd/bd_test.h5', type=str)
parser.add_argument("--bad_model", help="B model filename", default='./models/bd_net.h5', type=str)
parser.add_argument("--bad_model_weights", help="B model weights filename", default='./models/bd_weights.h5', type=str)
parser.add_argument("--threshold", help="B' model prune threshold", type=int)
parser.add_argument("--pruned_bad_model_weights", help="B' model weights filename", default='./models/bd_10p_weights.h5', type=str)
args = parser.parse_args()

clean_data_filename = args.clean_set
poisoned_data_filename = args.bad_set
B_model_filename = args.bad_model
B_model_weights_filename = args.bad_model_weights

if args.threshold:
    B_dash_model_weights_filename = f'./models/bd_{args.threshold}p_weights.h5'
else:
    B_dash_model_weights_filename = args.pruned_bad_model_weights

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def main():
cl_x_test, cl_y_test = data_loader(clean_data_filename)
bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

# Initializing GoodNet Model
G = GoodNet(B_model_filename)
G.load_weights(B_model_weights_filename, B_dash_model_weights_filename)

cl_label_p = G.predict(cl_x_test)
clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test))*100
print('Clean Classification accuracy:', clean_accuracy)

bd_label_p = G.predict(bd_x_test)
asr = np.mean(np.equal(bd_label_p, bd_y_test))*100
print('Attack Success Rate:', asr)

if __name__ == '__main__':
    main()