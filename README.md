# Backdoor Attacks
## Machine Learning for Cyber Security (ECE-GY 9163), New York University
### Suriya Prakash Jambunathan - sj3828

## Introduction
Backdoor attacks pose a significant threat in today's AI landscape. When companies fine-tune pre-trained models downloaded online, they risk inheriting backdoor connections that compromise accuracy on specific samples. These vulnerabilities stem from Backdoor Neural Networks (BadNets), containing neurons activated only by backdoored inputs. While these models achieve seemingly perfect accuracy on clean samples, they perform poorly on backdoored ones, predicting a target class different from the true class in targeted attacks. This project focuses on defining and implementing a pruning strategy to rectify BadNet models.

## Objective
The primary goal of this project is to define and implement a rectification strategy for the BadNet model. The methodology we are employing is pruning. Once the model is pruned, we will construct another GoodNet model that accurately predicts clean samples, detects backdoored samples, and assigns them to a separate class.

## Methodology
We will be pruning channels based on the order of mean activations on the last pooling layer. As explained, backdoor networks contain certain neurons that are only activated under backdoored samples. One way to observe the effect of this is by examining the outputs of the last pooling layer before the fully connected layers. The idea is to prune channels based on the order of mean activations on the clean validation set. This means that we are pruning channels that are less relevant to the clean dataset prediction, thereby not significantly affecting prediction accuracy. However, these channels may have relevance to backdoored samples, hence reducing the attack success rate.
The steps are as follows:
- Extract a neural network from the BadNet model, ensuring that the output layer is the last pooling layer.
- Compute activations for the entire clean validation set on this extracted model.
- Calculate mean activation values with respect to the channel axis.
- Sort the mean activations so that the least important channels are ordered first.
- Prune the channels by iterating over the sorted channel indices.
    - Set the weights and biases for the particular channel index to 0 to prune the channel.
- Evaluate accuracy on clean test and validation sets and calculate the attack success rate on BadNet validation and test sets.
- Measure the accuracy drop from the original BadNet model to the current pruned BadNet model on the clean validation set.
- Store three sets of pruned model weights for the following accuracy difference thresholds:
    - 2%
    - 4%
    - 10%
- Create a GoodNet model (G) that passes inputs to both the original BadNet model (B) and the pruned BadNet model (B') and predicts based on the following two conditions:
    - If both the original and pruned BadNet models predict the same class, return that class.
    - If the original and pruned BadNet models predict different classes, return N+1 as the class (where N is the total number of classes).

## Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals and split into validation and test datasets.
   3. bd_valid.h5 and bd_test.h5 contains validation and test images with sunglasses trigger respectively, that activates the backdoor for bd_net.h5. 

## Usage
### Command Line Interface (CLI)
```console
python3 new_eval.py --clean_set <clean set filepath> --bad_set <bad set filepath> --bad_model <B model filepath> --bad_model_weights <B model weights filepath> --threshold <% threshold of the pruned model> --pruned_bad_model_weights <B' model weights filepath>
```

Arguments:
- `--clean_set` : File path of the clean dataset to test the GoodNet Model on.
- `--bad_set` : File path of the backdoored dataset to test the GoodNet Model on.
- `--bad_model` : File path of the BadNet model architecture.
- `--bad_model_weights` : File path of the BadNet model weights.
- `--threshold` : Accuracy drop percentage threshold of the pruning stage.
- `--pruned_bad_model_weights` : File path of the Pruned BadNet model weights.

### Python Program
```python
import keras
import h5py
import numpy as np
from new_eval import data_loader
from GoodNet import GoodNet
    
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
```