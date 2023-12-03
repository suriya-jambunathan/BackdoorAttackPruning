import keras
import numpy as np

# Defining GoodNet (G) Model which uses both B and B' for prediction. It identifies and labels the detected Bad Samples as n+1 class.
class GoodNet():
    
    # Class Initialization
    def __init__(self, BD_Net_path, num_classes = 1282):
        self.B = keras.models.load_model(BD_Net_path)
        self.B_dash = keras.models.load_model(BD_Net_path)
        self.bd_class = num_classes + 1
    
    # Load Weights Function
    def load_weights(self, B_weights_path, B_dash_weights_path):
        self.B.load_weights(B_weights_path)
        self.B_dash.load_weights(B_dash_weights_path)
    
    # Predict Function 
    def predict(self, X):
        B_pred = np.argmax(self.B.predict(X, verbose = 1), axis=1)
        B_dash_pred = np.argmax(self.B_dash.predict(X, verbose = 1), axis=1)
                
        y_preds = []
        for i in range(len(B_pred)):
            
            # Predict same class if both B and B' predict the same class
            if B_pred[i] == B_dash_pred[i]:
                y_preds.append(B_pred[i])
                
            # Predict N+1 as the class if the predictions from B and B' are different
            else:
                y_preds.append(self.bd_class)
        
        return(y_preds)