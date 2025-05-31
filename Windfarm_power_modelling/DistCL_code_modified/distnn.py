import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import *

class qdnn(object):

    """
    Class to train a feedforward neural network with Gaussian NLL loss.
    """
    
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, n_hidden, n_nodes, drop, iters, learning_rate):
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.n_hidden = n_hidden
        self.n_nodes = n_nodes
        self.drop = drop
        self.iters = iters
        self.learning_rate = learning_rate
        
    
    def train(self,patience = None):
        
        # wrap Data into Dataloades
        train_ds = TabularDataset(self.X_train, self.y_train)
        val_ds = TabularDataset(self.X_val, self.y_val)
        test_ds = TabularDataset(self.X_test, self.y_test)
        batchsize = 256
        train_dl = DataLoader(train_ds, batch_size = batchsize, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=len(val_ds))
        test_dl = DataLoader(test_ds, batch_size=len(test_ds))
        
        lr = self.learning_rate
        
        # set seed and empty cache
        seed = 0 #################### Seed Hardcoded !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.empty_cache()

        # Define Gaussian Negative Log Likelihood as Lodd function
        criterion = nn.MSELoss()
        
        # define model as Distributional Neural Network as defined in utils.py
        model = DistFCNN(input_size=self.X_train.shape[1], output_size=1, hidden_layers=self.n_hidden, hidden_size=self.n_nodes, drop=self.drop).to(device)
        
        # Define Optimizer (Adam)
        wd = 0
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
        
        print(model)
        print(device)

        # Train the model
        best_loss = float("inf")
        last_save = 0 

        for epoch in range(self.iters):
            ###### Training ######
            model = model.train()
            for y, cont_x in train_dl:
                optimizer.zero_grad()
                cont_x = cont_x.to(device)
                y = y.to(device)
                pred = model(cont_x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            ###### Validation ######
            model = model.eval()
            val_loss = 0
            with torch.no_grad():
                for y, cont_x in val_dl:
                    cont_x = cont_x.to(device)
                    y  = y.to(device)
                    pred = model(cont_x)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
            
            val_loss /= float(len(val_dl))  # average validation loss
            
            if val_loss < best_loss:
                torch.save(model.state_dict(), 'best_model.pt')
                best_loss = val_loss
                last_save = epoch
            
            # Print loss every 500 epochs
            if (epoch % 500) == 0:
                print('epoch', epoch, 'loss', val_loss)

            # Early stopping condition
            if patience is not None:
                if epoch - last_save >= patience:
                    print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.4f}")
                    break

        
        # Load best model and evaluate on test set
        model.load_state_dict(torch.load('best_model.pt'))
        model = model.eval()
        with torch.no_grad():
            for y, cont_x in test_dl:
                cont_x = cont_x.to(device)
                y  = y.to(device)
                pred = model(cont_x)
        preds_test = pred.detach().cpu().numpy()
        y_test = y.detach().cpu().numpy()
        
        print('NN fitting process finished with a validation MSE loss of', best_loss, 'in epoch', last_save)
        return model, preds_test, y_test
    


