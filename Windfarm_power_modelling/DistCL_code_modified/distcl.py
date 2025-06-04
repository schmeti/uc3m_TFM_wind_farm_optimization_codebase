import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from distnn import *
from pyomo import environ
from pyomo.environ import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
scaler_X = StandardScaler()
scaler_y = StandardScaler()



class distcl(object):
    '''
    X, y for fitting a Distributional Neural Network are expected.
    n_preds: number of predictions needed in the optimization problem.
    '''
    
    def __init__(self, X, y, n_preds, val_ind = None, test_ind = None):
        
        # check if validation indexces given
        try:
            val_ind
        except NameError:
            ind = False
        else:
            ind = True
        
        # sort and reindex X 
        X = X.reindex(sorted(X.columns), axis=1)
        
        ### Split data
        # if validation index given, split accordingly, else split randomly
        if ind == True:
            self.random_split = False
            
            self.X_val = X[X.index.isin(val_ind)]
            self.X_test = X[X.index.isin(test_ind)]
            self.X_train = X[(~X.index.isin(val_ind)) & (~X.index.isin(test_ind))]
            
            self.y_val = y[y.index.isin(val_ind)]
            self.y_test = y[y.index.isin(test_ind)]
            self.y_train = y[(~y.index.isin(val_ind)) & (~y.index.isin(test_ind))]
        else:
            self.random_split = True
            
            msk = np.random.rand(len(X)) < 0.7
            self.X_train = X[msk]
            self.y_train = y[msk]
            self.X_test = X[~msk]
            self.y_test = y[~msk]
            
            msk_val = np.random.rand(len(self.X_test)) < 0.5
            self.X_val = self.X_test[msk_val]
            self.y_val = self.y_test[msk_val]
            self.X_test = self.X_test[~msk_val]
            self.y_test = self.y_test[~msk_val]
            
            #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            #self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(self.X_test, self.y_test, test_size=0.5, random_state=0)
        
        ### Scale Data

        # fit scaler to training data: X
        self.X_train = pd.DataFrame(scaler_X.fit_transform(self.X_train), columns = scaler_X.feature_names_in_)
        
        self.X_mean = scaler_X.mean_ # mean
        self.X_std = scaler_X.scale_ # std
        self.scaler_X = scaler_X


        # scale validation and test data: X
        self.X_val = pd.DataFrame(scaler_X.transform(self.X_val), columns = scaler_X.feature_names_in_)
        self.X_test = pd.DataFrame(scaler_X.transform(self.X_test), columns = scaler_X.feature_names_in_)
        
        # fit scaler to training data: y
        self.y_train = pd.DataFrame(scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)), columns = [y.name])
        self.y_mean = scaler_y.mean_
        self.y_std = scaler_y.scale_
        
        # scale validation and test data: y
        self.y_val = pd.DataFrame(scaler_y.transform(self.y_val.values.reshape(-1, 1)), columns = [y.name])
        self.y_test = pd.DataFrame(scaler_y.transform(self.y_test.values.reshape(-1, 1)), columns = [y.name])
        
        # init params
        self.n_preds = n_preds
        self.alpha = None
        self.scaler_y = scaler_y

        
    # train model defined in distnn.py
    def train(self, n_hidden = 2, n_nodes = 50, drop = 0.05, iters = 4000, learning_rate = 1e-3, patience = None):
        nn_tool = qdnn(self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, n_hidden = n_hidden, n_nodes = n_nodes, drop = drop, iters = iters, learning_rate = learning_rate)
        model, preds_test, y_test = nn_tool.train(patience = patience)
        
        # save model
        self.model = model

        return model, preds_test, y_test
    
    def optimize_nn(self, param_grid, iters=5000, drop=0.05, learning_rate=1e-4, pacience = None,plot=False):

        best_params = None
        best_rmse = float('inf')
        grid_search = []

        for n_hidden in param_grid['n_hidden']:
            for n_nodes in param_grid['n_nodes']:
                print(f"Training with n_hidden={n_hidden}, n_nodes={n_nodes}")


                model, preds_test, y_test = self.train(
                    n_hidden=n_hidden,
                    n_nodes=n_nodes,
                    iters=iters,
                    drop=drop,
                    learning_rate=learning_rate,
                    patience = pacience
                )

                rmse = np.sqrt(mean_squared_error(y_test, preds_test))
                grid_search.append({'n_hidden': n_hidden, 'n_nodes': n_nodes, 'rmse': rmse})

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'n_hidden': n_hidden, 'n_nodes': n_nodes}
            
        if plot:
            # Convert grid_search to a DataFrame for easier plotting
            grid_search_df = pd.DataFrame(grid_search)

            # Plot the grid search results
            plt.figure(figsize=(10, 6))
            for n_hidden in grid_search_df['n_hidden'].unique():
                subset = grid_search_df[grid_search_df['n_hidden'] == n_hidden]
                plt.plot(subset['n_nodes'], subset['rmse'], marker='o', label=f'n_hidden={n_hidden}')

            plt.xlabel('Number of Nodes')
            plt.ylabel('RMSE')
            plt.title('Grid Search Results')
            plt.legend()
            plt.show()

        return best_params, best_rmse, grid_search

    def load_model(self, path):
        self.model = torch.load(path)
        return self.model

    def predict(self, X):
        """
        Predict using the trained DistFCNN model.

        Parameters:
            X: pandas DataFrame or numpy array of input features

        Returns:
            numpy.ndarray: Predicted values
        """
        X = X.reindex(sorted(X.columns), axis=1)
        X_pred_scaled = pd.DataFrame(self.scaler_X.transform(X), columns=scaler_X.feature_names_in_)

        # Convert to torch tensor
        X_tensor = torch.from_numpy(X_pred_scaled.values.astype(np.float32))
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)
        
        y_pred = preds.numpy()
        y_pred_invscaled = self.scaler_y.inverse_transform(y_pred)

        return y_pred_invscaled
    

    def expected_power_for_location(self, x_turb2_value, y_turb2_value, wind_df):

        """
        Calculate the expected power for a given location based on wind distribution data.
        Parameters:
            x_turb2_value: x-coordinate of the turbine location
            y_turb2_value: y-coordinate of the turbine location
            wind_df: pandas DataFrame containing wind distribution data with columns:
                     'wind_direction', 'wind_speed', 'turbulence_intensity', 'probability'
        Returns:
            expected_power: float, the expected power at the given location
        """
        # Get wind distribution data
        wind_directions = wind_df['wind_direction'].values
        wind_speed_value = wind_df['wind_speed'].values
        turbulence_intensity_value = wind_df['turbulence_intensity'].values
        probabilities = wind_df['probability'].values

        # Create the DataFrame
        data_pred = pd.DataFrame({
            'x_turb2': x_turb2_value,
            'y_turb2': y_turb2_value,
            'wind_speed': wind_speed_value,
            'wind_direction': wind_directions,
            'turbulence_intensity': turbulence_intensity_value,
            'probability': probabilities
        })
        generated_data = pd.DataFrame(data_pred, columns=["x_turb2", "y_turb2", "wind_speed", "wind_direction", "turbulence_intensity"])
        
        # predict
        data_pred["pred_power"] = self.predict(X=generated_data[["x_turb2", "y_turb2", "wind_speed", "wind_direction", "turbulence_intensity"]])
        
        # calculate expectation
        expected_power = (data_pred["pred_power"] * data_pred["probability"]).sum()

        return expected_power
    
    # build constraints for the optimization problem from Neural Network
    def constraint_build(self, fitted_model):
        
        # initi empty param lists
        weight_names = []
        weight_values = []
        bias_names = []
        bias_values = []

        # extract weights and biases from fitted model
        for name, param in fitted_model.named_parameters():
            if 'weight' in name:
                weight_names.append(name)
                weight_values.append(param.cpu().detach().numpy())
            else:
                bias_names.append(name)
                bias_values.append(param.cpu().detach().numpy())
        
        # generate constraints df
        constraints = constraint_extrapolation_MLP(weight_values,bias_values,weight_names)
        
        return constraints
    

    
    def const_embed(self, opt_model, constaints, outcome, n_scenarios = 1, deterministic = False):
        
        '''
        This function embdeds the fitted prediction model within the optimization problem.
        Expecting a defined optimization model "opt_model", constraint dataframe "constraints",
        a name for an "outcome", and the number of scenarios to generate.
        '''
        
        # Predefining variable y
        #opt_model.y = Var(Any, dense=False, domain=Reals)
        
        # Defining Variables 
        M_l=-1e3 # M lower bound
        M_u=1e3 # M upper bound

        opt_model.v = Var(Any, dense=False, domain=NonNegativeReals) # activation value 
        opt_model.v_ind = Var(Any, dense=False, domain=Binary) #binary variable for constraints composing activation value
        
        # Loop over predictions
        for n in range(self.n_preds):
            
            # Get Variables 
            max_layer = max(constaints['layer'])
            nodes_input = range(len(self.X_train.columns))


            # scale X
            v_input = [(opt_model.x[name]-self.X_mean[i])/self.X_std[i] for i,name in enumerate(self.X_train.columns)]
            # TODO: Modify input to have decision variable 
            
            # Loop over layers
            for l in range(max_layer+1):
                # get layer from df
                df_layer = constaints.query('layer == %d' % l)

                # get attributes
                max_nodes = [k for k in df_layer.columns if 'node_' in k] # TODO: Works with NaNs ? 
                # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
                coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                intercepts_layer = np.array(df_layer['intercept'])
                nodes = df_layer['node']
            
                # if output layer, add constraints corresponding to Mean and Standard deviation (equality constraints)
                if l == max_layer:
                    node = nodes.iloc[0]

                    # constraint for mean 
                    opt_model.add_component('Mean_est' + str(n), Constraint(rule=opt_model.y[outcome, n,'output'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                    
                    # # constraint for sd
                    # l = l+1
                    # df_layer = constaints.query('layer == %d' % l)
                    # max_nodes = [k for k in df_layer.columns if 'node_' in k]
                    # coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                    # intercepts_layer = np.array(df_layer['intercept'])
                    # opt_model.add_component('Var_est' + str(n), Constraint(rule=opt_model.y[outcome, n,'sd'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                
                # else add constraints moving through network (inequality constraints)
                else:
                    # Save v_pos for input to next layer
                    v_pos_list = []
                    # Loop over nodes in layer
                    for node in nodes:
                        ## Initialize variables
                        v_pos_list.append(opt_model.v[(outcome, l, node, n)])
                        opt_model.add_component('constraint_1_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] >= sum((v_input[i]) * (coeffs_layer[node, i]) for i in nodes_input) + intercepts_layer[node]))
                        opt_model.add_component('constraint_2_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] <= M_u * (opt_model.v_ind[(outcome, l, node, n)])))
                        opt_model.add_component('constraint_3_layer' + str(l) + '_node' + str(node)+ '_pred'+ str(n)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, n)] <= sum((v_input[i]) * (coeffs_layer[node, i]) for i in nodes_input) + intercepts_layer[node] - M_l * (1 - opt_model.v_ind[(outcome, l, node, n)])))
                    ## Prepare nodes_input for next layer
                    nodes_input = nodes
                    v_input = v_pos_list

                print("Layer ", l, "embedded for prediction ", n, " of outcome ", outcome)
        
        # If Stochastic, add noise term and scale back to original scale
        if deterministic == False:
            np.random.seed(0) # Seed hardcoded !!!!!!
            z_vals = np.random.normal(size=n_scenarios) # draw samples from standard normal (generate noise)

            for n in range(self.n_preds):
                for w in range(1, n_scenarios + 1):
                    opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == (z_vals[w-1] * opt_model.y[outcome, n,'sd'] + opt_model.y[outcome, n,'mean'])*self.y_std + self.y_mean))
        # Elif deterministic, take mean value and scale back to original scale
        else:
            for n in range(self.n_preds):
                for w in range(1, n_scenarios + 1):
                    opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == opt_model.y[outcome, n,'output']*self.y_std + self.y_mean))




   
    def const_embed_modified(self, opt_model, constaints, outcome, n_scenarios = 1, deterministic = False):
        
        '''
        This function embdeds the fitted prediction model within the optimization problem.
        Expecting a defined optimization model "opt_model", constraint dataframe "constraints",
        a name for an "outcome", and the number of scenarios to generate.
        '''
        
        # Predefining variable y
        #opt_model.y = Var(Any, dense=False, domain=Reals)
        
        # Defining Variables 
        M_l=-1e3 # M lower bound
        M_u=1e3 # M upper bound

        opt_model.v = Var(Any, dense=False, domain=NonNegativeReals) # activation value 
        opt_model.v_ind = Var(Any, dense=False, domain=Binary) #binary variable for constraints composing activation value
        
        # Loop over predictions
        for w in range(n_scenarios):
            
            print("Secenario ", w, "embedded")
            n = 0
            
            # Get Variables 
            max_layer = max(constaints['layer'])
            nodes_input = range(len(self.X_train.columns))


            # scale X
            v_input = [(opt_model.x[name,w]-self.X_mean[i])/self.X_std[i] for i,name in enumerate(self.X_train.columns)]
            # TODO: Modify input to have decision variable 
            
            # Loop over layers
            for l in range(max_layer+1):
                # get layer from df
                df_layer = constaints.query('layer == %d' % l)

                # get attributes
                max_nodes = [k for k in df_layer.columns if 'node_' in k] # TODO: Works with NaNs ? 
                # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
                coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                intercepts_layer = np.array(df_layer['intercept'])
                nodes = df_layer['node']
                
                # if output layer, add constraints corresponding to Mean and Standard deviation (equality constraints)
                if l == max_layer:
                    node = nodes.iloc[0]

                    # constraint for mean 
                    opt_model.add_component('Mean_est' + str(w), Constraint(rule=opt_model.y[outcome, w,'mean'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                    
                    # # constraint for sd
                    # l = l+1
                    # df_layer = constaints.query('layer == %d' % l)
                    # max_nodes = [k for k in df_layer.columns if 'node_' in k]
                    # coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
                    # intercepts_layer = np.array(df_layer['intercept'])
                    # opt_model.add_component('Var_est' + str(w), Constraint(rule=opt_model.y[outcome, w,'sd'] == (sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node])))
                
                # else add constraints moving through network (inequality constraints)
                else:
                    # Save v_pos for input to next layer
                    v_pos_list = []
                    # Loop over nodes in layer
                    for node in nodes:
                        ## Initialize variables
                        v_pos_list.append(opt_model.v[(outcome, l, node, w)])
                        opt_model.add_component('constraint_1_layer' + str(l) + '_node' + str(node)+ '_sce'+ str(w)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, w)] >= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                        opt_model.add_component('constraint_2_layer' + str(l) + '_node' + str(node)+ '_sce'+ str(w)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, w)] <= M_u * (opt_model.v_ind[(outcome, l, node, w)])))
                        opt_model.add_component('constraint_3_layer' + str(l) + '_node' + str(node)+ '_sce'+ str(w)+ outcome,
                                            Constraint(rule=opt_model.v[(outcome, l, node, w)] <= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node] - M_l * (1 - opt_model.v_ind[(outcome, l, node, w)])))
                    
                    ## Prepare nodes_input for next layer
                    nodes_input = nodes
                    v_input = v_pos_list
                
                print("Layer ", l, "embedded for prediction ", n, " of outcome ", outcome)
  

        # Elif deterministic, take mean value and scale back to original scale

        for n in range(self.n_preds):
            for w in range(n_scenarios):
                opt_model.add_component(outcome + '_pred' + str(n) + '_sce' + str(w), Constraint(rule=opt_model.y[outcome, n, w] == opt_model.y[outcome, w,'mean']*self.y_std + self.y_mean))

