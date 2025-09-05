import pandas as pd
import numpy as np
import sys
from sklearn.metrics import r2_score, mean_squared_error
import rmsp
import itertools
import random
import time
from sklearn.model_selection import KFold

class GridSearch_KF:

    def __init__(self, estimator, param_grid, n_folds, random_state):
        """
        Initialize the class.

        Parameters:
        - estimator: obj
            An object of that type is instantiated for each iteration. It is the base model for hyperparameter tunning.
        - param_grid: dict
            Dictionary with parameters names (str) as keys and distributions or lists of parameters to try.
        - n_folds: int
            Number of non-overlapping splits to be performed on the k-fold cross-validation framework.
        - random_state: int
            Seed for reproducibility in splitting the data into training and test.

        """
        self.base_model = estimator
        self.param_grid = param_grid
        self.n_folds = n_folds
        self.random_state = random_state
        self.results_dict = {}
        self.best_r2_test = -float('inf') # Initialize worst R² as very low
        self.best_r2_train = -float('inf') # Initialize worst R² as very low
        self.best_params = None
        self.worst_r2_test = float('inf')  # Initialize worst R² as very high
        self.worst_r2_train = float('inf')  # Initialize worst R² as very high
        self.worst_params = None
        self.refitted_model = None  # Stores the final model trained on all data

    def fit(self, data, var_x, var_y):
        """
        Run k-fold cross-validation for multiple scenarios and store results.

        Parameters:
        - data: pandas.DataFrame
            Dataset containing input features and target variable in original units.
        - var_x: list[str]
            List of input feature column names.
        - var_y: str
            Name of the (single) target variable in the dataframe.

        Returns:
        - results_dict: dict
            Dictionary containing results for all iterations.
        """

        var_full = var_x + [var_y]

        # Generate all combinations for tuning parameters (excluding fixed ones)
        param_combinations = list(itertools.product(*self.param_grid.values()))
        print(f'Initializing... there are {len(param_combinations)} possible combinations of hyperparameters')
        print(f'Running all {len(param_combinations)} combinations...\n')
        
        # Initialize timer
        start_time = time.time()
                  
        # Iterate through the shuffled combinations
        for index, combination in enumerate(param_combinations):
            
            # Initialize timer for iteration
            iter_start_time = time.time()
            
            # Create a dictionary for the current hyperparameter configuration
            param_config = dict(zip(self.param_grid.keys(), combination))
        
            # Merge fixed parameters from base_model with current param_config
            full_params = {**self.base_model.get_params(), **param_config}
        
            # Instantiating model object correctly with fixed + dynamic parameters
            model = type(self.base_model)(**full_params)
    
            # Initialize K-Fold object
            kf = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
            
            # Initialize arrays to store metrics for the run with the selected combination of hyperparameters
            r2_comb_test, rmse_comb_test = np.array([]), np.array([])
            r2_comb_train, rmse_comb_train = np.array([]), np.array([])
        
            # Loop over k non-overlapping splits
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
    
               # Split data into training and test sets using indices
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
        
                # Add prefixes "NS_" and "SP_DSPK_" to each variable to be compatible after transformation
                var_x_ns = ['NS_' + i for i in var_x]
                var_y_ns = f'NS_{var_y}'

                # NS fit_transform for multiple variables
                nstransformer_mv = rmsp.NSTransformerMV(warn_no_wt=False)
                train_prepared = nstransformer_mv.fit_transform(train_data[var_full])
                X_train, y_train = train_prepared[var_x_ns].values, train_prepared[var_y_ns].values
        
                # Transform test data, applying the parameters from the training set, thus avoiding data leakage
                test_prepared = nstransformer_mv.transform(test_data[var_full])
                X_test, y_test = test_prepared[var_x_ns].values, test_prepared[var_y_ns].values
        
                # Fit model with transformed training data
                model.fit(X_train, y_train)
        
                # Predict test and training data
                pred_test = model.predict(X_test)
                pred_train = model.predict(X_train)
        
                # Compute metrics inside the fold and append to the array relative to the iteration's
                r2_fold_test = r2_score(y_test, pred_test)
                r2_comb_test = np.append(r2_comb_test, r2_fold_test)       
                rmse_fold_test = np.sqrt(mean_squared_error(y_test, pred_test))
                rmse_comb_test = np.append(rmse_comb_test, rmse_fold_test)

                r2_fold_train = r2_score(y_train, pred_train)
                r2_comb_train = np.append(r2_comb_train, r2_fold_train)       
                rmse_fold_train = np.sqrt(mean_squared_error(y_train, pred_train))
                rmse_comb_train = np.append(rmse_comb_train, rmse_fold_train)
    
            # Calculate the average value for the run with the selected combination of hyperparameters
            r2_comb_test_avg = np.mean(r2_comb_test)
            rmse_comb_test_avg = np.mean(rmse_comb_test)
            r2_comb_train_avg = np.mean(r2_comb_train)
            rmse_comb_train_avg = np.mean(rmse_comb_train)
        
            # Finish timer for iteration with the selected combination of hyperparameters  
            iter_end_time = time.time()
            iter_elapsed_time = (iter_end_time - iter_start_time) / 60
        
            # Print status
            sys.stdout.write(f"\rIteration {index+1}/{len(param_combinations)} - Avg test R²: {r2_comb_test_avg:.4f} | Avg train R²: {r2_comb_train_avg:.4f} - Time: {iter_elapsed_time:.2f} min ")
            sys.stdout.flush()

            # Store results
            self.results_dict[index + 1] = {
                "r2_test": r2_comb_test_avg,
                "rmse_test": rmse_comb_test_avg,
                "r2_train": r2_comb_train_avg,
                "rmse_train": rmse_comb_train_avg,
                "params": full_params
            }

            # Check if this is the best R² so far
            if r2_comb_test_avg > self.best_r2_test:
                self.best_r2_test = r2_comb_test_avg
                self.best_r2_train = r2_comb_train_avg
                self.best_params = full_params  # Store best parameters

            # Check if this is the worst R² so far
            if r2_comb_test_avg < self.worst_r2_test:
                self.worst_r2_test = r2_comb_test_avg
                self.worst_r2_train = r2_comb_train_avg
                self.worst_params = full_params  # Store worst parameters
    
        # Sort and store the top 3 configurations
        sorted_results = sorted(self.results_dict.values(), key=lambda x: x["r2_test"], reverse=True)
        self.top_3_results = sorted_results[:3]  # Take top 3 configurations

        # Finish timer
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f"\n\nTotal time for {len(param_combinations)} iterations: {elapsed_time:.2f} minutes")

    def get_top_3_params(self):
        """
        Retrieve the top 3 hyperparameter combinations based on R² score.

        Returns:
        - list[dict]: List of top 3 configurations.
        """
        if not self.top_3_results:
            print("No top results available. Run `fit()` first.")
            return None

        return self.top_3_results
       
    def print_best_params(self):
        """
        Retrieve the best hyperparameter combination found.

        Returns:
        - dict: Best hyperparameters and R² score.
        """
        if self.best_params is None:
            print("No model has been trained yet. Run `fit()` first.")
            return None

        # Print best hyperparameters
        print(f"\nBest test R² score: {self.best_r2_test:.4f}")
        print(f"Associated training R² score: {self.best_r2_train:.4f}")
        print("\nBest hyperparameter combination:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

    def print_best_score(self):
        """
        Retrieve the R2 obtained from the best hyperparameter combination found.

        Returns:
        - Best R² score.
        """
        if self.best_params is None:
            print("No model has been trained yet. Run `fit()` first.")
            return None

        # Print best hyperparameters
        print(f"\nBest test R² score: {self.best_r2_test:.4f}\n")
        print(f"Associated training R² score: {self.best_r2_train:.4f}")

    def print_worst_score(self):
        """
        Retrieve the R2 obtained from the worst hyperparameter combination found.

        Returns:
        - Worst R² score.
        """
        if self.worst_params is None:
            print("No model has been trained yet. Run `fit()` first.")
            return None

        # Print best hyperparameters
        print(f"\nWorst test R² score: {self.worst_r2_test:.4f}\n")
        print(f"Associated training R² score: {self.worst_r2_train:.4f}")

    def print_worst_params(self):
        """
        Retrieve the worst hyperparameter combination found.

        Returns:
        - dict: Worst hyperparameters and R² score.
        """
        if self.worst_params is None:
            print("No model has been trained yet. Run `fit()` first.")
            return None
        
        # Print worst hyperparameters
        print(f"\Worst R² score: {self.worst_r2:.4f}")
        print("Worst hyperparameter combination:")
        for key, value in self.worst_params.items():
            print(f"  {key}: {value}")

    def best_model(self):
        """
        Return a new instance of the model with the best hyperparameters.

        Returns:
        - obj: A new model instance with the best hyperparameters.
        """
        if self.best_params is None:
            print("No model has been trained yet. Run `fit()` first.")
            return None

        return type(self.base_model)(**self.best_params)

    def refit_best_model(self, data, var_x, var_y):
        """
        Refit the model with the best hyperparameters using all available data.

        Parameters:
        - data: pandas.DataFrame
            Full dataset for training.
        - var_x: list[str]
            List of input feature column names.
        - var_y: str
            Name of the (single) target variable in the dataframe.

        Returns:
        - obj: The final trained model.
        """

        var_full = var_x + [var_y]

        if self.best_params is None:
            print("No best model found. Run `fit()` first.")
            return None

        # Instantiate the model with best parameters
        self.refitted_model = type(self.base_model)(**self.best_params)

        # NS fit_transform for multiple variables
        nstransformer_mv = rmsp.NSTransformerMV(warn_no_wt=False)
        data_prepared = nstransformer_mv.fit_transform(data[var_full])

        # Extract transformed features and target
        var_x_ns = ['NS_' + i for i in var_x]
        var_y_ns = ['NS_' + var_y]
        X_train, y_train = data_prepared[var_x_ns].values, data_prepared[var_y_ns].values.flatten()

        # Train the final model
        self.refitted_model.fit(X_train, y_train)

        # Calculate R2 from the training data for reference
        pred = self.refitted_model.predict(X_train)
        r2_train = r2_score(y_train, pred)

        print("\nBest model has been refitted using all data.")
        print(f"R2 from training data (just for reference): {r2_train:.4f}")
        return self.refitted_model
