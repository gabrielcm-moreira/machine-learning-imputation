import numpy as np
import pandas as pd
import rmsp
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import time

class Multi_Scenario_CV:
    """
    Class to run a workflow for multiple scenarios of cross-validation analysis, overlapping or not, using 
    Multi-Gaussian (MG) and Machine Learning (ML) models, varying the number of runs (splits), the proportion 
    of the test set or the number of folds in a k-fold cross validation framework.
    The workflow applies custom transformations from the CompleteTransformer Class, which can be wraped 
    in a Pipeline from Sklearn, followed by the application of MultiGaussian and Machine Learning approaches 
    (through their respective classes) to build conditional distributions in the Gaussian unit space.
    In each split, the transformers are fit and applied to the training, and applied to the test set, without 
    refitting, which avoids data leakage. 
    Subsets' statistics are calculated and stored, as well as R-Squared and RMSE values from the multiple runs.
    """

    def __init__(self, random_state, model_ml, model_mg):
        """
        Initialize the class with core workflow parameters.

        Parameters:
        - random_state: int
            Seed for random number generation for splitting the data.
        - model_ml: object
            Model instance using ConditionalML Class.
        - model_mg: object
            Model instance using ConditionalMG Class.
        """
        self.random_state = random_state
        self.model_ml = model_ml
        self.model_mg = model_mg

    def multi_overlap_splitting(self, data, var_x, var_y, n_splits_list, test_prop):
        """
        Run the workflow analysis for multiple overlapping splits (n_splits_list) with a fixed test proportion.

        Parameters:
        - data: pandas.DataFrame
            Dataset containing input features and target variable in original units.
        - var_x: list[str]
            List of input feature column names.
        - var_y: list[str]
            List of target variable column names.
        - n_splits_list: list[int]
            List of n_splits values to iterate over.These are the number of times that the model is going to
            be trained and testd in the workflow, each time reshuffling the split.
        - test_prop: float
            Proportion of test data. Must be between 0.0 and 1.0

        Returns:
        - results_dict_1: dict
            Dictionary containing results for each n_splits value.
        """

        var_full = var_x + [var_y]

        self.results_dict_1 = {}
       
        for n_splits in n_splits_list:
            print(f"\nRunning workflow for: n_splits={n_splits} | test_prop={test_prop}")

            # Initialize timer
            start_time = time.time()
            
            # Initialize arrays to store statistics
            mean_value_ts, var_value_ts, max_value_ts, min_value_ts = np.array([]), np.array([]), np.array([]), np.array([])
            mean_value_tr, var_value_tr, max_value_tr, min_value_tr = np.array([]), np.array([]), np.array([]), np.array([])

            # Initialize arrays for MG and ML metrics
            r2_values_ts_mg, r2_values_tr_mg, rmse_values_ts_mg, rmse_values_tr_mg, condvar_ts_mg, condvar_tr_mg = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            r2_values_ts_ml, r2_values_tr_ml, rmse_values_ts_ml, rmse_values_tr_ml = np.array([]), np.array([]), np.array([]), np.array([])

            # Loop over k overlapping splits
            for split_idx in range(n_splits):
                sys.stdout.write(f"\rIteration {split_idx+1}/{n_splits} running...")

                # Train-test split
                train_data, test_data = train_test_split(data, test_size=test_prop, random_state=self.random_state * split_idx)

                # Add prefix "NS_" to each variable to be compatible after the transformation pipeline
                var_x_ns = ['NS_' + i for i in var_x]
                var_y_ns = f'NS_{var_y}'

                # NS fit_transform for multiple variables
                nstransformer_mv = rmsp.NSTransformerMV(warn_no_wt=False)
                train_prepared = nstransformer_mv.fit_transform(train_data[var_full])
                X_train, y_train = train_prepared[var_x_ns].values, train_prepared[var_y_ns].values
        
                # Transform test data, applying the parameters from the training set, thus avoiding data leakage
                test_prepared = nstransformer_mv.transform(test_data[var_full])
                X_test, y_test = test_prepared[var_x_ns].values, test_prepared[var_y_ns].values

                # Collect statistics from training and test data
                mean_value_ts = np.append(mean_value_ts, y_test.mean())
                var_value_ts = np.append(var_value_ts, y_test.var())
                min_value_ts = np.append(min_value_ts, y_test.min())
                max_value_ts = np.append(max_value_ts, y_test.max())

                mean_value_tr = np.append(mean_value_tr, y_train.mean())
                var_value_tr = np.append(var_value_tr, y_train.var())
                min_value_tr = np.append(min_value_tr, y_train.min())
                max_value_tr = np.append(max_value_tr, y_train.max())

                # Fit MG and ML models
                self.model_mg.fit(X_train, y_train)
                self.model_ml.fit(X_train, y_train)

                # Predictions for MG
                pred_ts_mg, var_ts_mg, _ = self.model_mg.predict(X_test, return_pdf=False)
                pred_tr_mg, var_tr_mg, _ = self.model_mg.predict(X_train, return_pdf=False)

                # Predictions for ML
                pred_ts_ml, _, _ = self.model_ml.predict(X_test, return_pdf=False)
                pred_tr_ml, _, _ = self.model_ml.predict(X_train, return_pdf=False)

                # Metrics for MG
                r2_values_ts_mg = np.append(r2_values_ts_mg, r2_score(y_test, pred_ts_mg))
                r2_values_tr_mg = np.append(r2_values_tr_mg, r2_score(y_train, pred_tr_mg))
                rmse_values_ts_mg = np.append(rmse_values_ts_mg, np.sqrt(mean_squared_error(y_test, pred_ts_mg)))
                rmse_values_tr_mg = np.append(rmse_values_tr_mg, np.sqrt(mean_squared_error(y_train, pred_tr_mg)))
                condvar_ts_mg = np.append(condvar_ts_mg, var_ts_mg)
                condvar_tr_mg = np.append(condvar_tr_mg, var_tr_mg)

                # Metrics for ML
                r2_values_ts_ml = np.append(r2_values_ts_ml, r2_score(y_test, pred_ts_ml))
                r2_values_tr_ml = np.append(r2_values_tr_ml, r2_score(y_train, pred_tr_ml))
                rmse_values_ts_ml = np.append(rmse_values_ts_ml, np.sqrt(mean_squared_error(y_test, pred_ts_ml)))
                rmse_values_tr_ml = np.append(rmse_values_tr_ml, np.sqrt(mean_squared_error(y_train, pred_tr_ml)))

            # Ensure nested dictionary structure
            if n_splits not in self.results_dict_1:
                self.results_dict_1[n_splits] = {}
            
            # Store results for current n_splits
            self.results_dict_1[n_splits][test_prop] = {
                # MG statistics
                'r2_stats_mg': self._compute_stats(r2_values_ts_mg, r2_values_tr_mg),
                'rmse_stats_mg': self._compute_stats(rmse_values_ts_mg, rmse_values_tr_mg),

                # ML statistics
                'r2_stats_ml': self._compute_stats(r2_values_ts_ml, r2_values_tr_ml),
                'rmse_stats_ml': self._compute_stats(rmse_values_ts_ml, rmse_values_tr_ml),

                # Data statistics
                'data_stats': {
                    'test': {
                        'min': min_value_ts,
                        'max': max_value_ts,
                        'mean': mean_value_ts,
                        'var': var_value_ts,
                    },
                    'train': {
                        'min': min_value_tr,
                        'max': max_value_tr,
                        'mean': mean_value_tr,
                        'var': var_value_tr,
                    },
                },

                # All metrics for visualization
                'all_metrics': {
                    'r2_mg': {'test': r2_values_ts_mg, 'train': r2_values_tr_mg},
                    'r2_ml': {'test': r2_values_ts_ml, 'train': r2_values_tr_ml},
                    'rmse_mg': {'test': rmse_values_ts_mg, 'train': rmse_values_tr_mg},
                    'rmse_ml': {'test': rmse_values_ts_ml, 'train': rmse_values_tr_ml},
                },
            }

            # Finish timer
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print(f"\nElapsed Time: {elapsed_time:.2f} minutes")

        return self.results_dict_1
    
# --------------------------------------------------------------
# --------------------------------------------------------------

    def multi_test_proportions(self, data, var_x, var_y, n_splits, test_props):
        """
        Run workflow for multiple test proportions while keeping n_splits fixed.

        Parameters:
        - data: pandas.DataFrame
            Dataset containing input features and target variable in original units.
        - var_x: list[str]
            List of input feature column names.
        - var_y: list[str]
            List of target variable column names.
        - n_splits: int
            Number of overlapping splits for the workflow. This is the number of times the model is
            going to be trained and tested in the workflow, each time shuffling the combination of
            training and testing data.
        - test_props: list[float]
            List of test proportions to iterate over.

        Returns:
        - results_dict_2: dict
            Dictionary containing results for each test_prop value.
        """

        var_full = var_x + [var_y]

        self.results_dict_2 = {}

        for test_prop in test_props:
            print(f"\nRunning workflow for: n_splits={n_splits} | test_prop={test_prop}")
            start_time = time.time()
            
            # Initialize arrays to store statistics
            mean_value_ts, var_value_ts, max_value_ts, min_value_ts = np.array([]), np.array([]), np.array([]), np.array([])
            mean_value_tr, var_value_tr, max_value_tr, min_value_tr = np.array([]), np.array([]), np.array([]), np.array([])

            # Initialize arrays for MG and ML metrics
            r2_values_ts_mg, r2_values_tr_mg, rmse_values_ts_mg, rmse_values_tr_mg, condvar_ts_mg, condvar_tr_mg = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            r2_values_ts_ml, r2_values_tr_ml, rmse_values_ts_ml, rmse_values_tr_ml = np.array([]), np.array([]), np.array([]), np.array([])

            # Loop over k splits
            for split_idx in range(n_splits):
                sys.stdout.write(f"\rIteration {split_idx+1}/{n_splits} running...")
                # Train-test split
                train_data, test_data = train_test_split(data, test_size=test_prop, random_state=self.random_state * split_idx)

                # Add prefix "NS_" to each variable to be compatible after transformation pipeline
                var_x_ns = ['NS_' + i for i in var_x]
                var_y_ns = f'NS_{var_y}'

                # NS fit_transform for multiple variables
                nstransformer_mv = rmsp.NSTransformerMV(warn_no_wt=False)
                train_prepared = nstransformer_mv.fit_transform(train_data[var_full])
                X_train, y_train = train_prepared[var_x_ns].values, train_prepared[var_y_ns].values
        
                # Transform test data, applying the parameters from the training set, thus avoiding data leakage
                test_prepared = nstransformer_mv.transform(test_data[var_full])
                X_test, y_test = test_prepared[var_x_ns].values, test_prepared[var_y_ns].values
                
                # # Fit and transform training data
                # train_prepared = self.trans_pipeline.fit_transform(train_data)
                # X_train, y_train = train_prepared[var_x_ns].values, train_prepared[var_y_ns].values

                # # Transform test data, applying the parameters from the training set, thus avoiding data leakage
                # test_prepared = self.trans_pipeline.transform(test_data)
                # X_test, y_test = test_prepared[var_x_ns].values, test_prepared[var_y_ns].values

                # Collect statistics for training and test data
                mean_value_ts = np.append(mean_value_ts, y_test.mean())
                var_value_ts = np.append(var_value_ts, y_test.var())
                min_value_ts = np.append(min_value_ts, y_test.min())
                max_value_ts = np.append(max_value_ts, y_test.max())

                mean_value_tr = np.append(mean_value_tr, y_train.mean())
                var_value_tr = np.append(var_value_tr, y_train.var())
                min_value_tr = np.append(min_value_tr, y_train.min())
                max_value_tr = np.append(max_value_tr, y_train.max())

                # Fit MG and ML models
                self.model_mg.fit(X_train, y_train)
                self.model_ml.fit(X_train, y_train)

                # Predictions for MG
                pred_ts_mg, var_ts_mg, _ = self.model_mg.predict(X_test, return_pdf=False)
                pred_tr_mg, var_tr_mg, _ = self.model_mg.predict(X_train, return_pdf=False)

                # Predictions for ML
                pred_ts_ml, _, _ = self.model_ml.predict(X_test, return_pdf=False)
                pred_tr_ml, _, _ = self.model_ml.predict(X_train, return_pdf=False)

                # Metrics for MG
                r2_values_ts_mg = np.append(r2_values_ts_mg, r2_score(y_test, pred_ts_mg))
                r2_values_tr_mg = np.append(r2_values_tr_mg, r2_score(y_train, pred_tr_mg))
                rmse_values_ts_mg = np.append(rmse_values_ts_mg, np.sqrt(mean_squared_error(y_test, pred_ts_mg)))
                rmse_values_tr_mg = np.append(rmse_values_tr_mg, np.sqrt(mean_squared_error(y_train, pred_tr_mg)))
                condvar_ts_mg = np.append(condvar_ts_mg, var_ts_mg.mean())
                condvar_tr_mg = np.append(condvar_tr_mg, var_tr_mg.mean())

                # Metrics for ML
                r2_values_ts_ml = np.append(r2_values_ts_ml, r2_score(y_test, pred_ts_ml))
                r2_values_tr_ml = np.append(r2_values_tr_ml, r2_score(y_train, pred_tr_ml))
                rmse_values_ts_ml = np.append(rmse_values_ts_ml, np.sqrt(mean_squared_error(y_test, pred_ts_ml)))
                rmse_values_tr_ml = np.append(rmse_values_tr_ml, np.sqrt(mean_squared_error(y_train, pred_tr_ml)))

            # Ensure nested dictionary structure
            if n_splits not in self.results_dict_2:
                self.results_dict_2[n_splits] = {}
            
            # Store results for current n_splits
            self.results_dict_2[n_splits][test_prop] = {
                # MG statistics
                'r2_stats_mg': self._compute_stats(r2_values_ts_mg, r2_values_tr_mg),
                'rmse_stats_mg': self._compute_stats(rmse_values_ts_mg, rmse_values_tr_mg),

                # ML statistics
                'r2_stats_ml': self._compute_stats(r2_values_ts_ml, r2_values_tr_ml),
                'rmse_stats_ml': self._compute_stats(rmse_values_ts_ml, rmse_values_tr_ml),

                # Data statistics
                'data_stats': {
                    'test': {
                        'min': min_value_ts,
                        'max': max_value_ts,
                        'mean': mean_value_ts,
                        'var': var_value_ts,
                    },
                    'train': {
                        'min': min_value_tr,
                        'max': max_value_tr,
                        'mean': mean_value_tr,
                        'var': var_value_tr,
                    },
                },

                # All metrics for visualization
                'all_metrics': {
                    'r2_mg': {'test': r2_values_ts_mg, 'train': r2_values_tr_mg},
                    'r2_ml': {'test': r2_values_ts_ml, 'train': r2_values_tr_ml},
                    'rmse_mg': {'test': rmse_values_ts_mg, 'train': rmse_values_tr_mg},
                    'rmse_ml': {'test': rmse_values_ts_ml, 'train': rmse_values_tr_ml},
                    'condvar_mg': {'test': condvar_ts_mg, 'train': condvar_tr_mg}
                },
            }

            # Finish timer
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print(f"\nElapsed Time: {elapsed_time:.2f} minutes")

        return self.results_dict_2

# --------------------------------------------------------------
# --------------------------------------------------------------

    def multi_kfold_splitting(self, data, var_x, var_y, n_splits_list):
        """
        Run the workflow using k-fold cross-validation for multiple scenarios.

        Parameters:
        - data: pandas.DataFrame
            Dataset containing input features and target variable in original units.
        - var_x: list[str]
            List of input feature column names.
        - var_y: list[str]
            List of target variable column names.
        - n_splits_list: list[int]
            List of n_splits (list of k's in the multi k-fold splitting iterations) values to iterate over.
            These are the number of times that k-fold splitting is going to be performed.

        - n_splits_list: list[int]
            List of k values (number of folds) to iterate over.

        Returns:
        - results_dict_3: dict
            Dictionary containing results for all k-fold configurations.
        """

        var_full = var_x + [var_y]

        self.results_dict_3 ={}

        for n_splits in n_splits_list:
            print(f"\nRunning k-fold CV workflow for k={n_splits}")
            start_time = time.time()
            
            # Initialize arrays to store statistics
            mean_value_ts, var_value_ts, max_value_ts, min_value_ts = np.array([]), np.array([]), np.array([]), np.array([])
            mean_value_tr, var_value_tr, max_value_tr, min_value_tr = np.array([]), np.array([]), np.array([]), np.array([])
    
            # Initialize arrays for MG and ML metrics
            r2_values_ts_mg, r2_values_tr_mg, rmse_values_ts_mg, rmse_values_tr_mg = np.array([]), np.array([]), np.array([]), np.array([])
            r2_values_ts_ml, r2_values_tr_ml, rmse_values_ts_ml, rmse_values_tr_ml = np.array([]), np.array([]), np.array([]), np.array([])
    
            # Initialize K-Fold object
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            # Create test_prop variable to record in the resulst dictionary
            test_prop = (100 / n_splits) / 100
    
            # Loop over k non-overlapping splits
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
                sys.stdout.write(f"\rIteration {fold_idx+1}/{n_splits} running...")

                # Split data into training and test sets using indices
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
    
                # Add prefix "NS_" to each variable to be compatible after transformation
                var_x_ns = ['NS_' + i for i in var_x]
                var_y_ns = f'NS_{var_y}'

                # NS fit_transform for multiple variables
                nstransformer_mv = rmsp.NSTransformerMV(warn_no_wt=False)
                train_prepared = nstransformer_mv.fit_transform(train_data[var_full])
                X_train, y_train = train_prepared[var_x_ns].values, train_prepared[var_y_ns].values
        
                # Transform test data, applying the parameters from the training set, thus avoiding data leakage
                test_prepared = nstransformer_mv.transform(test_data[var_full])
                X_test, y_test = test_prepared[var_x_ns].values, test_prepared[var_y_ns].values
                
                # # Fit and transform training data
                # train_prepared = self.trans_pipeline.fit_transform(train_data)
                # X_train, y_train = train_prepared[var_x_ns].values, train_prepared[var_y_ns].values
    
                # # Transform test data, applying the parameters from the training set, thus avoiding data leakage
                # test_prepared = self.trans_pipeline.transform(test_data)
                # X_test, y_test = test_prepared[var_x_ns].values, test_prepared[var_y_ns].values
    
                # Collect statistics for training and test data
                mean_value_ts = np.append(mean_value_ts, y_test.mean())
                var_value_ts = np.append(var_value_ts, y_test.var())
                min_value_ts = np.append(min_value_ts, y_test.min())
                max_value_ts = np.append(max_value_ts, y_test.max())
    
                mean_value_tr = np.append(mean_value_tr, y_train.mean())
                var_value_tr = np.append(var_value_tr, y_train.var())
                min_value_tr = np.append(min_value_tr, y_train.min())
                max_value_tr = np.append(max_value_tr, y_train.max())
    
                # Fit MG and ML models
                self.model_mg.fit(X_train, y_train)
                self.model_ml.fit(X_train, y_train)
    
                # Predictions for MG
                pred_ts_mg, _, _ = self.model_mg.predict(X_test, return_pdf=False)
                pred_tr_mg, _, _ = self.model_mg.predict(X_train, return_pdf=False)
    
                # Predictions for ML
                pred_ts_ml, _, _ = self.model_ml.predict(X_test, return_pdf=False)
                pred_tr_ml, _, _ = self.model_ml.predict(X_train, return_pdf=False)
    
                # Metrics for MG
                r2_values_ts_mg = np.append(r2_values_ts_mg, r2_score(y_test, pred_ts_mg))
                r2_values_tr_mg = np.append(r2_values_tr_mg, r2_score(y_train, pred_tr_mg))
                rmse_values_ts_mg = np.append(rmse_values_ts_mg, np.sqrt(mean_squared_error(y_test, pred_ts_mg)))
                rmse_values_tr_mg = np.append(rmse_values_tr_mg, np.sqrt(mean_squared_error(y_train, pred_tr_mg)))
    
                # Metrics for ML
                r2_values_ts_ml = np.append(r2_values_ts_ml, r2_score(y_test, pred_ts_ml))
                r2_values_tr_ml = np.append(r2_values_tr_ml, r2_score(y_train, pred_tr_ml))
                rmse_values_ts_ml = np.append(rmse_values_ts_ml, np.sqrt(mean_squared_error(y_test, pred_ts_ml)))
                rmse_values_tr_ml = np.append(rmse_values_tr_ml, np.sqrt(mean_squared_error(y_train, pred_tr_ml)))

            # Ensure nested dictionary structure
            if n_splits not in self.results_dict_3:
                self.results_dict_3[n_splits] = {}
           
            # Store results for current run
            self.results_dict_3[n_splits][test_prop] = {
                # MG statistics
                'r2_stats_mg': self._compute_stats(r2_values_ts_mg, r2_values_tr_mg),
                'rmse_stats_mg': self._compute_stats(rmse_values_ts_mg, rmse_values_tr_mg),
    
                # ML statistics
                'r2_stats_ml': self._compute_stats(r2_values_ts_ml, r2_values_tr_ml),
                'rmse_stats_ml': self._compute_stats(rmse_values_ts_ml, rmse_values_tr_ml),
    
                # Data statistics
                'data_stats': {
                    'test': {
                        'min': min_value_ts,
                        'max': max_value_ts,
                        'mean': mean_value_ts,
                        'var': var_value_ts,
                    },
                    'train': {
                        'min': min_value_tr,
                        'max': max_value_tr,
                        'mean': mean_value_tr,
                        'var': var_value_tr,
                    },
                },
    
                # All metrics for visualization
                'all_metrics': {
                    'r2_mg': {'test': r2_values_ts_mg, 'train': r2_values_tr_mg},
                    'r2_ml': {'test': r2_values_ts_ml, 'train': r2_values_tr_ml},
                    'rmse_mg': {'test': rmse_values_ts_mg, 'train': rmse_values_tr_mg},
                    'rmse_ml': {'test': rmse_values_ts_ml, 'train': rmse_values_tr_ml},
                },
            }
    
            # Finish timer
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print(f"\nElapsed Time: {elapsed_time:.2f} minutes")
    
        return self.results_dict_3
    
    @staticmethod
    def _compute_stats(test_values, train_values):
        """
        Compute minimum, maximum, mean, and variance for test and train values.

        Parameters:
        - test_values: np.array
            Metrics from the test set.
        - train_values: np.array
            Metrics from the train set.

        Returns:
        - dict: Dictionary of statistics for test and train sets.
        """
        return {
            'test': {
                'min': np.min(test_values),
                'max': np.max(test_values),
                'mean': np.mean(test_values),
                'var': np.var(test_values),
            },
            'train': {
                'min': np.min(train_values),
                'max': np.max(train_values),
                'mean': np.mean(train_values),
                'var': np.var(train_values),
            },
        }
    
def get_metrics_cv(results_dict, n_splits, test_prop):
    """
    Access metrics for a specific n_splits and test_prop scenario.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - n_splits: int
        The n_splits scenario to retrieve results for.
    - test_prop: flt
        The test_prop to retrieve results for.

    Returns:
    - dict: Metrics for the specified n_splits and test_prop.
    """
    if n_splits not in results_dict or test_prop not in results_dict[n_splits]:
        raise ValueError(f"Results for either n_splits={n_splits} or test_prop={test_prop} not found.\
                           \nIf accessing k-folding results, test_prop must be compatible with n_splits!")
    return results_dict[n_splits][test_prop]

# --------------------------------------------------------------
# --------------------------------------------------------------

def print_result_table_cv(results_dict, n_splits, test_prop, approach, subset):
    """
    Create a DataFrame of statistics (R² and RMSE) for the chosen approach and subset.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - n_splits: int
        The number of splits to retrieve results for.
    - test_prop: float
        The test proportion to retrieve results for.
    - approach: str
        Either "mg" or "ml" to specify the approach.
    - subset: str
        Either "test" or "train" to specify the data subset.
    
    Returns:
    - pd.DataFrame: DataFrame containing statistics for R² and RMSE.
    """

    # Check given parameters    
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")
    if subset not in ["test", "train"]:
        raise ValueError("Invalid subset. Choose 'test' or 'train'.")

    # Access metrics for a specific n_splits and test_prop scenario.
    if n_splits not in results_dict or test_prop not in results_dict[n_splits]:
        raise ValueError(f"Results for either n_splits={n_splits} or test_prop={test_prop} not found.\
                           \nIf accessing k-folding results, test_prop must be compatible with n_splits!")
    results = results_dict[n_splits][test_prop]

    # Extract R² and RMSE statistics
    r2_stats = results[f'r2_stats_{approach}'][subset]
    rmse_stats = results[f'rmse_stats_{approach}'][subset]
    
    # Create the DataFrame
    metrics = {
        "Metrics": ['Minimum', 'Maximum', 'Average', 'Variance'],
        "R²": [r2_stats['min'], r2_stats['max'], r2_stats['mean'], r2_stats['var']],
        "RMSE": [rmse_stats['min'], rmse_stats['max'], rmse_stats['mean'], rmse_stats['var']]
    }

    return pd.DataFrame(metrics).round(4)

# --------------------------------------------------------------
# --------------------------------------------------------------

def plot_r2_cv(results_dict, n_splits, test_prop, r2_ref, approach):
    """
    Plot R² values for both test and training data for the chosen approach.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - n_splits: int
        The number of splits to plot results for.
    - test_prop: float
        The test proportion to plot results for.
    - r2_ref: float
        Reference R² value for the full dataset.
    - approach: str
        Either "mg" or "ml" to specify the approach.
    """
    
    # Check given parameters    
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")

    # Access metrics for a specific n_splits and test_prop scenario.
    if n_splits not in results_dict or test_prop not in results_dict[n_splits]:
        raise ValueError(f"Results for either n_splits={n_splits} or test_prop={test_prop} not found.\
                           \nIf accessing k-folding results, test_prop must be compatible with n_splits!")
    results = results_dict[n_splits][test_prop]
    model_key = f'r2_{approach}'

    # Extract sorted R² values and average values for both subsets
    sorting_idx = np.argsort(results['all_metrics'][model_key]['test'])
    r2_values_tr_sorted = results['all_metrics'][model_key]['train'][sorting_idx]
    r2_values_ts_sorted = results['all_metrics'][model_key]['test'][sorting_idx]
    r2_tr_avg = results[f'r2_stats_{approach}']['train']['mean']
    r2_ts_avg = results[f'r2_stats_{approach}']['test']['mean']

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_splits + 1), r2_values_tr_sorted, linestyle='-', color='green', label="R² from train data")
    plt.plot(range(1, n_splits + 1), r2_values_ts_sorted, linestyle='--', color='green', label="R² from test data")
    plt.axhline(y=r2_ref, color='purple', linestyle='--', linewidth=1, label=f"Reference R² = {np.round(r2_ref, 4)} (full dataset)")
    plt.axhline(y=r2_tr_avg, color='grey', linestyle=':', linewidth=2, label=f"Average train R² = {np.round(r2_tr_avg, 4)}")
    plt.axhline(y=r2_ts_avg, color='dimgrey', linestyle='--', linewidth=1, label=f"Average test R² = {np.round(r2_ts_avg, 4)}")

    plt.title(f'R² Values for the {approach.upper()} case -- {n_splits} splits -- {test_prop*100}% test')
    plt.xlabel("Ordered Split Index (sorted by the R2 value on the test set)")
    plt.ylabel("R² Value")
    plt.legend(title="Legend")
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------
# --------------------------------------------------------------

def plot_rmse_cv(results_dict, n_splits, test_prop, rmse_ref, approach):
    """
    Plot RMSE values for both test and training data for the chosen approach.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - n_splits: int
        The number of splits to plot results for.
    - test_prop: float
        The test proportion to plot results for.
    - rmse_ref: float
        Reference RMSE value for the full dataset.
    - approach: str
        Either "mg" or "ml" to specify the approach.
    """

    # Check given parameters    
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")

    # Access metrics for a specific n_splits and test_prop scenario.
    if n_splits not in results_dict or test_prop not in results_dict[n_splits]:
        raise ValueError(f"Results for either n_splits={n_splits} or test_prop={test_prop} not found.\
                           \nIf accessing k-folding results, test_prop must be compatible with n_splits!")
    results = results_dict[n_splits][test_prop]
    model_key = f'r2_{approach}'

    # Extract sorted RMSE values and average values for both subsets
    sorting_idx = np.argsort(results['all_metrics'][f'r2_{approach}']['test'])
    rmse_values_tr_sorted = results['all_metrics'][model_key]['train'][sorting_idx]
    rmse_values_ts_sorted = results['all_metrics'][model_key]['test'][sorting_idx]
    rmse_tr_avg = results[f'rmse_stats_{approach}']['train']['mean']
    rmse_ts_avg = results[f'rmse_stats_{approach}']['test']['mean']

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_splits + 1), rmse_values_tr_sorted, linestyle='-', color='blue', label="RMSE from train data")
    plt.plot(range(1, n_splits + 1), rmse_values_ts_sorted, linestyle='--', color='blue', label="RMSE from test data")
    plt.axhline(y=rmse_ref, color='purple', linestyle='--', linewidth=1, label=f"Reference RMSE = {np.round(rmse_ref, 4)} (full dataset)")
    plt.axhline(y=rmse_tr_avg, color='grey', linestyle=':', linewidth=2, label=f"Average train RMSE = {np.round(rmse_tr_avg, 4)}")
    plt.axhline(y=rmse_ts_avg, color='dimgrey', linestyle='--', linewidth=1, label=f"Average test RMSE = {np.round(rmse_ts_avg, 4)}")

    plt.title(f'RMSE Values for the {approach.upper()} case -- {n_splits} splits -- {test_prop*100}% test')
    plt.xlabel("Ordered Split Index (sorted by the R-squared value on the test set)")
    plt.ylabel("RMSE Value")
    plt.legend(title="Legend")
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------
# --------------------------------------------------------------

def plot_variance_cv(results_dict, n_splits, test_prop, ref_variance, approach):
    """
    Plot variance values for both test and training data for the chosen approach.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - n_splits: int
        The number of splits to plot results for.
    - test_prop: float
        The test proportion to plot results for.
    - ref_variance: float
        Reference variance of a full dataset in a perfectlly Gaussian environment, declustering weights considered.
    - approach: str
        Either "mg" or "ml" to specify the approach.
    """
    
    # Check given parameters    
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")

    # Access metrics for a specific n_splits and test_prop scenario.
    if n_splits not in results_dict or test_prop not in results_dict[n_splits]:
        raise ValueError(f"Results for either n_splits={n_splits} or test_prop={test_prop} not found.\
                           \nIf accessing k-folding results, test_prop must be compatible with n_splits!")
    results = results_dict[n_splits][test_prop]

    # Extract sorted variance values for both subsets
    sorting_idx = np.argsort(results['all_metrics'][f'r2_{approach}']['test'])
    var_ts_sorted = results['data_stats']['test']['var'][sorting_idx]
    var_tr_sorted = results['data_stats']['train']['var'][sorting_idx]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_splits + 1), var_ts_sorted, linestyle='--', color='grey', label=f"NAIVE variance in the test set ({approach.upper()})")
    plt.plot(range(1, n_splits + 1), var_tr_sorted, linestyle='-', color='grey', label=f"NAIVE variance in the training set ({approach.upper()})")
    plt.axhline(y=ref_variance, color='purple', linestyle=':', linewidth=2, label="Reference variance in the perfect Gaussian environment")
    
    # Adding explanatory text
    plt.text(
        x=0.05, y=0.05,  # Position as a fraction of the axes (0.05 = 5% from left/bottom)
        s="Note: The plot shows the variances of the subsets\n"
          "           NOT considering declustering weights.",  # Explanatory text
        fontsize=10, color='red', ha='left', va='bottom', transform=plt.gca().transAxes
    )
    
    plt.title(f'Variance for the {approach.upper()} case -- {n_splits} splits -- {test_prop*100}% test')
    plt.xlabel("Ordered Split Index (sorted by the R-squared value on the test set)")
    plt.ylabel("Variance")
    plt.legend(title="Legend")
    plt.grid(False)
    plt.show()

# --------------------------------------------------------------
# --------------------------------------------------------------

def plot_mean_cv(results_dict, n_splits, test_prop, ref_mean, approach):
    """
    Plot mean for the test or training set.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - n_splits: int
        The number of splits to plot results for.
    - test_prop: float
        The test proportion to plot results for.
    - ref_mean: float
        Reference mean of a full dataset in a perfectlly Gaussian environment, declustering weights considered.
    - subset: str
        Either "test" or "train" to specify the data subset.
    """
    
    # Check given parameters    
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")

    # Access metrics for a specific n_splits and test_prop scenario.
    if n_splits not in results_dict or test_prop not in results_dict[n_splits]:
        raise ValueError(f"Results for either n_splits={n_splits} or test_prop={test_prop} not found.\
                           \nIf accessing k-folding results, test_prop must be compatible with n_splits!")
    results = results_dict[n_splits][test_prop]

    # Extract and sort mean values for each subset
    sorting_idx = np.argsort(results['all_metrics'][f'r2_{approach}']['test'])
    mean_ts_sorted = results['data_stats']['test']['mean'][sorting_idx]
    mean_tr_sorted = results['data_stats']['train']['mean'][sorting_idx]
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_splits + 1), mean_ts_sorted, linestyle='--', color='darkorange', label=f"NAIVE mean value from test set")
    plt.plot(range(1, n_splits + 1), mean_tr_sorted, linestyle='-', color='darkorange', label=f"NAIVE mean value from train set")
    plt.axhline(y=ref_mean, color='purple', linestyle=':', linewidth=2, label="Reference mean in the perfect Gaussian environment")
    
    # Adding explanatory text
    plt.text(
        x=0.05, y=0.05,  # Position as a fraction of the axes (0.05 = 5% from left/bottom)
        s="Note: The plot shows the mean values of the subsets\n"
          "           NOT considering declustering weights.",  # Explanatory text
        fontsize=10, color='red', ha='left', va='bottom', transform=plt.gca().transAxes
    )
    
    plt.title(f'Mean for the {approach.upper()} case -- {n_splits} splits -- {test_prop*100}% test')
    plt.xlabel("Ordered Split Index (sorted by the R-squared value on the test set)")
    plt.ylabel("Mean Value")
    plt.legend(title="Legend")
    plt.show()

# --------------------------------------------------------------
# --------------------------------------------------------------

def plot_r2_accross_kf(results_dict, r2_ref, approach, leg_loc='lower right'):
    """
    Plot R2 values accross multiple runs.
    
    Parameters:
    - results_dict: dict
        Dictionary containing results for different scenarios.
    - r2_ref: float
        Reference R² value for the full dataset.
    - approach: str
        Either "mg" or "ml" to specify the method.
    """
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")

    # Retrieving necessary results from full dictionary
    results = results_dict
    n_splits_list = [f'{i} splits ({100 / i:.0f}% test)' for i in list(results.keys())]
    r2_values_ts_avg = [results[i][(100/i)/100][f'r2_stats_{approach}']['test']['mean'] for i in results.keys()]
    r2_values_tr_avg = [results[i][(100/i)/100][f'r2_stats_{approach}']['train']['mean'] for i in results.keys()]
    r2_values_ts_min = [results[i][(100/i)/100][f'r2_stats_{approach}']['test']['min'] for i in results.keys()]
    r2_values_ts_max = [results[i][(100/i)/100][f'r2_stats_{approach}']['test']['max'] for i in results.keys()]

    # Convert split identifiers to integer indices for plotting
    x_ticks = range(len(n_splits_list))

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, r2_values_ts_avg, marker='o', label='Average Test R2', linestyle='--', linewidth=2)
    plt.plot(x_ticks, r2_values_tr_avg, marker='o', label='Average Train R2', linestyle='-', linewidth=1)
    plt.fill_between(x_ticks, r2_values_ts_min, r2_values_ts_max, color='blue', alpha=0.2, label='Min-Max Range (test)')
    
    # Adding a horizontal line for the average value
    plt.axhline(y=np.mean(r2_values_ts_avg), color='grey', linestyle='--', linewidth=2, label=f"Average test R² accross multiple splitting scenarios = {np.mean(np.mean(r2_values_ts_avg)):.4f}")
    
    # Adding a horizontal line for the reference value
    plt.axhline(y=np.mean(r2_ref), color='purple', linestyle=':', linewidth=2, label=f"Reference R² = {np.round(r2_ref, 4)} (full dataset)")
    
    # Add labels, title, and legend
    plt.title(f'R2 Accross multiple splitting scenarios - {approach.upper()}', fontsize=14)
    plt.xlabel("Number of non-overlapping splits (k-folds)", fontsize=12)
    plt.ylabel("Average R2 Value", fontsize=12)
    plt.xticks(x_ticks, n_splits_list, rotation=45, ha='right', fontsize=10)
    plt.ylim(-0.10, 1)
    plt.legend(loc=leg_loc)
    plt.grid(False)
    
    # Show the plot
    plt.show()

# --------------------------------------------------------------
# --------------------------------------------------------------

def plot_r2_accross_cv(results_dict, r2_ref, approach, leg_loc='lower right'):
    """
    Plot R2 values accross multiple runs of overlapping cross-validation multiple splitting scenarios.
    
    Parameters:
    - r2_ref: float
        Reference R² value for the full dataset.
    - subset: str
        Either "test" or "train" to specify the data subset.
    - leg_loc: str
        Location of the legend relative to the plot
    """
    if approach not in ["mg", "ml"]:
        raise ValueError("Invalid approach. Choose 'mg' or 'ml'.")

    # Retrieving necessary results from the nested dictionary
    results = results_dict
    n_splits_list = []
    r2_values_ts_avg = []
    r2_values_tr_avg = []
    r2_values_ts_min = []
    r2_values_ts_max = []

    for n_splits, test_props_dict in results.items():
        for test_prop, metrics in test_props_dict.items():
            n_splits_list.append(f"{n_splits} scenarios | {test_prop*100}% test")
            r2_values_ts_avg.append(metrics[f'r2_stats_{approach}']['test']['mean'])
            r2_values_tr_avg.append(metrics[f'r2_stats_{approach}']['train']['mean'])
            r2_values_ts_min.append(metrics[f'r2_stats_{approach}']['test']['min'])
            r2_values_ts_max.append(metrics[f'r2_stats_{approach}']['test']['max'])

    # Convert split identifiers to integer indices for plotting
    x_ticks = range(len(n_splits_list))

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_ticks, r2_values_ts_avg, marker='o', label='Average Test R²', linestyle='--', linewidth=2)
    plt.plot(x_ticks, r2_values_tr_avg, marker='o', label='Average Train R²', linestyle='-', linewidth=1)
    # plt.fill_between(x_ticks, r2_values_ts_min, r2_values_ts_max, color='blue', alpha=0.2, label='Min-Max Range (test)')

    # Adding a horizontal line for the reference value
    plt.axhline(y=r2_ref, color='purple', linestyle=':', linewidth=2,
                label=f"Reference R² = {np.round(r2_ref, 4)} (full dataset)")

    # Adding a horizontal line for the average value
    plt.axhline(y=np.mean(r2_values_ts_avg), color='grey', linestyle=':', linewidth=2,
                label=f"Overall avg test R² = {np.round(np.mean(r2_values_ts_avg), 4)} (accross the multiple runs)")

    # Add labels, title, and legend
    plt.title(f'Average R² Values Across Splits and Test Proportions - {approach.upper()} Case', fontsize=14)
    plt.xlabel("Splitting Scenarios (n_splits | test_prop)", fontsize=12)
    plt.ylabel("Average R² Value", fontsize=12)
    plt.xticks(x_ticks, n_splits_list, rotation=45, ha='right', fontsize=10)
    plt.legend(loc=leg_loc, fontsize=10)
    plt.grid(False)

    # Show the plot
    plt.tight_layout()
    plt.show()