import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal, norm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import rmsp
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

class ConditionalMG:
    """
    Initializes the ConditionalMG class to apply the Multivariate Gaussian approach using the 
    normal equations to calculate the conditional mean and variance of a variable given other variables.

    Parameters:
    - n_quantiles: int, number of quantiles to compute.
    """
    def __init__(self, n_quantiles=500):
        self.weights = None
        self.corr_primary = None
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.round(np.linspace(0.001, 0.999, n_quantiles), 3)
        self.X_mean = None
        self.X_stddev = None
        self.y_train = None
        self.variance_y_fit = None
        self.variance_y_fit_std = None

    def fit(self, X, y):
        """
        Train the model using the training data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), the input features.
        - y: numpy array, shape (n_samples,), the target variable.
        """
        # Remove rows with missing values for training
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]       
        self.y_train = y[mask]
        
        # Calculate the reference variance from fit
        self.variance_y_fit = self.y_train.var()

        # Standardize X and y
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_std = (self.y_train - np.mean(self.y_train)) / np.std(self.y_train)
        self.X_mean = X_std.mean()
        self.X_stddev = X_std.std()
        
        # Compute correlation matrix
        data = np.hstack((y_std.reshape(-1, 1), X_std))
        data_corr = np.corrcoef(data, rowvar=False)
        corr_secondary = data_corr[1:, 1:]  # Correlations between secondary variables
        self.corr_primary = data_corr[0, 1:].reshape(-1, 1)  # Correlations between primary and secondary variables

        # Solve normal equations
        self.weights = np.linalg.solve(corr_secondary, self.corr_primary)
        
    def predict(self, X, return_pdf=False):
        """
        Predict the conditional means, variances, quantiles, and optionally PDF values using the trained model.
        """
        # X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        X_std = (X - self.X_mean) / self.X_stddev
        conditional_means = np.zeros(X.shape[0])
        conditional_variances = np.zeros(X.shape[0])
        conditional_quantiles = np.zeros((X.shape[0], self.n_quantiles))
        conditional_pdfs = [] if return_pdf else None
        primary_ranges = [] if return_pdf else None

        for idx, row in enumerate(X_std):
            weighted_mean = np.nansum(self.weights.flatten() * row)
            weighted_corr_sum = np.nansum(self.weights.flatten() * self.corr_primary.flatten())
            
            cond_variance = 1.0 - weighted_corr_sum
            
            primary_range = np.linspace(weighted_mean - 3 * np.sqrt(cond_variance), 
                                        weighted_mean + 3 * np.sqrt(cond_variance), self.n_quantiles).reshape(-1, 1)
            
            pdf_values = (1 / np.sqrt(2 * np.pi * cond_variance)) * np.exp(-((primary_range - weighted_mean) ** 2) / (2 * cond_variance))
            
            cdf_values = np.cumsum(pdf_values) / pdf_values.sum()
            quantiles = [primary_range[np.searchsorted(cdf_values, q)][0] * np.std(self.y_train) + np.mean(self.y_train) for q in self.quantile_levels]
            
            conditional_means[idx] = weighted_mean * np.std(self.y_train) + np.mean(self.y_train)
            conditional_variances[idx] = cond_variance * self.variance_y_fit
            conditional_quantiles[idx] = quantiles
            
            if return_pdf:
                conditional_pdfs.append(pdf_values.flatten())
                primary_ranges.append(primary_range.flatten())

        if return_pdf:
            return conditional_means, conditional_variances, conditional_quantiles, np.array(conditional_pdfs), np.array(primary_ranges)
        else:
            return conditional_means, conditional_variances, conditional_quantiles
    
    def get_weights(self):
        """
        Get the weights obtained from the training data.

        Returns:
        - numpy array: The weights.
        """
        return self.weights

    def get_quantile_levels(self):
        """
        Get the quantile levels that were used.

        Returns:
        - numpy array: The quantile levels.
        """
        return self.quantile_levels

#############################################################################################
#############################################################################################
#############################################################################################

class ConditionalGMM:
    def __init__(self, n_components, n_quantiles=500, random_state=None):
        """
        Initializes the ConditionalGMM class to apply the Gaussian Mixure Models approach to calculate 
        the mean, variance and quantiles of a conditional distribution of a variable given other variables.
        
        Parameters:
        - n_components: int, number of mixture components.
        - n_quantiles: int, number of quantiles to compute.
        - random_state: int, random seed for reproducibility.
        """
        self.n_components = n_components
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=random_state)
        self.quantile_levels = np.round(np.linspace(0.001, 0.999, n_quantiles), 3)

    def fit(self, X, y):
        """
        Fits the Gaussian Mixture Model to the data.
        
        Parameters:
        - X: np.ndarray, array of secondary variables.
        - y: np.ndarray, array of primary variable.
        """

        # Define data
        self.X = X
        self.y = y

       # Remove rows with missing values for training
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # Preparing data and fitting gmm using sklearn
        data = np.column_stack((y, X))
        self.gmm.fit(data)



    def predict(self, X, return_pdf=False):
        """
        Predict the conditional means, variances, quantiles, and optionally PDF values using the trained model.
        
        Parameters:
        - X: numpy array, shape (n_samples, n_features), the input features.
        - return_pdf: bool, if True, returns the PDF values.
        
        Returns:
        - conditional_means: np.ndarray, array of conditional means for each sample.
        - conditional_variances: np.ndarray, array of conditional variances for each sample.
        - conditional_quantiles: np.ndarray, array of conditional quantiles for each sample.
        - conditional_pdfs (optional): np.ndarray, array of PDF values for each sample (if return_pdf is True).
        - primary_ranges (optional): np.ndarray, array of reference values to calculate the PDF values for each sample (if return_pdf is True).
        """

        # Initialize arrays to store the results
        num_samples = X.shape[0]
        conditional_means = np.zeros(num_samples)
        conditional_variances = np.zeros(num_samples)
        conditional_quantiles = np.zeros((num_samples, self.n_quantiles))
        conditional_pdfs = [] if return_pdf else None
        primary_ranges = [] if return_pdf else None

        for i, row in enumerate(X):
            # Assigning the attributes of the fitted GMM
            weights = self.gmm.weights_
            means = self.gmm.means_
            covariances = self.gmm.covariances_

            # Initializing the lists to keep the results
            cond_means = []
            cond_vars = []
            cond_weights = []
            
            for k in range(self.gmm.n_components):
                mean_k = means[k]
                cov_k = covariances[k]

                # Partition the mean vector
                mean_1 = mean_k[0]
                mean_2 = mean_k[1:]

                # Partition the covariance matrix
                cov_11 = cov_k[0, 0]
                cov_22 = cov_k[1:, 1:]
                cov_12 = cov_k[0, 1:]

                # Ensure cov_12 is reshaped to be a column vector
                cov_12 = cov_12.reshape(-1, 1)

                # Compute the conditional mean and covariance
                cond_mean = mean_1 + cov_12.T @ np.linalg.inv(cov_22) @ (row - mean_2)
                cond_cov = cov_11 - cov_12.T @ np.linalg.inv(cov_22) @ cov_12

                cond_means.append(cond_mean.item())  # Ensure it's a scalar
                cond_vars.append(cond_cov.item())  # Ensure it's a scalar

                # Update the weights
                cond_weight = weights[k] * multivariate_normal.pdf(row, mean=mean_2, cov=cov_22)
                cond_weights.append(cond_weight)
            
            # Normalize the weights
            cond_weights = np.array(cond_weights)
            cond_weights /= cond_weights.sum()

            # Compute the overall conditional mean and variance
            overall_cond_mean = np.sum(np.array(cond_means) * cond_weights)
            overall_cond_var = np.sum(np.array(cond_vars) * cond_weights)
            
            # Define a range of values for the primary variable to estimate conditional density
            primary_range = np.linspace(overall_cond_mean - 3, overall_cond_mean + 3, self.n_quantiles).reshape(-1, 1)
            
            # Initialize the list with the values of the pdf
            pdf_values = np.zeros_like(primary_range)
            
            # Get pdf
            for weight, mean, cov in zip(cond_weights, cond_means, cond_vars):
                component_pdf_wt = weight * norm.pdf(primary_range, mean, np.sqrt(cov))
                pdf_values += component_pdf_wt

            # Get cdf
            cdf_values = np.cumsum(pdf_values) / pdf_values.sum()

            # Calculating quantiles
            for j, quant in enumerate(self.quantile_levels):
                quantile_value = primary_range[np.searchsorted(cdf_values, quant)][0]  # Looks for the nth element in the primary_range array for each of the quantiles in the list
                conditional_quantiles[i, j] = quantile_value

            # Store the results           
            conditional_means[i] = overall_cond_mean
            conditional_variances[i] = overall_cond_var
        
            if return_pdf:
                conditional_pdfs.append(pdf_values.flatten())
                primary_ranges.append(primary_range.flatten())

        if return_pdf:
            return conditional_means, conditional_variances, conditional_quantiles, np.array(conditional_pdfs), np.array(primary_ranges)
        else:
            return conditional_means, conditional_variances, conditional_quantiles

    def get_weights(self):
        """
        Get the weights obtained from the training data.

        Returns:
        - numpy array: The weights.
        """
        return self.gmm.weights_

    def get_ccovariances(self):
        """
        Get the covariances from each gmm component obtained from the training data.

        Returns:
        - numpy array: The covariances.
        """
        return self.gmm.covariances_

    def get_cmeans(self):
        """
        Get the means from each gmm component obtained from the training data.

        Returns:
        - numpy array: The means.
        """
        return self.gmm.means_

    def get_quantile_levels(self):
        """
        Get the quantile levels that were used.

        Returns:
        - numpy array: The quantile levels.
        """
        return self.quantile_levels
    
    def get_params(self):
        """
        Get the hyperparameters that were used.

        Returns:
        - dict: Parameter names mapped to their values.
        """
        return {'n_components': self.n_components}

#############################################################################################
#############################################################################################
#############################################################################################

class ConditionalKDE:
    def __init__(self, bandwidth, n_quantiles=500):
        """
        Initializes the ConditionalKDE class to apply the Kernel Density Approach to calculate 
        the mean, variance and quantiles of a conditional distribution of a variable given other variables.
        
        Parameters:
        - bandwidth (float): Bandwidth parameter for Kernel Density Estimation.
        - n_quantiles (int): Number of quantiles to compute.
        """
        self.bandwidth = bandwidth
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.round(np.linspace(0.001, 0.999, self.n_quantiles), 3)
        self.kde = None
        self.kde_marginal = None

    def fit(self, X, y):
        """
        Fit the Conditional KDE model on given data.

        Parameters:
        - X (numpy.ndarray): Secondary variables array of shape (n_samples, n_features).
        - y (numpy.ndarray): Main variable array of shape (n_samples,).
        """
        # Ensure y and X are numpy arrays
        y = np.asarray(y).reshape(-1, 1) if len(y.shape) == 1 else np.asarray(y)
        X = np.asarray(X)

        # Remove rows with missing values for training
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).ravel()
        X = X[mask]
        y = y[mask]
      
        # Fit the KDE model on the joint data of the entire dataset
        data_train = np.hstack((y, X))
        self.kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde.fit(data_train)

        # Fit KDE for the marginal density of the secondary variables of the entire dataset
        self.kde_marginal = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth)
        self.kde_marginal.fit(X)

        # Store the primary range for predictions
        self.primary_range = np.linspace(y.min() - 3, y.max() + 3, self.n_quantiles).reshape(-1, 1)
        self.spacing = self.primary_range[1] - self.primary_range[0]

    def predict(self, X, return_pdf=False):
        """
        Predict conditional means, variances, and quantiles for new data.

        Parameters:
        - X (numpy.ndarray): New data array of secondary variables of shape (n_samples, n_features).
        - return_pdf (bool): Whether to return the PDF values for each sample.

        Returns:
        - conditional_means (numpy.ndarray): Array of conditional means for each sample.
        - conditional_variances (numpy.ndarray): Array of conditional variances for each sample.
        - conditional_quantiles (numpy.ndarray): Array of conditional quantiles for each sample.
        - conditional_pdfs (optional): np.ndarray, array of PDF values for each sample (if return_pdf is True).
        - primary_ranges (optional): np.ndarray, array of reference values to calculate the PDF values for each sample (if return_pdf is True).
        """
        # Ensure X is a numpy array
        X = np.asarray(X)

        # Initialize arrays to store the results
        conditional_means = np.zeros(X.shape[0])
        conditional_variances = np.zeros(X.shape[0])
        conditional_quantiles = np.zeros((X.shape[0], self.n_quantiles))
        conditional_pdfs = [] if return_pdf else None
        primary_ranges = [] if return_pdf else None

        # Compute the log marginal density for each data point
        log_marginal_density = self.kde_marginal.score_samples(X)

        for i in range(X.shape[0]):
            xi = X[i].reshape(1, -1)
            joint_density = np.exp(self.kde.score_samples(np.hstack((self.primary_range, np.repeat(xi, self.primary_range.shape[0], axis=0)))))
            conditional_density = joint_density / np.exp(log_marginal_density[i])
            pdf_values = conditional_density / (np.sum(conditional_density) * self.spacing)
            conditional_density /= np.sum(conditional_density)

            # Calculate conditional means and variances from the distribution
            conditional_means[i] = np.sum(conditional_density * self.primary_range.ravel())
            conditional_variances[i] = np.sum(conditional_density * (self.primary_range.ravel() - conditional_means[i])**2)

            # Calculate the cumulative distribution function (CDF)
            cdf = np.cumsum(conditional_density)

            # Calculate quantiles
            for j, quant in enumerate(self.quantile_levels):
                quantile_value = self.primary_range[np.searchsorted(cdf, quant)][0]
                conditional_quantiles[i, j] = quantile_value

            # Store PDF values if requested
            if return_pdf:
                conditional_pdfs.append(pdf_values)
                primary_ranges.append(self.primary_range.ravel())

        if return_pdf:
            return conditional_means, conditional_variances, conditional_quantiles, np.array(conditional_pdfs), np.array(primary_ranges)
        else:
            return conditional_means, conditional_variances, conditional_quantiles
        
    def get_quantile_levels(self):
        """
        Get the quantile levels that were used.

        Returns:
        - numpy array: The quantile levels.
        """
        return self.quantile_levels
    
    def get_params(self):
        """
        Get the hyperparameters that were used.

        Returns:
        - dict: Parameter names mapped to their values.
        """
        return {'bandwidth': self.bandwidth}

#############################################################################################
#############################################################################################
#############################################################################################

class ConditionalML:
    """
    Initializes the ConditionalML class to apply a machine learning algorithm to predict values
    and assume a Gaussian distribution of variance '1 - r_squared' centered on the predicted value
    to construct a conditional distribution and calculate the quantiles.

    Parameters:
    - n_quantiles: int, number of quantiles to compute.
    """
    def __init__(self, regressor, n_quantiles=500, reg_hyperparams=None, r_squared=0):
        """
        Initialize the ConditionalML class with the provided parameters.
        
        Parameters:
        - regressor: The regression algorithm to be used for conditional distribution (LinearRegression, RandomForestRegressor, GradientBoostingRegressor, KNN or SVR).
        - n_quantiles (int): The number of quantiles to use for conditional distributions.
        - reg_hyperparams (dict): Hyperparameters specific to the chosen regressor.
        - r_squared (float): R-squared value for calculating the conditional variance 
        """
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.round(np.linspace(0.001, 0.999, n_quantiles), 3)
        self.regressor = regressor(**(reg_hyperparams if reg_hyperparams is not None else {}))
        self.r_squared = r_squared

    def fit(self, X, y):
        """
        Train the model using the training data.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), the input features.
        - y: numpy array, shape (n_samples,), the target variable.
        """

        # Remove rows with missing values for training
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]

        # Fitting the model with the chosen algorithm
        self.regressor.fit(X, y)
      
    def predict(self, X, return_pdf=False):
        """
        Predict the conditional means, variances, quantiles, and optionally PDF values using the trained model.

        Parameters:
        - X: numpy array, shape (n_samples, n_features), the input features.
        - return_pdf: bool, if True, returns the PDF values.

        Returns:
        - conditional_means: numpy array, the conditional means.
        - conditional_variances: numpy array, the conditional variances.
        - conditional_quantiles: np.ndarray, array of conditional quantiles for each sample.
        - conditional_pdfs (optional): np.ndarray, array of PDF values for each sample (if return_pdf is True).
        - primary_ranges (optional): np.ndarray, array of reference values to calculate the PDF values for each sample (if return_pdf is True).
        """
        # Predict the means
        conditional_means = self.regressor.predict(X)
        
        # Calculate the conditional variance
        conditional_variances = np.ones(X.shape[0]) * (1.0 - self.r_squared)
        
        # Initialize arrays to store the quantiles and PDFs if needed
        conditional_quantiles = np.zeros((X.shape[0], self.n_quantiles))
        conditional_pdfs = [] if return_pdf else None
        primary_ranges = [] if return_pdf else None

        # Define a range of values for the primary variable to estimate the conditional density
        for idx, mean in enumerate(conditional_means):
            variance = conditional_variances[idx]
            primary_range = np.linspace(mean - 3 * np.sqrt(variance), mean + 3 * np.sqrt(variance), self.n_quantiles).reshape(-1, 1)
            pdf_values = (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((primary_range - mean) ** 2) / (2 * variance))
            cdf_values = np.cumsum(pdf_values) / np.sum(pdf_values)
            
            for j, quant in enumerate(self.quantile_levels):
                quantile_value = primary_range[np.searchsorted(cdf_values, quant)][0]
                conditional_quantiles[idx, j] = quantile_value
            
            if return_pdf:
                conditional_pdfs.append(pdf_values.flatten())
                primary_ranges.append(primary_range.flatten())
        
        if return_pdf:
            return conditional_means, conditional_variances, conditional_quantiles, np.array(conditional_pdfs), np.array(primary_ranges)
        else:
            return conditional_means, conditional_variances, conditional_quantiles
    
    def get_quantile_levels(self):
        """
        Get the quantile levels that were used.

        Returns:
        - numpy array: The quantile levels.
        """
        return self.quantile_levels

#############################################################################################
#############################################################################################
#############################################################################################

class ConditionalDistCV:
    def __init__(self, data, y_var, X_vars, model, model_params=None, declus_csize=100, n_quantiles=500, nscore_extrapolation=20, nscore_extrapolation_y=20, n_folds=10, random_state=42):
        """
        Initialize the ConditionalDistCV class with the provided parameters.
        
        Parameters:
        data (pd.DataFrame): The input data frame.
        y_var (str): The name of the target variable.
        X_vars (list): A list of secondary variables.
        model: The model to be used for conditional distribution (ConditionalMG, ConditionalGMM, ConditionalKDE or ConditionalML).
        model_params (dict): Parameters specific to the chosen model.
        declus_csize (int): Cell size for declustering.
        n_quantiles (int): The number of quantiles to use for conditional distributions.
        nscore_extrapolation (int): extrapolation below minimum and above maximum data values to make sure fitted nscore object will include all test set values.
        nscore_extrapolation_y (int): extrapolation (for the target variable) below minimum and above maximum data values to make sure fitted nscore object will include all test set values.
        n_folds (int): The number of folds for cross-validation.
        random_state (int): The random state for reproducibility.
        """
        self.data = data
        self.y_var = y_var
        self.X_vars = X_vars
        self.model = model
        self.model_params = model_params if model_params is not None else {}
        self.declus_csize = declus_csize
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.round(np.linspace(0.001, 0.999, self.n_quantiles), 3)
        self.nscore_extrapolation = nscore_extrapolation
        self.nscore_extrapolation_y = nscore_extrapolation_y
        self.n_folds = n_folds
        self.random_state = random_state

    def run_cv(self):
        """
        Perform k-fold cross-validation to calculate conditional distributions and back-transform the results.
        
        Returns:
        dict: A dictionary containing results such as conditional means, variances, quantiles, and transformed values.
        """
        # Combine the secondary variables and the target variable
        variables = self.X_vars + [self.y_var]
        
        # Define the k-fold cross-validation split
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Initialize arrays to store results for the entire dataset
        conditional_means = np.zeros(len(self.data))
        conditional_variances = np.zeros(len(self.data))
        conditional_quantiles = np.zeros((self.data.shape[0], self.n_quantiles))
        conditional_pdfs = np.zeros((self.data.shape[0], self.n_quantiles))
        primary_ranges = np.zeros((self.data.shape[0], self.n_quantiles))

        # Initialize arrays to store back-transformed quantiles, means, and variances
        conditional_quantiles_original = np.zeros((self.data.shape[0], self.n_quantiles))
        conditional_means_original = np.zeros(self.data.shape[0])
        conditional_variances_original = np.zeros(self.data.shape[0])

        # DataFrames to store the normal score transformed values and weights
        ns_transformed_df_train = pd.DataFrame(index=self.data.index)
        ns_transformed_df_test = pd.DataFrame(index=self.data.index)

        # Array to store the fold label for each data point
        fold_labels = np.full(len(self.data), -1)  # Default label is -1, indicating no fold assignment initially

        # Perform k-fold cross-validation
        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.data)):
            # Split the data into training and testing sets
            train_data, test_data = self.data.iloc[train_index].copy(), self.data.iloc[test_index].copy()
            
            # Assign the current fold index to the test samples in the fold_labels array
            fold_labels[test_index] = fold_idx
            
            # Calculate declustering weights for the training data
            declus = rmsp.DeclusterCell(num_origins=100)
            for var in variables:
                train_data[f'{var}_wt'] = declus.decluster(train_data, var, self.declus_csize)

            # Despike the training data
            despike = rmsp.DespikeMVSpatial().fit(train_data[variables])
            temp = despike.transform(train_data, variables)
            for var, col in zip(variables, temp.columns):
                train_data[var] = temp[col]
            
            # Apply normal score transformation to the training data
            nscorers = {}
            for var in variables:
                if var == self.y_var:
                    # Use nscore_extrapolation_y for the target variable
                    tail_values_y = (train_data[var].min() - self.nscore_extrapolation_y, train_data[var].max() + self.nscore_extrapolation_y)
                    nscorers[var] = rmsp.NSTransformer()
                    train_data["NS_" + var] = nscorers[var].fit_transform(train_data[var], train_data[var + "_wt"], tail_values=tail_values_y, tail_powers=(5.0, 5.5))
                else:
                    # Use nscore_extrapolation for secondary variables
                    tail_values = (train_data[var].min() - self.nscore_extrapolation, train_data[var].max() + self.nscore_extrapolation)
                    nscorers[var] = rmsp.NSTransformer()
                    train_data["NS_" + var] = nscorers[var].fit_transform(train_data[var], train_data[var + "_wt"], tail_values=tail_values, tail_powers=(5.0, 5.0))
            
            ns_vars = ["NS_" + var for var in variables]

            # Apply the same preprocessing to the test data
            for var in variables:
                test_data[f'{var}_wt'] = declus.decluster(test_data, var, self.declus_csize)
            temp = despike.transform(test_data, variables)
            for var, col in zip(variables, temp.columns):
                test_data[var] = temp[col]
            for var in variables:
                test_data["NS_" + var] = nscorers[var].transform(test_data[var])
            
            # Save the normal score transformed values and weights in the DataFrames
            ns_transformed_df_train.loc[train_index, ns_vars + [f'{var}_wt' for var in variables]] = train_data[ns_vars + [f'{var}_wt' for var in variables]]
            ns_transformed_df_test.loc[test_index, ns_vars + [f'{var}_wt' for var in variables]] = test_data[ns_vars + [f'{var}_wt' for var in variables]]
                
            # Define training and test sets for the model
            X_train, X_test = train_data[ns_vars[:-1]].values, test_data[ns_vars[:-1]].values  # Secondary variables
            y_train, y_test = train_data[ns_vars[-1]].values, test_data[ns_vars[-1]].values  # Primary variable

            # Instantiate and fit the conditional model
            model_instance = self.model(n_quantiles=self.n_quantiles, **self.model_params)
            model_instance.fit(X_train, y_train)

            # Predict and store the results
            fold_conditional_means, fold_conditional_variances, fold_conditional_quantiles, fold_conditional_pdfs, fold_primary_ranges = model_instance.predict(X_test, return_pdf=True)
            conditional_means[test_index] = fold_conditional_means
            conditional_variances[test_index] = fold_conditional_variances
            conditional_quantiles[test_index] = fold_conditional_quantiles
            conditional_pdfs[test_index] = fold_conditional_pdfs
            primary_ranges[test_index] = fold_primary_ranges
            
            # Back-transform the quantiles to the original scale and store the results
            conditional_quantiles_original[test_index] = np.array([nscorers[self.y_var].inverse_transform(quantile) for quantile in fold_conditional_quantiles])
            conditional_means_original[test_index] = conditional_quantiles_original[test_index].mean(axis=1)
            conditional_variances_original[test_index] = conditional_quantiles_original[test_index].var(axis=1)

        # Store results in a dictionary for easy access
        results = {
            "conditional_means": conditional_means,
            "conditional_variances": conditional_variances,
            "conditional_quantiles": conditional_quantiles,
            "conditional_pdfs": conditional_pdfs,
            "primary_ranges": primary_ranges,
            "conditional_quantiles_original": conditional_quantiles_original,
            "conditional_means_original": conditional_means_original,
            "conditional_variances_original": conditional_variances_original,
            "ns_transformed_df_train": ns_transformed_df_train,
            "ns_transformed_df_test": ns_transformed_df_test,
            "fold_labels": fold_labels  # Adding fold labels to results
        }
        
        return results

    def get_quantile_levels(self):
        """
        Get the quantile levels that were used.

        Returns:
        - numpy array: The quantile levels.
        """
        return self.quantile_levels

#############################################################################################
#############################################################################################
#############################################################################################