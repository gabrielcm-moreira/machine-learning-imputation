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
