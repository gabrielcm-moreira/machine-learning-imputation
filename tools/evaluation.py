import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import skew

class PredictionEval:
    def __init__(self, df, true_value, predicted_value, conditional_variance=None):
        """
        Initialize the PredictionEval class with the required data and column names.

        Parameters:
        - df: DataFrame containing the data
        - true_value: str, column name for the true values
        - predicted_value: str, column name for the predicted values
        - conditional_variance: (Optional) Column name for the conditional variances
        """
        self.df = df
        self.true_value = true_value
        self.predicted_value = predicted_value
        self.conditional_variance = conditional_variance
        # Initialize a dictionary to store evaluation metrics
        self.results = {
            'R_squared': [],
            'MSE': [],
            'Diff_Btwn_Avg': [],
            'Mean_Rel_Diff(%)': [],
            'Correlation': [],
        }
        # If conditional variance is provided, add a key for it in the results dictionary
        if conditional_variance is not None:
            self.results['Avg_Cond_Variance'] = []

    def calc_r_squared(self):
        """
        Calculate R-squared value and append it to the results.
        """
        r2 = r2_score(self.df[self.true_value], self.df[self.predicted_value])
        self.results['R_squared'].append(r2)
    
    def calc_mse(self):
        """
        Calculate Mean Squared Error (MSE) and append it to the results.
        """
        mse = mean_squared_error(self.df[self.true_value], self.df[self.predicted_value])
        self.results['MSE'].append(mse)
    
    def calc_diff_between_avg(self):
        """
        Calculate Mean Absolute Difference (MAD) between the mean of true values and predicted values, and append it to the results.
        """
        dba = np.abs(self.df[self.true_value].mean() - self.df[self.predicted_value].mean())
        self.results['Diff_Btwn_Avg'].append(dba)

    def calc_mean_rel_diff(self):
        """
        Calculate Mean Relative Difference (MRD) between the mean of true values and predicted values, and append it to the results.
        """
        mrd = (np.abs(self.df[self.true_value].mean() - self.df[self.predicted_value].mean())) / self.df[self.true_value].mean()
        self.results['Mean_Rel_Diff(%)'].append(mrd * 100)
    
    def calc_correlation(self):
        """
        Calculate Pearson Correlation Coefficient between true values and predicted values, and append it to the results.
        """
        correlation, _ = scipy.stats.pearsonr(self.df[self.true_value], self.df[self.predicted_value])
        self.results['Correlation'].append(correlation)
    
    def calc_avg_conditional_variance(self):
        """
        Calculate the average conditional variance and append it to the results, if conditional variance is provided.
        """
        if self.conditional_variance is not None:
            avg_cond_var = self.df[self.conditional_variance].mean()
            self.results['Avg_Cond_Variance'].append(avg_cond_var)
    
    def evaluate(self, include_conditional_variance=False):
        """
        Evaluate the predictions by calculating various metrics.
        Optionally include the average conditional variance if specified.

        Parameters:
        - include_conditional_variance: Boolean flag to include conditional variance in the results (default is False)
        """
        self.calc_r_squared()
        self.calc_mse()
        self.calc_diff_between_avg()
        self.calc_mean_rel_diff()
        self.calc_correlation()
        if include_conditional_variance and self.conditional_variance is not None:
            self.calc_avg_conditional_variance()
        
    def get_results(self):
        """
        Return the evaluation results as a DataFrame.
        """
        return pd.DataFrame(self.results).reset_index(drop=True)

######################################################################################
######################################################################################
######################################################################################

class QuantilePlotter:
    def __init__(self, data, predicted_value, true_value, conditional_variance, quantile_levels, conditional_quantiles):
        """
        Initialize the QuantilePlotter class with the required data and column names.

        Parameters:
        - data: DataFrame containing the data
        - predicted_value: Column name for the predicted values
        - true_value: Column name for the true values
        - conditional_variance: Column name for the conditional variances
        - quantile_levels: np.array containing qualtile levels related to the calculated (conditional) quantiles
        - conditional_quantiles: np.array containing arrays with the calculated conditional quantiles for each sample in the DataFrame
        """
        self.data = data
        self.predicted_value = predicted_value
        self.true_value = true_value
        self.conditional_variance = conditional_variance
        self.quantile_levels = quantile_levels
        self.conditional_quantiles = conditional_quantiles

    def plot_quantiles(self, num_samples=100, nrows=10, ncols=10, figsize=(30, 20), sample_index=0, leg_loc='upper right', leg_visible=True):
        """
        Plot the conditional distributions based on quantiles for each sample.

        Parameters:
        - num_samples: Number of samples to plot (default is 100)
        - nrows: Number of rows in the subplot grid (default is 10)
        - ncols: Number of columns in the subplot grid (default is 10)
        - figsize: Size of the figure (default is (30, 20))
        - leg_loc: location of the legend if num_samples=1 (default is 'upper right')
        - leg_visible: set if legend is to be shown or not (default is 'True')
        """
        # Check if only one sample is to be plotted
        if num_samples == 1:
            # Create a single plot
            fig, ax = plt.subplots(figsize=figsize)
            # Get the conditional mean, reference mean, and conditional variance for the first sample
            cond_mean = self.data[self.predicted_value].iloc[sample_index]
            true_value = self.data[self.true_value].iloc[sample_index]
            cond_variance = self.data[self.conditional_variance].iloc[sample_index]

            # Plot the conditional quantiles
            ax.plot(self.conditional_quantiles[sample_index], self.quantile_levels, label='Conditional Quantiles', color='red', linestyle='--')
            # Plot the conditional mean (predicted value)
            ax.plot([cond_mean, cond_mean], [0, 1], color='red', label='Conditional Mean (Predicted)')
            # Plot the reference value (true value)
            ax.plot([true_value, true_value], [0, 1], color='grey', label='Reference Value (True)')

            # Set the y-axis limit
            ax.set_ylim([0, 1.0])
            # Show the legend
            ax.legend(fontsize=10, loc=leg_loc).set_visible(leg_visible)
        else:
            # Create a grid of subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            axs = axs.flatten()

            for i in range(num_samples):
                # Get the conditional mean, reference mean, and conditional variance for each sample
                cond_mean = self.data[self.predicted_value].iloc[i]
                true_value = self.data[self.true_value].iloc[i]
                cond_variance = self.data[self.conditional_variance].iloc[i]

                ax = axs[i]
                # Plot the conditional quantiles
                ax.plot(self.conditional_quantiles[i], self.quantile_levels, label='Cond PDF', color='red', linestyle='--')
                # Plot the conditional mean (predicted value)
                ax.plot([cond_mean, cond_mean], [0, 1], color='red', label='conditional mean')
                # Plot the reference value (true value)
                ax.plot([true_value, true_value], [0, 1], color='grey', label='true value')

                # Set the y-axis limit
                ax.set_ylim([0, 1.0])
                # Hide the legend to avoid clutter
                ax.legend().set_visible(False)

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        # Show the plot
        plt.show()

######################################################################################
######################################################################################
######################################################################################

class PDFPlotter:
    def __init__(self, conditional_means, true_value, reference_variance=1.00, primary_ranges=None, pdf_values=None):
        """
        Initialize the PDFPlotter class with the required data and column names.

        Parameters:
        - conditional_means: np.ndarray containing the calculated conditional means
        - true_value: np.array containing the reference (true) values
        - reference_variance: float, value used for reference for plotting a Gaussian distribution around the reference value
        - primary_ranges: np.ndarray, name of the array of arrays containing the primary variable values used for calculating the pdf values
        - pdf_values: np.ndarray, name the array of arrays containing the calculated (conditional) pdf
        """
        self.conditional_means = conditional_means
        self.true_value = true_value
        self.reference_variance = reference_variance
        self.primary_ranges = primary_ranges
        self.pdf_values = pdf_values

    def plot_pdf_values(self, num_samples=100, nrows=10, ncols=10, figsize=(30, 20), sample_index=0, leg_loc='upper right', leg_visible=True, hor_plot_bounds=[-4, 4]):
        """
        Plot the conditional distributions based on the pdf values for each sample.

        Parameters:
        - num_samples: Number of samples to plot (default is 100)
        - nrows: Number of rows in the subplot grid (default is 10)
        - ncols: Number of columns in the subplot grid (default is 10)
        - figsize: Size of the figure (default is (30, 20))
        - sample_index: index of the sample to be plotted if num_samples=1 (default is 0)
        - leg_loc: location of the legend if num_samples=1 (default is 'upper right')
        - leg_visible: set if legend is to be shown or not (default is 'True')
        - hor_plot_bounds: bounds of the horizontal axis of the plot if num_samples=1
        """
        # Check if only one sample is to be plotted
        if num_samples == 1:
            # Create a single plot
            fig, ax = plt.subplots(figsize=figsize)
            # Get the conditional mean, reference mean, and conditional variance for each sample
            cond_mean = self.conditional_means[sample_index]
            true_value = self.true_value[sample_index]
            ref_variance = self.reference_variance

            # Generate data points for the x-axis
            x_ref_values = np.linspace(true_value - 4*np.sqrt(ref_variance), true_value + 4*np.sqrt(ref_variance), 500)  # Range of x-axis values

            # Compute the reference Gaussian distribution based on reference value and variance
            ref_pdf = (1 / np.sqrt(2 * np.pi * ref_variance)) * np.exp(-((x_ref_values - true_value) ** 2) / (2 * ref_variance))

            # Plot the conditional pdf
            ax.plot(self.primary_ranges[sample_index], self.pdf_values[sample_index], label='Conditional Distribution', color='red', linestyle='--')
            # Plot the reference distribution
            ax.plot(x_ref_values, ref_pdf, label='Reference Distribution', color='grey', linestyle='--')
            # Plot the conditional mean (predicted value)
            ax.plot([cond_mean, cond_mean], [0, self.pdf_values[sample_index].max()], color='red', label='Conditional Mean (Predicted)')
            # Plot the reference value (true value)
            ax.plot([true_value, true_value], [0, ref_pdf.max()], color='grey', label='Reference Value (True)')

            # Set the limits of the axes
            ax.set_xlim(hor_plot_bounds)
            ax.set_ylim([0, self.pdf_values[sample_index].max() + 0.1])
            # Show the legend
            ax.legend(fontsize=10, loc=leg_loc).set_visible(leg_visible)
        else:
            # Create a grid of subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
            axs = axs.flatten()

            for i in range(num_samples):
                # Get the conditional mean, reference mean, and conditional variance for each sample
                cond_mean = self.conditional_means[i]
                true_value = self.true_value[i]
                ref_variance = self.reference_variance

                # Generate data points for the x-axis
                x_ref_values = np.linspace(true_value - 4*np.sqrt(ref_variance), true_value + 4*np.sqrt(ref_variance), 500)  # Range of x-axis values

                # Compute the reference Gaussian distribution based on reference value and variance
                ref_pdf = (1 / np.sqrt(2 * np.pi * ref_variance)) * np.exp(-((x_ref_values - true_value) ** 2) / (2 * ref_variance))

                ax = axs[i]
                # Plot the conditional pdf
                ax.plot(self.primary_ranges[i], self.pdf_values[i], label='Conditional Distribution', color='red', linestyle='--')
                # Plot the reference distribution
                ax.plot(x_ref_values, ref_pdf, label='Reference Distribution', color='grey', linestyle='--')
                # Plot the conditional mean (predicted value)
                ax.plot([cond_mean, cond_mean], [0, self.pdf_values[i].max()], color='red', label='Conditional Mean (Predicted)')
                # Plot the reference value (true value)
                ax.plot([true_value, true_value], [0, ref_pdf.max()], color='grey', label='Reference Value (True)')

                # Set the limits of the axes
                ax.set_xlim([-4, 4])
                ax.set_ylim([0, self.pdf_values[i].max() + 0.1])
                # Hide the legend to avoid clutter
                ax.legend().set_visible(False)

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        # Show the plot
        plt.show()

######################################################################################
######################################################################################
######################################################################################

class MapTrueQuantiles:
    def __init__(self, quantile_levels, quantile_values, true_values):
        """
        Initialize the MapTrueQuantiles class with quantile levels, quantile values, and true values.

        Parameters:
        quantile_levels (np.ndarray): A (n_quantiles,) array of quantile levels (e.g., [0.1, 0.2, ..., 0.9]).
        quantile_values (np.ndarray): A (n_samples, n_quantiles) array of quantile values for each sample.
        true_values (np.ndarray): A (n_samples,) array of true/reference values for each sample.
        """
        self.quantile_levels = quantile_levels
        self.quantile_values = quantile_values
        self.true_values = true_values
        self.true_quantile_levels = None

    def get_true_quantile_levels(self):
        """
        Find the quantile level for each true value in the conditional distributions.

        Returns:
        np.ndarray: An array of the same length as true_values, with the quantile level for each true value.
        """
        n_samples, n_quantiles = self.quantile_values.shape

        # Initialize an array to store the quantile indices
        true_quantile_levels = np.zeros(n_samples)

        for i in range(n_samples):
            # Get the quantile distribution for the current sample
            quantiles = self.quantile_values[i]
            
            # Find where the true value falls within the quantiles
            quantile_index = np.searchsorted(quantiles, self.true_values[i], side='right') - 1

            # Find quantile level that corresponds to the true quantile index (approximated)
            true_quantile_levels[i] = self.quantile_levels[quantile_index]

        self.true_quantile_levels = true_quantile_levels
        return true_quantile_levels

    def plot_histogram(self, bins=10, figsize=(6, 3), show_results=True):
        """
        Plot a histogram of the true quantile levels.

        Parameters:
        bins (int): Number of bins for the histogram.
        figsize (tuple): Figure size for plotting the histograms.
        show_results: whether or not to show statistics (mean, accuracy, skewness) 

        """
        if self.true_quantile_levels is None:
            self.get_true_quantile_levels()
        
        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.hist(self.true_quantile_levels, bins=bins, edgecolor='black')
        plt.title("Histogram of True Value's Corresponding Quantiles", loc='left')
        plt.xlabel('Quantile')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(self.true_quantile_levels, bins=1000, cumulative=True, histtype='step', edgecolor='black')
        plt.xlabel('Quantile')
        plt.ylabel('Cumulative Frequency')

        if show_results:
            plt.text(0.70, 0.30, f'Mean = {self.true_quantile_levels.mean():.2f}', transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.22, f'Variance = {self.true_quantile_levels.var():.2f}', transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.16, f'Skewness = {skew(self.true_quantile_levels):.2f}', transform=plt.gca().transAxes, fontsize=9, verticalalignment='top', horizontalalignment='center')

        
        plt.tight_layout()
        plt.show()

######################################################################################
######################################################################################
######################################################################################

class AccuracyEvaluator:
    def __init__(self, quantile_levels, quantile_values, true_values):
        self.quantile_levels = quantile_levels
        self.quantile_values = quantile_values
        self.true_values = true_values

        # Initialize a dictionary to store evaluation metrics
        self.results = {
            'Accuracy': [],
            'Accuracy_Alternative': [],
            'Precision': [],
            'Goodness': [],
            'Fitness': [],
        }
        self._results_calculated = False  # A flag to check if results have been calculated

    def get_proportions(self, num_intervals=20):
        n_samples, n_quantiles = self.quantile_values.shape
        true_quantile_levels = np.zeros(n_samples) # Initialize array that will store the quantile levels that correspond to the true values

        # Iterate over samples
        for i in range(n_samples):
            quantiles = self.quantile_values[i] # Get true value for ith sample  
            quantile_index = np.searchsorted(quantiles, self.true_values[i], side='right') - 1 # Get index of the closest quantile level (to the right)

            # Find quantile level that corresponds to the true quantile index (approximated)
            true_quantile_levels[i] = self.quantile_levels[quantile_index]

            # # Handle edge cases where the true value is outside the quantile range 
            # if quantile_index < 0:
            #     quantile_index = 0
            # elif quantile_index >= n_quantiles - 1:
            #     quantile_index = n_quantiles - 2

            # # Linear interpolation to find the exact quantile level
            # lower_bound = quantiles[quantile_index] # Get lower bound of quantile values
            # upper_bound = quantiles[quantile_index + 1] # Get upper bound of quantile values 
            # lower_level = self.quantile_levels[quantile_index] # Get lower bound of quantile levels
            # upper_level = self.quantile_levels[quantile_index + 1] # Get upper bound of quantile levels

            # # Calculate exact quantile level, corresponding to the true value (rule of three)
            # if lower_bound != upper_bound:
            #     quantile_value = lower_level + (self.true_values[i] - lower_bound) * (upper_level - lower_level) / (upper_bound - lower_bound)
            # else:
            #     quantile_value = lower_level

            # print(lower_level)

            # quantile_indices[i] = quantile_value

        self.proportions = {}
        self.probabilities = []
        interval_step = 100 // num_intervals
        for p in range(interval_step, 101, interval_step):
            lower_bound = (100 - p) / 200
            upper_bound = 1 - lower_bound
            within_interval = (true_quantile_levels >= lower_bound) & (true_quantile_levels <= upper_bound)
            proportion = np.mean(within_interval)
            self.proportions[f"{p}% symmetric interval"] = proportion
            self.probabilities.append(p)
        
        return self.proportions

    def _calculate_results(self, num_intervals=20):
        if not self._results_calculated:
            proportions = self.get_proportions(num_intervals=num_intervals)
            intervals = list(range(100 // num_intervals, 101, 100 // num_intervals))
            proportions = [proportions[f"{p}% symmetric interval"] for p in intervals]
            indicators, absolute_differences, goodness_term1_list, goodness_term2_list = [], [], [], []

            for prop, prob in zip(proportions, self.probabilities): #prop is the proportion of true values, prob is the associated probability interval
                prob = prob / 100
                indicator = int(prop >= prob)
                abs_diff = abs(prop - prob)
                goodness_term1 = indicator * abs_diff
                goodness_term2 = (1 - indicator) * abs_diff
                weighted_diff = abs_diff * indicator
                indicators.append(indicator)
                absolute_differences.append(weighted_diff)
                goodness_term1_list.append(goodness_term1)
                goodness_term2_list.append(goodness_term2)

            # Calculate accuracy
            accuracy = sum(indicators) / len(self.probabilities)
            accuracy_alternative = (1 - np.sum(goodness_term2_list) / np.count_nonzero(goodness_term2_list)) if np.count_nonzero(goodness_term2_list) != 0 else 1
            
            # Calculate precision
            if accuracy == 0:
                precision = np.nan
            else:
                precision = 1 - 2 * (sum(absolute_differences) / len(self.probabilities))

            # Calculate goodness
            goodness = 1 - ((sum(goodness_term1_list) / len(self.probabilities)) + (2 * (sum(goodness_term2_list) / len(self.probabilities))))

            # Calculate fitness level
            fitness = 1 - ((sum(goodness_term1_list) / len(self.probabilities)) + ((sum(goodness_term2_list) / len(self.probabilities))))

            # Consolidate results
            self.results['Accuracy'].append(accuracy)
            self.results['Accuracy_Alternative'].append(accuracy_alternative)
            self.results['Precision'].append(precision)
            self.results['Goodness'].append(goodness)
            self.results['Fitness'].append(fitness)

            self._results_calculated = True  # Set the flag to True after calculation

    def accuracy_plot(self, figsize=(5, 5), num_intervals=20, show_results=True, leg_loc='upper right', leg_visible=True):

        """
        Accuracy plot based on the count of true values within probability intervals.

        Parameters:
        - figsize: Size of the figure (default is (30, 20))
        - num_intervals: Number of probability intervals to consider
        - show_results: whether or not to show the metrics (accuracy, precision, goodness) 
        - leg_loc: location of the legend if num_samples=1 (default is 'upper right')
        - leg_visible: set if legend is to be shown or not (default is 'True')
        """
        self._calculate_results(num_intervals)
        
        plt.figure(figsize=figsize)
        plt.scatter(self.probabilities, list(self.proportions.values()), s=20, color='black', label='Prop. of True Values in Prob. Interval')
        plt.plot([0, 100], [0, 1], color='blue', linestyle='--', label='Reference Line')
        plt.xlabel('Probability interval', fontsize=12)
        plt.ylabel('Proportion of true values', fontsize=12)
        plt.grid(True, color='grey', linestyle='--', linewidth=0.2)
        if show_results:
            plt.text(0.70, 0.40, f'Count = {len(self.true_values)}', transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.35, f'Accuracy = {self.results["Accuracy"][-1]:.3f}', transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.30, f'Accuracy(alt.) = {self.results["Accuracy_Alternative"][-1]:.3f}', transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.25, f'Precision = {self.results["Precision"][-1]:.3f}', transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.20, f'Goodness = {self.results["Goodness"][-1]:.3f}', transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='center')
            plt.text(0.70, 0.20, f'Fitness = {self.results["Fitness"][-1]:.3f}', transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', horizontalalignment='center')
        plt.legend(fontsize=10, loc=leg_loc).set_visible(leg_visible)
        plt.show()

    def get_results(self, num_intervals=20):
        self._calculate_results(num_intervals)
        return pd.DataFrame(self.results)