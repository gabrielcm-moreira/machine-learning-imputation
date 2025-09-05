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
