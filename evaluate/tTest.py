import scipy.stats as stats
import numpy as np


class Ttest:
    """
    fill the tables comparing methods with mean and standard deviations
    of each metrics for each methods over the selected number of samples (>30).
    for the methods of higher interest or high uncertainty in comparisons,
    run the t-TEST
    """

    def __init__(self, method1_data: np.array, method2_data: np.array):
        """
        Initialize the Ttest object with two arrays of data.

        Parameters:
        method1_data (np.array): Data from method 1.
        method2_data (np.array): Data from method 2.
        """
        self.method1_data = method1_data
        self.method2_data = method2_data

    def check_normality(self, data: np.array, alpha: float = 0.05) -> bool:
        """
        Check the normality of data distribution using the Shapiro-Wilk test.
        Returns True if the data is normally distributed, False otherwise.

        Parameters:
        data (np.array): Data to be tested for normality.
        alpha (float): Significance level, default is 0.05.

        Returns:
        bool: True if data distribution is normal, False otherwise.
        """
        _, p_value = stats.shapiro(data)
        return p_value > alpha

    def check_variance(self) -> bool:
        """
        Check the variance between two methods using Levene's test.
        Returns True if the assumption of equal variances holds, False otherwise.

        Returns:
        bool: True if variances are equal, False otherwise.
        """
        _, p_value = stats.levene(self.method1_data, self.method2_data)
        return p_value > 0.05

    def sufficient_samples(self, threshold: int = 30) -> bool:
        """
        Check if the number of samples is sufficient to invoke the Central Limit Theorem.
        Returns True if the sample size in both methods is greater than or equal to the threshold, False otherwise.

        Parameters:
        threshold (int): The minimum number of samples required, default is 30.

        Returns:
        bool: True if sample size is sufficient, False otherwise.
        """
        return len(self.method1_data) >= threshold and len(self.method2_data) >= threshold

    def run_t_test(self) -> None:
        """
        Run the appropriate t-test based on variance and normality,
        and output the result regarding the null hypothesis and
        implications for the two methods.
        """
        # Check sample size
        if not self.sufficient_samples():
            print('Sample size is not sufficient to invoke the Central Limit Theorem.')
            return

        # Check normality
        normality_method1 = self.check_normality(self.method1_data)
        normality_method2 = self.check_normality(self.method2_data)

        if not normality_method1 or not normality_method2:
            print('Data is not normally distributed, consider using a non-parametric test.')
            return

        # Check variance and run t-test
        equal_variance = self.check_variance()
        if equal_variance:
            t_stat, p_value_t_test = stats.ttest_ind(self.method1_data, self.method2_data)
        else:
            t_stat, p_value_t_test = stats.ttest_ind(self.method1_data, self.method2_data, equal_var=False)

        if p_value_t_test < 0.05:
            print(f'The null hypothesis is rejected (p-value: {p_value_t_test:.3f}).')
            print('There is a statistically significant difference between the two methods.')
        else:
            print(f'The null hypothesis holds (p-value: {p_value_t_test:.3f}).')
            print('There is not a statistically significant difference between the two methods.')

    def get_means_stds(self) -> tuple:
        """
        Calculate and return the mean and standard deviation of each method
        over the number of samples for each, and also print these statistics.

        Returns:
        tuple: A tuple of dictionaries with the mean and std for each method.
        """
        method1_mean = np.mean(self.method1_data)
        method1_std = np.std(self.method1_data, ddof=1)
        method2_mean = np.mean(self.method2_data)
        method2_std = np.std(self.method2_data, ddof=1)

        print(f'Method 1: Mean = {method1_mean:.2f}, Standard Deviation = {method1_std:.2f}')
        print(f'Method 2: Mean = {method2_mean:.2f}, Standard Deviation = {method2_std:.2f}')

        return ({"mean": method1_mean, "std": method1_std},
                {"mean": method2_mean, "std": method2_std})


# To test the class, we need to create some dummy data.
np.random.seed(0)  # Seed for reproducibility
method1_data = np.random.normal(100, 15, 35)  # Method 1 data: normal distribution around
