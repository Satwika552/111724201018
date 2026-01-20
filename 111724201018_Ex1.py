import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Union, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics"""
    r2: float
    adj_r2: float
    mae: float
    mse: float
    rmse: float
    mape: float
    mae_percent: float
    explained_variance: float
    mean_absolute_percentage_error: float
    median_absolute_error: float
    max_error: float

class RegressionEvaluator:
    """
    Comprehensive regression model evaluation class
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 X: Optional[np.ndarray] = None, 
                 feature_names: Optional[List[str]] = None,
                 model_name: str = "Model"):
        """
        Initialize evaluator with true and predicted values
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            X: Feature matrix (required for adjusted R-squared)
            feature_names: Names of features (for residual analysis)
            model_name: Name of the model being evaluated
        """
        self.y_true = np.array(y_true).flatten()
        self.y_pred = np.array(y_pred).flatten()
        self.X = X
        self.feature_names = feature_names
        self.model_name = model_name
        self.n = len(y_true)
        self.residuals = self.y_true - self.y_pred
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input arrays"""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(f"Length mismatch: y_true({len(self.y_true)}) != y_pred({len(self.y_pred)})")
        
        if self.n <= 1:
            raise ValueError("Number of samples must be greater than 1")
            
        if np.any(np.isnan(self.y_true)) or np.any(np.isnan(self.y_pred)):
            raise ValueError("Input contains NaN values")
            
    def calculate_all_metrics(self, k: Optional[int] = None) -> ModelMetrics:
        """
        Calculate all regression metrics
        
        Args:
            k: Number of predictors/features (for adjusted R-squared)
               If None and X is provided, uses number of features from X
        
        Returns:
            ModelMetrics object containing all calculated metrics
        """
        if k is None and self.X is not None:
            k = self.X.shape[1] if len(self.X.shape) > 1 else 1
        
        metrics = ModelMetrics(
            r2=self.r_squared(),
            adj_r2=self.adjusted_r_squared(k) if k is not None else None,
            mae=self.mean_absolute_error(),
            mse=self.mean_squared_error(),
            rmse=self.root_mean_squared_error(),
            mape=self.mean_absolute_percentage_error(),
            mae_percent=self.mae_as_percentage(),
            explained_variance=self.explained_variance_score(),
            mean_absolute_percentage_error=self.mean_absolute_percentage_error(),
            median_absolute_error=self.median_absolute_error(),
            max_error=self.max_error()
        )
        
        return metrics
    
    # ============ CORE METRICS IMPLEMENTATIONS ============
    
    def r_squared(self) -> float:
        """
        Calculate R-squared (Coefficient of Determination)
        
        R² = 1 - (SS_res / SS_tot)
        where:
        SS_res = Σ(y_true - y_pred)²
        SS_tot = Σ(y_true - mean(y_true))²
        """
        ss_res = np.sum(self.residuals ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
    
    def adjusted_r_squared(self, k: int) -> float:
        """
        Calculate Adjusted R-squared
        
        Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - k - 1)]
        where:
        n = number of samples
        k = number of predictors
        
        Returns:
            Adjusted R-squared value
        """
        if k >= self.n - 1:
            warnings.warn(f"k ({k}) >= n-1 ({self.n-1}), adjusted R-squared may be unreliable")
            return np.nan
            
        r2 = self.r_squared()
        adj_r2 = 1 - ((1 - r2) * (self.n - 1) / (self.n - k - 1))
        return float(adj_r2)
    
    def mean_absolute_error(self) -> float:
        """
        Calculate Mean Absolute Error (MAE)
        
        MAE = (1/n) * Σ|y_true - y_pred|
        """
        mae = np.mean(np.abs(self.residuals))
        return float(mae)
    
    def mean_squared_error(self) -> float:
        """
        Calculate Mean Squared Error (MSE)
        
        MSE = (1/n) * Σ(y_true - y_pred)²
        """
        mse = np.mean(self.residuals ** 2)
        return float(mse)
    
    def root_mean_squared_error(self) -> float:
        """
        Calculate Root Mean Squared Error (RMSE)
        
        RMSE = √(MSE)
        """
        rmse = np.sqrt(self.mean_squared_error())
        return float(rmse)
    
    def mean_absolute_percentage_error(self) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE)
        
        MAPE = (100%/n) * Σ|(y_true - y_pred) / y_true|
        
        Note: Returns percentage (e.g., 10 for 10%)
        """
        # Avoid division by zero
        mask = self.y_true != 0
        if not np.any(mask):
            return np.nan
            
        mape = np.mean(np.abs(self.residuals[mask] / self.y_true[mask])) * 100
        return float(mape)
    
    def mae_as_percentage(self) -> float:
        """
        Calculate MAE as percentage of mean target
        
        Returns MAE as percentage of mean(y_true)
        """
        if np.mean(self.y_true) == 0:
            return np.nan
            
        mae_pct = (self.mean_absolute_error() / np.abs(np.mean(self.y_true))) * 100
        return float(mae_pct)
    
    def explained_variance_score(self) -> float:
        """
        Calculate Explained Variance Score
        
        Explained Variance = 1 - (Var(y_true - y_pred) / Var(y_true))
        """
        var_residuals = np.var(self.residuals)
        var_true = np.var(self.y_true)
        
        if var_true == 0:
            return 1.0 if var_residuals == 0 else 0.0
            
        explained_variance = 1 - (var_residuals / var_true)
        return float(explained_variance)
    
    def median_absolute_error(self) -> float:
        """
        Calculate Median Absolute Error
        """
        median_ae = np.median(np.abs(self.residuals))
        return float(median_ae)
    
    def max_error(self) -> float:
        """
        Calculate Maximum Error
        """
        max_err = np.max(np.abs(self.residuals))
        return float(max_err)
    
    # ============ COMPARISON FUNCTIONS ============
    
    def compare_with_baseline(self, baseline_method: str = 'mean') -> Dict[str, float]:
        """
        Compare model performance with baseline predictions
        
        Args:
            baseline_method: 'mean' or 'median' for baseline prediction
        
        Returns:
            Dictionary with comparison metrics
        """
        if baseline_method == 'mean':
            baseline_pred = np.full_like(self.y_true, np.mean(self.y_true))
        elif baseline_method == 'median':
            baseline_pred = np.full_like(self.y_true, np.median(self.y_true))
        else:
            raise ValueError("baseline_method must be 'mean' or 'median'")
        
        baseline_residuals = self.y_true - baseline_pred
        model_residuals = self.residuals
        
        comparison = {
            'baseline_mse': np.mean(baseline_residuals ** 2),
            'model_mse': self.mean_squared_error(),
            'mse_improvement': (np.mean(baseline_residuals ** 2) - self.mean_squared_error()) / np.mean(baseline_residuals ** 2) * 100,
            'baseline_mae': np.mean(np.abs(baseline_residuals)),
            'model_mae': self.mean_absolute_error(),
            'mae_improvement': (np.mean(np.abs(baseline_residuals)) - self.mean_absolute_error()) / np.mean(np.abs(baseline_residuals)) * 100
        }
        
        return comparison
    
    # ============ VISUALIZATION METHODS ============
    
    def plot_residuals(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive residual analysis plots
        
        Args:
            figsize: Figure size for the plots
        """
        fig = plt.figure(figsize=figsize)
        
        # 1. Residuals vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.y_pred, self.residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals Histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(self.residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot of Residuals
        ax3 = plt.subplot(2, 3, 3)
        from scipy import stats
        stats.probplot(self.residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # 4. Actual vs Predicted
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(self.y_true, self.y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Actual vs Predicted')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals vs Index (for time series)
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(range(len(self.residuals)), self.residuals, marker='o', alpha=0.6)
        ax5.axhline(y=0, color='r', linestyle='--')
        ax5.set_xlabel('Index')
        ax5.set_ylabel('Residuals')
        ax5.set_title('Residuals vs Index')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cook's Distance (if X is provided)
        if self.X is not None:
            ax6 = plt.subplot(2, 3, 6)
            # Simplified influence measure
            leverage = self._calculate_leverage()
            ax6.scatter(range(len(leverage)), leverage, alpha=0.6)
            ax6.axhline(y=np.mean(leverage) * 2, color='r', linestyle='--', label='2× Mean')
            ax6.set_xlabel('Observation Index')
            ax6.set_ylabel('Leverage (approx)')
            ax6.set_title('Observation Influence')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {self.model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def _calculate_leverage(self) -> np.ndarray:
        """Calculate leverage scores (simplified)"""
        if self.X is None:
            return np.zeros(self.n)
        
        try:
            # For linear regression, leverage = diag(X(X'X)^{-1}X')
            X = np.column_stack([np.ones(self.n), self.X])  # Add intercept
            H = X @ np.linalg.pinv(X.T @ X) @ X.T
            leverage = np.diag(H)
            return leverage
        except:
            # Simplified version if matrix inversion fails
            return np.ones(self.n) / self.n
    
    def plot_metrics_comparison(self, other_evaluators: List['RegressionEvaluator'], 
                                metric_names: List[str] = None):
        """
        Compare metrics across multiple models
        
        Args:
            other_evaluators: List of other RegressionEvaluator objects
            metric_names: List of metric names to compare
        """
        if metric_names is None:
            metric_names = ['r2', 'adj_r2', 'mae', 'rmse', 'mape']
        
        all_evaluators = [self] + other_evaluators
        model_names = [e.model_name for e in all_evaluators]
        
        metrics_data = {}
        for metric in metric_names:
            metrics_data[metric] = []
            for evaluator in all_evaluators:
                metrics_dict = evaluator.calculate_all_metrics()
                metrics_data[metric].append(getattr(metrics_dict, metric))
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metric_names[:6]):
            ax = axes[idx]
            bars = ax.bar(model_names, metrics_data[metric])
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Model Metrics Comparison', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def summary_report(self) -> pd.DataFrame:
        """
        Generate comprehensive summary report
        
        Returns:
            DataFrame with detailed metrics
        """
        metrics = self.calculate_all_metrics()
        
        report_data = {
            'Metric': [
                'R-squared (R²)', 'Adjusted R²', 
                'Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)',
                'Root Mean Squared Error (RMSE)', 
                'Mean Absolute Percentage Error (MAPE)', 'MAE % of Mean',
                'Explained Variance', 'Median Absolute Error',
                'Maximum Error'
            ],
            'Value': [
                metrics.r2, metrics.adj_r2, 
                metrics.mae, metrics.mse, 
                metrics.rmse, 
                metrics.mape, metrics.mae_percent,
                metrics.explained_variance, metrics.median_absolute_error,
                metrics.max_error
            ],
            'Interpretation': [
                f"{metrics.r2:.1%} of variance explained",
                f"Adjusted for {self.X.shape[1] if self.X is not None else 'N/A'} predictors" if metrics.adj_r2 is not None else "N/A",
                f"Average error: ±{metrics.mae:.3f} units",
                f"Average squared error: {metrics.mse:.3f}",
                f"Standard deviation of errors: {metrics.rmse:.3f}",
                f"Average percentage error: {metrics.mape:.1f}%",
                f"MAE is {metrics.mae_percent:.1f}% of target mean",
                f"{metrics.explained_variance:.1%} of variance explained",
                f"Median error: {metrics.median_absolute_error:.3f} units",
                f"Worst error: {metrics.max_error:.3f} units"
            ]
        }
        
        df_report = pd.DataFrame(report_data)
        df_report['Value'] = df_report['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
        
        return df_report

# ============ SCKIT-LEARN COMPATIBLE WRAPPER ============

class SklearnStyleEvaluator:
    """
    Scikit-learn style evaluator for easy integration
    """
    
    @staticmethod
    def evaluate_model(y_true, y_pred, X=None, model_name="Model"):
        """
        Quick evaluation function similar to scikit-learn
        
        Args:
            y_true: True values
            y_pred: Predicted values
            X: Feature matrix
            model_name: Name of model
            
        Returns:
            Dictionary with metrics
        """
        evaluator = RegressionEvaluator(y_true, y_pred, X, model_name=model_name)
        metrics = evaluator.calculate_all_metrics()
        
        results = {
            'model': model_name,
            'r2': metrics.r2,
            'adj_r2': metrics.adj_r2,
            'mae': metrics.mae,
            'mse': metrics.mse,
            'rmse': metrics.rmse,
            'mape': metrics.mape,
            'explained_variance': metrics.explained_variance
        }
        
        return results
    
    @staticmethod
    def cross_val_evaluate(model, X, y, cv=5, scoring=None):
        """
        Perform cross-validation evaluation
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metrics (default: ['r2', 'neg_mean_squared_error'])
        
        Returns:
            DataFrame with cross-validation results
        """
        from sklearn.model_selection import cross_validate
        
        if scoring is None:
            scoring = {
                'r2': 'r2',
                'neg_mse': 'neg_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error',
                'explained_variance': 'explained_variance'
            }
        
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        # Process results
        results_summary = {}
        for metric_name in scoring.keys():
            if f'test_{metric_name}' in cv_results:
                scores = cv_results[f'test_{metric_name}']
                if 'neg' in metric_name:
                    scores = -scores  # Convert negative metrics to positive
                    metric_name_clean = metric_name.replace('neg_', '')
                else:
                    metric_name_clean = metric_name
                
                results_summary[f'{metric_name_clean}_mean'] = np.mean(scores)
                results_summary[f'{metric_name_clean}_std'] = np.std(scores)
                results_summary[f'{metric_name_clean}_min'] = np.min(scores)
                results_summary[f'{metric_name_clean}_max'] = np.max(scores)
        
        return pd.DataFrame([results_summary])

# ============ EXAMPLE USAGE ============

def example_regression_evaluation():
    """Demonstrate the regression evaluation framework"""
    
    print("="*60)
    print("REGRESSION MODEL EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    
    # True relationship: y = 2*x1 + 3*x2 - 1.5*x3 + noise
    true_coeffs = np.array([2, 3, -1.5])
    y_true = X @ true_coeffs + np.random.randn(n_samples) * 0.5
    
    # Create different model predictions
    # Model 1: Good predictions
    y_pred_good = y_true + np.random.randn(n_samples) * 0.3
    
    # Model 2: Bad predictions
    y_pred_bad = y_true + np.random.randn(n_samples) * 2.0
    
    # Model 3: Systematic bias
    y_pred_bias = y_true * 0.8 + 1.0 + np.random.randn(n_samples) * 0.3
    
    print("\n1. EVALUATING GOOD MODEL")
    print("-"*40)
    
    evaluator_good = RegressionEvaluator(y_true, y_pred_good, X, 
                                         model_name="Good Model")
    
    # Get all metrics
    metrics_good = evaluator_good.calculate_all_metrics()
    
    print(f"R² Score: {metrics_good.r2:.4f}")
    print(f"Adjusted R²: {metrics_good.adj_r2:.4f}")
    print(f"MAE: {metrics_good.mae:.4f}")
    print(f"MSE: {metrics_good.mse:.4f}")
    print(f"RMSE: {metrics_good.rmse:.4f}")
    print(f"MAPE: {metrics_good.mape:.2f}%")
    print(f"Explained Variance: {metrics_good.explained_variance:.4f}")
    
    # Generate summary report
    print("\nSummary Report:")
    print(evaluator_good.summary_report().to_string(index=False))
    
    # Compare with baseline
    print("\nComparison with Baseline (mean prediction):")
    baseline_comparison = evaluator_good.compare_with_baseline('mean')
    for key, value in baseline_comparison.items():
        print(f"{key}: {value:.2f}")
    
    print("\n2. COMPARING MULTIPLE MODELS")
    print("-"*40)
    
    # Create evaluators for all models
    evaluator_bad = RegressionEvaluator(y_true, y_pred_bad, X, model_name="Bad Model")
    evaluator_bias = RegressionEvaluator(y_true, y_pred_bias, X, model_name="Biased Model")
    
    # Compare metrics
    evaluator_good.plot_metrics_comparison(
        [evaluator_bad, evaluator_bias],
        metric_names=['r2', 'mae', 'rmse', 'mape']
    )
    
    # Residual analysis
    print("\n3. RESIDUAL ANALYSIS FOR GOOD MODEL")
    print("-"*40)
    evaluator_good.plot_residuals()
    
    print("\n4. SCIKIT-LEARN STYLE EVALUATION")
    print("-"*40)
    
    # Using scikit-learn style wrapper
    results = SklearnStyleEvaluator.evaluate_model(
        y_true, y_pred_good, X, "Good Model"
    )
    
    print("Metrics from scikit-learn style evaluator:")
    for key, value in results.items():
        if key != 'model':
            print(f"{key}: {value:.4f}")
    
    print("\n5. CROSS-VALIDATION EXAMPLE (Simulated)")
    print("-"*40)
    
    # Simulate cross-validation results
    cv_results = {
        'r2_mean': 0.85, 'r2_std': 0.05,
        'mse_mean': 0.32, 'mse_std': 0.08,
        'mae_mean': 0.45, 'mae_std': 0.06
    }
    
    cv_df = pd.DataFrame([cv_results])
    print("Cross-Validation Results:")
    print(cv_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
    R-squared (R²): 
      - 1.0: Perfect prediction
      - 0.8-1.0: Excellent
      - 0.6-0.8: Good
      - 0.4-0.6: Fair
      - 0.0-0.4: Poor
      - < 0.0: Worse than horizontal line
    
    Adjusted R²:
      - Adjusts R² for number of predictors
      - Use when comparing models with different numbers of features
    
    MAE (Mean Absolute Error):
      - Average absolute error in original units
      - Less sensitive to outliers than MSE
    
    MSE (Mean Squared Error):
      - Average squared error
      - Penalizes large errors more heavily
    
    RMSE (Root Mean Squared Error):
      - Standard deviation of residuals
      - In same units as target variable
    
    MAPE (Mean Absolute Percentage Error):
      - Average percentage error
      - Useful for business context
      - Warning: Undefined when y_true = 0
    
    Residual Analysis Checklist:
      1. Residuals should be randomly scattered around zero
      2. No clear patterns in residuals vs predicted plot
      3. Residuals should be normally distributed
      4. No outliers with high leverage/influence
    """)

def real_world_example():
    """Example with real-world style data"""
    
    print("\n" + "="*60)
    print("REAL-WORLD EXAMPLE: HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Simulate house price data
    np.random.seed(123)
    n_houses = 200
    
    # Features
    square_feet = np.random.randint(800, 4000, n_houses)
    bedrooms = np.random.randint(1, 6, n_houses)
    bathrooms = np.random.randint(1, 4, n_houses)
    age = np.random.randint(0, 50, n_houses)
    
    # True price calculation (in thousands)
    price = (
        50 +  # Base price
        0.15 * square_feet +
        20 * bedrooms +
        15 * bathrooms -
        0.5 * age +
        np.random.randn(n_houses) * 30  # Noise
    )
    
    # Feature matrix
    X = np.column_stack([square_feet, bedrooms, bathrooms, age])
    feature_names = ['Square Feet', 'Bedrooms', 'Bathrooms', 'Age']
    
    # Create model predictions
    # Model predictions with different levels of accuracy
    price_pred_good = price + np.random.randn(n_houses) * 25
    price_pred_ok = price * 0.9 + np.random.randn(n_houses) * 40 + 10
    
    print(f"\nDataset: {n_houses} houses")
    print(f"True price range: ${price.min():.0f}k - ${price.max():.0f}k")
    print(f"Mean price: ${price.mean():.0f}k")
    
    print("\nEvaluating Model Performance:")
    print("-"*40)
    
    # Evaluate good model
    evaluator_good = RegressionEvaluator(
        price, price_pred_good, X, feature_names, "Premium Model"
    )
    
    metrics_good = evaluator_good.calculate_all_metrics()
    
    print("\nPremium Model Results:")
    print(f"R²: {metrics_good.r2:.3f} ({metrics_good.r2:.1%} variance explained)")
    print(f"MAE: ${metrics_good.mae:.1f}k")
    print(f"RMSE: ${metrics_good.rmse:.1f}k")
    print(f"MAPE: {metrics_good.mape:.1f}%")
    print(f"MAE as % of mean price: {metrics_good.mae_percent:.1f}%")
    
    # Business interpretation
    print("\nBusiness Interpretation:")
    print(f"• The model explains {metrics_good.r2:.1%} of price variation")
    print(f"• Average prediction error: ±${metrics_good.mae:.0f}k")
    print(f"• 68% of predictions are within ±${metrics_good.rmse:.0f}k")
    print(f"• Average percentage error: {metrics_good.mape:.1f}%")
    
    # Compare with acceptable model
    evaluator_ok = RegressionEvaluator(
        price, price_pred_ok, X, feature_names, "Standard Model"
    )
    
    print("\nModel Comparison:")
    comparison = pd.DataFrame({
        'Metric': ['R²', 'MAE ($k)', 'RMSE ($k)', 'MAPE (%)'],
        'Premium Model': [
            f"{metrics_good.r2:.3f}",
            f"{metrics_good.mae:.1f}",
            f"{metrics_good.rmse:.1f}",
            f"{metrics_good.mape:.1f}"
        ],
        'Standard Model': [
            f"{evaluator_ok.r_squared():.3f}",
            f"{evaluator_ok.mean_absolute_error():.1f}",
            f"{evaluator_ok.root_mean_squared_error():.1f}",
            f"{evaluator_ok.mean_absolute_percentage_error():.1f}"
        ]
    })
    
    print(comparison.to_string(index=False))
    
    # Visualization
    evaluator_good.plot_residuals()
    
    # Feature importance analysis (simplified)
    if evaluator_good.feature_names:
        print("\nFeature Analysis (based on correlations):")
        correlations = []
        for i, feature_name in enumerate(feature_names):
            corr = np.corrcoef(X[:, i], price)[0, 1]
            correlations.append((feature_name, corr))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feature_name, corr in correlations:
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            direction = "positive" if corr > 0 else "negative"
            print(f"  {feature_name}: {corr:.3f} ({strength} {direction} correlation)")

if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Run examples
    example_regression_evaluation()
    real_world_example()
    
    print("\n" + "="*60)
    print("IMPLEMENTATION FEATURES SUMMARY")
    print("="*60)
    print("""
    1. COMPREHENSIVE METRICS:
       • R-squared (R²) and Adjusted R²
       • MAE, MSE, RMSE
       • MAPE and percentage metrics
       • Explained variance
       • Median absolute error
       • Maximum error
    
    2. ADVANCED FEATURES:
       • Residual analysis visualization
       • Model comparison tools
       • Baseline comparison
       • Business-friendly interpretations
       • Cross-validation support
    
    3. PRACTICAL TOOLS:
       • Scikit-learn compatible interface
       • Detailed summary reports
       • Real-world examples
       • Error distribution analysis
       • Feature correlation analysis
    
    4. PRODUCTION READY:
       • Input validation
       • Error handling
       • Type hints
       • Comprehensive documentation
       • Modular design 
    """)
