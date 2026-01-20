import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass
import warnings
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy import stats
import itertools
warnings.filterwarnings('ignore')

@dataclass
class ClassificationMetrics:
    """Data class to store classification evaluation metrics"""
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f2_score: float
    fbeta_score: float
    
    # Advanced metrics
    specificity: float
    npv: float  # Negative Predictive Value
    fpr: float  # False Positive Rate
    fnr: float  # False Negative Rate
    
    # Composite metrics
    mcc: float  # Matthews Correlation Coefficient
    g_mean: float  # Geometric Mean
    balanced_accuracy: float
    youden_j: float  # Youden's J statistic
    
    # Class-wise metrics for multi-class
    class_wise_metrics: Optional[Dict] = None
    
    # Probabilistic metrics
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None  # Precision-Recall AUC

class ClassificationEvaluator:
    """
    Comprehensive classification model evaluation class
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_prob: Optional[np.ndarray] = None,
                 class_names: Optional[List[str]] = None,
                 model_name: str = "Model",
                 pos_label: int = 1,
                 threshold: float = 0.5):
        """
        Initialize evaluator with true and predicted values
        
        Args:
            y_true: True labels (0/1 or multi-class)
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for ROC, AUC, etc.)
            class_names: Names of classes
            model_name: Name of the model being evaluated
            pos_label: Positive class label for binary classification
            threshold: Classification threshold for probabilities
        """
        self.y_true = np.array(y_true).flatten()
        self.y_pred = np.array(y_pred).flatten()
        self.y_prob = y_prob
        self.class_names = class_names
        self.model_name = model_name
        self.pos_label = pos_label
        self.threshold = threshold
        self.n_classes = len(np.unique(self.y_true))
        self.is_binary = self.n_classes == 2
        
        # Apply threshold if probabilities are provided
        if self.y_prob is not None and self.is_binary:
            self.y_pred_from_prob = (self.y_prob >= self.threshold).astype(int)
            if y_pred is None:
                self.y_pred = self.y_pred_from_prob
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate confusion matrix
        self.confusion_matrix = self._calculate_confusion_matrix()
        
    def _validate_inputs(self):
        """Validate input arrays"""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(f"Length mismatch: y_true({len(self.y_true)}) != y_pred({len(self.y_pred)})")
        
        if self.y_prob is not None and len(self.y_true) != len(self.y_prob):
            raise ValueError(f"Length mismatch: y_true({len(self.y_true)}) != y_prob({len(self.y_prob)})")
        
        if self.n_classes < 2:
            raise ValueError("At least 2 classes required for classification")
        
        unique_true = np.unique(self.y_true)
        unique_pred = np.unique(self.y_pred)
        
        if not np.array_equal(np.sort(unique_true), np.sort(unique_pred)):
            warnings.warn("True and predicted labels have different unique values")
    
    def _calculate_confusion_matrix(self) -> np.ndarray:
        """Calculate confusion matrix"""
        if self.is_binary:
            # Binary confusion matrix
            tp = np.sum((self.y_true == self.pos_label) & (self.y_pred == self.pos_label))
            tn = np.sum((self.y_true != self.pos_label) & (self.y_pred != self.pos_label))
            fp = np.sum((self.y_true != self.pos_label) & (self.y_pred == self.pos_label))
            fn = np.sum((self.y_true == self.pos_label) & (self.y_pred != self.pos_label))
            return np.array([[tn, fp], [fn, tp]])
        else:
            # Multi-class confusion matrix
            n = self.n_classes
            cm = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    cm[i, j] = np.sum((self.y_true == i) & (self.y_pred == j))
            return cm
    
    # ============ BASIC METRICS ============
    
    def accuracy(self) -> float:
        """Calculate accuracy"""
        return np.mean(self.y_true == self.y_pred)
    
    def precision_score(self, average: str = 'binary') -> Union[float, Dict]:
        """
        Calculate precision
        
        Args:
            average: 'binary', 'micro', 'macro', 'weighted', or 'none'
        
        Returns:
            Precision score(s)
        """
        if self.is_binary or average == 'binary':
            tp = self.confusion_matrix[1, 1] if self.is_binary else self._true_positives()
            fp = self.confusion_matrix[0, 1] if self.is_binary else self._false_positives()
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return self._multi_class_metric(self._precision_per_class, average)
    
    def recall_score(self, average: str = 'binary') -> Union[float, Dict]:
        """
        Calculate recall (sensitivity)
        
        Args:
            average: 'binary', 'micro', 'macro', 'weighted', or 'none'
        
        Returns:
            Recall score(s)
        """
        if self.is_binary or average == 'binary':
            tp = self.confusion_matrix[1, 1] if self.is_binary else self._true_positives()
            fn = self.confusion_matrix[1, 0] if self.is_binary else self._false_negatives()
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return self._multi_class_metric(self._recall_per_class, average)
    
    def f1_score(self, average: str = 'binary') -> Union[float, Dict]:
        """
        Calculate F1 score (harmonic mean of precision and recall)
        
        Args:
            average: 'binary', 'micro', 'macro', 'weighted', or 'none'
        
        Returns:
            F1 score(s)
        """
        if self.is_binary or average == 'binary':
            precision = self.precision_score('binary')
            recall = self.recall_score('binary')
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return self._multi_class_metric(self._f1_per_class, average)
    
    def fbeta_score(self, beta: float = 1.0, average: str = 'binary') -> Union[float, Dict]:
        """
        Calculate F-beta score
        
        Args:
            beta: Weight of recall in harmonic mean (beta > 1 favors recall)
            average: 'binary', 'micro', 'macro', 'weighted', or 'none'
        
        Returns:
            F-beta score(s)
        """
        if self.is_binary or average == 'binary':
            precision = self.precision_score('binary')
            recall = self.recall_score('binary')
            beta2 = beta ** 2
            return (1 + beta2) * (precision * recall) / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0
        
        def fbeta_per_class(cls):
            precision = self._precision_per_class(cls)
            recall = self._recall_per_class(cls)
            beta2 = beta ** 2
            return (1 + beta2) * (precision * recall) / (beta2 * precision + recall) if (beta2 * precision + recall) > 0 else 0.0
        
        return self._multi_class_metric(fbeta_per_class, average)
    
    def f2_score(self, average: str = 'binary') -> Union[float, Dict]:
        """Calculate F2 score (emphasizes recall)"""
        return self.fbeta_score(beta=2.0, average=average)
    
    # ============ ADVANCED METRICS ============
    
    def specificity(self) -> float:
        """Calculate specificity (True Negative Rate)"""
        if not self.is_binary:
            raise ValueError("Specificity is only defined for binary classification")
        
        tn = self.confusion_matrix[0, 0]
        fp = self.confusion_matrix[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def negative_predictive_value(self) -> float:
        """Calculate Negative Predictive Value (NPV)"""
        if not self.is_binary:
            raise ValueError("NPV is only defined for binary classification")
        
        tn = self.confusion_matrix[0, 0]
        fn = self.confusion_matrix[1, 0]
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    def false_positive_rate(self) -> float:
        """Calculate False Positive Rate (FPR)"""
        if not self.is_binary:
            raise ValueError("FPR is only defined for binary classification")
        
        fp = self.confusion_matrix[0, 1]
        tn = self.confusion_matrix[0, 0]
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    def false_negative_rate(self) -> float:
        """Calculate False Negative Rate (FNR)"""
        if not self.is_binary:
            raise ValueError("FNR is only defined for binary classification")
        
        fn = self.confusion_matrix[1, 0]
        tp = self.confusion_matrix[1, 1]
        return fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    def matthews_correlation_coefficient(self) -> float:
        """
        Calculate Matthews Correlation Coefficient (MCC)
        
        MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        """
        if self.is_binary:
            tp = self.confusion_matrix[1, 1]
            tn = self.confusion_matrix[0, 0]
            fp = self.confusion_matrix[0, 1]
            fn = self.confusion_matrix[1, 0]
            
            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            
            return numerator / denominator if denominator != 0 else 0.0
        else:
            # Multi-class MCC
            n = self.n_classes
            cov_xy = 0
            sum_row = np.sum(self.confusion_matrix, axis=1)
            sum_col = np.sum(self.confusion_matrix, axis=0)
            
            for i in range(n):
                for k in range(n):
                    for l in range(n):
                        cov_xy += (self.confusion_matrix[i, i] * self.confusion_matrix[k, l] - 
                                  self.confusion_matrix[i, k] * self.confusion_matrix[l, i])
            
            denominator = np.sqrt(np.sum(sum_row) * np.sum(sum_col))
            for i in range(n):
                denominator *= np.sqrt(sum_row[i] * sum_col[i])
            
            return cov_xy / denominator if denominator != 0 else 0.0
    
    def geometric_mean(self) -> float:
        """Calculate Geometric Mean of sensitivity and specificity"""
        if not self.is_binary:
            # For multi-class, use average of class-wise G-means
            g_means = []
            for i in range(self.n_classes):
                tp = self.confusion_matrix[i, i]
                fn = np.sum(self.confusion_matrix[i, :]) - tp
                fp = np.sum(self.confusion_matrix[:, i]) - tp
                tn = np.sum(self.confusion_matrix) - tp - fn - fp
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                g_means.append(np.sqrt(sensitivity * specificity))
            
            return np.mean(g_means)
        else:
            sensitivity = self.recall_score()
            specificity = self.specificity()
            return np.sqrt(sensitivity * specificity)
    
    def balanced_accuracy(self) -> float:
        """Calculate Balanced Accuracy"""
        if self.is_binary:
            sensitivity = self.recall_score()
            specificity = self.specificity()
            return (sensitivity + specificity) / 2
        else:
            # Average of per-class recall
            recalls = []
            for i in range(self.n_classes):
                tp = self.confusion_matrix[i, i]
                actual = np.sum(self.confusion_matrix[i, :])
                recalls.append(tp / actual if actual > 0 else 0)
            return np.mean(recalls)
    
    def youden_j_statistic(self) -> float:
        """Calculate Youden's J statistic (Sensitivity + Specificity - 1)"""
        if not self.is_binary:
            raise ValueError("Youden's J is only defined for binary classification")
        
        sensitivity = self.recall_score()
        specificity = self.specificity()
        return sensitivity + specificity - 1
    
    # ============ PROBABILISTIC METRICS ============
    
    def log_loss(self, eps: float = 1e-15) -> float:
        """
        Calculate Logarithmic Loss (Cross-Entropy Loss)
        
        Args:
            eps: Small value to avoid log(0)
        
        Returns:
            Log loss value
        """
        if self.y_prob is None:
            raise ValueError("Probabilities required for log loss calculation")
        
        # Clip probabilities to avoid log(0)
        y_prob_clipped = np.clip(self.y_prob, eps, 1 - eps)
        
        if self.is_binary:
            # Binary log loss
            return -np.mean(self.y_true * np.log(y_prob_clipped) + 
                           (1 - self.y_true) * np.log(1 - y_prob_clipped))
        else:
            # Multi-class log loss (one-hot encoding needed)
            # For simplicity, assume y_prob is already in correct shape
            n_samples = len(self.y_true)
            n_classes = y_prob_clipped.shape[1] if len(y_prob_clipped.shape) > 1 else 2
            
            # Create one-hot encoding
            y_true_onehot = np.zeros((n_samples, n_classes))
            y_true_onehot[np.arange(n_samples), self.y_true.astype(int)] = 1
            
            return -np.mean(np.sum(y_true_onehot * np.log(y_prob_clipped), axis=1))
    
    def brier_score(self) -> float:
        """Calculate Brier Score (mean squared error of probabilities)"""
        if self.y_prob is None:
            raise ValueError("Probabilities required for Brier score calculation")
        
        if self.is_binary:
            return np.mean((self.y_prob - self.y_true) ** 2)
        else:
            # Multi-class Brier score
            n_samples = len(self.y_true)
            n_classes = self.y_prob.shape[1] if len(self.y_prob.shape) > 1 else 2
            
            # Create one-hot encoding
            y_true_onehot = np.zeros((n_samples, n_classes))
            y_true_onehot[np.arange(n_samples), self.y_true.astype(int)] = 1
            
            return np.mean(np.sum((self.y_prob - y_true_onehot) ** 2, axis=1))
    
    def roc_auc_score(self) -> float:
        """
        Calculate Area Under ROC Curve
        
        Returns:
            AUC-ROC score
        """
        if self.y_prob is None:
            raise ValueError("Probabilities required for AUC calculation")
        
        if not self.is_binary:
            # Multi-class AUC (one-vs-rest)
            from sklearn.metrics import roc_auc_score as sk_roc_auc_score
            return sk_roc_auc_score(self.y_true, self.y_prob, multi_class='ovr')
        
        # Binary AUC
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob, pos_label=self.pos_label)
        return auc(fpr, tpr)
    
    def precision_recall_auc_score(self) -> float:
        """
        Calculate Area Under Precision-Recall Curve
        
        Returns:
            AUC-PR score
        """
        if self.y_prob is None:
            raise ValueError("Probabilities required for PR AUC calculation")
        
        if not self.is_binary:
            # Multi-class AUC-PR (average precision)
            from sklearn.metrics import average_precision_score
            n_classes = self.y_prob.shape[1] if len(self.y_prob.shape) > 1 else 2
            y_true_onehot = np.zeros((len(self.y_true), n_classes))
            y_true_onehot[np.arange(len(self.y_true)), self.y_true.astype(int)] = 1
            return average_precision_score(y_true_onehot, self.y_prob, average='macro')
        
        # Binary AUC-PR
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob, 
                                                     pos_label=self.pos_label)
        return auc(recall, precision)
    
    # ============ HELPER METHODS ============
    
    def _true_positives(self, class_idx: int = None) -> int:
        """Calculate true positives for a class"""
        if class_idx is None:
            class_idx = self.pos_label if self.is_binary else 0
        return self.confusion_matrix[class_idx, class_idx]
    
    def _false_positives(self, class_idx: int = None) -> int:
        """Calculate false positives for a class"""
        if class_idx is None:
            class_idx = self.pos_label if self.is_binary else 0
        return np.sum(self.confusion_matrix[:, class_idx]) - self.confusion_matrix[class_idx, class_idx]
    
    def _false_negatives(self, class_idx: int = None) -> int:
        """Calculate false negatives for a class"""
        if class_idx is None:
            class_idx = self.pos_label if self.is_binary else 0
        return np.sum(self.confusion_matrix[class_idx, :]) - self.confusion_matrix[class_idx, class_idx]
    
    def _true_negatives(self, class_idx: int = None) -> int:
        """Calculate true negatives for a class"""
        if class_idx is None:
            class_idx = self.pos_label if self.is_binary else 0
        total = np.sum(self.confusion_matrix)
        tp = self._true_positives(class_idx)
        fp = self._false_positives(class_idx)
        fn = self._false_negatives(class_idx)
        return total - tp - fp - fn
    
    def _precision_per_class(self, class_idx: int) -> float:
        """Calculate precision for a specific class"""
        tp = self._true_positives(class_idx)
        fp = self._false_positives(class_idx)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def _recall_per_class(self, class_idx: int) -> float:
        """Calculate recall for a specific class"""
        tp = self._true_positives(class_idx)
        fn = self._false_negatives(class_idx)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _f1_per_class(self, class_idx: int) -> float:
        """Calculate F1 score for a specific class"""
        precision = self._precision_per_class(class_idx)
        recall = self._recall_per_class(class_idx)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _multi_class_metric(self, metric_func, average: str = 'macro'):
        """Calculate multi-class metric with specified averaging"""
        scores = [metric_func(i) for i in range(self.n_classes)]
        
        if average == 'none':
            return {i: score for i, score in enumerate(scores)}
        elif average == 'micro':
            # Micro averaging: aggregate TP, FP, FN across all classes
            total_tp = sum(self._true_positives(i) for i in range(self.n_classes))
            total_fp = sum(self._false_positives(i) for i in range(self.n_classes))
            total_fn = sum(self._false_negatives(i) for i in range(self.n_classes))
            
            if metric_func.__name__ == '_precision_per_class':
                return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            elif metric_func.__name__ == '_recall_per_class':
                return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            else:
                # For F1, calculate from micro precision and recall
                micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                return 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        elif average == 'macro':
            return np.mean(scores)
        elif average == 'weighted':
            # Weight by class support
            supports = np.sum(self.confusion_matrix, axis=1)
            total = np.sum(supports)
            return np.sum([scores[i] * supports[i] for i in range(self.n_classes)]) / total
    
    def calculate_all_metrics(self, include_probabilistic: bool = True) -> ClassificationMetrics:
        """
        Calculate all classification metrics
        
        Args:
            include_probabilistic: Include probabilistic metrics if probabilities available
        
        Returns:
            ClassificationMetrics object
        """
        # Calculate probabilistic metrics if available
        log_loss_val = None
        brier_score_val = None
        auc_roc_val = None
        auc_pr_val = None
        
        if include_probabilistic and self.y_prob is not None:
            try:
                log_loss_val = self.log_loss()
                brier_score_val = self.brier_score()
                auc_roc_val = self.roc_auc_score()
                auc_pr_val = self.precision_recall_auc_score()
            except:
                pass
        
        # Calculate class-wise metrics for multi-class
        class_wise_metrics = None
        if not self.is_binary:
            class_wise_metrics = {}
            for i in range(self.n_classes):
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                class_wise_metrics[class_name] = {
                    'precision': self._precision_per_class(i),
                    'recall': self._recall_per_class(i),
                    'f1': self._f1_per_class(i),
                    'support': np.sum(self.confusion_matrix[i, :])
                }
        
        return ClassificationMetrics(
            accuracy=self.accuracy(),
            precision=self.precision_score('binary' if self.is_binary else 'macro'),
            recall=self.recall_score('binary' if self.is_binary else 'macro'),
            f1_score=self.f1_score('binary' if self.is_binary else 'macro'),
            f2_score=self.f2_score('binary' if self.is_binary else 'macro'),
            fbeta_score=self.fbeta_score(beta=1.0, average='binary' if self.is_binary else 'macro'),
            specificity=self.specificity() if self.is_binary else None,
            npv=self.negative_predictive_value() if self.is_binary else None,
            fpr=self.false_positive_rate() if self.is_binary else None,
            fnr=self.false_negative_rate() if self.is_binary else None,
            mcc=self.matthews_correlation_coefficient(),
            g_mean=self.geometric_mean(),
            balanced_accuracy=self.balanced_accuracy(),
            youden_j=self.youden_j_statistic() if self.is_binary else None,
            class_wise_metrics=class_wise_metrics,
            log_loss=log_loss_val,
            brier_score=brier_score_val,
            auc_roc=auc_roc_val,
            auc_pr=auc_pr_val
        )
    
    # ============ VISUALIZATION METHODS ============
    
    def plot_confusion_matrix(self, normalize: bool = False, 
                             figsize: Tuple[int, int] = (10, 8),
                             title: str = None):
        """
        Plot confusion matrix
        
        Args:
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            title: Plot title
        """
        cm = self.confusion_matrix
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   cbar=True, square=True, linewidths=0.5)
        
        # Set labels
        if self.class_names:
            tick_labels = self.class_names
        else:
            tick_labels = [f'Class {i}' for i in range(self.n_classes)]
        
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(np.arange(self.n_classes) + 0.5, tick_labels, rotation=45, ha='right')
        plt.yticks(np.arange(self.n_classes) + 0.5, tick_labels, rotation=0)
        
        if title is None:
            title = f'Confusion Matrix - {self.model_name}'
            if normalize:
                title += ' (Normalized)'
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot ROC Curve
        
        Args:
            figsize: Figure size
        """
        if self.y_prob is None:
            raise ValueError("Probabilities required for ROC curve")
        
        if not self.is_binary:
            self._plot_multi_class_roc_curve(figsize)
            return
        
        # Binary ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob, pos_label=self.pos_label)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
        
        # Plot optimal threshold (Youden's J)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100, 
                   label=f'Optimal threshold: {optimal_threshold:.3f}\n(FPR={fpr[optimal_idx]:.3f}, TPR={tpr[optimal_idx]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print threshold information
        print(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.3f}")
        print(f"At this threshold:")
        print(f"  Sensitivity (TPR): {tpr[optimal_idx]:.3f}")
        print(f"  Specificity: {1 - fpr[optimal_idx]:.3f}")
        print(f"  FPR: {fpr[optimal_idx]:.3f}")
    
    def _plot_multi_class_roc_curve(self, figsize: Tuple[int, int] = (10, 8)):
        """Plot multi-class ROC curves (one-vs-rest)"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output
        y_true_bin = label_binarize(self.y_true, classes=range(self.n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), self.y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.figure(figsize=figsize)
        
        # Plot all ROC curves
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_classes))
        
        for i, color in zip(range(self.n_classes), colors):
            class_name = self.class_names[i] if self.class_names else f'Class {i}'
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                    label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=3)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Multi-class ROC Curves - {self.model_name}', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot Precision-Recall Curve
        
        Args:
            figsize: Figure size
        """
        if self.y_prob is None:
            raise ValueError("Probabilities required for PR curve")
        
        if not self.is_binary:
            self._plot_multi_class_pr_curve(figsize)
            return
        
        # Binary PR curve
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_prob, pos_label=self.pos_label
        )
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=figsize)
        
        # Plot PR curve
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Plot baseline (prevalence)
        prevalence = np.mean(self.y_true == self.pos_label)
        plt.axhline(y=prevalence, color='navy', linestyle='--', 
                   label=f'Baseline (Prevalence = {prevalence:.3f})')
        
        # Find optimal threshold (F1-score maximization)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100,
                   label=f'Optimal (F1-max): threshold={optimal_threshold:.3f}\n'
                         f'Precision={precision[optimal_idx]:.3f}, Recall={recall[optimal_idx]:.3f}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensitivity)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_multi_class_pr_curve(self, figsize: Tuple[int, int] = (10, 8)):
        """Plot multi-class Precision-Recall curves"""
        from sklearn.preprocessing import label_binarize
        
        # Binarize the output
        y_true_bin = label_binarize(self.y_true, classes=range(self.n_classes))
        
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(self.n_classes):
            precision[i], recall[i], _ = precision_recall_curve(
                y_true_bin[:, i], self.y_prob[:, i]
            )
            average_precision[i] = average_precision_score(
                y_true_bin[:, i], self.y_prob[:, i]
            )
        
        plt.figure(figsize=figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_classes))
        
        for i, color in zip(range(self.n_classes), colors):
            class_name = self.class_names[i] if self.class_names else f'Class {i}'
            plt.plot(recall[i], precision[i], color=color, lw=1.5,
                    label=f'{class_name} (AP = {average_precision[i]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Multi-class Precision-Recall Curves - {self.model_name}', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_summary(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot comprehensive metrics summary
        
        Args:
            figsize: Figure size
        """
        metrics = self.calculate_all_metrics()
        
        fig = plt.figure(figsize=figsize)
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        cm = self.confusion_matrix
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.figure.colorbar(im, ax=ax1)
        
        if self.class_names:
            tick_marks = np.arange(len(self.class_names))
            ax1.set_xticks(tick_marks)
            ax1.set_yticks(tick_marks)
            ax1.set_xticklabels(self.class_names)
            ax1.set_yticklabels(self.class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Confusion Matrix')
        
        # 2. Key Metrics Bar Chart
        ax2 = plt.subplot(2, 3, 2)
        key_metrics = {
            'Accuracy': metrics.accuracy,
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1-Score': metrics.f1_score,
            'MCC': metrics.mcc,
            'Balanced Acc': metrics.balanced_accuracy
        }
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(key_metrics)))
        bars = ax2.bar(range(len(key_metrics)), list(key_metrics.values()), 
                      color=colors, alpha=0.8)
        ax2.set_xticks(range(len(key_metrics)))
        ax2.set_xticklabels(list(key_metrics.keys()), rotation=45, ha='right')
        ax2.set_ylabel('Score')
        ax2.set_title('Key Metrics')
        ax2.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bar, value in zip(bars, key_metrics.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. ROC Curve (if probabilities available)
        if self.y_prob is not None and self.is_binary:
            ax3 = plt.subplot(2, 3, 3)
            fpr, tpr, _ = roc_curve(self.y_true, self.y_prob, pos_label=self.pos_label)
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC (AUC = {roc_auc:.3f})')
            ax3.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve')
            ax3.legend(loc='lower right')
            ax3.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curve (if probabilities available)
        if self.y_prob is not None and self.is_binary:
            ax4 = plt.subplot(2, 3, 4)
            precision, recall, _ = precision_recall_curve(
                self.y_true, self.y_prob, pos_label=self.pos_label
            )
            pr_auc = auc(recall, precision)
            prevalence = np.mean(self.y_true == self.pos_label)
            ax4.plot(recall, precision, color='darkgreen', lw=2,
                    label=f'PR Curve (AUC = {pr_auc:.3f})')
            ax4.axhline(y=prevalence, color='navy', linestyle='--',
                       label=f'Baseline: {prevalence:.3f}')
            ax4.set_xlim([0.0, 1.0])
            ax4.set_ylim([0.0, 1.05])
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve')
            ax4.legend(loc='lower left')
            ax4.grid(True, alpha=0.3)
        
        # 5. Class Distribution
        ax5 = plt.subplot(2, 3, 5)
        unique, counts = np.unique(self.y_true, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
        wedges, texts, autotexts = ax5.pie(counts, labels=[f'Class {i}' for i in unique], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax5.set_title('Class Distribution')
        
        # 6. Error Analysis
        ax6 = plt.subplot(2, 3, 6)
        if self.is_binary:
            error_types = ['TP', 'TN', 'FP', 'FN']
            error_counts = [
                self.confusion_matrix[1, 1],  # TP
                self.confusion_matrix[0, 0],  # TN
                self.confusion_matrix[0, 1],  # FP
                self.confusion_matrix[1, 0]   # FN
            ]
            colors = ['green', 'blue', 'orange', 'red']
            bars = ax6.bar(error_types, error_counts, color=colors, alpha=0.8)
            ax6.set_ylabel('Count')
            ax6.set_title('Error Analysis')
            
            # Add value labels
            for bar, count in zip(bars, error_counts):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}', ha='center', va='bottom')
        
        plt.suptitle(f'Model Evaluation Summary - {self.model_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def classification_report(self, digits: int = 3) -> pd.DataFrame:
        """
        Generate detailed classification report
        
        Args:
            digits: Number of decimal places to display
        
        Returns:
            DataFrame with classification report
        """
        if self.is_binary:
            report_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall (Sensitivity)', 
                          'Specificity', 'F1-Score', 'F2-Score',
                          'MCC', 'Balanced Accuracy', 'G-Mean',
                          'Youden\'s J', 'FPR', 'FNR', 'NPV'],
                'Value': [
                    self.accuracy(),
                    self.precision_score(),
                    self.recall_score(),
                    self.specificity(),
                    self.f1_score(),
                    self.f2_score(),
                    self.matthews_correlation_coefficient(),
                    self.balanced_accuracy(),
                    self.geometric_mean(),
                    self.youden_j_statistic(),
                    self.false_positive_rate(),
                    self.false_negative_rate(),
                    self.negative_predictive_value()
                ],
                'Interpretation': [
                    f"Correct predictions: {self.accuracy():.1%}",
                    f"Precision: {self.precision_score():.1%}",
                    f"Recall/Sensitivity: {self.recall_score():.1%}",
                    f"Specificity: {self.specificity():.1%}",
                    f"Harmonic mean of Precision and Recall",
                    f"F2 score (recall-weighted): {self.f2_score():.3f}",
                    f"Matthews Correlation Coefficient: {self.matthews_correlation_coefficient():.3f}",
                    f"Average of sensitivity and specificity",
                    f"Geometric mean: {self.geometric_mean():.3f}",
                    f"Youden's J: {self.youden_j_statistic():.3f}",
                    f"False Positive Rate: {self.false_positive_rate():.1%}",
                    f"False Negative Rate: {self.false_negative_rate():.1%}",
                    f"Negative Predictive Value: {self.negative_predictive_value():.1%}"
                ]
            }
            
            if self.y_prob is not None:
                prob_metrics = {
                    'AUC-ROC': self.roc_auc_score(),
                    'AUC-PR': self.precision_recall_auc_score(),
                    'Log Loss': self.log_loss(),
                    'Brier Score': self.brier_score()
                }
                
                for metric_name, value in prob_metrics.items():
                    report_data['Metric'].append(metric_name)
                    report_data['Value'].append(value)
                    report_data['Interpretation'].append(f"{metric_name}: {value:.3f}")
            
        else:
            # Multi-class report
            report_data = {
                'Class': [],
                'Precision': [],
                'Recall': [],
                'F1-Score': [],
                'Support': []
            }
            
            for i in range(self.n_classes):
                class_name = self.class_names[i] if self.class_names else f'Class {i}'
                report_data['Class'].append(class_name)
                report_data['Precision'].append(self._precision_per_class(i))
                report_data['Recall'].append(self._recall_per_class(i))
                report_data['F1-Score'].append(self._f1_per_class(i))
                report_data['Support'].append(np.sum(self.confusion_matrix[i, :]))
            
            # Add averages
            report_data['Class'].append('Macro Avg')
            report_data['Precision'].append(self.precision_score('macro'))
            report_data['Recall'].append(self.recall_score('macro'))
            report_data['F1-Score'].append(self.f1_score('macro'))
            report_data['Support'].append(len(self.y_true))
            
            report_data['Class'].append('Weighted Avg')
            report_data['Precision'].append(self.precision_score('weighted'))
            report_data['Recall'].append(self.recall_score('weighted'))
            report_data['F1-Score'].append(self.f1_score('weighted'))
            report_data['Support'].append(len(self.y_true))
        
        df_report = pd.DataFrame(report_data)
        
        # Format numeric columns
        if 'Value' in df_report.columns:
            df_report['Value'] = df_report['Value'].apply(
                lambda x: f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)
            )
        elif 'Precision' in df_report.columns:
            for col in ['Precision', 'Recall', 'F1-Score']:
                df_report[col] = df_report[col].apply(
                    lambda x: f"{x:.{digits}f}" if isinstance(x, (int, float)) else str(x)
                )
        
        return df_report

# ============ EXAMPLE USAGE ============

def binary_classification_example():
    """Demonstrate binary classification evaluation"""
    
    print("="*60)
    print("BINARY CLASSIFICATION EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels (imbalanced: 70% class 0, 30% class 1)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Model 1: Good predictions (with probabilities)
    y_prob_good = np.where(y_true == 1, 
                          np.random.beta(8, 2, n_samples),  # High for positives
                          np.random.beta(2, 8, n_samples))  # Low for negatives
    
    y_pred_good = (y_prob_good >= 0.5).astype(int)
    
    # Model 2: Random predictions
    y_pred_random = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    y_prob_random = np.random.uniform(0, 1, n_samples)
    
    print("\n1. EVALUATING GOOD MODEL")
    print("-"*40)
    
    evaluator_good = ClassificationEvaluator(
        y_true, y_pred_good, y_prob_good,
        class_names=['Negative', 'Positive'],
        model_name="Good Model"
    )
    
    # Get all metrics
    metrics_good = evaluator_good.calculate_all_metrics()
    
    print(f"Accuracy: {metrics_good.accuracy:.3f}")
    print(f"Precision: {metrics_good.precision:.3f}")
    print(f"Recall: {metrics_good.recall:.3f}")
    print(f"F1-Score: {metrics_good.f1_score:.3f}")
    print(f"Specificity: {metrics_good.specificity:.3f}")
    print(f"MCC: {metrics_good.mcc:.3f}")
    print(f"AUC-ROC: {metrics_good.auc_roc:.3f}")
    print(f"AUC-PR: {metrics_good.auc_pr:.3f}")
    
    # Generate comprehensive report
    print("\nDetailed Classification Report:")
    print(evaluator_good.classification_report().to_string(index=False))
    
    # Visualizations
    print("\nGenerating Visualizations...")
    evaluator_good.plot_confusion_matrix()
    evaluator_good.plot_roc_curve()
    evaluator_good.plot_precision_recall_curve()
    evaluator_good.plot_metrics_summary()
    
    print("\n2. EVALUATING RANDOM MODEL")
    print("-"*40)
    
    evaluator_random = ClassificationEvaluator(
        y_true, y_pred_random, y_prob_random,
        class_names=['Negative', 'Positive'],
        model_name="Random Model"
    )
    
    # Compare the two models
    print("\nModel Comparison:")
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUC-ROC'],
        'Good Model': [
            metrics_good.accuracy,
            metrics_good.precision,
            metrics_good.recall,
            metrics_good.f1_score,
            metrics_good.mcc,
            metrics_good.auc_roc
        ],
        'Random Model': [
            evaluator_random.accuracy(),
            evaluator_random.precision_score(),
            evaluator_random.recall_score(),
            evaluator_random.f1_score(),
            evaluator_random.matthews_correlation_coefficient(),
            evaluator_random.roc_auc_score() if evaluator_random.y_prob is not None else np.nan
        ]
    })
    
    print(comparison.to_string(index=False))
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE FOR BINARY CLASSIFICATION")
    print("="*60)
    print("""
    Key Metrics Interpretation:
    
    1. ACCURACY:
       - Overall correctness
       - Can be misleading with imbalanced data
    
    2. PRECISION (Positive Predictive Value):
       - Of all predicted positives, how many are actually positive?
       - Important when FP cost is high (e.g., spam detection)
    
    3. RECALL (Sensitivity):
       - Of all actual positives, how many did we catch?
       - Important when FN cost is high (e.g., disease detection)
    
    4. F1-SCORE:
       - Harmonic mean of Precision and Recall
       - Good overall metric for imbalanced data
    
    5. SPECIFICITY (True Negative Rate):
       - Of all actual negatives, how many did we correctly identify?
    
    6. MCC (Matthews Correlation Coefficient):
       - Balanced measure even with imbalanced classes
       - Range: -1 (worst) to 1 (best)
    
    7. AUC-ROC:
       - Ability to distinguish between classes
       - 0.5: Random, 1.0: Perfect
       - Good for balanced and imbalanced data
    
    8. AUC-PR:
       - Better for imbalanced data than AUC-ROC
       - Focuses on performance on positive class
    
    Business Context Examples:
    
    • MEDICAL DIAGNOSIS (High Recall needed):
       - Recall (Sensitivity) is critical
       - Better to have false alarms than miss cases
    
    • SPAM DETECTION (High Precision needed):
       - Precision is critical
       - Better to miss some spam than block legitimate emails
    
    • FRAUD DETECTION (Balance needed):
       - Both Precision and Recall important
       - F1-Score or MCC are good choices
    """)

def multi_class_classification_example():
    """Demonstrate multi-class classification evaluation"""
    
    print("\n" + "="*60)
    print("MULTI-CLASS CLASSIFICATION EVALUATION DEMONSTRATION")
    print("="*60)
    
    # Generate sample data (3 classes)
    np.random.seed(123)
    n_samples = 500
    n_classes = 3
    
    # True labels
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Create predictions (with some error patterns)
    y_pred = y_true.copy()
    
    # Introduce errors: Class 0 sometimes predicted as Class 1
    mask_0_to_1 = (y_true == 0) & (np.random.random(n_samples) < 0.2)
    y_pred[mask_0_to_1] = 1
    
    # Class 1 sometimes predicted as Class 2
    mask_1_to_2 = (y_true == 1) & (np.random.random(n_samples) < 0.15)
    y_pred[mask_1_to_2] = 2
    
    # Class 2 sometimes predicted as Class 0
    mask_2_to_0 = (y_true == 2) & (np.random.random(n_samples) < 0.1)
    y_pred[mask_2_to_0] = 0
    
    # Generate probabilities for ROC curves
    y_prob = np.random.dirichlet([1, 1, 1], n_samples)
    # Make probabilities somewhat aligned with predictions
    for i in range(n_samples):
        y_prob[i, y_pred[i]] += 0.3
        y_prob[i] = y_prob[i] / y_prob[i].sum()
    
    print(f"\nDataset: {n_samples} samples, {n_classes} classes")
    print(f"Class distribution:")
    for i in range(n_classes):
        count = np.sum(y_true == i)
        print(f"  Class {i}: {count} samples ({count/n_samples:.1%})")
    
    evaluator = ClassificationEvaluator(
        y_true, y_pred, y_prob,
        class_names=['Class A', 'Class B', 'Class C'],
        model_name="Multi-Class Model"
    )
    
    print("\nEVALUATION RESULTS")
    print("-"*40)
    
    metrics = evaluator.calculate_all_metrics()
    
    print(f"Overall Accuracy: {metrics.accuracy:.3f}")
    print(f"Macro-average Precision: {metrics.precision:.3f}")
    print(f"Macro-average Recall: {metrics.recall:.3f}")
    print(f"Macro-average F1-Score: {metrics.f1_score:.3f}")
    print(f"Matthews Correlation Coefficient: {metrics.mcc:.3f}")
    print(f"Balanced Accuracy: {metrics.balanced_accuracy:.3f}")
    
    print("\nClass-wise Metrics:")
    for class_name, class_metrics in metrics.class_wise_metrics.items():
        print(f"\n  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.3f}")
        print(f"    Recall: {class_metrics['recall']:.3f}")
        print(f"    F1-Score: {class_metrics['f1']:.3f}")
        print(f"    Support: {class_metrics['support']}")
    
    print("\nAveraging Methods Comparison:")
    print(f"  Macro-average F1: {evaluator.f1_score('macro'):.3f}")
    print(f"  Micro-average F1: {evaluator.f1_score('micro'):.3f}")
    print(f"  Weighted-average F1: {evaluator.f1_score('weighted'):.3f}")
    
    # Generate visualizations
    print("\nGenerating Visualizations...")
    evaluator.plot_confusion_matrix()
    evaluator.plot_confusion_matrix(normalize=True)
    
    if evaluator.y_prob is not None:
        evaluator.plot_roc_curve()
        evaluator.plot_precision_recall_curve()
    
    evaluator.plot_metrics_summary()
    
    # Detailed report
    print("\nDetailed Classification Report:")
    report = evaluator.classification_report()
    print(report.to_string(index=False))
    
    print("\n" + "="*60)
    print("MULTI-CLASS METRICS INTERPRETATION")
    print("="*60)
    print("""
    Averaging Methods:
    
    1. MACRO-AVERAGE:
       - Calculate metric for each class, then average
       - Treats all classes equally
       - Good when classes are balanced
    
    2. MICRO-AVERAGE:
       - Aggregate TP, FP, FN across all classes first
       - Weighted by class frequency
       - Good for imbalanced data
    
    3. WEIGHTED-AVERAGE:
       - Calculate macro-average, then weight by class support
       - Accounts for class imbalance
       - Most commonly used
    
    4. NONE/PER-CLASS:
       - Report metric for each class separately
       - Best for understanding per-class performance
    
    Key Insights:
    
    • Check confusion matrix for error patterns
    • Compare per-class metrics to identify weak classes
    • Use weighted average for imbalanced datasets
    • Consider business costs of different error types
    • For probabilistic metrics, use AUC-ROC or AUC-PR
    """)

def threshold_analysis_example():
    """Demonstrate threshold analysis for binary classification"""
    
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # True labels
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Generate probabilities
    y_prob = np.where(y_true == 1,
                     np.random.beta(6, 2, n_samples),  # Positives
                     np.random.beta(2, 6, n_samples))  # Negatives
    
    # Analyze different thresholds
    thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        evaluator = ClassificationEvaluator(y_true, y_pred, y_prob, 
                                           model_name=f"Threshold={threshold:.1f}")
        
        metrics = evaluator.calculate_all_metrics()
        
        results.append({
            'Threshold': threshold,
            'Accuracy': metrics.accuracy,
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1-Score': metrics.f1_score,
            'Specificity': metrics.specificity,
            'FPR': metrics.fpr,
            'Youden_J': metrics.youden_j
        })
    
    results_df = pd.DataFrame(results)
    
    print("\nThreshold Analysis Results:")
    print(results_df.round(3).to_string(index=False))
    
    # Plot threshold analysis
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results_df['Threshold'], results_df['Accuracy'], 'bo-', label='Accuracy')
    plt.plot(results_df['Threshold'], results_df['F1-Score'], 'ro-', label='F1-Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Accuracy and F1-Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(results_df['Threshold'], results_df['Precision'], 'go-', label='Precision')
    plt.plot(results_df['Threshold'], results_df['Recall'], 'mo-', label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(results_df['Recall'], results_df['Precision'], 'ko-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Trade-off')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(results_df['FPR'], results_df['Recall'], 'co-')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Trade-off')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Threshold Analysis', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Find optimal threshold by different criteria
    print("\nOptimal Thresholds by Different Criteria:")
    print("-"*40)
    
    # Maximum F1-Score
    opt_f1_idx = results_df['F1-Score'].idxmax()
    print(f"Maximum F1-Score: threshold={results_df.loc[opt_f1_idx, 'Threshold']:.2f}")
    print(f"  F1-Score: {results_df.loc[opt_f1_idx, 'F1-Score']:.3f}")
    print(f"  Precision: {results_df.loc[opt_f1_idx, 'Precision']:.3f}")
    print(f"  Recall: {results_df.loc[opt_f1_idx, 'Recall']:.3f}")
    
    # Maximum Youden's J (Sensitivity + Specificity - 1)
    opt_youden_idx = results_df['Youden_J'].idxmax()
    print(f"\nMaximum Youden's J: threshold={results_df.loc[opt_youden_idx, 'Threshold']:.2f}")
    print(f"  Youden's J: {results_df.loc[opt_youden_idx, 'Youden_J']:.3f}")
    print(f"  Sensitivity: {results_df.loc[opt_youden_idx, 'Recall']:.3f}")
    print(f"  Specificity: {results_df.loc[opt_youden_idx, 'Specificity']:.3f}")
    
    # Balanced Accuracy
    balanced_acc = (results_df['Recall'] + results_df['Specificity']) / 2
    opt_bal_idx = balanced_acc.idxmax()
    print(f"\nMaximum Balanced Accuracy: threshold={results_df.loc[opt_bal_idx, 'Threshold']:.2f}")
    print(f"  Balanced Accuracy: {balanced_acc[opt_bal_idx]:.3f}")
    
    # Business context example
    print("\n" + "="*60)
    print("BUSINESS CONTEXT: CHOOSING THE RIGHT THRESHOLD")
    print("="*60)
    print("""
    Scenario 1: SPAM DETECTION (High Precision needed)
      - Cost of FP (legitimate email marked as spam) is HIGH
      - Cost of FN (spam not caught) is LOW
      - Choose HIGHER threshold (e.g., 0.7-0.9)
      - Result: High Precision, Lower Recall
    
    Scenario 2: DISEASE SCREENING (High Recall needed)
      - Cost of FN (missed disease) is VERY HIGH
      - Cost of FP (false alarm) is MEDIUM
      - Choose LOWER threshold (e.g., 0.1-0.3)
      - Result: High Recall, Lower Precision
    
    Scenario 3: FRAUD DETECTION (Balance needed)
      - Both FP and FN have significant costs
      - Choose MODERATE threshold (e.g., 0.4-0.6)
      - Optimize for F1-Score or MCC
    """)

if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Run all examples
    binary_classification_example()
    multi_class_classification_example()
    threshold_analysis_example()
    
    print("\n" + "="*60)
    print("IMPLEMENTATION SUMMARY")
    print("="*60)
    print("""
    IMPLEMENTED FEATURES:
    
    1. COMPREHENSIVE METRICS:
       • Accuracy, Precision, Recall, F1-Score
       • Specificity, NPV, FPR, FNR
       • MCC, G-Mean, Balanced Accuracy, Youden's J
       • AUC-ROC, AUC-PR, Log Loss, Brier Score
    
    2. MULTI-CLASS SUPPORT:
       • Macro, Micro, Weighted, and None averaging
       • Per-class metrics
       • Multi-class ROC and PR curves
    
    3. ADVANCED VISUALIZATIONS:
       • Confusion Matrix (normalized and raw)
       • ROC Curves with optimal threshold
       • Precision-Recall Curves
       • Comprehensive metrics summary
       • Threshold analysis plots
    
    4. PRACTICAL TOOLS:
       • Detailed classification reports
       • Threshold optimization
       • Business context interpretation
       • Error analysis
    
    5. PRODUCTION FEATURES:
       • Input validation and error handling
       • Flexible configuration
       • Type hints and documentation
       • Modular and extensible design
    
    USAGE:
    
    # Basic usage
    evaluator = ClassificationEvaluator(y_true, y_pred, y_prob)
    metrics = evaluator.calculate_all_metrics()
    
    # Generate visualizations
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc_curve()
    evaluator.plot_metrics_summary()
    
    # Get detailed report
    report = evaluator.classification_report()
    
    # Multi-class with class names
    evaluator_multi = ClassificationEvaluator(
        y_true, y_pred, y_prob,
        class_names=['Class A', 'Class B', 'Class C']
    )
    """)