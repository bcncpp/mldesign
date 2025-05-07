## ✅ Offline Metrics

### 🔢 Classification Metrics

* **Accuracy**: Proportion of correctly predicted instances out of all predictions.
* **Precision**: Proportion of true positives among all predicted positives.
* **Recall**: Proportion of true positives among all actual positives.
* **F1-score**: Harmonic mean of precision and recall.
* **ROC-AUC**: Area under the curve plotting true vs false positive rates.
* **PR-AUC**: Area under the precision-recall curve, useful for imbalanced classes.
* **Log Loss**: Penalizes wrong confident predictions using cross-entropy.
* **Cohen’s Kappa**: Measures agreement between predicted and true labels adjusted for chance.
* **MCC**: Correlation coefficient between true and predicted classes (-1 to 1).

### 📈 Regression Metrics

* **MAE**: Average of absolute differences between predicted and true values.
* **MSE**: Average of squared differences between predicted and true values.
* **RMSE**: Square root of MSE, more sensitive to large errors.
* **R² Score**: Proportion of variance explained by the model.
* **MAPE**: Average percentage error between predicted and actual values.

### 📌 Ranking / Recommendation

* **Precision\@K**: Fraction of top-K recommended items that are relevant.
* **Recall\@K**: Fraction of all relevant items that appear in top-K recommendations.
* **NDCG**: Rewards ranked relevance based on position in the result list.
* **MRR**: Inverse of the rank of the first relevant result, averaged across queries.
* **Hit Rate**: Binary indicator if at least one relevant item is in top-K.

### 🧠 Embedding / Similarity

* **Cosine similarity**: Measure of angle between two vectors; closer to 1 means more similar.
* **Triplet Loss**: Ensures anchor is closer to positive than to negative embedding.
* **Contrastive Loss**: Penalizes dissimilar pairs that are close and similar pairs that are far.
* **Recall\@K (ANN)**: Measures if the true nearest neighbor is among the top-K approximate ones.
* **MAP**: Average precision across multiple queries or tasks.

### 🧪 Clustering

* **Silhouette Score**: Measures how close each sample is to its own cluster vs others.
* **Davies–Bouldin Index**: Lower values mean better clustering separation.
* **Adjusted Rand Index**: Measures similarity between true and predicted cluster assignments.
* **Calinski-Harabasz**: Ratio of between-cluster dispersion to within-cluster dispersion.

### ⚖️ Fairness / Bias

* **Demographic Parity**: Outcome rates should be similar across groups.
* **Equal Opportunity**: True positive rates should be equal across groups.
* **Equalized Odds**: Both TPR and FPR should be equal across groups.
* **Disparate Impact**: Ratio of outcomes between groups should be close to 1.
* **Calibration by group**: Predicted probabilities reflect actual outcomes per group.

---

## 🌐 Online Metrics

### 🌟 User / Business Impact

* **CTR**: Fraction of users who clicked out of those who saw an item.
* **Conversion Rate**: Fraction of users who took desired action (buy, sign up, etc.).
* **Revenue per User**: Average revenue generated per active user or session.
* **Dwell Time**: Time user spends engaging with content.
* **Retention Rate**: Percent of users returning after initial interaction.
* **Churn Rate**: Percent of users who stopped using the product.

### 📊 Production Model Performance

* **Prediction Drift**: Change in distribution of model predictions over time.
* **Label Agreement**: Percent of predictions matching actual outcomes (once known).
* **A/B Test Uplift**: Difference in key metrics between control and treatment groups.
* **Online Accuracy**: Accuracy calculated on live data once ground truth is available.

### ⚙️ System-Level

* **Latency (P50/P95/P99)**: Response time percentiles for inference.
* **QPS (Queries Per Second)**: Load handled by the model/server per second.
* **Memory/CPU/GPU usage**: Resource consumption for inference or batch jobs.
* **Model Load Time**: Time it takes to load a model into memory.
* **Cache Hit Rate**: Fraction of inference requests served from cache.

### 📉 Drift / Quality

* **Feature Drift**: Change in input feature distribution over time.
* **Concept Drift**: Change in relationship between inputs and outputs over time.
* **Missing Value Rate**: Percent of missing/null values in features.
* **Outlier Rate**: Frequency of inputs outside expected value range.

### 🔀 Feedback Loop

* **Feedback Capture Rate**: Proportion of inferences where user feedback was received.
* **Retrain Gain**: Improvement after retraining the model on new data.
* **Rollout Stability**: Measures if a new model performs stably over time.
* **Serving Consistency**: Detects discrepancies between offline predictions and live inference.

