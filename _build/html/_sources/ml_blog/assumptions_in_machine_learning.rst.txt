.. _assumptions_in_machine_learning:

A Deep Dive into ML Model Assumptions
====================================

Understanding the assumptions behind machine learning models is a critical skill for any data scientist. These assumptions, are the underlying principles that guide how a model learns from data. When these assumptions align with the structure of your data, a model can perform brilliantly. When they are violated, the model's results can be misleading or outright incorrect.

This guide explores the core assumptions of several major classes of machine learning models, what happens when those assumptions break down, and how to address the issues.

Assumptions of Linear Models
----------------------------

The Gauss-Markov Assumptions & Normality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

For an Ordinary Least Squares (OLS) regression model to be the Best Linear Unbiased Estimator (BLUE), it must meet a set of conditions known as the Gauss-Markov assumptions. A further assumption, normality of errors, is needed for valid statistical inference.

* **Linearity:** The relationship between the features and the target variable is linear.
* **No Perfect Multicollinearity:** No feature is a perfect linear combination of other features.
* **Exogeneity of Features (:math:`E[\epsilon|X] = 0`):** The error term has a conditional mean of zero. This implies that the features are not correlated with the error term.
* **Homoscedasticity:** The error term has a constant variance for all levels of the features (:math:`\text{Var}(\epsilon_i) = \sigma^2`).
* **No Autocorrelation:** The error terms are uncorrelated with each other (:math:`\text{Cov}(\epsilon_i, \epsilon_j) = 0` for :math:`i \neq j`).
* **Normality of Errors (Optional, for inference):** The error terms are normally distributed. This is not required for OLS to be BLUE, but it is required for hypothesis tests (t-tests, F-tests) and confidence intervals to be valid.

**The Problem: When They Break Down**

Violating these assumptions can invalidate the results of a linear model.

* **Non-linearity:** The model will be biased and systematically underfit the data.
* **Heteroscedasticity:** OLS estimates are still unbiased, but the standard errors will be incorrect. This makes all hypothesis tests and confidence intervals unreliable. You might think a feature is significant when it's not, or vice-versa.
* **Autocorrelation:** Common in time-series data. Similar to heteroscedasticity, it leads to incorrect standard errors and invalid inference.
* **Non-normality of Errors:** For small sample sizes, this invalidates p-values and confidence intervals.

**Solutions & Mitigation Strategies**

Detecting and correcting violations is a standard part of the regression workflow.

* **Detection:** Use diagnostic plots. A plot of residuals vs. fitted values is excellent for spotting non-linearity (patterns) and heteroscedasticity (funnel shapes). A Q-Q plot of residuals can check for normality. Formal tests like the Breusch-Pagan test (for heteroscedasticity) and the Durbin-Watson test (for autocorrelation) can also be used.
* **Correction:** For non-linearity, one can apply transformations to the features or target (e.g., log, polynomial) or use a more complex model. For heteroscedasticity and autocorrelation, the best practice is often to use **Robust Standard Errors** (e.g., Huber-White standard errors), which provide a corrected estimate. Alternatively, one could use Weighted Least Squares (WLS) or Generalized Least Squares (GLS).


Assumptions of Tree-Based Models
--------------------------------

Implicit Assumptions of Trees (Non-Parametric Models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

Tree-based models like Decision Trees, Random Forests, and Gradient Boosted Trees are non-parametric. This means they do not make strong assumptions about the functional form of the relationship between features and the target (e.g., they don't assume linearity). This flexibility is a major advantage. However, they are not assumption-free; their assumptions are implicit in their structure and are better described as **inductive biases**.

* **Axis-Aligned Splits:** The core assumption is that the feature space can be effectively partitioned using a series of axis-aligned splits (e.g., ``feature_A > 5``).
* **Hierarchical Structure:** They assume the data has a hierarchical structure that can be captured by a tree.
* **Data Independence:** Like most models, they still assume the training samples are independent (IID). A violation (e.g., time-series data) requires special handling.

**The Problem: When They Break Down**

The implicit assumptions of trees can be "gotchas" in certain scenarios.

* **Diagonal Decision Boundaries:** If the true decision boundary in the data is diagonal (e.g., ``feature_A + feature_B > 10``), a tree model will struggle. It will be forced to approximate the diagonal line with a jagged, inefficient "staircase" of many axis-aligned splits, which can lead to overfitting and a loss of predictive power.
* **High-Cardinality Categorical Features:** Trees can be biased towards selecting high-cardinality features during splitting, as these features offer more potential split points, increasing the chance of a "good" split purely by chance.
* **Instability:** Small changes in the training data can lead to a completely different tree structure being learned. This is a sign of high variance, which is what ensemble methods like Random Forest are designed to mitigate.

**Solutions & Mitigation Strategies**

* **Feature Engineering:** For diagonal boundaries, creating new features that are combinations of existing ones (e.g., ``feature_C = feature_A + feature_B``) can allow the tree to find the correct boundary with a single split.
* **Ensemble Methods:** Use Random Forests or Gradient Boosting instead of a single decision tree. Bagging (in Random Forests) and boosting both dramatically reduce the instability of individual trees and improve generalization.
* **Categorical Feature Handling:** For high-cardinality features, consider using target encoding or other more sophisticated methods instead of one-hot encoding, which can make the feature space unwieldy for trees.


Assumptions of Time Series Models
---------------------------------

The Assumption of Stationarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

Many classical time series models, like ARIMA, operate under the assumption that the underlying time series is **stationary**. A stationary series is one whose statistical properties do not change over time. More formally:

* **Constant Mean:** The mean of the series is not a function of time.
* **Constant Variance:** The variance of the series is not a function of time (this is a form of homoscedasticity).
* **Constant Autocovariance:** The covariance between two observations depends only on the lag (the distance between them), not on time itself.

This assumption is crucial because it implies that the patterns learned from the past are repeatable and will continue into the future, which is the entire basis for forecasting.

**The Problem: When They Break Down**

Most real-world time series are non-stationary.

* **Trends:** The series has a long-term upward or downward movement (violates constant mean). A classic example is a company's stock price over years.
* **Seasonality:** The series exhibits predictable, repeating patterns over a fixed period (e.g., daily, weekly, yearly). This also violates the constant mean assumption.
* **Changing Variance:** The volatility of the series changes over time. For example, a stock market might be much more volatile during a financial crisis.

Applying a model like ARIMA to a non-stationary series will lead to a fundamentally flawed model that produces unreliable and nonsensical forecasts.

**Solutions & Mitigation Strategies**

The goal is to transform the non-stationary series into a stationary one before modeling.

* **Detection:** Visually inspect the time series plot for obvious trends or seasonality. Analyze the Autocorrelation Function (ACF) plot; for a non-stationary series, the ACF will decay very slowly. Use formal statistical tests like the **Augmented Dickey-Fuller (ADF) test**, where the null hypothesis is that the series is non-stationary.
* **Correction:**
    * **For Trends:** The most common method is **differencing**. That is, instead of modeling the value :math:`Y_t`, you model the change from the previous period, :math:`\Delta Y_t = Y_t - Y_{t-1}`. This is the 'I' (Integrated) part of an ARIMA model.
    * **For Seasonality:** You can use seasonal differencing, where you subtract the value from the previous season (e.g., :math:`Y_t - Y_{t-12}` for monthly data).
    * **For Changing Variance:** Apply a transformation like taking the logarithm or using a Box-Cox transformation to stabilize the variance.


Assumptions of Instance-Based Models
-----------------------------------

K-Nearest Neighbors (KNN)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

K-Nearest Neighbors (KNN) is a simple, non-parametric algorithm used for classification and regression. It makes predictions for a new data point based on the majority class (for classification) or average value (for regression) of its 'k' nearest neighbors in the feature space. Its core assumptions are implicit in the distance calculation.

* **The "Neighborhood" Assumption:** The most critical implicit assumption is that data points close to each other in the feature space are likely to have the same target value.
* **Feature Relevance:** The model assumes all features are equally important and relevant to the outcome, as they all contribute to the distance metric.
* **Scale Dependency:** The distance metric (e.g., Euclidean distance) is highly sensitive to the scale of the features.

**The Problem: When They Break Down**

KNN's simplicity is deceptive, and it fails spectacularly when its implicit assumptions are violated.

* **The Curse of Dimensionality:** As the number of features (dimensions) increases, the concept of "distance" becomes less meaningful. In high-dimensional space, all points tend to be far away from each other, and the nearest neighbor might not be "near" at all.
* **Irrelevant Features:** If the dataset contains many irrelevant features, they add noise to the distance calculation. Two points might be close in the irrelevant dimensions but far apart in the important ones, leading to incorrect predictions.
* **Unscaled Features:** If features are on different scales (e.g., age in years and income in dollars), the feature with the largest scale will dominate the distance calculation, effectively ignoring the contribution of other features.

**Solutions & Mitigation Strategies**

* **Feature Scaling:** This is not optional for KNN; it is mandatory. Features must be scaled to a common range before applying KNN. Common methods include StandardScaler (to zero mean and unit variance) or MinMaxScaler (to a [0, 1] range).
* **Dimensionality Reduction:** Before using KNN on a high-dimensional dataset, it's crucial to perform dimensionality reduction using techniques like Principal Component Analysis (PCA) or feature selection.
* **Distance Metric Selection:** While Euclidean distance is the default, other metrics like Manhattan distance or Cosine similarity might be more appropriate depending on the data's structure.


Assumptions of Probabilistic Models
-----------------------------------

The 'Naive' Conditional Independence Assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

This is the core assumption of the **Naive Bayes** family of classifiers. The model is 'naive' because it assumes that all features are **conditionally independent** of each other, given the class label. For example, in a spam classifier, it assumes that the presence of the word "viagra" is independent of the presence of the word "free", given that the email is spam. Mathematically, for features :math:`X_1, ..., X_n` and class :math:`Y`, it assumes :math:`P(X_i | Y, X_j) = P(X_i | Y)` for all :math:`i \neq j`.

**The Problem: When It Breaks Down**

In nearly all real-world problems, this assumption is false. Words in a sentence are not independent; pixels in an image are not independent; a patient's symptoms are not independent.

**Why it Still Works (Often)**

Naive Bayes often performs surprisingly well even when the independence assumption is clearly violated.

* **Classification vs. Probability Estimation:** The goal of a classifier is only to get the correct class, not to produce accurate probability estimates. The independence assumption can lead to poorly calibrated probabilities, but as long as the correct class has the highest probability, the classification is correct.
* **Bias-Variance Tradeoff:** The strong, incorrect assumption gives the model a high bias, but this comes with very low variance. In situations with limited data, this high bias can be a form of regularization that prevents overfitting.


Assumptions of Support Vector Machines
-------------------------------------

Data Separability & The Kernel Trick
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

Support Vector Machines (SVMs) are a powerful class of supervised learning models. Their core assumption, or inductive bias, changes depending on the kernel used.

* **Linear SVM:** The fundamental assumption is that the data is **linearly separable** (or nearly linearly separable, in the case of the soft-margin SVM). The model's objective is to find the hyperplane that separates the classes with the maximum possible margin.
* **Kernel SVM:** For data that is not linearly separable, SVMs use the **kernel trick**. The implicit assumption here is that the data *will become* linearly separable if it is mapped to a higher-dimensional feature space. The kernel function allows the SVM to operate in this high-dimensional space without ever having to explicitly compute the coordinates of the data in that space.

**The Problem: When They Break Down**

* **Heavily Overlapping Data:** If the classes are heavily mixed and not separable even in a high-dimensional space, SVMs can perform poorly.
* **Kernel and Parameter Choice:** The performance of a kernel SVM is critically dependent on the choice of the kernel (e.g., Polynomial, Radial Basis Function - RBF) and its hyperparameters (like ``C`` for regularization and ``gamma`` for the RBF kernel).
* **Computational Complexity:** Training a kernel SVM is typically between :math:`O(n^2 d)` and :math:`O(n^3 d)`. This makes them computationally intractable for very large datasets.

**Solutions & Mitigation Strategies**

* **Hyperparameter Tuning:** Rigorous cross-validation is essential to find the best combination of the kernel, the regularization parameter ``C``, and kernel-specific parameters like ``gamma``.
* **Scaling:** Like KNN, SVMs are sensitive to the scale of the features, so proper feature scaling (e.g., with StandardScaler) is a mandatory preprocessing step.
* **For Large Datasets:** For large datasets, one can use online or linear SVM implementations (like ``SGDClassifier`` in scikit-learn with a hinge loss) which scale much better.


Assumptions of Deep Learning Models
-----------------------------------

General Neural Networks (Compositionality)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Concept Definition & Importance**

The most fundamental inductive bias of a standard feed-forward neural network is the assumption of **compositionality** or **hierarchical structure** in the data. The layered architecture is designed to learn features in a hierarchical fashion: the first layer learns simple patterns, the second layer combines these to learn more complex patterns, and so on.

**The Problem: When It Breaks Down**

This assumption is very general, but it can break down for problems that lack a clear hierarchical structure. More importantly, without a more specific inductive bias tailored to the data modality (like for images or text), a standard Multi-Layer Perceptron (MLP) can be very inefficient.

**Solutions & Mitigation Strategies**

The solution is to choose a network architecture whose inductive bias matches the data's specific structure.

* **For Spatial Data (Images):** Use a **Convolutional Neural Network (CNN)**.
* **For Sequential Data (Text, Time Series):** Use a **Recurrent Neural Network (RNN)** or a **Transformer**.
* **For Graph/Network Data:** Use a **Graph Neural Network (GNN)**.


Convolutional Neural Networks (CNNs)
------------------------------------

**Concept Definition & Importance**

CNNs are built on two powerful inductive biases for grid-like data:

* **Locality:** The assumption that features are local. In an image, a pixel is most strongly related to its immediate neighbors.
* **Translation Equivariance:** The assumption that the identity of an object does not change when its location does. This is achieved through **parameter sharing**: the same filter is slid across the entire image to detect the same feature everywhere.

**The Problem: When They Break Down**

* **Non-Grid Data:** For data where features have no spatial relationship (e.g., tabular data), the locality assumption is meaningless.
* **Positional Importance:** For problems where the absolute position of a feature is critical, the translation equivariance can be a disadvantage.
* **Rotational/Scale Variance:** Standard CNNs are not inherently invariant to rotation or changes in scale.

**Solutions & Mitigation Strategies**

* **Data Augmentation:** To handle variations like rotation and scale, heavy data augmentation is used during training.
* **Specialized Architectures:** For problems requiring equivariance to other transformations, specialized architectures like **Group Equivariant CNNs (G-CNNs)** can be used.


Recurrent Neural Networks (RNNs)
--------------------------------

**Concept Definition & Importance**

RNNs are designed for sequential data. Their core inductive bias is **sequential dependence** and **time invariance**.

* **Sequential Dependence:** The model assumes that the output at a given time step :math:`t` is a function of the current input and the model's hidden state from the previous time step, :math:`h_t = f(x_t, h_{t-1})`.
* **Time Invariance:** An RNN applies the same transition function (the same set of weights) at every time step.

**The Problem: When They Break Down**

* **Long-Range Dependencies:** The basic RNN architecture struggles to maintain information over long sequences due to the **vanishing and exploding gradient problem**.
* **Non-Sequential Data:** Applying an RNN to data where the order is arbitrary is inappropriate.

**Solutions & Mitigation Strategies**

* **Gated Architectures:** The vanishing gradient problem is primarily solved by using more sophisticated gated architectures like the **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)**.
* **Attention Mechanisms & Transformers:** For very long-range dependencies, attention mechanisms, and the Transformer architecture which is based entirely on them, relax the strict sequential assumption and allow the model to directly attend to any part of the sequence.

