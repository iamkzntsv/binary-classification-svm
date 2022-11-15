# Binary Classification of High-dimensional data using Support Vector Machines

This project addresses the problem of binary classification on high-dimensional data. The data consists of 4608 features. 4096 of these were extracted from the fc7 activation layer of CaffeNet[^1] and the remaining 512 are gist features[^2]

## Approach
In our experiments, we used two variations of the Support Vector Machine classifier to determine which approach is best for a given task.

The first one is SVM with a linear kernel. SVM tries to find the hyperplane which separates classes from each other with a maximum margin between support vectors (data points closest to the hyperplane). The goal of SVM is to find the optimal set of weights in such a way that only the support vectors determine the weights and thus the decision boundary. It can be formulated mathematically as follows:

Find:
$$\min_{w}\frac{1}{2}\|w\| ^2$$
Subject to:
$$w^Tx^{(i)} \geq 1 \quad \text{if}\, y^{(i)} = 1 $$
$$w^Tx^{(i)} \leq -1 \,\, \text{if}\, y^{(i)} = 0 $$
For any given sample $i$

The main motivation for using this approach is that high-dimensional data can often be linearly separable[^3], which means that we can use a hyperplane to separate classes.

The second approach is an SVM with a radial basis function kernel. The use of the kernel trick allows to introduce transformations of the data, which are represented as pairwise comparisons of similarity between observations:

$$ K(x, x')=exp(-\gamma\|x-x'\|^2) $$

The motivation for using this method is that since the number of features will be reduced during our experiments, it may break the linear separability of the data, and hence we may need some feature engineering to make it linearly separable again.

## Methods
### Pre-processing
1. We look for missing values in our dataset and fill them in. We can also assume that if any feature has more than 20\% missing values, it can be removed. During imputation, we replace each missing value in a given feature vector with the mean of that vector.

2. We standardize our data to have a mean of 0 and st.d. of 1.

3. We then normalize it so that all values are in the $[0,1]$ range.

After that, we can plot the distribution of some of the features to better understand how the data is organized. Since we have two types of features, we choose 3 features from each category.

FIGURE

It can be seen that the CNN features follow a skewed distribution with a peak at 0, with only a few observations taking on a higher value. The distribution of gist features is also left-skewed, but with a smaller standard deviation.

### Dimensionality Reduction

Since a large number of dimensions can often lead to unnecessary complexity and overfitting [^4], we can reduce the number of features in our dataset. We consider two approaches: feature selection and feature transformation.

Feature selection is used o select the most important features in a dataset. Common feature selection methods include Filters and Wrappers. Filters use correlation measures to determine which features are more statistically significant and wrappers create multiple models with varying amounts of features and choose the best model based on performance. Although the latter approach is quite efficient, it is often computationally intensive, and since the number of observations and features in our data set is significantly large, we will use filters that require less CPU resources[^5]. Since our data is numerical and class labels are categorical, the general approach is to do univariate feature selection using ANOVA correlation coefficient. The idea is that each feature is compared to the target variable to determine if there is any statistically significant relationship between them. Then the most significant features are selected, and the rest are eliminated.

Feature transformation implies projecting the existing number of features onto a new space with fewer dimensions. Principal component analysis (PCA) is one of the most common techniques used for this kind of task. It considers the features based on their variance and combines them to reduce the dimensionality and retain as much information as possible. In our experiments, we apply both methods and compare which one gives the highest accuracy.

### Hyperparameter Tuning
The C parameter of SVM is a positive constant that introduces regularization, allowing some data points to be misclassified. The value of C corresponds to the penalty added for each misclassified observation, and the higher this value, the more the penalty is added and the more likely that our model will overfit. $\gamma$ (gamma) parameter controls the amount of influence two points have on each other during kernel trick. A higher gamma value will mean that data points need to be close to each other to be considered similar, and therefore more likely to cause overfitting. In our experiments, we used grid search for the following hyperparameters to determine at what values our model achieves the best performance:

$$\text{C} \in \lbrace 10^{-2}, 10^{-1}, 10^{0}, 10^{1} \rbrace$$

$$\gamma \in \lbrace \text{"scale"}, 10^{-3}, 10^{-2}, 10^{-1} \rbrace $$

## Experimental Results
For our experiments we use 5-fold cross-validation to split our training dataset into 5 mutually exclusive subsets and test our model. The performance of the classifiers changes as the number of features/components increases. We see that the best performance for ANOVA is achieved with SVM-RBF when the number of features is $\approx 2300$.  SVM with linear kernel performs better when the number of features is higher, as expected, but its best result is still slightly worse than SVM-RBF. PCA shows several peaks and achieves about the same accuracy as ANOVA. For further experiments, we will use ANOVA to keep the data in its original format.

FIGURE

We then use grid search to find the hyperparameters that provide the best performance for a given set of features. Our dataset also contains confidence scores that we can use as sample weights for the classifier. This means that for points with higher confidence, SVM will pay more emphasis to get these points right. We scaled the confidence scores by a factor of 2 to make the effect a bit more significant.

FIGURE

## Conclusion

Although ANOVA is not very sensitive to moderate deviations, it still assumes that the data is "gaussian-like". Therefore, one possible way to improve results is to use data transformations (such as log or box cox) to make the distribution of the data closer to normal before performing feature selection. In this case, we need to think about how to deal with zero values in our data, which can be a problem for such transformations. In addition, we can consider using bootstraping methods instead of cross validation for model selection, which can improve ANOVA-SVMs performance \cite{vrigazova2019optimization}. Results might also be improved by applying PCA transformation after ANOVA feature selection.

[^1]: http://caffe.berkeleyvision.org
[^2]: http://people.csail.mit.edu/torralba/courses/6.870/papers/IJCV01-Oliva-Torralba.pdf
[^3]: Leon Bobrowski. Feature selection based on linear separability and a cpl criterion function. Task Quarterly, 8(2):183–192, 2004.
[^4]:Andreas Gr ̈unauer and Markus Vincze. Using dimension reduction to improve the classification of high-dimensional data. arXiv preprint arXiv:1505.06907, 2015.
[^5]:Mar ́ıa Jos ́e Blanca Mena, Rafael Alarc ́on Postigo, Jaume Arnau Gras, Roser Bono Cabr ́e, Rebecca Bendayan, et al. Non-normal data: Is anova still a valid option? Psicothema, 2017.2

