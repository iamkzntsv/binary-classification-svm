# Binary Classification of High-dimensional Data using Support Vector Machines

This project addresses the problem of binary classification on high-dimensional data. The data consists of 4608 features. 4096 of these were extracted from the fc7 activation layer of CaffeNet[^1] and the remaining 512 are gist features[^2]

## Approach
In our experiments, we used two variations of the Support Vector Machine classifier to determine which approach is best for a given task.

The first one is SVM with a linear kernel. SVM tries to find the hyperplane which separates classes from each other with a maximum margin between support vectors (data points closest to the hyperplane). The goal of SVM is to find the optimal set of weights in such a way that only the support vectors determine the weights and thus the decision boundary. It can be formulated mathematically as follows:

Find:
$$\begin{equation}
\min_{w}\frac{1}{2}\|w\| ^2 \\
\end{equation}
\quad Subject to:
\begin{equation}
w^Tx^{(i)} \geq 1 \quad \text{if}\, y^{(i)} = 1 \\
\end{equation}
\begin{equation}
w^Tx^{(i)} \leq -1 \,\, \text{if}\, y^{(i)} = 0
\end{equation}$$
For any given sample $i$

The main motivation for using this approach is that high-dimensional data can often be linearly separable[^3], which means that we can use a hyperplane to separate classes.

The second approach is an SVM with a radial basis function kernel. The use of the kernel trick allows to introduce transformations of the data, which are represented as pairwise comparisons of similarity between observations:

$$ K(x, x')=exp(-\gamma\|x-x'\|^2) $$

The motivation for using this method is that since the number of features will be reduced during our experiments, it may break the linear separability of the data, and hence we may need some feature engineering to make it linearly separable again.

[^1]: http://caffe.berkeleyvision.org
[^2]: http://people.csail.mit.edu/torralba/courses/6.870/papers/IJCV01-Oliva-Torralba.pdf
[^3]: Leon Bobrowski. Feature selection based on linear separabil-
ity and a cpl criterion function. Task Quarterly, 8(2):183–192,
2004.

