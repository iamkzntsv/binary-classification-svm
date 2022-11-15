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


[^1]: http://caffe.berkeleyvision.org
[^2]: http://people.csail.mit.edu/torralba/courses/6.870/papers/IJCV01-Oliva-Torralba.pdf

