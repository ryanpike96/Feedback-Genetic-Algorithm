# Feedback-Genetic-Algorithm
A control loop consisting of a neuro-evolution GA followed by a feature selection GA
a robust, light-weight framework for detecting malignancy in breast cancer
lesions. Taking inspiration from control theory
the implementation has two genetic algorithms
in a feedback loop. The first algorithm in the
pipeline attempts to optimise the hyperparameters of an artificial neural network with the feature space fixed. The topology, parameters and
rules are then fed into a second genetic algorithm
that acts as a feature pruning procedure to find
the most pertinent features. The cost function
used here allows the algorithm to favour less features, or a greater classification accuracy with
an ω value set by the user. In balancing classification performance with sparsity in features, we
found an ω value equal to 0.7 had the best performance. It achieved an accuracy, sensitivity and
specificity of 97.1%, 96.76% and 97.31% respectively. This was able to compete with the standard classification algorithms such as linear RBF,
Kernel RBF whilst getting better results than the
K-nearest neighbour algorithm. What is particularly impressive is its ability to constrain the
dimension of the feature space by almost 3-fold.
This was applied to the Wisconsin Breast Cancer
Database (WBCD) which contains 569 samples
in a 30-dimensional feature space. Although the
problem formulation was motivated by mammography this iterative approach can be applied to
most classification algorithms with its strength
stemming from the unnecessity of hand-crafted
hyperparameters.

![Pipeline](https://github.com/ryanpike96/img/blob/master/flowdiagram.eps)
