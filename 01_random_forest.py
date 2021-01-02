""" RANDOM FOREST
    -------------
    Implementation of a Classification And Regression Random Forest algorithm for binary classification.

    Code reference: https://machinelearningmastery.com/implement-random-forest-scratch-python/
"""


# -------------------------------------------------------------------------------
# 0. IMPORT LIBRARIES
# -------------------------------------------------------------------------------


import pandas as pd
from random import seed, randrange
from math import sqrt


# Set the seed

seed(2)


# -------------------------------------------------------------------------------
# 1. PREPARE THE DATA
# -------------------------------------------------------------------------------


# Import the dataset

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data', header=None)
print(data.head())

data = data.replace(['R', 'M'], [0, 1])
dataset = data.values.tolist()


# -------------------------------------------------------------------------------
# 2. CREATE A SPLIT
# -------------------------------------------------------------------------------


# Calculate the gini score to evaluate the cost of a split

def gini_score(groups, unique_classes):

    number_samples = float(sum([len(group) for group in groups]))
    gini = 0.0

    for group in groups:

        group_size = float(len(group))

        if group_size == 0:

            continue

        score = 0.0

        for class_val in unique_classes:

            p = [sample[-1] for sample in group].count(class_val) / group_size
            score += p * p

        gini += (1.0 - score) * (group_size / number_samples)

    return gini


# Split a dataset based on a feature and a feature value

def test_split(dataset, feature_index, value):

    left, right = list(), list()

    for sample in dataset:

        if sample[feature_index] < value:

            left.append(sample)

        else:

            right.append(sample)

    return left, right


# Select the best split point for a dataset

def get_split(dataset, n_features):

    features = list()
    unique_classes = list(set(sample[-1] for sample in dataset))
    b_feature, b_value, b_score, b_groups = 999, 999, 999, None

    while len(features) < n_features:

        feature_index = randrange(len(dataset[0])-1)

        if feature_index not in features:

            features.append(feature_index)

    for feature_index in features:

        for sample in dataset:

            groups = test_split(dataset=dataset, feature_index=feature_index, value=sample[feature_index])
            gini = gini_score(groups=groups, unique_classes=unique_classes)

            if gini < b_score:

                b_feature, b_value, b_score, b_groups = feature_index, sample[feature_index], gini, groups

    return {'feature': b_feature, 'value': b_value, 'groups': b_groups}


# -------------------------------------------------------------------------------
# 3. BUILD A TREE
# -------------------------------------------------------------------------------


# Create a terminal node value

def terminal_node_value(group):

    outcomes = [sample[-1] for sample in group]

    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal

def split(node, max_depth, min_size, n_features, depth):

    left, right = node['groups']
    del(node['groups'])

    if not left or not right:

        node['left'] = node['right'] = terminal_node_value(left + right)
        return

    if depth >= max_depth:

        node['left'], node['right'] = terminal_node_value(left), terminal_node_value(right)
        return

    if len(left) <= min_size:

        node['left'] = terminal_node_value(left)

    else:

        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:

        node['right'] = terminal_node_value(right)

    else:

        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# Build a decision tree

def build_tree(training_data, max_depth, min_size, n_features):

    node = get_split(training_data, n_features)
    split(node, max_depth, min_size, n_features, 1)

    return node


# Print a decision tree

def print_tree(node, depth=0):

    if isinstance(node, dict):

        print('{}[X{} < {:.3f}]'.format(depth*' ', node['feature'] + 1, node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)

    else:

        print('{}[{}]'.format(depth*' ', node))


# -------------------------------------------------------------------------------
# 4. MAKE PREDICTIONS
# -------------------------------------------------------------------------------


# Make a prediction with a decision tree

def predict(node, sample):

    if sample[node['feature']] < node['value']:

        if isinstance(node['left'], dict):

            return predict(node['left'], sample)

        else:

            return node['left']

    else:

        if isinstance(node['right'], dict):

            return predict(node['right'], sample)

        else:

            return node['right']


# Create a random subsample from the dataset with replacement

def subsample(dataset, ratio):

    sample = list()
    n_sample = round(len(dataset) * ratio)

    while len(sample) < n_sample:

        index = randrange(len(dataset))
        sample.append(dataset[index])

    return sample


# Make a prediction with a list of bagged trees

def bagging_predict(trees, sample):

    predictions = [predict(tree, sample) for tree in trees]

    return max(set(predictions), key=predictions.count)


# -------------------------------------------------------------------------------
# 5. EVALUATE THE MODEL
# -------------------------------------------------------------------------------


# Classification and Regression Forest Algorithm

def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):

    trees = list()

    for i in range(n_trees):

        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    predictions = [bagging_predict(trees, row) for row in test]

    return(predictions)

# Classification and Regression Tree Algorithm

def decision_tree(train, test, max_depth, min_size):

    tree = build_tree(train, max_depth, min_size)
    predictions = list()

    for row in test:

        prediction = predict(tree, row)
        predictions.append(prediction)

    return(predictions)


# Split a dataset into k folds

def cross_validation_split(dataset, n_folds):

    dataset_copy = list(dataset)
    fold_size = int(len(dataset_copy) / n_folds)
    dataset_split = list()

    for i in range(n_folds):

        fold = list()

        while len(fold) < fold_size:

            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))

        dataset_split.append(fold)

    return dataset_split


# Calculate accuracy percentage

def accuracy_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i] == predicted[i]:

            correct += 1

    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split

def evaluate_algorithm(dataset, algorithm, n_folds, *args):

    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for fold in folds:

        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()

        for sample in fold:

            sample_copy = list(sample)
            test_set.append(sample_copy)
            sample_copy[-1] = None

        predicted = algorithm(train_set, test_set, *args)
        actual = [sample[-1] for sample in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


# Evaluate algorithm

n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))

for n_trees in [1, 5, 10]:

    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    print('Trees: {}'.format(n_trees))
    print('Scores: {}'.format(scores))
    print('Mean Accuracy: {:.3}%'.format(sum(scores)/float(len(scores))))