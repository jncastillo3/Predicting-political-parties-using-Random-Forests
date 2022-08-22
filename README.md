# Random-Forests

Based on congressional votes across 16 difference issues, the following algorithm aims to predict which political party each congressperson belongs to. We make use of a random forest, rather than a single decision tree as this will allow us to use "wisom of crowds" theory to our advantage. By having multiple trees, we reduce bias in our model, and increase accuracy.

In addition to making predictions about political parties, this code also evaluated the impact of the number of trees in a random forest. We use metrics such as accuracy, precision, recall, and F1 ratio to test for the best value of our parameter nTrees, the number of trees. Bar graphs are provided at the bottom of the code that visualize each of the aforementioned metrics, across each value of nTree.

Included is the algorithm code, and the csv file corresponding to the dataset.
