from sklearn.datasets import load_iris
import numpy as np

class node:
    def __init__(self, feature, treshold, ):
        return 0
        
class tree:
    def __init__(self, root):
        self.root = root

class decision_tree():
    def __init__(self):
        self.tree = []
    def train(self, data):
        
        return 0
    def test(self, test_data):
        return 0


if __name__ == "__main__":  
    data = load_iris()
    X = data.data.astype(float)
    Y = data.target.astype(int)
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)

