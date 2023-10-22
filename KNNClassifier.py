import numpy as np


def read_data_from_file(file_path):
    x_train, x_test, y_train, y_test = [], [], [], []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:
        line = line.strip().split(',')
        if 'train' in line:
            x_train.append(line[0:4])
            y_train.append(line[4])
        else:
            x_test.append(line[0:4])
            y_test.append(line[4])

    x_train = np.array(x_train, dtype=float)
    x_test = np.array(x_test, dtype=float)

    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)

    return x_train, x_test, y_train, y_test


def write_list_to_file(file_path, values):

    with open(file_path, 'w') as file:
        file.write('\n'.join(map(str, values)))


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, X):
        predicted = []
        for x in X:
            dist = list(map(lambda x_train: self.calc_euclid_dist(x, x_train), self.x_train))

            x_indices = list(map(lambda x: x[0], sorted(enumerate(dist), key=lambda x: x[1])[:self.k]))
            group_labels = [self.y_train[i] for i in x_indices]

            majority_label = max(set(group_labels), key=group_labels.count)
            predicted.append(int(majority_label))

        return predicted

    def calc_euclid_dist(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))


x_train, x_test, y_train, y_test = read_data_from_file('src/LR3_input2.txt')

knn = KNNClassifier()
knn.fit(x_train, y_train)
y_predicted = knn.predict(x_test)

write_list_to_file('output.txt', y_predicted)

