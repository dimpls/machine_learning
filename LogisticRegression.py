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


def sigm(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:

    def __init__(self, learn_speed=0.001, count_of_iters=1000):
        self.count_of_iters = count_of_iters
        self.learn_speed = learn_speed
        self.weights = None
        self.bias = None

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.count_of_iters):
            z = X.dot(self.weights) + self.bias
            a = sigm(z)

            derivative_weights = np.dot(X.T, a - y) / m
            derivative_bias = np.sum(a - y) / m

            self.weights -= self.learn_speed * derivative_weights
            self.bias -= self.learn_speed * derivative_bias

    def predict(self, X):
        return [0 if y <= 0.5 else 1 for y in sigm(np.dot(X, self.weights) + self.bias)]



x_train, x_test, y_train, y_test = read_data_from_file('src/LR1_input3.txt')

log_reg = LogisticRegression()
log_reg.train(x_train, y_train)
y_pred = log_reg.predict(x_test)

write_list_to_file('output.txt', y_pred)
