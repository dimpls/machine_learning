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


class NBClassifier:

    def __init__(self):
        self.class_priors = None
        self.class_vars = None
        self.class_means = None
        self.classes = None
        self.m = None
        self.n = None

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.classes = np.unique(y, return_counts=False)

        self.class_means = np.zeros((len(self.classes), self.n))
        self.class_vars = np.zeros((len(self.classes), self.n))
        self.class_priors = np.zeros(len(self.classes))

        for class_index, class_label in enumerate(self.classes):
            class_mask = (y == class_label)
            X_samples = X[class_mask]

            self.class_means[class_index, :] = X_samples.mean(axis=0)
            self.class_vars[class_index, :] = X_samples.var(axis=0)
            self.class_priors[class_index] = X_samples.shape[0] / float(self.m)

    def predict(self, X):
        y_predicted = []
        for x in X:
            posteriors = []

            for class_index in range(len(self.classes)):
                prior = np.log(self.class_priors[class_index])
                likelihood = np.sum(np.log(self.probability_density_calculations(class_index, x)))
                likelihood += prior
                posteriors.append(likelihood)

            max_posterior_index = np.argmax(posteriors)
            predicted_class = self.classes[max_posterior_index]
            y_predicted.append(int(predicted_class))

        return np.array(y_predicted)

    def probability_density_calculations(self, class_index, x):
        return np.exp(-((x - self.class_means[class_index]) ** 2) /
                      (2 * self.class_vars[class_index])) / np.sqrt(2 * np.pi * self.class_vars[class_index])


x_train, x_test, y_train, y_test = read_data_from_file('src/LR2_input7.txt')

nb_classifier = NBClassifier()
nb_classifier.fit(x_train, y_train)
y_predicted = nb_classifier.predict(x_test)

write_list_to_file('output.txt', y_predicted)

