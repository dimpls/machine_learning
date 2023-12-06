from scipy.io import loadmat
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import numpy as np


def get_palette(num_classes):
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", num_classes - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))

    return palette


def convert_to_color(arr_2d, palette=None):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        palette = get_palette(np.max(arr_2d)+1)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


triple_coffee = loadmat('./triple/triple_coffee.mat')['image']
triple_coffee_gt = loadmat('./triple/triple_coffee_mask.mat')['img']

spectral_data_triple = triple_coffee.reshape(-1, triple_coffee.shape[2])

double_coffee = loadmat('./double/double_coffee.mat')['image']
double_coffee_gt = loadmat('./double/double_coffee_gt.mat')['img'].astype('uint8')

spectral_data_double = double_coffee.reshape(-1, double_coffee.shape[2])

X_train = spectral_data_triple
y_train = triple_coffee_gt.ravel()

X_test = spectral_data_double
y_test = double_coffee_gt.ravel()

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

pca = PCA(n_components=150)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

print(X_train_pca.shape, y_train.shape, X_test_pca.shape, y_test.shape)

rf = RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_split=200, class_weight='balanced')

rf.fit(X_train_pca, y_train)

y_pred = rf.predict(X_test_pca)
accuracy = np.mean(y_pred == y_test)
print("Точность:", accuracy_score(y_test, y_pred))













