import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
import tensorflow as tf

# load dataset
X = pd.read_csv("dataset/train_classifier.csv")
y = X.iloc[:, -1]
X = X.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.25, stratify=y
)

input_shape = X_train.shape[1]
output_shape = 5

# build model
clf_svc = SVC(probability=True, verbose=True)
clf_dt = DecisionTreeClassifier(criterion="entropy")
clf_rf = RandomForestClassifier(n_jobs=-1, n_estimators=1000, criterion="entropy")
# clf_nb = CategoricalNB()

clf_dnn = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(output_shape, activation="softmax"),
    ],
    name="DNN",
)

clf_dnn.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    metrics=["accuracy"],
)

# train model
clf_svc.fit(X_train, y_train)
print("trained SVC")
clf_dt.fit(X_train, y_train)
print("trained DT")
clf_rf.fit(X_train, y_train)
print("trained RF")
# clf_nb.fit(X_train, y_train)
# print("trained NB")
clf_dnn.fit(X_train, y_train, verbose=1, epochs=35)

# save model
with open("models/SVM.pickle", "wb") as handle:
    pickle.dump(clf_svc, handle)

with open("models/DT.pickle", "wb") as handle:
    pickle.dump(clf_dt, handle)

with open("models/RF.pickle", "wb") as handle:
    pickle.dump(clf_rf, handle)

# with open("models/NB.pickle", "wb") as handle:
#     pickle.dump(clf_nb, handle)

clf_dnn.save("models/DNN.h5")
