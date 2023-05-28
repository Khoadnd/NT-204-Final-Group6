import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from attackgan import AttackGan
import pandas as pd
from const import *
import sys


def main():
    match sys.argv[1]:
        case "0":
            attackGan = AttackGan(
                blackbox_path="models/RF.pickle",
                blackbox_type="sklearn",
                n_epochs=35,
                log_save_path="log/RF_attackgan_log.feather",
            )
        case "1":
            attackGan = AttackGan(
                blackbox_path="models/SVM.pickle",
                blackbox_type="sklearn",
                n_epochs=35,
                log_save_path="log/SVM_attackgan_log.feather",
            )
        case "2":
            attackGan = AttackGan(
                blackbox_path="models/DT.pickle",
                blackbox_type="sklearn",
                n_epochs=35,
                log_save_path="log/DT_attackgan_log.feather",
            )
        case "3":
            attackGan = AttackGan(
                blackbox_path="models/NB.pickle",
                blackbox_type="sklearn",
                n_epochs=35,
                log_save_path="log/NB_attackgan_log.feather",
            )
        case "4":
            attackGan = AttackGan(
                blackbox_path="models/DNN.h5",
                blackbox_type="tf",
                n_epochs=35,
                log_save_path="log/DNN_attackgan_log.feather",
            )
        case _:
            exit(1)

    X = pd.read_csv("dataset/train_gan.csv")
    X = (
        X.drop(columns=["label"])[X["label"] == 1]
        .reset_index(drop=True)
        .astype("float32")
    )
    X_org = X.copy()
    X = X.loc[:, content_feature]

    attackGan.train(X, None)
    attackGan.eval(X_org, None)


if __name__ == "__main__":
    main()
