from const import *
import pandas as pd

real_sample = pd.read_csv("dataset/train_gan.csv")
real_sample = (
    real_sample.drop(columns=["label"])[real_sample["label"] == 1]
    .reset_index(drop=True)
    .astype("float32")
)


def mapping(feat):
    feat = pd.DataFrame(feat, columns=content_feature, dtype="float32")
    real = real_sample.sample(n=feat.shape[0]).reset_index(drop=True).astype("float32")
    real[content_feature] = feat[content_feature]
    return real.to_numpy()
