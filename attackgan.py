from __future__ import annotations
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from helper import mapping

from const import *


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class AttackGan:
    def __init__(
        self,
        n_epochs: int = 5,
        latent_dim: int = 13,
        blackbox_path: str = "models/RF.pickle",
        blackbox_type: str = "sklearn",
        generator_optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        critic_optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        n_critic: int = 5,
        batchsize: int = 64,
        lambada: float = 0.5,
        ids_loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        log_save_path: str = "log/attackgan_log.feather",
        clip_value=0.01,
    ) -> None:
        self.n_epochs = n_epochs
        self.latent_dim = latent_dim
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.generator = self.__build_generator()
        self.critic = self.__build_critic()
        self.blackbox_type = blackbox_type
        self.blackbox = self.__load_blackbox(blackbox_path, blackbox_type)
        self.n_critics = n_critic
        self.batchsize = batchsize
        self.lambada = lambada
        self.clip_value = clip_value
        self.ids_loss = ids_loss
        self.log = pd.DataFrame(
            columns=["step", "epoch", "loss_critic", "loss_gen", "loss_blackbox"],
            dtype="float32",
        )
        self.log_save_path = log_save_path
        pass

    def __save_log(self):
        self.log.to_feather(self.log_save_path)
        pass

    def __logging(self, step, epoch, loss_critic, loss_gen, loss_blackbox):
        self.log = self.log.append(
            {
                "step": step,
                "epoch": epoch,
                "loss_critic": loss_critic,
                "loss_gen": loss_gen,
                "loss_blackbox": loss_blackbox,
            },
            ignore_index=True,
        )
        pass

    def __load_blackbox(self, blackbox_path, blackbox_type):
        if blackbox_type == "sklearn":
            with open(blackbox_path, "rb") as f:
                blackbox = pickle.load(f)
            return blackbox
        elif blackbox_type == "tf":
            blackbox = tf.keras.models.load_model(blackbox_path)
            return blackbox

    def __build_generator(self) -> tf.keras.models.Sequential:
        generator = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(units=128, input_shape=(self.latent_dim,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(units=128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(units=self.latent_dim),
            ],
            name="Generator",
        )
        self.generator_optimizer.build(generator.trainable_variables)
        return generator

    def __build_critic(self) -> tf.keras.models.Sequential:
        critic = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Dense(units=128, input_shape=(self.latent_dim,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(units=128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Dense(units=1),
            ],
            name="Critic",
        )
        self.critic_optimizer.build(critic.trainable_variables)
        return critic

    @tf.function
    def __train_generator(self, real_sample, n_batch):
        with tf.GradientTape() as tape:
            noise = tf.random.normal([n_batch, self.latent_dim])
            fake = self.generator(noise, training=True)
            fake_pred = self.critic(fake, training=True)
            fake_map = tf.numpy_function(mapping, [fake], Tout=tf.float32)
            if self.blackbox_type == "sklearn":
                blackbox_pred = tf.numpy_function(
                    self.blackbox.predict_proba, [fake_map], Tout=tf.double
                )
                blackbox_pred = tf.cast(blackbox_pred, tf.float32)
            elif self.blackbox_type == "tf":
                fake_map.set_shape([n_batch, 122])
                blackbox_pred = self.blackbox(fake_map, training=False)
            target = tf.zeros(n_batch)
            blackbox_loss = self.ids_loss(target, blackbox_pred)
            generator_loss = tf.reduce_mean(-fake_pred + self.lambada * blackbox_loss)

        gen_grad = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_grad, self.generator.trainable_variables)
        )

        return blackbox_loss, generator_loss

    @tf.function
    def __train_critic(self, real_sample, n_batch):
        with tf.GradientTape() as tape:
            noise = tf.random.normal([n_batch, self.latent_dim])
            fake = self.generator(noise, training=True)
            fake_pred = self.critic(fake, training=True)
            real_pred = self.critic(real_sample, training=True)
            critic_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables)
        )
        for var in self.critic.trainable_variables:
            var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))

        return critic_loss

    def train(self, X, y):
        train_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(X.values))
        train_data = train_data.shuffle(buffer_size=1024).batch(self.batchsize)

        for epoch in range(self.n_epochs):
            for step, X_real in enumerate(train_data):
                for _ in range(self.n_critics):
                    critic_loss = self.__train_critic(X_real, len(X_real))
                blackbox_loss, generator_loss = self.__train_generator(
                    X_real, len(X_real)
                )

                self.__logging(
                    step,
                    epoch,
                    critic_loss.numpy(),
                    generator_loss.numpy(),
                    blackbox_loss.numpy(),
                )

                sys.stdout.write("\r")
                sys.stdout.write(
                    "Step: {}, Epoch: {}, Loss Critic: {}, Loss Gen: {}, Loss Blackbox: {}".format(
                        step, epoch, critic_loss, generator_loss, blackbox_loss
                    )
                )
                sys.stdout.flush()
            sys.stdout.write("\n")

        print("\nTraining Finished")
        self.__save_log()

    def eval(self, X, y):
        detection_accuracy = lambda n, n_d: n_d / n
        attack_success_rate = lambda d_o, d_a: d_o - d_a
        evade_increase_rate = lambda d_o, d_a: 1 - d_a / d_o

        n = len(X)
        pred = self.blackbox.predict(X)

        if self.blackbox_type == "tf":
            pred = np.argmax(pred, axis=1)

        num_detect_org = np.sum(pred == 1)

        noise = tf.random.normal([n, self.latent_dim])
        adv_sample = self.generator(noise, training=False)
        # adv_sample = self.generator(X[content_feature].to_numpy(), training=False)
        adv_sample = mapping(adv_sample)
        pred_adv = self.blackbox.predict(adv_sample)

        if self.blackbox_type == "tf":
            pred_adv = np.argmax(pred_adv, axis=1)

        num_detect_adv = np.sum(pred_adv == 1)

        detection_rate_org = detection_accuracy(n, num_detect_org)
        detection_rate_adv = detection_accuracy(n, num_detect_adv)

        asr = attack_success_rate(detection_rate_org, detection_rate_adv)
        eir = evade_increase_rate(detection_rate_org, detection_rate_adv)

        print("Detection Rate Original: {}".format(detection_rate_org))
        print("Detection Rate Adversarial: {}".format(detection_rate_adv))
        print("Attack Success Rate: {}".format(asr))
        print("Evade Increase Rate: {}".format(eir))
