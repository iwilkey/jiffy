""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

import threading
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from gif import GIF
from hypers import *

class JiffyGenerator(tf.keras.Model):
    
    def __init__(self, filters=[128, 64], kernel_sizes=[(5,5,5), (5,5,5)], strides=[(1,2,2), (2,2,2)]):
        super(JiffyGenerator, self).__init__()
        self.dense = layers.Dense(filters[0] * (GIF_FIXED_FRAMES // 2) * (GIF_SAMPLE_SIZE[0] // 4) * (GIF_SAMPLE_SIZE[1] // 4), use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU()
        self.convs = []
        for i in range(len(filters)):
            self.convs.append(layers.Conv3DTranspose(filters[i], kernel_sizes[i], strides=strides[i], padding='same', use_bias=False))
            self.convs.append(layers.BatchNormalization())
            self.convs.append(layers.LeakyReLU())
        self.final_conv = layers.Conv3DTranspose(CHANNELS, (5, 5, 5), strides=(1, 1, 1), padding='same', use_bias=False, activation='tanh')
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        x = tf.reshape(x, (-1, GIF_FIXED_FRAMES // 2, GIF_SAMPLE_SIZE[0] // 4, GIF_SAMPLE_SIZE[1] // 4, 128))
        for layer in self.convs:
            x = layer(x)
        x = self.final_conv(x)
        return x
    
class JiffyDiscriminator(tf.keras.Model):
    
    def __init__(self, filters=[64, 128], kernel_sizes=[(5,5,5), (5,5,5)], strides=[(2,2,2), (2,2,2)]):
        super(JiffyDiscriminator, self).__init__()
        self.convs = []
        for i in range(len(filters)):
            self.convs.append(layers.Conv3D(filters[i], kernel_sizes[i], strides=strides[i], padding='same'))
            self.convs.append(layers.LeakyReLU())
            self.convs.append(layers.Dropout(0.3))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = inputs
        for layer in self.convs:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
class TrainingArtifacts:
    """ Data shared between the training session and the main thread.
    """
    
    def __init__(self, **kwargs):
        self.latest_gif : GIF = GIF()
        self.epoch : int = 0
        self.d_loss : float = 0.0
        self.g_loss : float = 0.0
    
    # Setters and getters.
    
    def set_epoch(self, epoch : int):
        self.epoch = epoch
        
    def set_latest_gif_frames(self, frames):
        self.latest_gif.frames = frames
        
    def set_d_loss(self, loss : float):
        self.d_loss = loss
    
    def set_g_loss(self, loss : float):
        self.g_loss = loss
    
    def get_epoch(self) -> int:
        return self.epoch
    
    def get_latest_generated_gif(self) -> GIF:
        return self.latest_gif
    
    def get_d_loss(self) -> float:
        return self.d_loss
    
    def get_g_loss(self) -> float:
        return self.g_loss

class JiffyGenerativeAdversarialNetwork:
    
    def __init__(self, training_artifacts : TrainingArtifacts):
        # Training artifacts.
        self.artifacts = training_artifacts
        # Initialize the generator and discriminator.
        self.generator = JiffyGenerator()
        self.discriminator = JiffyDiscriminator()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, BETA_1)

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([images.shape[0], LATENT_DIM])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) 
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def session(self, dataset : tf.Tensor, epochs : int, steps_per_epoch : int, stop_event : threading.Event):
        for epoch in range(1, epochs + 1):
            # Keep track of the current epoch.
            self.artifacts.set_epoch(epoch)
            # Check for the stop event.
            if stop_event.is_set():
                return
            # Run training steps.
            for step, image_batch in enumerate(dataset.take(steps_per_epoch)):
                gen_loss, disc_loss = self.train_step(image_batch)
                gen_loss_float = float(gen_loss.numpy())
                disc_loss_float = float(disc_loss.numpy())
                self.artifacts.set_d_loss(disc_loss_float)
                self.artifacts.set_g_loss(gen_loss_float)
            # Generate a gif to render after each epoch.
            noise = tf.random.normal([1, LATENT_DIM])
            generated_images = self.generator(noise, training=False)
            arr = np.clip(generated_images.numpy() * 127.5 + 127.5, 0, 255).astype(np.uint8)
            self.artifacts.set_latest_gif_frames(arr)

class JiffyTrainer(threading.Thread):
    
    def __init__(self, **kwargs):
        """ Creates a new JIFF-Y training thread.
        """
        super().__init__()
        # Training artifacts, public for main thread.
        self.artifacts : TrainingArtifacts = TrainingArtifacts()
        # Create a new JIFF-Y GAN.
        self.gan = JiffyGenerativeAdversarialNetwork(self.artifacts)
        # Stop event.
        self.stop_event = threading.Event()
        
    def get_artifacts(self) -> TrainingArtifacts:
        return self.artifacts
    
    def run(self):
        """ Run the training process until the client sets the stop event.
        """
        print(f"[JiffyTrainer] SIGSTART RECEIVED. Starting.")
        
        # Load the target GIF.
        TARGET_PATH = "../data/data/target.gif"
        # Create a list of GIF objects to be used for training.
        gif_objects : GIF = []
        # Encapsulate the target GIF.
        gif = GIF()
        gif.create_from_path(TARGET_PATH, preprocess=True, augment=False, standardize=True)
        gif_objects.append(gif.get_numpy_gif_object_32_bit()[0])
        # Create a N number of augmented target GIFs for training.
        for _ in range(AUGMENTATIONS):
            gif = GIF()
            gif.create_from_path(TARGET_PATH, preprocess=True, augment=True, standardize=True)
            gif_objects.append(gif.get_numpy_gif_object_32_bit()[0])
        self.data = np.array(gif_objects)
        # Create TensorFlow data object.
        BATCH_SIZE = int(self.data.shape[0] / BATCH_FACTOR)
        BUFFER_SIZE = 10000
        train_dataset = tf.data.Dataset.from_tensor_slices(self.data)
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        
        # Start the session.
        self.gan.session(train_dataset, EPOCHS, len(self.data) // BATCH_SIZE, self.stop_event)
        
        # This loop will train the GAN until the stop event is set.
        if self.stop_event.is_set():
            return
        
    def stop(self):
        """ Stop the training process directly.
        """
        print(f"[JiffyTrainer] SIGKILL RECEIVED. Stopping.")
        self.stop_event.set()
        