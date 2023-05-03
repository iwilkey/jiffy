""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

###########################################
# JIFF-Y configuration and hyperparameters.
###########################################

# Windowed client dimensions (width, height).
WINDOWED_CLIENT_SIZE = (int(1920 / 1.5), int(1080 / 1.5))
# The size of the GIF sample (width, height).
GIF_SAMPLE_SIZE = (32, 32)
# The number of frames to normalize the GIF to.
GIF_FIXED_FRAMES = 6
# Number of target augmentations. The more, the better the GIF, longer to train.
AUGMENTATIONS = 2**6
# Batch size factor. The lower this number, the more images the GAN will see in an epoch.
BATCH_FACTOR = 4
# Epochs to train the GAN for.
EPOCHS = 10000
# The number of channels in the GIF.
CHANNELS = 3
# The latent dimension for the GAN.
LATENT_DIM = 150
# Learning rate for the GAN.
LEARNING_RATE = 0.0005
# Beta 1 for the GAN.
BETA_1 = 0.01
