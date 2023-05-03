""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

import pygame as engine

import numpy as np

import tensorflow as tf

from arch import JiffyTrainer
from gif import GIF
from client import WindowedClient
from hypers import *

JIFF_Y_VERSION = "v05.03.2023"

if __name__ == "__main__":
    # Initialize the architecture and start the training process.
    gan = JiffyTrainer()
    gan.start()
    # Store the current frame index of the latest generated gif.
    frame_idx = 0
    # Create a WindowedClient instance.
    client = WindowedClient(WINDOWED_CLIENT_SIZE[0], WINDOWED_CLIENT_SIZE[1])
    # Main application loop.
    while True:
        # Syncs engine to frame rate.
        client.sync()
        # Poll for events.
        events = client.poll()
        # Handle events.
        for event in events:
            if event.type == engine.QUIT:
                # Stop the GAN from training.
                gan.stop()
                gan.join()
                # Stop the engine.
                engine.quit()
                break
            
        # Clear the window.
        client.clear()
        # Capture the latest generated GIF.
        latest_gif = gan.get_artifacts().get_latest_generated_gif()
        # Draw the current frame (blow up to scale the entire screen using square ratio.)
        client.draw_gif_frame(latest_gif, frame_idx, (WINDOWED_CLIENT_SIZE[0] / GIF_SAMPLE_SIZE[0]) / 2)
        # Move to the next frame (and loop).
        frame_idx = (frame_idx + 1) % latest_gif.frame_count()
        # Draw info text.
        client.add_text_to_container(f"JIFF-Y {JIFF_Y_VERSION}")
        client.add_text_to_container(f"Copyright (C) 2023 Ian Wilkey.")
        client.add_text_to_container(f"All Rights Reserved.")
        client.add_text_to_container(f"GIF Dimensions: {GIF_SAMPLE_SIZE[0]} x {GIF_SAMPLE_SIZE[1]}")
        client.add_text_to_container(f"Augmentations: {AUGMENTATIONS}")
        client.add_text_to_container(f"Latent Dimensions: {LATENT_DIM}")
        client.add_text_to_container(f"Learning Rate: {LEARNING_RATE}")
        client.add_text_to_container(f"Beta 1: {BETA_1}")
        client.add_text_to_container(f"Batch Factor: {BATCH_FACTOR}")
        client.add_text_to_container(f"Frame: {frame_idx + 1} / {latest_gif.frame_count()}")
        client.add_text_to_container(f"Epoch: {gan.get_artifacts().get_epoch()} / {EPOCHS}")
        client.add_text_to_container(f"G_LOSS: {round(gan.get_artifacts().get_g_loss(), 4)}", color=(0, 255, 0))
        client.add_text_to_container(f"D_LOSS: {round(gan.get_artifacts().get_d_loss(), 4)}", color=(255, 0, 0))
        # Render the text.
        client.text_flush()
        # Flip the buffers for rendering.
        client.flip_buffers()
