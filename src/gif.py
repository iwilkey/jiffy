""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

# For directory and file access.
import os
# For random image augmentations.
import tensorflow as tf
# For rendering calculations.
import pygame as engine
# For standardized and powerful storage of GIF data.
import numpy as np
# For processing of external GIFs on host machine.
from PIL import Image, ImageSequence
# For standardized configuration values.
from hypers import *

class GIF:
    """ Contains the preprocessed frames of a JIFF-Y GIF.
    """
    
    def __init__(self, **kwargs):
        """ GIF constructor.
        """
        # The raw GIF, as imported by Pillow's Image object.
        self.raw_gif = None
        # The path to the GIF, if not created by computer.
        self.path = None
        # The GIF's frames in a NumPy array. Shape: (1, frames, width, height, channels).
        self.frames = []
        # When creating a brand new GIF object, it will always start as random noise.
        self.create_random()

    def create_from_path(self, path: str, preprocess=True, augment=True, standardize=True):
        """ Creates a GIF object from a GIF path.\n
        """
        # Check if GIF path exists.
        if os.path.exists(path) == False:
            raise FileNotFoundError("[JIFF-Y FATAL]: The specified GIF path does not exist.")
        # Set the path.
        self.path = path
        # Use Pillow to extract the GIFs.
        self.raw_gif = Image.open(path)
        # Extract Pillow gif object into NumPy array.
        self.__extract_frames()
        # If preprocess is true (by default), normalize all 8-bit integer values to [0, 1] as a 32-bit float for TensorFlow.
        if preprocess:
            self.__preprocess_frames(augment)
        # Standardizes the GIF duration.
        if standardize:
            self.__standardize_num_frames()
        # Ensure that the NumPy frames representation is always of the shape (1, frames, width, height, channels).
        self.__ensure_shape(num_frames = self.frame_count())

    def create_random(self):
        """ Replace the frames of the GIF with random noise in the shape (1, frames, width, height, channels) with data type uint8.
        """
        self.frames = np.random.randint(0, 255, (1, GIF_FIXED_FRAMES, GIF_SAMPLE_SIZE[0], GIF_SAMPLE_SIZE[1], 3), dtype=np.uint8)
    
    def frame_count(self):
        """ Returns the amount of frames in the GIF.
        """
        return self.frames.shape[(0 if len(self.frames.shape) == 4 else 1)]
    
    def get_numpy_gif_object_8_bit(self):
        """ Returns the NumPy unsigned 8-bit integer array of the GIF with shape (1, frames, width, height, channels).
        NOTE: This format should be used for rendering, as all of the color data will make sense to the renderer.
        """
        ret = self.get_numpy_representation()
        # If the frames are currently in a normalized 32-bit float format, convert them back to 8-bit integers.
        if self.frames.dtype == np.float32:
            ret = np.array(self.frames * ((2**8) - 1), dtype=np.uint8)
        return ret
    
    def get_numpy_gif_object_32_bit(self):
        """ Returns the NumPy 32-bit float array of the GIF with shape (1, frames, width, height, channels).
        NOTE: This format should be used for training, as it is normalized to [0, 1] and will be easier for the GAN to learn from.
        """
        ret = self.get_numpy_representation()
        # If the frames are currently in a normalized 32-bit float format, convert them back to 8-bit integers.
        if self.frames.dtype == np.uint8:
            ret = np.array(self.frames / ((2**8) - 1), dtype=np.float32)
        return ret
    
    def get_numpy_representation(self):
        """ Returns a copy of the NumPy array of the GIF. This ensures that no transformations are made to the original array.
        NOTE: One should NEVER modify the original NumPy array, as it could cause the GIF data to be corrupted.
        """
        return self.frames.copy()
    
    def create_renderables(self, scale : float):
        """ Takes the current GIF and returns renderables for the WindowedClient.
        """
        # Extract the GIF with shape (frames, width, height, channels).
        raw_frames = self.get_numpy_gif_object_8_bit()[0]
        # Check to see if the raw_frames are of the correct data type. The renderer should not render 32-bit floats.
        assert raw_frames.dtype == np.uint8, "[JIFF-Y FATAL]: The GIF frames are not of the correct data type (uint8). The renderer should not render 32-bit floats."
        # Rotate the frames to be in the correct orientation.
        raw_frames = raw_frames.transpose((0, 2, 1, 3))
        # Scaled gif dimensions.
        scaled_width, scaled_height = int(raw_frames.shape[1] * scale), int(raw_frames.shape[2] * scale)
        # Convert the numpy arrays to Surfaces and scale them.
        frames = [engine.transform.scale(engine.surfarray.make_surface(frame), (scaled_width, scaled_height)) for frame in raw_frames]
        # Return renderables.
        return frames, scaled_width, scaled_height
    
    def __extract_frames(self):
        """ Extracts the frames from the GIF and appends them to a NumPy frames list.
        """
        self.frames = []
        for frame in ImageSequence.Iterator(self.raw_gif):
            # Convert to RGB.
            frame = frame.convert('RGB')
            # Resize the frame to the specified size.
            frame = frame.resize(GIF_SAMPLE_SIZE)
            # Append the frame to the list of frames.
            self.frames.append(frame)
        # Convert to NumPy array of 8-bit type.
        self.frames = np.array([np.array(frame) for frame in self.frames], dtype=np.uint8)
    
    def __generate_random_augmentation_parameters(self):
        """ Generates random augmentation parameters for the GIF.
        """
        # Random brightness
        brightness_delta = tf.random.uniform([], minval=-0.5, maxval=0.5)
        # Random contrast
        contrast_factor = tf.random.uniform([], minval=0.5, maxval=2.0)
        # Random saturation (only applicable to RGB images)
        saturation_factor = tf.random.uniform([], minval=0.5, maxval=2.0)
        # Random hue (only applicable to RGB images)
        hue_delta = tf.random.uniform([], minval=-0.5, maxval=0.5)
        # Random gamma (only applicable to RGB images)
        gamma_factor = tf.random.uniform([], minval=1, maxval=1.5)
        # Random quality of the GIF.
        quality = tf.random.uniform([], minval=0, maxval=100, dtype=tf.int32)
        return brightness_delta, contrast_factor, saturation_factor, hue_delta, gamma_factor, quality

    def __apply_random_augmentation_from_parameters(self, image : tf.Tensor, 
                                                    brightness_delta : float, 
                                                    contrast_factor : float, saturation_factor : float, 
                                                    hue_delta : float, gamma_factor : float, quality : int):
        """ Apply an augmentation to the image using the given parameters.
        """
        # Apply easy transformations.
        image = tf.image.adjust_brightness(image, brightness_delta)
        image = tf.image.adjust_contrast(image, contrast_factor)
        image = tf.image.adjust_saturation(image, saturation_factor)
        image = tf.image.adjust_hue(image, hue_delta)
        image = tf.image.adjust_gamma(image, gamma_factor)
        # Cast working image to uint8.
        u8im = tf.clip_by_value(image, 0, 1)  # Ensure values are within [0, 1]
        u8im = tf.multiply(u8im, 255)  # Scale values to [0, 255]
        u8im = tf.cast(u8im, tf.uint8)  # Cast tensor to uint8
        # Encode the image to JPEG format with desired quality.
        quality_image = tf.image.encode_jpeg(u8im, quality=quality)
        u8im = tf.image.decode_jpeg(quality_image, channels=3)
        # Convert back to float32.
        u8im = tf.divide(u8im, 255)
        image = tf.cast(u8im, tf.float32)
        return image

    @tf.autograph.experimental.do_not_convert
    def __preprocess_frames(self, augment : bool):
        """ Pre-processes the frames by converting them to NumPy arrays and normalizing the pixel values. 
        NOTE: This will make the GIF 32-bit float with pixel range [0, 1], not 8-bit integer [0, 255].
        """
        # Convert to 32-bit float NumPy array and normalize pixel values [0, 1].
        self.frames = np.array([np.array(frame) for frame in self.frames], dtype=np.float32)
        # Normalize pixel values between 0 and 1.
        self.frames = self.frames / 255.0
        # Augment the GIF frames, only if specified in GIF constructor that augmentation should be applied.
        if augment:
            # Convert NumPy array to TensorFlow tensor.
            self.frames = tf.convert_to_tensor(self.frames)
            # Get random augmentation parameters
            augmentation_parameters = self.__generate_random_augmentation_parameters()
            # Apply the same random augmentations to all frames
            self.frames = tf.map_fn(lambda frame: self.__apply_random_augmentation_from_parameters(frame, *augmentation_parameters), self.frames)
            # Convert TensorFlow tensor back to NumPy array.
            self.frames = self.frames.numpy()

    def __standardize_num_frames(self):
        """ Ensures that all GIFs have the same number of frames by either padding or truncating.
        """
        current_num_frames = len(self.frames)
        if current_num_frames < GIF_FIXED_FRAMES:
            # Pad frames with the last frame.
            pad_frames = np.tile(self.frames[-1], (GIF_FIXED_FRAMES - current_num_frames, 1, 1, 1))
            self.frames = np.concatenate((self.frames, pad_frames), axis=0)
        elif current_num_frames > GIF_FIXED_FRAMES:
            # Truncate frames.
            self.frames = self.frames[: GIF_FIXED_FRAMES]
        # Ensure the frames array has the fixed number of frames.
        assert len(self.frames) == GIF_FIXED_FRAMES, f"[JIFF-Y FATAL] Expected {GIF_FIXED_FRAMES} frames, but got {len(self.frames)} frames."

    def __ensure_shape(self, num_frames : int = GIF_FIXED_FRAMES):
        """ Ensure the shape of self.frames is (1, frames, width, height, channels).
        """
        if len(self.frames.shape) != 5:
            self.frames = np.expand_dims(self.frames, axis=0)
        assert self.frames.shape == (1, num_frames, GIF_SAMPLE_SIZE[0], GIF_SAMPLE_SIZE[1], 3), \
            f"[JIFF-Y FATAL] Expected shape (1, {num_frames}, {GIF_SAMPLE_SIZE[0]}, {GIF_SAMPLE_SIZE[1]}, 3), but got {self.frames.shape}."
