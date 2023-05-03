""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

import pygame as engine
from gif import GIF
from hypers import *

class WindowedClient():
    """ A WindowedClient for any platform. Handles user interaction and renders images and sound.
    """
    
    def __init__(self, width, height, **kwargs):
        # Initialize engine.
        engine.init()
        # Window dimensions.
        self.width = width
        self.height = height
        self.screen = engine.display.set_mode((self.width, self.height))
        # Clock access.
        self.clock = engine.time.Clock()
        self.fps = 10
        # Init text container.
        self.__init_text_container(10)
        # Window metadata.
        engine.display.set_caption("JIFF-Y Standalone Client")
        self.__set_icon("../assets/icon/jiffy-icon.png")
    
    def __set_icon(self, icon_path):
        """ Sets the icon of the WindowedClient.
        """
        # Load the icon image
        icon = engine.image.load(icon_path)
        # Set the icon of the window
        engine.display.set_icon(icon)
        
    def sync(self):
        """ Syncs the main thread's timing to the target frame rate.
        """
        self.clock.tick(self.fps)
        
    def poll(self):
        """ Polls for events and returns them.
        """
        events = engine.event.get()
        return events

    def clear(self):
        self.screen.fill((0, 0, 0))
        self.container.fill((0, 0, 0))
    
    def draw_gif_frame(self, gif : GIF, frame : int, scale : float):
        """ Render a GIF object to the screen with given scale. It will render
        in the center by default. Must be called before flip_buffers().
        """
        renderables, scaled_width, scaled_height = gif.create_renderables(scale)
        # Center coordinates.
        center_x = ((self.width - scaled_width) // 2)
        center_y = (self.height - scaled_height) // 2
        # Draw current frame of GIF.
        self.screen.blit(renderables[frame], (center_x, center_y))
        
    def __init_text_container(self, padding):
        """ Creates a text container on the left hand side of the screen.
        """
        # Global font size for text.
        self.font_size = 20
        # Counter for number of texts drawn.
        self.texts_drawn = 0
        # Calculate the width of the container.
        gif_width = GIF_SAMPLE_SIZE[0] * ((WINDOWED_CLIENT_SIZE[0] / GIF_SAMPLE_SIZE[0]) / 2)
        width = (self.width // 2) - (gif_width // 2) - padding
        # Create the container surface.
        self.container = engine.Surface((width, self.height))
        
    def add_text_to_container(self, text : str, color : tuple = (255, 255, 255)):
        """ Draws white text to the left hand side of the screen.
        """
        # Set up the font
        font = engine.font.SysFont(None, self.font_size)
        # Create the text surface
        text = font.render(text, True, color)
        # Calculate the position of the text
        y = 10 + (self.texts_drawn * (self.font_size + 4))
        # Add the text to the container.
        self.container.blit(text, (10, y))
        # Add to the number of texts drawn.
        self.texts_drawn = self.texts_drawn + 1
    
    def text_flush(self):
        """ Flushes the text container to the renderer and resets the text counter.
        """
        self.screen.blit(self.container, (0, 0))
        self.texts_drawn = 0
        
    def flip_buffers(self):
        """ Flips the buffers for rendering.
        """
        engine.display.flip()
    
    def stop(self):
        """ Stops the WindowedClient.
        """
        engine.quit()
