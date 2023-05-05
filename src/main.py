""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

from arch import JiffyTrainer
from gif import GIF
from grapher import LossGrapher
from hypers import *

from typing import Tuple
import pygame
import imgui
from pimgu import Applet

version = "v05.04.2023"

if __name__ == "__main__":
    
    # Initialize the architecture and start the training process.
    gan = JiffyTrainer()
    gan.start()
    
    # Keep track of the current frame index for rendering.
    frame_idx = 0
    # Keep track of the number of ticks since the last frame change.
    ticks_since = 0
    
    # Surface to render the GIF onto.
    surface = pygame.Surface(WINDOWED_CLIENT_SIZE)
    
    # Create a new grapher to graph the losses as they occur.
    grapher = LossGrapher()
   
   # Create the application.
    app = Applet("JIFF-Y", WINDOWED_CLIENT_SIZE, "../assets/icon/jiffy-icon.png")
    app.set_target_fps(42)
    
    def on_gui():
        global grapher
        grapher.render()
        # Render the Hyperparameters window.
        imgui.begin("JIFF-Y General Information and Hyperparameters")
        imgui.text(f"JIFF-Y {version}")
        imgui.text("Copyright (C) 2023 Ian Wilkey. All rights reserved.")
        imgui.text("https://www.jiffy.iwilkey.com")
        imgui.separator()
        imgui.text(f"GIF Dimensions: {GIF_SAMPLE_SIZE[0]} x {GIF_SAMPLE_SIZE[1]}")
        imgui.text(f"Latent Dimension: {LATENT_DIM}")
        imgui.text(f"Augmentations: {AUGMENTATIONS}")
        imgui.text(f"Learning Rate: {LEARNING_RATE}")
        imgui.text(f"Beta 1: {BETA_1}")
        imgui.separator()
        imgui.text(f"Frame {frame_idx + 1} of {GIF_FIXED_FRAMES}")
        imgui.text(f"Epoch {gan.get_artifacts().get_epoch() - 1} of {EPOCHS}")
        imgui.text(f"G_LOSS: {gan.get_artifacts().get_g_loss()}")
        imgui.text(f"D_LOSS: {gan.get_artifacts().get_d_loss()}")
        imgui.end()
        
    def on_end():
        # Stop the training process right after the applet closes.
        gan.stop()
        gan.join()
    
    def on_render() -> Tuple[pygame.Surface, int, int]:
        # Access use of static variable, frame_idx.
        global frame_idx
        renderables, scaled_width, scaled_height = gan.get_artifacts().get_latest_generated_gif().create_renderables((WINDOWED_CLIENT_SIZE[0] / GIF_SAMPLE_SIZE[0]) / 4)
        # Center coordinates in surface.
        cx = (WINDOWED_CLIENT_SIZE[0] - scaled_width) // 2
        cy = (WINDOWED_CLIENT_SIZE[1] - scaled_height) // 2
        # BLIT the current frame onto the surface.
        surface.blit(renderables[frame_idx], (cx, cy))
        # Garbage control on renderables (good practice).
        del renderables
        # Send to OpenGL to be rendered.
        return surface, 0, 0

    def on_tick():
        # Access use of static variable, frame_idx.
        global frame_idx, ticks_since
        
        # Animate the GIF at 10 FPS.
        ticks_since = ticks_since + 1
        if ticks_since >= app.get_target_fps() // 10:   
            frame_idx = (frame_idx + 1) % GIF_FIXED_FRAMES
            ticks_since = 0
        
        # Update the grapher with the latest loss values.
        new_g_loss = gan.get_artifacts().get_g_loss()
        new_d_loss = gan.get_artifacts().get_d_loss()
        if new_g_loss and new_d_loss:
            grapher.tick(new_g_loss, new_d_loss)
            
    
    # Register callbacks.
    app.register_tick_callback(on_tick)
    app.register_pygame_render_callback(on_render)
    app.register_imgui_callback(on_gui)
    app.register_on_end_callback(on_end)
    
    # Run the application.
    app.run()
