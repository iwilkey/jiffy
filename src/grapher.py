""" JIFF-Y, an open-source GIF generator utilizing a robust GAN.
Author: Ian Wilkey
Copyright (C) 2023. All rights reserved.
https://www.jiffy.iwilkey.com
"""

import imgui
import numpy as np

class LossGrapher:
    
    def __init__(self, **kwargs):
        self.data_points = 100
        self.g_loss_data = np.zeros(self.data_points, dtype=np.float32)
        self.d_loss_data = np.zeros(self.data_points, dtype=np.float32)
    
    # Function to update data with new loss values.
    def __update_data(self, new_g_loss, new_d_loss):
        for data, value in zip([self.g_loss_data, self.d_loss_data], [new_g_loss, new_d_loss]):
            if len(data) < self.data_points:
                data = np.append(data, value)
            else:
                data[:-1] = data[1:]
                data[-1] = value
    
    # In your rendering loop, call ImGui functions to draw the real-time graph.
    def render(self):
        # Create a window and plot the data
        imgui.begin("Real-time Model Losses")
        
        # Find the max value in the data to scale the graph.
        mx = np.max([np.max(self.g_loss_data), np.max(self.d_loss_data)])
        
        imgui.plot_lines("G_LOSS", self.g_loss_data, graph_size=imgui.Vec2(300, 150), scale_min=0, scale_max=mx)
        imgui.same_line()
        imgui.plot_lines("D_LOSS", self.d_loss_data, graph_size=imgui.Vec2(300, 150), scale_min=0, scale_max=mx)
        imgui.end()
    
    # In your update loop or callback, update the data and call the render function.
    def tick(self, new_g_data : float, new_d_data : float):
        self.__update_data(new_g_data, new_d_data)
    