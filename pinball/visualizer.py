import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Visualizer(object):
    def __init__(self, action_label, option_label=None):
        self.action_label = action_label
        if option_label is not None:
            self.option_label = option_label
        else:
            self.option_label = [""]
        self.init_action_viz()
        self.init_option_q_vis()

    def init_action_viz(self):
        self.ax = plt.subplot2grid((1,2),(0,0))
        n_actions = len(self.action_label)
        x = np.arange(n_actions)
        y = np.zeros(n_actions)
        self.action_dist = self.ax.bar(x, y)
        plt.xticks(x, self.action_label)
        self.ax.set_ylim([0,1])
        # print(self.action_label)
    
    def init_option_q_vis(self):
        self.op_ax = plt.subplot2grid((1,2),(0,1))
        n_options = len(self.option_label)
        x = np.arange(n_options)
        y = np.zeros(n_options)
        self.opt_q = self.op_ax.bar(x,y)
        plt.xticks(x, self.option_label)
        self.op_ax.set_ylim([0,1])
        # print(self.option_label)
        
    def set_action_dist(self, action_dist, action):
        for i, i_action in enumerate(self.action_dist):
            i_action.set_height(action_dist[i])
            i_action.set_color("b")
        self.action_dist[action].set_color("r")


    def set_option_q(self, option_q, option):
        for i, i_option in enumerate(self.opt_q):
            i_option.set_height(option_q[i])
            i_option.set_color("b")
        min_y, max_y = self.op_ax.get_ylim()
        max_y_q = np.max(option_q)
        min_y_q = np.min(option_q)
        if max_y_q > max_y and min_y_q < min_y:
            self.op_ax.set_ylim(min_y_q, max_y_q)
        elif max_y_q > max_y and min_y_q > min_y:
            self.op_ax.set_ylim(min_y, max_y_q)
        elif max_y_q < max_y and min_y_q < min_y:
            self.op_ax.set_ylim(min_y_q, max_y)
        self.opt_q[option].set_color("r")

    def pause(self, second):
        plt.pause(second)
        



