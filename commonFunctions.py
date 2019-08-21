'''
Functions used in different files are gathered here to avoid redundance.
'''
import os
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Plots
import matplotlib.pyplot as plt
import sys
import pickle

def plotter(path,Ylabel,Title,log=False):
    open_= open(path,'rb')
    plot_= pickle.load(open_)
    if log:
        plt.semilogy(plot_)
    else:
        plt.plot(plot_)
    plt.ylabel(Ylabel)
    plt.xlabel("Epochs")
    plt.legend()
    plt.title(Title)
