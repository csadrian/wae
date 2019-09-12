import matplotlib
matplotlib.use('Agg')

import numpy as np
import syn_data
import matplotlib.pyplot as plt
import io
import PIL

def get_dataset(opts):
    dataset_name = opts['dataset'].replace('_', '-')
    dataset = syn_data.load(dataset_name, (28, 28), False)
    return dataset

def get_plots(x, opts, i):

    plot_dicts = []
    dataset = get_dataset(opts)
    #x = dataset.generate_samples(10000)
    nearest_params = dataset.get_nearest_params(x)
  
    if opts['dataset'] in ['syn_constant_uniform']:
        plot_dicts.append({'name': 'syn_constant_uniform_hist', 'plot': plot_hist(x, nearest_params, opts, i)})
    if opts['dataset'] in ['syn_2_constant_uniform']:
        plot_dicts.append({'name': 'syn_2_constant_uniform_2dhist', 'plot': plot_2dhist(x, nearest_params, opts, i)})
    return plot_dicts


def plot_to_image(plt):
    buffer = io.StringIO()
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    img = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    return img
    
def plot_hist(x, nearest_params, opts, i):
    plt.hist(nearest_params)
    plt.savefig(opts['name'] + '_' + str(i) + '_plot_hist.png')
    img = plot_to_image(plt)
    plt.clf()
    plt.close()
    return img


def plot_2dhist(x, nearest_params, opts, i):
    plt.hist(nearest_params)    
    plt.savefig(opts['name'] + '_' + str(i) + '_plot_2dhist.png')
    img = plot_to_image(plt)
    plt.clf()
    plt.close()
    return img


if __name__ == '__main__':
    get_plots(None, {"dataset": "syn_constant_uniform", "name": "a"})

