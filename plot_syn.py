import matplotlib
matplotlib.use('Agg')

import numpy as np
import syn_data
import synth_data # TODO sorry, should be merged into syn_data
import matplotlib.pyplot as plt
import io
import PIL

def get_dataset(opts):
    dataset_name = opts['dataset'].replace('_', '-')
    dataset = syn_data.load(dataset_name, (28, 28), False)
    return dataset

def get_plots(x, opts, i):

    plot_dicts = []

    dataset_name = opts['dataset']
    if dataset_name == 'checkers':
        # TODO sorry, should be merged into syn_data as subclass.
        nearest_params = synth_data.get_nearest_params(x)
    else:
        dataset = get_dataset(opts)
        nearest_params = dataset.get_nearest_params(x)

    if dataset_name in ['syn_constant_uniform']:
        plot_dicts.append({'name': 'syn_constant_uniform_hist', 'plot': plot_hist(x, nearest_params, opts, i)})
    if dataset_name in ['syn_2_constant_uniform', 'checkers']:
        plot_dicts.append({'name': dataset_name + '_2dhist', 'plot': plot_hist2d(x, nearest_params, opts, i)})
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


def plot_hist2d(x, nearest_params, opts, i):
    plt.hist2d(nearest_params[:,0], nearest_params[:,1])
    plt.savefig(opts['name'] + '_' + str(i) + '_plot_2dhist.png')
    img = plot_to_image(plt)
    plt.clf()
    plt.close()
    return img


if __name__ == '__main__':
    get_plots(None, {"dataset": "syn_constant_uniform", "name": "a"})

