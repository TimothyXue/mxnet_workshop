# import some useful packages
import numpy as np
import cPickle
import os
from bokeh.layouts import row, gridplot, column
from bokeh.models import CustomJS, ColumnDataSource, Slider, Button, RadioGroup, WidgetBox
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot, output_file, show
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from ipywidgets import interact_manual
import mxnet as mx
import warnings
warnings.simplefilter('ignore',DeprecationWarning)
from VisCostCallback import CostVisCallback


def get_iterators():
    
    data = dict()
    data['X_train'] = mx.nd.array(svhn['X_train'], dtype=np.float32).reshape((-1,3,64,64))
    data['X_test'] = mx.nd.array(svhn['X_test'], dtype=np.float32).reshape((-1,3,64,64))
    data['y_train'] = mx.nd.array(svhn['y_train'])
    data['y_test'] = mx.nd.array(svhn['y_test'])
    
    train_set = mx.io.NDArrayIter(data['X_train'], data['y_train'], batch_size=128)
    valid_set = mx.io.NDArrayIter(data['X_test'], data['y_test'], batch_size=128)

    return (train_set, valid_set)


def build_model():
    data = mx.symbol.Variable(name="data")
    label = mx.sym.Variable('softmax_label')
    
    convp = dict(kernel=(3, 3), pad=(1, 1))
    
    conv1_1 = mx.symbol.Convolution(data=data, num_filter=32, name="conv1_1", **convp)
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(data=relu1_1, num_filter=32, name="conv1_2", **convp)
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    dropout1 = mx.symbol.Dropout(data=pool1, p=0.5,name="dropout1")

    conv2_1 = mx.symbol.Convolution(data=dropout1, num_filter=64, name="conv2_1", **convp)
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(data=relu2_1, num_filter=64, name="conv2_2", **convp)
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    dropout2 = mx.symbol.Dropout(data=pool2, p=0.5,name="dropout2")

    conv3_1 = mx.symbol.Convolution(data=dropout2,  num_filter=128, name="conv3_1", **convp)
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(data=relu3_1, num_filter=128, name="conv3_2", **convp)
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    flatten = mx.symbol.Flatten(data = relu3_2, name = "flatten")
    
    linear1 = mx.symbol.FullyConnected(data= flatten ,num_hidden=4, name='linear1')
    pred = mx.sym.LinearRegressionOutput(data=linear1, label=label, name='lro')
    
    model = mx.mod.Module(pred, context=mx.cpu(0))
    return model

def train_model(learning_inputs,
                fig=None, handle=None, train_source=None, val_source=None):

    train_set, test_set = get_iterators()

    model = build_model()
    
    learning_rates = [10**(-1 * (10-f)) for f in learning_inputs]
 
    callbacks = CostVisCallback(h=300, w=300, nepochs=4.0, y_range=None, fig=fig, handle=handle, train_source=train_source, val_source=val_source, total_batches=52)

    
    
    model.fit(
        train_data          = train_set,
        eval_data           = test_set,
        eval_metric         = mx.metric.MSE(),
        optimizer           = 'sgd',
        optimizer_params    = {'learning_rate': 10**-4, 'momentum': 0.9},
        initializer         = mx.init.Normal(sigma=0.01),
        num_epoch           = 4,
        batch_end_callback  = callbacks)
    
    
    # run fit
    metric = model.score(test_set, mx.metric.MSE())
    
    test_set.reset()
    
    X = test_set.getdata()[0].asnumpy().reshape(128, -1)
    #T = test_set.getlabel()[0].asnumpy()

    #y = model.predict(valid_set, 1).asnumpy()
    
    #result = {'img': X, 'pred': y, 'gt': T}
 
    return (metric[0][1], None)


class Dashboard():

    def __init__(self):
        self.best_cost = 10000
        self.setup_dashboard()
        self.learn_inputs = 4 * [3]

    def setup_dashboard(self):
        output_notebook()

        x = [0,    1,     1,      2,     2,     3,    3,    4]
        y = 8*[10 ** -7]

        source = ColumnDataSource(data=dict(x=x, y=y))

        plot = figure(plot_width=300, plot_height=200, y_axis_type="log", y_range=[0.0000000001, 1], x_range=[0, 4],
                      x_axis_label='Epoch', y_axis_label='Learning Rate')
        plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

        learn_inputs = 4 * [3]

        base_code = """
                var data = source.data;
                var f = cb_obj.value
                x = data['x']
                y = data['y']
                y[{}] = Math.pow(10.0, -1.0 * (10-f))
                y[{}] = Math.pow(10.0, -1.0 * (10-f))
                source.trigger('change');

                var command = 'dashboard.learn_inputs[{}] = ' + f;
                var kernel = IPython.notebook.kernel;
                kernel.execute(command)
            """

        # set up figure
        fig = figure(name="cost", y_axis_label="Cost", x_range=(0, 4),
                     x_axis_label="Epoch", plot_width=300, plot_height=300)
        self.fig = fig
        train_source = ColumnDataSource(data=dict(x=[], y=[]))
        train_cost = fig.line('x', 'y', source=train_source)
        self.train_source = train_source

        val_source = ColumnDataSource(data=dict(x=[], y=[]))
        val_cost = fig.line('x', 'y', source=val_source, color='red')
        self.val_source = val_source

        # set up sliders and callback
        callbacks = [CustomJS(args=dict(source=source), code=base_code.format(k, k+1, k/2)) for k in [0, 2, 4, 6]]
        slider = [Slider(start=0.1, end=10, value=3, step=.1, title=None, callback=C, orientation='vertical', width=80, height=50) for C in callbacks]

        radio_group = RadioGroup(labels=[""], active=0, width=65)

        def train_model_button(run=True):
            train_model(slider, fig=fig, handle=fh, train_source=train_source, val_source=val_source)
            
        sliders = row(radio_group, slider[0], slider[1], slider[2], slider[3])
        settings = column(plot, sliders)


        layout = gridplot([[settings, fig]], sizing_mode='fixed', merge_tools=True, toolbar_location=None)

        self.fh = show(layout, notebook_handle=True)

    def plot_results(self, result):
        plt.figure(2)
        imgs_to_plot = [0, 1, 2, 3]
        for i in imgs_to_plot:
            plt.subplot(2, 2, i+1)

            title = "test {}".format(i)
            plt.imshow(result['img'][:, i].reshape(3, 64, 64).transpose(1, 2, 0))
            y = result['pred']
            T = result['gt']
            ax = plt.gca()
            ax.add_patch(plt.Rectangle((y[0,i], y[1,i]), y[2,i], y[3,i], fill=False, edgecolor="red"))
            ax.add_patch(plt.Rectangle((T[0,i], T[1,i]), T[2,i], T[3,i], fill=False, edgecolor="blue"))
            plt.title(title)
            plt.axis('off')

    def train(self):
        learn_inputs = self.learn_inputs
        (cost, result) = train_model(learn_inputs, fig=self.fig, handle=self.fh, train_source=self.train_source, val_source=self.val_source)
        if cost < self.best_cost:
            self.best_cost = cost
        print "This run has Cost: {}".format(cost)
        print "Your best run so far has Cost: {}".format(self.best_cost)
        print "Note: lower is better."
        #self.plot_results(result)

    def show_button(self):
        interact_manual(self.train)

print "Setting up the data..."
fileName = 'data/svhn_64.p'
with open(fileName) as f:
    svhn = cPickle.load(f)

