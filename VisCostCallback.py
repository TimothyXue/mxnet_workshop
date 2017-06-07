from bokeh.plotting import output_notebook, figure, ColumnDataSource, show
from bokeh.io import push_notebook
from timeit import default_timer
import math

class CostVisCallback(object):
    """
    Callback providing a live updating console based progress bar.
    """

    def __init__(self, epoch_freq=1, y_range=(0, 4.5), fig=None, handle=None,
                 update_thresh_s=0.65, w=400, h=300, nepochs=1.0, total_batches=10.0,
                 train_source=None, val_source=None):

        self.update_thresh_s = update_thresh_s
        self.w = w
        self.h = h
        self.nepochs = nepochs
        self.total = total_batches
        self.last_update = 0
        self.epoch = 0
        
        if handle is None:
            output_notebook()
            self.handle = None
        else:
            self.handle = handle

        if fig is None:
            self.fig = figure(name="cost", y_axis_label="Cost", x_range=(0, self.nepochs), y_range=y_range,
                              x_axis_label="Epoch", plot_width=self.w, plot_height=self.h)
        else:
            self.fig = fig

        if train_source is None:
            self.train_source = ColumnDataSource(data=dict(x=[], y=[]))
        else:
            self.train_source = train_source
            self.train_source.data = dict(x=[], y=[])

        self.train_cost = self.fig.line('x', 'y', source=self.train_source)

        if val_source is None:
            self.val_source = ColumnDataSource(data=dict(x=[], y=[]))
        else:
            self.val_source = val_source
            self.val_source.data = dict(x=[], y=[])

        self.val_cost = self.fig.line('x', 'y', source=self.val_source, color='red')
 
    def __call__(self, param):
        if self.handle is None:
            self.handle = show(self.fig, notebook_handle=True)
   
        now = default_timer()
            
        time = float(param.nbatch) / self.total + self.epoch
        if param.nbatch == self.total:
            self.epoch = self.epoch + 1
            
        if param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            param.eval_metric.reset()
            cost = name_value[0][1]
            self.train_source.data['x'].append(time)
            self.train_source.data['y'].append(cost)
        
        if (now - self.last_update > self.update_thresh_s):
            self.last_update = now
            
            if self.handle is not None:
                push_notebook(handle=self.handle)
            else:
                push_notebook()

            
# def on_epoch_begin(self, callback_data, model, epoch):
#         """
#         Since the number of minibatches per epoch is not constant, calculate it here.
#         """
#         self.start_epoch = self.last_update = default_timer()
#         self.nbatches = model.nbatches
# 
#     def on_minibatch_end(self, callback_data, model, epoch, minibatch):
#         """
#         Read the training cost already computed by the TrainCostCallback out of 'callback_data', and display it.
#         """
#         now = default_timer()
#         mb_complete = minibatch + 1
# 
#         mbstart = callback_data['time_markers/minibatch'][epoch-1] if epoch > 0 else 0
#         train_cost = callback_data['cost/train'][mbstart + minibatch]
# 
#         if math.isnan(train_cost) or train_cost > 20000:
#             tc = 5000
#         else:
#             tc = train_cost
# 
#         mb_epoch_scale = epoch + minibatch / float(self.nbatches)
#         self.train_source.data['x'].append(mb_epoch_scale)
#         self.train_source.data['y'].append(tc)
# 
#         if (now - self.last_update > self.update_thresh_s or mb_complete == self.nbatches):
#             self.last_update = now
# 
#             if self.handle is not None:
#                 push_notebook(handle=self.handle)
#             else:
#                 push_notebook()
