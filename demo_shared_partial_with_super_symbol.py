import logging
import mxnet as mx


logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

mnist = mx.test_utils.get_mnist()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# Shared symbol 
def get_shared_symbol(data):
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='act1', act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type="relu")
    return act2

# Build master module
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
act2 = get_shared_symbol(data)
mlp_master = mx.mod.Module(symbol=act2, label_names=None, context=mx.cpu())
mlp_master.bind(data_shapes=train_iter.provide_data, label_shapes=None)
mlp_master.init_params()


# Module 1
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)

act2 = get_shared_symbol(data)
fc3 = mx.sym.FullyConnected(data=act2, name='fc3_1', num_hidden=5)
mlps = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

mlp_model = mx.mod.Module(symbol=mlps, context=mx.cpu())
# ========================================================
# DON'T WORK this line crash with :
# > terminate called after throwing an instance of 'std::out_of_range'
# >   what():  _Map_base::at
# > Abandon
mlp_model.bind(shared_module=mlp_master, data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mlp_model.init_params()

# Module 2
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)

act2 = get_shared_symbol(data)
fc3 = mx.sym.FullyConnected(data=act2, name='fc3_2', num_hidden=10)
mlps = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

mlp_model2 = mx.mod.Module(symbol=mlps, context=mx.cpu())
mlp_model2.bind(shared_module=mlp_master, data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mlp_model2.init_params()

# Train module 1
print("\n===Training module1===\n")
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=5)  # train for at most 10 dataset passes


# Train module 2
# We expect the shared module to start where the first module finished
print("\n===Training module2===\n")
mlp_model2.fit(train_iter,  # train data
               eval_data=val_iter,  # validation data
               optimizer='sgd',  # use SGD to train
               optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
               eval_metric='acc',  # report accuracy during training
               batch_end_callback=mx.callback.Speedometer(batch_size, 100),
               num_epoch=5)  # train for at most 10 dataset passes


# Making sure that fit doesn't always overwrite parameters by returning to module 1
print("\n===Training module1===\n")
mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=5)  # train for at most 10 dataset passes