import logging
import mxnet as mx


logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

mnist = mx.test_utils.get_mnist()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# Shared symbol 
def get_all_symbols(data):
    # Shared
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='act1', act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type="relu")

    # Module 1
    fc3_1 = mx.sym.FullyConnected(data=act2, name='fc3_1', num_hidden=5)
    mlps1 = mx.sym.SoftmaxOutput(data=fc3_1, name='softmax1')
    
    # Module 2
    fc3_2 = mx.sym.FullyConnected(data=act2, name='fc3_2', num_hidden=10)
    mlps2 = mx.sym.SoftmaxOutput(data=fc3_2, name='softmax2')

    return mx.sym.Group([mlps1, mlps2])

def get_module1_symbols(data):
    # Shared
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='act1', act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type="relu")

    # Module 1
    fc3_1 = mx.sym.FullyConnected(data=act2, name='fc3_1', num_hidden=5)
    mlps1 = mx.sym.SoftmaxOutput(data=fc3_1, name='softmax1')
    
    return mlps1

def get_module2_symbols(data):
    # Shared
    fc1 = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='act1', act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, name='act2', act_type="relu")

    # Module 1
    fc3_2 = mx.sym.FullyConnected(data=act2, name='fc3_2', num_hidden=10)
    mlps2 = mx.sym.SoftmaxOutput(data=fc3_2, name='softmax2')
    
    return mlps2

# Build master module
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
act2 = get_all_symbols(data)
mlp_master = mx.mod.Module(symbol=act2, label_names=None, context=mx.cpu())
mlp_master.bind(data_shapes=train_iter.provide_data, label_shapes=None)
# ========================================================
# DON'T WORK this line crash with :
# > ValueError: Unknown initialization pattern for softmax1_label
mlp_master.init_params()
# ========================================================

# Module 1
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
mlps1 = get_module1_symbols(data)
mlp_model1 = mx.mod.Module(symbol=mlps1, context=mx.cpu())
mlp_model1.bind(shared_module=mlp_master, data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mlp_model1.init_params()

# Module 2
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
mlps2 = get_module2_symbols(data)
mlp_model2 = mx.mod.Module(symbol=mlps2, context=mx.cpu())
mlp_model2.bind(shared_module=mlp_master, data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mlp_model2.init_params()

# Train module 1
print("\n===Training module1===\n")
mlp_model1.fit(train_iter,  # train data
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
mlp_model1.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=5)  # train for at most 10 dataset passes