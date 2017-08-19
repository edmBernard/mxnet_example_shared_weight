import logging
import mxnet as mx

mnist = mx.test_utils.get_mnist()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# Set up variables to share parameters for all layers
w1 = mx.sym.Variable('1_weight')
w2 = mx.sym.Variable('2_weight')
w3 = mx.sym.Variable('3_weight')
b1 = mx.sym.Variable('1_bias')
b2 = mx.sym.Variable('2_bias')
b3 = mx.sym.Variable('3_bias')

# Shared symbol 
def get_shared_symbol(data, weight, bias):
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128, weight=weight[0], bias=bias[0])
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64, weight=weight[0], bias=bias[0])
    act2 = mx.sym.Activation(data=fc2, act_type="relu")
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10, weight=weight[0], bias=bias[0])
    return fc3


# Module 1: used as master module
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
shared = get_shared_symbol(data)
mlps = mx.sym.SoftmaxOutput(data=shared, name='softmax')
mlp_model = mx.mod.Module(symbol=mlps, context=mx.cpu())
mlp_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mlp_model.init_params()

# Module 2
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
shared = get_shared_symbol(data)
mlps = mx.sym.SoftmaxOutput(data=shared, name='softmax')
mlp_model2 = mx.mod.Module(symbol=mlps, context=mx.cpu())
mlp_model2.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
mlp_model2.init_params()

# Train module 1
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
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