import logging
import mxnet as mx

logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

mnist = mx.test_utils.get_mnist()

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# Shared symbol 
def get_shared_symbol(data, weight, bias):
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128, weight=weight[0], bias=bias[0])
    act1 = mx.sym.Activation(data=fc1, act_type="relu")
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64, weight=weight[1], bias=bias[1])
    act2 = mx.sym.Activation(data=fc2, act_type="relu")
    return act2

# Set up variables to share parameters for shared network
fc_weight = []
fc_bias = []
for i in range(2):
    fc_weight.append(mx.sym.Variable(str(i) + '_weight'))
    fc_bias.append(mx.sym.Variable(str(i) + '_bias'))

# Module 1 
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
shared = get_shared_symbol(data, fc_weight, fc_bias)
fc3 = mx.sym.FullyConnected(data=shared, num_hidden=10)
mlps = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
mlp_model1 = mx.mod.Module(symbol=mlps, context=mx.cpu())
mlp_model1.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# mlp_model1.init_params()

# Module 2
data = mx.sym.Variable('data')
data = mx.sym.flatten(data=data)
shared = get_shared_symbol(data, fc_weight, fc_bias)
fc3 = mx.sym.FullyConnected(data=shared, num_hidden=10)
mlps = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
mlp_model2 = mx.mod.Module(symbol=mlps, context=mx.cpu())
mlp_model2.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# mlp_model2.init_params()

# Train module 1
print("\n===Training module1===\n")
mlp_model1.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=5,
              allow_missing=True)  # train for at most 10 dataset passes


# Train module 2
# We expect the shared module to start where the first module finished
print("\n===Training module2===\n")
arg_param, aux_param = mlp_model1.get_params()
mlp_model2.fit(train_iter,  # train data
               eval_data=val_iter,  # validation data
               optimizer='sgd',  # use SGD to train
               optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
               eval_metric='acc',  # report accuracy during training
               batch_end_callback=mx.callback.Speedometer(batch_size, 100),
               num_epoch=5,
               arg_params=arg_param,
               allow_missing=True)  # train for at most 10 dataset passes


# Making sure that fit doesn't always overwrite parameters by returning to module 1
print("\n===Training module1===\n")
arg_param, aux_param = mlp_model2.get_params()
mlp_model1.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              num_epoch=5,
              arg_params=arg_param,
              allow_missing=True)  # train for at most 10 dataset passes
