import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

ctx = mx.cpu()

batch_size = 100
num_inputs = 784
num_outputs = 10

# Get MNIST Data
def transform(data, label):
    return data.astype(np.float32)/255., label.astype(np.float32)

train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

net_shared = gluon.nn.Sequential()
with net_shared.name_scope():
    net_shared.add(gluon.nn.Dense(128, activation='relu'))
    net_shared.add(gluon.nn.Dense(64, activation='relu'))
    net_shared.add(gluon.nn.Dense(num_outputs, activation='relu'))

net_shared.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer_shared = gluon.Trainer(net_shared.collect_params(), 'sgd', {'learning_rate': 0.005})

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

def evaluate_accurac_v2(datas, labels, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(zip(datas, labels)):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

epochs = 4
moving_loss = 0.
smoothing_constant = .01

for e in range(epochs):
    # Train Branch with mod1 on dataset 1 
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net_shared(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer_shared.step(batch_size)

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        # print("Epoch %s. Batch [%s]. Loss: %-9f, Train_acc: %s " % (e, i*batch_size, curr_loss, evaluate_accurac_v2(data, label, net_shared)))

    train_accuracy = evaluate_accuracy(train_data, net_shared)
    test_accuracy = evaluate_accuracy(test_data, net_shared)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
