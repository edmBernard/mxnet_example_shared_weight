import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

def main():
    ctx = mx.cpu()

    batch_size = 100
    num_inputs = 784
    num_outputs = 10

    # Get MNIST Data
    def transform(data, label):
        return data.astype(np.float32)/255, label.astype(np.float32)
    train_data1 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
    test_data1 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)
    train_data2 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
    test_data2 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)

    net_shared = gluon.nn.Sequential()
    with net_shared.name_scope():
        net_shared.add(gluon.nn.Dense(128, activation='relu'))
        net_shared.add(gluon.nn.Dense(64, activation='relu'))

    net_mod1 = gluon.nn.Sequential()
    with net_mod1.name_scope():
        net_mod1.add(gluon.nn.Dense(num_outputs))

    net_mod2 = gluon.nn.Sequential()
    with net_mod2.name_scope():
        net_mod2.add(gluon.nn.Dense(num_outputs))

    net_shared.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    net_mod1.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    net_mod2.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)

    softmax_cross_entropy_1 = gluon.loss.SoftmaxCrossEntropyLoss()
    softmax_cross_entropy_2 = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer_shared = gluon.Trainer(net_shared.collect_params(), 'sgd', {'learning_rate': 0.05})
    trainer_mod1 = gluon.Trainer(net_mod1.collect_params(), 'sgd', {'learning_rate': 0.05})
    trainer_mod2 = gluon.Trainer(net_mod2.collect_params(), 'sgd', {'learning_rate': 0.05})

    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
            output = net(data)
            acc.update([label], [output])
        return acc.get()

    epochs = 4
    moving_loss = 0.
    smoothing_constant = .01
    metric = mx.metric.Accuracy()

    print("#### Before Training ####")
    _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_mod1(net_shared(x)))
    _, train_accuracy = evaluate_accuracy(train_data1, lambda x: net_mod1(net_shared(x)))
    print("Mod1: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    _, test_accuracy = evaluate_accuracy(test_data2, lambda x: net_mod2(net_shared(x)))
    _, train_accuracy = evaluate_accuracy(train_data2, lambda x: net_mod2(net_shared(x)))
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))

    print("\n#### Shared+Module1 Training ####")
    for e in range(epochs):
        metric.reset()
        # Train Branch with mod1 on dataset 1 
        for i, (data, label) in enumerate(train_data1):
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net_mod1(net_shared(data))
                loss = softmax_cross_entropy_1(output, label)
                loss.backward()
            trainer_shared.step(batch_size)
            trainer_mod1.step(batch_size)

            metric.update([label], [output])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Loss: %s Training: %s=%f'%(e, i, moving_loss, name, acc))

        _, train_accuracy = metric.get()
        _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_mod1(net_shared(x)))
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s\n" % (e, moving_loss, train_accuracy, test_accuracy))

    # We expect the shared module to start where the first module finished
    # There will be a small accuracy decrease since one layer was not trained
    _, test_accuracy = evaluate_accuracy(test_data2, lambda x: net_mod2(net_shared(x)))
    _, train_accuracy = evaluate_accuracy(train_data2, lambda x: net_mod2(net_shared(x)))
    print("\n#### Shared+Module2 Result after Mod1 Training ####")
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    print("\n#### Shared+Module2 Training ####")
    for e in range(epochs):
        metric.reset()
        # Train Branch with mod2 on dataset 2 
        for i, (data, label) in enumerate(train_data2):
            data = data.as_in_context(ctx).reshape((-1,784))
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net_mod2(net_shared(data))
                loss = softmax_cross_entropy_2(output, label)
                loss.backward()
            trainer_shared.step(batch_size)
            trainer_mod2.step(batch_size)

            metric.update([label], [output])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Loss: %s Training: %s=%f'%(e, i, moving_loss, name, acc))

        _, train_accuracy = metric.get()
        _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_mod2(net_shared(x)))
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s\n" % (e, moving_loss, train_accuracy, test_accuracy))

    print("\n#### After Training ####")
    _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_mod1(net_shared(x)))
    _, train_accuracy = evaluate_accuracy(train_data1, lambda x: net_mod1(net_shared(x)))
    print("Mod1: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    _, test_accuracy = evaluate_accuracy(test_data2, lambda x: net_mod2(net_shared(x)))
    _, train_accuracy = evaluate_accuracy(train_data2, lambda x: net_mod2(net_shared(x)))
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()