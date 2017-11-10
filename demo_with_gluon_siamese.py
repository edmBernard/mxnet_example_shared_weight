import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

def main():
    ctx = mx.gpu()

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

    net_siamese = gluon.nn.Sequential()
    with net_siamese.name_scope():
        net_siamese.add(gluon.nn.Dense(256, activation='relu'))
        net_siamese.add(gluon.nn.Dense(128, activation='relu'))

    net_out = gluon.nn.Sequential()
    with net_out.name_scope():
        net_out.add(gluon.nn.Dense(128, activation='relu'))
        net_out.add(gluon.nn.Dense(64, activation='relu'))
        net_out.add(gluon.nn.Dense(num_outputs))


    net_siamese.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    net_out.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer_siamese = gluon.Trainer(net_siamese.collect_params(), 'sgd', {'learning_rate': 0.05})
    trainer_out = gluon.Trainer(net_out.collect_params(), 'sgd', {'learning_rate': 0.05})

    def evaluate_accuracy(data_iterator1, data_iterator2, net):
        acc = mx.metric.Accuracy()
        for i, ((data1, label1), (data2, label2)) in enumerate(zip(data_iterator1, data_iterator2)):
            data1 = data1.as_in_context(ctx).reshape((-1, 784))
            data2 = data2.as_in_context(ctx).reshape((-1, 784))
            label1 = label1.as_in_context(ctx)
            inter1 = net_siamese(data1)
            inter2 = net_siamese(data2)
            output = net_out(nd.concat(inter1, inter2))
            acc.update([label1], [output])
        return acc.get()

    epochs = 4
    moving_loss = 0.
    smoothing_constant = .01
    metric = mx.metric.Accuracy()

    print("\n#### Shared+Module1 Training ####")
    for e in range(epochs):
        metric.reset()
        # Train Branch with mod1 on dataset 1 
        for i, ((data1, label1), (data2, label2)) in enumerate(zip(train_data1, train_data2)):
            data1 = data1.as_in_context(ctx).reshape((-1, 784))
            data2 = data2.as_in_context(ctx).reshape((-1, 784))
            label1 = label1.as_in_context(ctx)
            with autograd.record():
                inter1 = net_siamese(data1)
                inter2 = net_siamese(data2)
                output = net_out(nd.concat(inter1, inter2))
                loss = softmax_cross_entropy(output, label1)
                loss.backward()
            trainer_siamese.step(batch_size)
            trainer_out.step(batch_size)

            metric.update([label1], [output])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

            if i % 100 == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Loss: %s Training: %s=%f'%(e, i, moving_loss, name, acc))

        _, train_accuracy = metric.get()
        _, test_accuracy = evaluate_accuracy(test_data1, test_data2, lambda x, y: net_out(nd.concat(net_siamese(x), net_siamese(y))))
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s\n" % (e, moving_loss, train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
