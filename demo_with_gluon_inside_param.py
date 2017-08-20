import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np

@profile
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


    class Model(gluon.Block):
        def __init__(self, num_outputs, **kwargs):
            super(Model, self).__init__(**kwargs)
            # use name_scope to give child Blocks appropriate names.
            # It also allows sharing Parameters between Blocks recursively.
            with self.name_scope():
                self.dense0 = gluon.nn.Dense(128)
                self.dense1 = gluon.nn.Dense(64)
                self.dense3_1 = gluon.nn.Dense(num_outputs)
                self.dense3_2 = gluon.nn.Dense(num_outputs)

        def forward(self, x, use_branch1):
            root = nd.F.relu(self.dense1(nd.F.relu(self.dense0(x))))
            if use_branch1:
                out = self.dense3_1(root)
            else:
                out = self.dense3_2(root)
            return out

    model = Model()
    model.initialize(mx.init.Uniform(scale=0.1), ctx=ctx)

    softmax_cross_entropy_1 = gluon.loss.SoftmaxCrossEntropyLoss()
    softmax_cross_entropy_2 = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(model, 'sgd', {'learning_rate': 0.05})

    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(ctx).reshape((-1,784))
            label = label.as_in_context(ctx)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    epochs = 4
    moving_loss = 0.
    smoothing_constant = .01

    print("#### Before Training ####")
    test_accuracy = evaluate_accuracy(test_data1, lambda x: model(x, True))
    train_accuracy = evaluate_accuracy(train_data1, lambda x :model(x, True))
    print("Mod1: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    test_accuracy = evaluate_accuracy(test_data2, lambda x: model(x, False))
    train_accuracy = evaluate_accuracy(train_data2, lambda x :model(x, False))
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))

    print("\n#### Shared+Module1 Training ####")
    for e in range(epochs):
        # Train Branch with mod1 on dataset 1 
        for i, (data, label) in enumerate(train_data1):
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data, True)
                loss = softmax_cross_entropy_1(output, label)
                loss.backward()
            trainer.step(batch_size)

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
        test_accuracy = evaluate_accuracy(test_data1, lambda x: model(x, True))
        train_accuracy = evaluate_accuracy(train_data1, lambda x :model(x, True))
        print("Mod1: Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    # We expect the shared module to start where the first module finished
    # There will be a small accuracy decrease since one layer was not trained
    test_accuracy = evaluate_accuracy(test_data2, lambda x: model(x, False))
    train_accuracy = evaluate_accuracy(train_data2, lambda x :model(x, False))
    print("\n#### Shared+Module2 Result after Mod1 Training ####")
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    print("\n#### Shared+Module2 Training ####")
    for e in range(epochs):
        # Train Branch with mod2 on dataset 2 
        for i, (data, label) in enumerate(train_data2):
            data = data.as_in_context(ctx).reshape((-1,784))
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data, False)
                loss = softmax_cross_entropy_1(output, label)
                loss.backward()
            trainer.step(batch_size)

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                        else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        test_accuracy = evaluate_accuracy(test_data2, lambda x: model(x, False))
        train_accuracy = evaluate_accuracy(train_data2, lambda x: model(x, False))
        print("Mod2: Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))

    print("\n#### After Training ####")
    test_accuracy = evaluate_accuracy(test_data1, lambda x: model(x, True))
    train_accuracy = evaluate_accuracy(train_data1, lambda x :model(x, True))
    print("Mod1: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    test_accuracy = evaluate_accuracy(test_data2, lambda x: model(x, False))
    train_accuracy = evaluate_accuracy(train_data2, lambda x :model(x, False))
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()