import datetime
import os

import mxnet
from mxnet import gluon, nd, autograd
from tqdm import tqdm


def test(metric, ctx, net, val_data, num_views=1, num_class=None):
    assert num_views >= 1, "'num_views' should be greater or equal to 1"
    metric.reset()

    print('evaluating...')
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    for data, label, *rest in tqdm(val_data):
        if data.shape[0] == 1:
            Xs = [data.as_in_context(ctx[0])]
            Ys = [label.as_in_context(ctx[0])]
        else:
            Xs = gluon.utils.split_and_load(data,
                                            ctx_list=ctx, batch_axis=0, even_split=False)
            Ys = gluon.utils.split_and_load(label,
                                            ctx_list=ctx, batch_axis=0, even_split=False)

        if num_views > 1:
            outputs = [net(X).reshape(-1, num_views, num_class).mean(axis=1) for X in Xs]
        else:
            outputs = [net(X) for X in Xs]
        metric.update(Ys, outputs)
    return metric.get()


def get_format_time_string(time_interval):
    h, remainder = divmod((time_interval).seconds, 3600)
    m, s = divmod(remainder, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def log_string(log_out, out_str):
    log_out.write(out_str + '\n')
    log_out.flush()


def smooth(label, classes, eta=0.1):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx=label.context)
    res += eta / classes
    res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1 - eta + eta / classes
    return res


def save_checkpoint(net, current_epoch, checkpoint_prefix):
    net.save_parameters(os.path.join(checkpoint_prefix, 'Epoch%s.params' % current_epoch))


def train(net, train_data, test_data, loss_fun, kvstore, log_out, checkpoint_prefix, train_args):
    trainer_dict = {'learning_rate': train_args.lr, 'wd': train_args.wd}
    if train_args.optimizer == 'sgd':
        trainer_dict['momentum'] = 0.9
    trainer = gluon.Trainer(net.collect_params(), train_args.optimizer, trainer_dict, kvstore=kvstore)
    best_test_acc = 0
    metric = mxnet.metric.Accuracy()

    ctx = [mxnet.gpu(gpu_id) for gpu_id in train_args.gpu]

    log_string(log_out, str(datetime.datetime.now()))
    log_string(log_out, str(train_args))
    print('start training on %s' % train_args.dataset_path)

    for epoch in range(train_args.from_epoch, train_args.epoch):
        train_loss = 0.0
        metric.reset()
        tic = datetime.datetime.now()

        if epoch > 0 and epoch % train_args.decay_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * train_args.decay_rate)
        iter_time = 1

        for data, label, sample_weights in tqdm(train_data):

            if train_args.label_smoothing:
                label_smooth = smooth(label, train_args.num_classes)
            else:
                label_smooth = label

            gpu_data = gluon.utils.split_and_load(data, ctx, even_split=False)
            gpu_label = gluon.utils.split_and_load(label_smooth.astype('float32'), ctx, even_split=False)
            gpu_weights = gluon.utils.split_and_load(sample_weights.astype('float32'), ctx, even_split=False)

            outputs = []
            Ls = []
            with autograd.record():

                for x, y, weight in zip(gpu_data, gpu_label, gpu_weights):
                    out = net(x)
                    if train_args.use_sample_weights:
                        loss = loss_fun(out, nd.repeat(y, repeats=train_args.num_views, axis=0),
                                        nd.repeat(weight, repeats=train_args.num_views, axis=0))
                    else:
                        loss = loss_fun(out, nd.repeat(y, repeats=train_args.num_views, axis=0))
                    outputs.append(out)
                    Ls.append(loss)
            autograd.backward(Ls)

            if iter_time % train_args.batch_update_period == 0:
                trainer.step(train_args.batch_size * train_args.batch_update_period)
                net.collect_params().zero_grad()
            elif iter_time == len(train_data):
                trainer.step(train_args.batch_size * (len(train_data) % train_args.batch_update_period))
                net.collect_params().zero_grad()
            if train_args.label_smoothing:
                origin_label = gluon.utils.split_and_load(label.astype('float32'), ctx, even_split=False)
            else:
                origin_label = gpu_label
            avg_output = [output.reshape((-1, train_args.num_views, train_args.num_classes)).mean(axis=1) for output
                          in outputs]
            metric.update(preds=avg_output, labels=origin_label)
            train_loss += nd.sum(loss).asscalar()
            iter_time += 1

        _, train_acc = metric.get()
        _, test_acc = test(metric, ctx, net, test_data, num_views=train_args.num_views,
                           num_class=train_args.num_classes)

        save_checkpoint(net, '_latest', checkpoint_prefix)
        if test_acc > best_test_acc:
            save_checkpoint(net, '_best', checkpoint_prefix)
            best_test_acc = test_acc
        toc = datetime.datetime.now()
        time_cost = get_format_time_string(toc - tic)
        epoch_str = "Epoch %d. Loss: %f, Train acc: %f, Valid acc: %f, Best acc: %f, lr: %f, Time: %s" % (
            epoch, train_loss / len(train_data) / train_args.batch_size,
            train_acc, test_acc, best_test_acc, trainer.learning_rate, time_cost)
        print(epoch_str)
        log_string(log_out, epoch_str)
