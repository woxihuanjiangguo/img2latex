class Logger:
    def __init__(self):
        pass

    def log_train(self, epoch, train_loss, val_loss, time, train_err, train_acc, val_err, val_acc):
        print(f'epoch: {epoch} time: {time / 60} minutes')
        print(f'train_loss: {train_loss} train_err: {train_err} train_acc: {train_acc}')
        print(f'val_loss: {val_loss} val_err: {val_err} val_acc: {val_acc}')

    def log_test_metrics(self, loss, err, acc):
        print(f'loss: {loss} err: {err} acc: {acc}')
