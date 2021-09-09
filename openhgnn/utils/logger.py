
def printInfo(metric, epoch, train_score, train_loss, val_score, val_loss):
    if metric == 'f1_lr':
        print((
            f"Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, Train_macro_f1: {train_score[0]:.4f}, Train_micro_f1: {train_score[1]:.4f}, "
            f"Val_macro_f1: {val_score[0]:.4f}, Val_micro_f1: {val_score[1]:.4f}, ValLoss:{val_loss: .4f}"
        ))
    # use acc
    elif metric == 'acc':
        print((
            f"Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, Train_acc: {train_score:.4f},  "
            f"Val_acc: {val_score:.4f}, ValLoss:{val_loss: .4f}"
        ))
    elif metric == 'acc-ogbn-mag':
        print((
            f"Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, Train_acc: {train_score:.4f},  "
            f"Val_acc: {val_score:.4f}, ValLoss:{val_loss: .4f}"
        ))
    else:
        print((
            f"Epoch: {epoch:03d}, Train_loss: {train_loss:.4f}, Train_macro_f1: {train_score[0]:.4f}, Train_micro_f1: {train_score[1]:.4f}, "
            f"Val_macro_f1: {val_score[0]:.4f}, Val_micro_f1: {val_score[1]:.4f}, ValLoss:{val_loss: .4f}"
        ))


def printMetric(metric, score, mode):
    if isinstance(score, tuple):
        print(f"{mode}_macro_{metric} = {score[0]:.4f}, {mode}_micro_{metric}: {score[1]:.4f}")
    else:
        print(f"{mode}_{metric} = {score:.4f}")
