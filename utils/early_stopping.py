def early_stopping(epoch, value, best_value, stopping_step, stopping_step_cnt=10):
    # early stopping strategy:

    if value >= best_value:
        stopping_step = 0
        best_value = value
        best_epoch = epoch
    else:
        stopping_step += 1

    if stopping_step >= stopping_step_cnt:
        should_stop = True
    else:
        should_stop = False
    return best_epoch, best_value, stopping_step, should_stop
