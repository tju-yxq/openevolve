"""Pure scheduling helpers for data-epoch-based multi-fidelity training."""


def steps_per_data_epoch(dataset_size, batch_size, drop_last=True):
    if dataset_size <= 0 or batch_size <= 0:
        raise ValueError("dataset_size and batch_size must be positive")
    if drop_last:
        steps = dataset_size // batch_size
    else:
        steps = (dataset_size + batch_size - 1) // batch_size
    if steps <= 0:
        raise ValueError("dataset is smaller than one batch")
    return steps


def epoch_validation_interval_steps(steps_per_epoch, interval_epochs):
    if steps_per_epoch <= 0 or interval_epochs <= 0:
        raise ValueError("epoch validation inputs must be positive")
    return steps_per_epoch * interval_epochs


def next_epoch_validation_step(global_step, origin_step, steps_per_epoch, interval_epochs):
    if global_step < origin_step:
        raise ValueError("global_step cannot precede the data phase origin")
    interval = epoch_validation_interval_steps(steps_per_epoch, interval_epochs)
    relative_step = global_step - origin_step
    return origin_step + ((relative_step // interval) + 1) * interval


def is_epoch_validation_step(global_step, origin_step, steps_per_epoch, interval_epochs):
    if global_step < origin_step:
        return False
    interval = epoch_validation_interval_steps(steps_per_epoch, interval_epochs)
    return (global_step - origin_step) % interval == 0