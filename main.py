# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
import logging
import os

import click
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy
from torch.nn import CrossEntropyLoss

from quantization.quant_param_utils import separate_quantized_model_params
from quantization.quant_utils import (
    pass_data_for_range_estimation,
    set_range_estimators,
)
from utils import CosineTempDecay, DotDict
from utils.click_options import (
    base_options,
    get_prune_kwargs,
    multi_optimizer_options,
    pruning_options,
    qat_options,
    quant_params_dict,
    quantization_options,
)
from utils.optimizer_utils import optimizer_lr_factory
from utils.oscillation_tracking_utils import add_oscillation_trackers
from utils.qat_utils import (
    CompositeLoss,
    DampeningLoss,
    MethodPropagator,
    ReestimateBNStats,
    UpdateDampeningLossWeighting,
    UpdateFreezingThreshold,
    get_dataloaders_and_model,
)
from utils.supervised_driver import (
    create_trainer_engine,
    log_metrics,
    setup_tensorboard_logger,
)


# setup stuff
class Config(DotDict):
    pass


@click.group()
def oscillations():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


pass_config = click.make_pass_decorator(Config, ensure=True)


@oscillations.command()
@pass_config
@base_options
@multi_optimizer_options()
@quantization_options
@pruning_options
@qat_options
def train_compressed(config):
    """
    Main QAT function
    """

    print("Setting up network and data loaders")
    qparams = quant_params_dict(config)

    dataloaders, model = get_dataloaders_and_model(config, **qparams)
    del model
    prune_kwargs = get_prune_kwargs(config, dataloaders.train_loader)
    qparams.update(prune_kwargs)
    config.prune_kwargs = prune_kwargs
    qparams.update(prune_kwargs)

    dataloaders, model = get_dataloaders_and_model(config, **qparams)

    # Estimate ranges using training data
    pass_data_for_range_estimation(
        loader=dataloaders.train_loader,
        model=model,
        act_quant=config.quant.act_quant,
        weight_quant=config.quant.weight_quant,
        max_num_batches=config.quant.num_est_batches,
    )

    # Put quantizers in desirable state
    set_range_estimators(config, model)

    print("Loaded model:\n{}".format(model))

    # Get all models parameters in  subcategories
    quantizer_params, model_params, grad_params = separate_quantized_model_params(model)
    model_optimizer, quant_optimizer = None, None
    if config.qat.sep_quant_optimizer:
        # Separate optimizer for model and quantization parameters
        model_optimizer, model_lr_scheduler = optimizer_lr_factory(
            config.optimizer, model_params, config.base.max_epochs
        )
        quant_optimizer, quant_lr_scheduler = optimizer_lr_factory(
            config.quant_optimizer, quantizer_params, config.base.max_epochs
        )

        optimizer = MethodPropagator([model_optimizer, quant_optimizer])
        lr_schedulers = [s for s in [model_lr_scheduler, quant_lr_scheduler] if s is not None]
        lr_scheduler = MethodPropagator(lr_schedulers) if len(lr_schedulers) else None
    else:
        optimizer, lr_scheduler = optimizer_lr_factory(
            config.optimizer, quantizer_params + model_params, config.base.max_epochs
        )

    print("Optimizer:\n{}".format(optimizer))
    print(f"LR scheduler\n{lr_scheduler}")

    # Define metrics for ingite engine
    metrics = {"top_1_accuracy": Accuracy(), "top_5_accuracy": TopKCategoricalAccuracy()}

    # Set-up losses
    task_loss_fn = CrossEntropyLoss()
    dampening_loss = None
    if config.osc_damp.weight is not None:
        # Add dampening loss to task loss
        dampening_loss = DampeningLoss(model, config.osc_damp.weight, config.osc_damp.aggregation)
        loss_dict = {"task_loss": task_loss_fn, "dampening_loss": dampening_loss}
        loss_func = CompositeLoss(loss_dict)
        loss_metrics = {
            "task_loss": Loss(task_loss_fn),
            "dampening_loss": Loss(dampening_loss),
            "loss": Loss(loss_func),
        }
    else:
        loss_func = task_loss_fn
        loss_metrics = {"loss": Loss(loss_func)}

    metrics.update(loss_metrics)

    # Set up ignite trainer and evaluator
    trainer, evaluator = create_trainer_engine(
        model=model,
        optimizer=optimizer,
        criterion=loss_func,
        data_loaders=dataloaders,
        metrics=metrics,
        lr_scheduler=lr_scheduler,
        save_checkpoint_dir=config.base.save_checkpoint_dir,
        device="cuda" if config.base.cuda else "cpu",
    )

    if config.base.progress_bar:
        pbar = ProgressBar()
        pbar.attach(trainer)
        pbar.attach(evaluator)

    # Create TensorboardLogger
    if config.base.tb_logging_dir:
        if config.qat.sep_quant_optimizer:
            optimizers_dict = {"model": model_optimizer, "quant_params": quant_optimizer}
        else:
            optimizers_dict = optimizer
        tb_logger = setup_tensorboard_logger(
            trainer, evaluator, config.base.tb_logging_dir, optimizers_dict
        )

    if config.osc_damp.weight_final:
        # Apply cosine annealing of dampening loss
        total_iterations = len(dataloaders.train_loader) * config.base.max_epochs
        annealing_schedule = CosineTempDecay(
            t_max=total_iterations,
            temp_range=(config.osc_damp.weight, config.osc_damp.weight_final),
            rel_decay_start=config.osc_damp.anneal_start,
        )
        print(f"Weight gradient parameter cosine annealing schedule:\n{annealing_schedule}")
        trainer.add_event_handler(
            Events.ITERATION_STARTED,
            UpdateDampeningLossWeighting(dampening_loss, annealing_schedule),
        )

    # Evaluate model
    print("Running evaluation before training")
    evaluator.run(dataloaders.val_loader)
    log_metrics(evaluator.state.metrics, "Evaluation", trainer.state.epoch)

    # BN Re-estimation
    if config.qat.reestimate_bn_stats:
        evaluator.add_event_handler(
            Events.EPOCH_STARTED, ReestimateBNStats(model, dataloaders.train_loader)
        )

    # Add oscillation trackers to the model and set up oscillation freezing
    if config.osc_freeze.threshold:
        oscillation_tracker_dict = add_oscillation_trackers(
            model,
            max_bits=config.osc_freeze.max_bits,
            momentum=config.osc_freeze.ema_momentum,
            freeze_threshold=config.osc_freeze.threshold,
            use_ema_x_int=config.osc_freeze.use_ema,
        )

        if config.osc_freeze.threshold_final:
            # Apply cosine annealing schedule to the freezing threshdold
            total_iterations = len(dataloaders.train_loader) * config.base.max_epochs
            annealing_schedule = CosineTempDecay(
                t_max=total_iterations,
                temp_range=(config.osc_freeze.threshold, config.osc_freeze.threshold_final),
                rel_decay_start=config.osc_freeze.anneal_start,
            )
            print(f"Oscillation freezing annealing schedule:\n{annealing_schedule}")
            trainer.add_event_handler(
                Events.ITERATION_STARTED,
                UpdateFreezingThreshold(oscillation_tracker_dict, annealing_schedule),
            )

    print("Starting training")

    trainer.run(dataloaders.train_loader, max_epochs=config.base.max_epochs)

    print("Finished training")


if __name__ == "__main__":
    oscillations()
