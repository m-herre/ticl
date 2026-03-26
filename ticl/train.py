import time, wandb
from contextlib import nullcontext

import torch
from torch import nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import ticl.utils as utils
from ticl.utils import ExponentialLR, ReduceLROnSpike, init_dist

import pdb


def eval_criterion(criterion, targets, output, device, n_out):
    if isinstance(criterion, nn.GaussianNLLLoss):
        assert output.shape[-1] == 2, \
            'need to write a little bit of code to handle multiple regression targets at once'

        mean_pred = output[..., 0]
        var_pred = output[..., 1].abs()
        losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
        losses = criterion(output.flatten(), targets.to(device).flatten())
    elif isinstance(criterion, nn.CrossEntropyLoss):
        losses = criterion(output.reshape(-1, n_out)[:, :int(targets.max()) + 1], targets.to(device).long().flatten())
    else:
        losses = criterion(output, targets)
    losses = losses.view(*output.shape[0:2])
    return utils.torch_nanmean(losses.mean(0), return_nanshare=True)


def train_epoch(
    model, 
    aggregate_k_gradients, 
    using_dist, 
    scaler, 
    dl, 
    device, 
    optimizer, 
    criterion, 
    n_out, 
    progress_bar=False
):
    model.train()  # Turn on the train mode
    total_loss = torch.tensor(0., device = device)
    nan_steps = torch.tensor(0., device = device)
    ignore_steps = torch.tensor(0., device = device)
    train_eval_pos_sum = 0.0
    test_eval_pos_sum = 0.0
    steps_per_epoch = len(dl)
    assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
    if progress_bar:
        dl = tqdm(dl)
    for batch, (data, targets, single_eval_pos) in enumerate(dl):
        # change the description of the progress bar
        if progress_bar:
            dl.set_description(f'| train sample number: {single_eval_pos} | test sample number: {data[1].shape[0] - single_eval_pos}')
        train_eval_pos_sum += float(single_eval_pos)
        test_eval_pos_sum += float(data[1].shape[0] - single_eval_pos)

        if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
            cm = model.no_sync()
        else:
            cm = nullcontext()
        with cm:
            with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if scaler is not None else nullcontext():
                # for mothernet, la_mothernet, model is MLPModelPredictor from ticl.py
                output = model(
                    tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                    if isinstance(data, tuple) else data.to(device), 
                    single_eval_pos=single_eval_pos
                )

                if single_eval_pos is not None:
                    targets = targets[single_eval_pos:]
                loss, nan_share = eval_criterion(
                    criterion, 
                    targets, 
                    output, 
                    device=device, 
                    n_out=n_out
                )
                loss = loss / aggregate_k_gradients

            loss.backward()

            if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
                optimizer.step()
                optimizer.zero_grad()                

            if torch.isnan(loss):
                raise ValueError("NAN loss encountered")
            else:
                total_loss += loss.mean().cpu().detach().item()
            nan_steps += nan_share
            ignore_steps += (targets == -100).float().mean()
            
    return (total_loss / steps_per_epoch * aggregate_k_gradients,
            nan_steps.cpu().item() / steps_per_epoch,
            ignore_steps.cpu().item()/steps_per_epoch,
            {
                "batch_loss": total_loss / steps_per_epoch * aggregate_k_gradients,
                "train_train_sample_number": train_eval_pos_sum / steps_per_epoch,
                "train_test_sample_number": test_eval_pos_sum / steps_per_epoch,
            })


def train(dl, model, criterion, optimizer_state=None, scheduler=None,
          epochs=10, stop_after_epochs=None, learning_rate=None, min_lr=None, weight_decay=0.0, warmup_epochs=10,
          device='cuda:0',
          aggregate_k_gradients=1, verbose=True, epoch_callback=None, train_mixed_precision=False, adaptive_batch_size=False,
          learning_rate_schedule='cosine', lr_decay=0.99, adam_beta1=0.9, reduce_lr_on_spike=False,
          spike_tolerance=4, progress_bar=False,
          ):
    using_dist, rank, device = init_dist(device)

    # Disable flash SDP on non-A100/H100 GPUs — PyTorch's flash attention kernel
    # requires exactly SM80 or SM90, and fails on e.g. A40 (SM86).
    if device != 'cpu:0' and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability not in ((8, 0), (9, 0)):
            torch.backends.cuda.enable_flash_sdp(False)
            if rank == 0 and verbose:
                print(f"GPU SM{capability[0]}{capability[1]}: disabled flash SDP (requires SM80/SM90)")

    if rank == 0 and verbose:
        print(f'Using {device} device')

    model.to(device)
    criterion.to(device)

    n_out = model.n_out
    if using_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        if rank == 0:
            print("Distributed training")
    elif "cuda" in device:
        print(f"Single GPU training on {torch.cuda.get_device_name()}")
    elif "cpu" in device:
        pass
    else:
        raise ValueError(f"Invalid device: {device}")

    if rank == 0:
        model.learning_rates = getattr(model, 'learning_rates', [])
        model.losses = getattr(model, 'losses', [])
        model.wallclock_times = getattr(model, 'wallclock_times', [])
        model.start_time = time.time()
        if len(model.wallclock_times):
            model.start_time -= model.wallclock_times[-1]
        if epoch_callback is not None:
            epoch_callback(model, None, None, "start")

    dl.model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(adam_beta1, 0.999))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    spike_scheduler = None
    if scheduler is None:
        if learning_rate_schedule == 'cosine':
            base_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
        elif learning_rate_schedule == 'exponential':
            base_scheduler = ExponentialLR(optimizer, gamma=lr_decay, min_lr=min_lr)
        elif learning_rate_schedule == 'constant':
            base_scheduler = ExponentialLR(optimizer, gamma=1, min_lr=min_lr)
        else:
            raise ValueError(f"Invalid learning rate schedule: {learning_rate_schedule}")
        # add linear warmup to scheduler
        scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=1e-10, end_factor=1, total_iters=warmup_epochs),
                                             base_scheduler], milestones=[warmup_epochs])

        start_epoch = 1
    else:
        start_epoch = scheduler.last_epoch + 1

    if reduce_lr_on_spike:
        # In this case we're not properly restarting the scheduler when we load a checkpoint, sad
        spike_scheduler = ReduceLROnSpike(optimizer, smoothing=10, factor=0.5, min_lr=min_lr, tolerance=spike_tolerance, verbose=True)
    scaler = GradScaler() if train_mixed_precision and device != "cpu" else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    total_loss = float('inf')
    increased_batch_size = 0
    epoch = start_epoch
    if stop_after_epochs is not None:
        epochs = min(epochs, stop_after_epochs)
    if "cuda" in device:
        gpu_start_time = torch.cuda.Event(enable_timing=True)
        gpu_end_time = torch.cuda.Event(enable_timing=True)

    profile_model = model.module if hasattr(model, "module") else model
    grande_diagnostics_enabled = (
        rank == 0
        and getattr(profile_model, "child_model", None) == "grande"
        and getattr(profile_model, "grande_diagnostics", False)
    )
    grande_diagnostics_snapshot = None
    if grande_diagnostics_enabled:
        from ticl.grande_diagnostics import (
            prepare_grande_diagnostic_snapshot,
            run_grande_diagnostics,
        )

        grande_diagnostics_snapshot = prepare_grande_diagnostic_snapshot(
            dl,
            seed=getattr(profile_model, "grande_diagnostics_seed", 0),
        )
        if wandb.run:
            init_metrics = run_grande_diagnostics(
                model=profile_model,
                snapshot=grande_diagnostics_snapshot,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                n_out=n_out,
                epoch=max(start_epoch - 1, 0),
                level=getattr(profile_model, "grande_diagnostics_level", "scalars_small_hists"),
                hist_max_points=getattr(profile_model, "grande_diagnostics_hist_max_points", 2048),
                base_seed=getattr(profile_model, "grande_diagnostics_seed", 0),
                collect_gradients=getattr(profile_model, "grande_diagnostics_gradients", False),
            )
            if init_metrics:
                wandb.log(init_metrics, step=max(start_epoch - 1, 0))

    try:
        train_time, inference_time, train_gpu_time = [], [], []
        for epoch in range(start_epoch, epochs + 1):
            if verbose:
                print(f"start of epoch {epoch}")
            if hasattr(profile_model, "reset_grande_profile"):
                profile_model.reset_grande_profile()

            epoch_start_time = time.time()
            if "cuda" in device:
                gpu_start_time.record()
            
            new_loss, nan_share, ignore_share, train_metrics = train_epoch(
                model, 
                aggregate_k_gradients, 
                using_dist, 
                scaler, 
                dl, 
                device, 
                optimizer, 
                criterion, 
                n_out,
                progress_bar=progress_bar,
            )

            total_loss = new_loss
            if spike_scheduler is not None:
                last_lr = spike_scheduler.get_last_lr()[0]
            else:
                last_lr = scheduler.get_last_lr()[0]

            train_time.append(time.time() - epoch_start_time)
            if "cuda" in device:
                gpu_end_time.record()
                torch.cuda.synchronize()
                train_gpu_time.append(gpu_start_time.elapsed_time(gpu_end_time)/1000)
            else:
                train_gpu_time.append(0)

            grande_profile_stats = {}
            if hasattr(profile_model, "get_grande_profile_stats"):
                grande_profile_stats = profile_model.get_grande_profile_stats(reset=True)
            wandb_metrics = {
                "avg_train_time": sum(train_time) / len(train_time),
                "train_time": train_time[-1],
                "avg_train_gpu_time": sum(train_gpu_time) / len(train_gpu_time),
                "train_gpu_time": train_gpu_time[-1],
                "avg_batch_loss": float(train_metrics["batch_loss"]),
                "train_train_sample_number": train_metrics["train_train_sample_number"],
                "train_test_sample_number": train_metrics["train_test_sample_number"],
            }
            wandb_metrics.update(grande_profile_stats)

            if grande_diagnostics_enabled and wandb.run:
                diagnostics_metrics = run_grande_diagnostics(
                    model=profile_model,
                    snapshot=grande_diagnostics_snapshot,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    n_out=n_out,
                    epoch=epoch,
                    level=getattr(profile_model, "grande_diagnostics_level", "scalars_small_hists"),
                    hist_max_points=getattr(profile_model, "grande_diagnostics_hist_max_points", 2048),
                    base_seed=getattr(profile_model, "grande_diagnostics_seed", 0),
                    collect_gradients=getattr(profile_model, "grande_diagnostics_gradients", False),
                )
                wandb_metrics.update(diagnostics_metrics)

            if wandb.run:
                wandb.log(wandb_metrics, step=epoch)

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | Wallclock time: {train_time[-1]:5.2f}s | GPU time: {train_gpu_time[-1]:5.2f}s | mean loss {total_loss:5.4f} | ')

                if grande_profile_stats:
                    print(
                        " GRANDE profile "
                        + " ".join(
                            f"{name}={value:0.4f}s"
                            for name, value in grande_profile_stats.items()
                            if name.endswith("_s")
                        )
                    )

                print(
                    f' lr {last_lr}'
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}')
                print('-' * 89)
                
            if new_loss > 1.5 * total_loss:
                print("LOSS DIVERGED")
                return total_loss, model.to('cpu'), dl, epoch
            
            if adaptive_batch_size:
                if increased_batch_size == 0 and epoch >= 20:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 1
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 1 and epoch >= 50:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 2
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 2 and epoch >= 200:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 3
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                elif increased_batch_size == 3 and total_loss >= 1000:
                    aggregate_k_gradients *= 2
                    increased_batch_size = 4
                    print("increased aggregate_k_gradients size to", aggregate_k_gradients)
                    
            scheduler.step()
            if spike_scheduler is not None:
                spike_scheduler.step(metrics=total_loss)
            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                model.learning_rates.append(last_lr)
                model.losses.append(total_loss)
                model.wallclock_times.append(time.time() - model.start_time)
                output = epoch_callback(model, optimizer, scheduler, epoch)
                if output: 
                    inference_time.append(output)
                    if wandb.run:
                        wandb.log(
                            {
                                "avg_inference_time": sum(inference_time)/len(inference_time),
                                "inference_time": inference_time[-1],
                            },
                            step=epoch,
                        )


    except KeyboardInterrupt:
        pass

    if rank == 0:  # trivially true for non-parallel training
        return total_loss, model.to('cpu'), dl, epoch
