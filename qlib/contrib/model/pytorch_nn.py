# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
from collections import defaultdict

import os
import gc
import numpy as np
import pandas as pd
from typing import Callable, Optional, Text, Union, Dict, List
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.weight import Reweighter
from ...utils import (
    auto_filter_kwargs,
    init_instance_by_config,
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    get_or_create_path,
)
from ...log import get_module_logger
from ...workflow import R
from qlib.contrib.meta.data_selection.utils import ICLoss
from torch.nn import DataParallel

# Custom Dataset for PyTorch DataLoader


class PyTorchDataset(Dataset):
    """
    Custom PyTorch Dataset to wrap features, labels, and weights.
    Assumes data is already loaded into numpy arrays or tensors.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, weights: np.ndarray, index: Optional[pd.Index] = None):
        """
        Initialize the dataset.

        Parameters
        ----------
        features : np.ndarray
            Numpy array of features.
        labels : np.ndarray
            Numpy array of labels.
        weights : np.ndarray
            Numpy array of sample weights.
        index : pd.Index, optional
            Pandas index corresponding to the data, by default None.
            Useful if metrics require original index information.
        """
        assert len(features) == len(labels) == len(weights)
        # Convert numpy arrays to torch tensors
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.weights = torch.from_numpy(weights).float()
        self.index = index  # Store index if provided

    def __len__(self):
        """Return the total number of samples."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve a sample (features, label, weight) at the given index.
        Optionally includes the original index if available.
        """
        # Here we keep the index for ICLoss calculation, but it's not used in the DataLoader directly
        # Return data tensors for the given index
        return self.features[idx], self.labels[idx], self.weights[idx]


class DNNModelPytorch(Model):
    """DNN Model

    Parameters
    ----------
    lr : float
        Learning rate.
    max_steps : int
        Maximum number of training steps (epochs).
    batch_size : int
        Number of samples per batch.
    early_stop_rounds : int
        Stop training if validation loss doesn't improve for this many evaluation steps.
    eval_steps : int
        Evaluate the model on the validation set every `eval_steps` training steps.
    optimizer : str
        Optimizer name (e.g., 'adam', 'gd' for SGD).
    loss : str
        Loss function type ('mse' or 'binary').
    GPU : int or str
        GPU ID to use (-1 for CPU, or device string like 'cuda:0').
    seed : int, optional
        Random seed for reproducibility.
    weight_decay : float
        Weight decay (L2 regularization) factor.
    data_parall : bool
        Whether to use nn.DataParallel for multi-GPU training.
        Consider DistributedDataParallel for better performance.
    scheduler : Optional[Union[Callable]], optional
        Learning rate scheduler configuration. 'default' uses ReduceLROnPlateau,
        None uses no scheduler, a Callable receives the optimizer.
    init_model : nn.Module, optional
        Use this pre-initialized model instead of creating one.
    eval_train_metric : bool
        Whether to compute evaluation metrics on the training set during validation steps (can be slow).
    pt_model_uri : str
        Class path for the PyTorch model to instantiate (e.g., 'qlib.contrib.model.pytorch_nn.Net').
    pt_model_kwargs : dict
        Keyword arguments passed when instantiating the PyTorch model.
        Should now include dropout rates if Net is configured for it.
    valid_key : str
        Data key from DataHandlerLP used for preparing the validation set.
    num_workers : int
        Number of subprocesses to use for data loading with DataLoader.
    pin_memory : bool
        If True, DataLoader will copy Tensors into CUDA pinned memory before returning them (may speed up GPU transfer).
    infer_batch_size : int, optional
        Batch size for inference (`predict` method). If None, uses `batch_size * 4`.
    """

    def __init__(
        self,
        lr=0.001,
        max_steps=300,
        batch_size=2000,
        early_stop_rounds=50,
        eval_steps=20,
        optimizer="gd",
        loss="mse",
        GPU=0,
        seed=None,
        weight_decay=0.0,
        data_parall=False,
        # when it is Callable, it accept one argument named optimizer
        scheduler: Optional[Union[Callable]] = "default",
        init_model=None,
        eval_train_metric=False,
        pt_model_uri="qlib.contrib.model.pytorch_nn.Net",
        pt_model_kwargs={
            "input_dim": 360,
            "layers": (256,),
            "dropout_input": 0.05,
            "dropout_hidden": 0.05,
            "act": "LeakyReLU",
        },
        valid_key=DataHandlerLP.DK_L,
        # TODO: Infer Key is a more reasonable key. But it requires more detailed processing on label processing
        # Default to 0 for simplicity, increase for performance if I/O is bottleneck
        num_workers: int = 0,
        pin_memory: bool = False,  # Set to True if using GPU and num_workers > 0
        infer_batch_size: Optional[int] = None,
    ):
        # Set logger.
        self.logger = get_module_logger("DNNModelPytorch")
        self.logger.info("DNN pytorch version...")

        # set hyper-parameters.
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.early_stop_rounds = early_stop_rounds
        self.eval_steps = eval_steps
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        if isinstance(GPU, str):
            self.device = torch.device(GPU)
        else:
            self.device = torch.device("cuda:%d" % (
                GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.weight_decay = weight_decay
        self.data_parall = data_parall
        self.eval_train_metric = eval_train_metric
        self.valid_key = valid_key
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.infer_batch_size = infer_batch_size if infer_batch_size is not None else self.batch_size * 4

        self.best_step = None

        self.logger.info(
            "DNN parameters setting:"
            f"\nlr : {lr}"
            f"\nmax_steps : {max_steps}"
            f"\nbatch_size : {batch_size}"
            f"\nearly_stop_rounds : {early_stop_rounds}"
            f"\neval_steps : {eval_steps}"
            f"\noptimizer : {optimizer}"
            f"\nloss_type : {loss}"
            f"\nseed : {seed}"
            f"\ndevice : {self.device}"
            f"\nuse_GPU : {self.use_gpu}"
            f"\nweight_decay : {weight_decay}"
            f"\nenable data parall : {self.data_parall}"
            f"\npt_model_uri: {pt_model_uri}"
            f"\npt_model_kwargs: {pt_model_kwargs}"
            f"\nnum_workers: {self.num_workers}, pin_memory: {self.pin_memory}"
            f"\ninfer_batch_size: {self.infer_batch_size}"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if loss not in {"mse", "binary"}:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        self._scorer = mean_squared_error if loss == "mse" else roc_auc_score

        # Initialize the base PyTorch model (use temporary variable)
        if init_model is None:
            _base_model = init_instance_by_config(
                {"class": pt_model_uri, "kwargs": pt_model_kwargs})
        else:
            _base_model = init_model

        # Apply DataParallel if requested AND possible
        if self.data_parall and torch.cuda.device_count() > 1:
            self.logger.info(
                f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
            # Assign wrapped model to self.model
            self.model = DataParallel(_base_model)
        else:
            if self.data_parall:  # Log warning if requested but not possible
                self.logger.warning(
                    "DataParallel requested but only one GPU available (or CPU). Not applying DataParallel.")
             # Assign base model directly to self.model
            self.model = _base_model

        self.model.to(self.device)

        self.logger.info("model:\n{:}".format(self.model))  # Log self.model
        self.logger.info("model size: {:.4f} MB".format(
            count_parameters(self.model)))  # Count parameters of self.model

        # Initialize optimizer
        if optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower() == "gd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(
                "optimizer {} is not supported!".format(optimizer))

        # Initialize scheduler
        if scheduler == "default":
            # In torch version 2.7.0, the verbose parameter has been removed. Reference Link:
            # https://github.com/pytorch/pytorch/pull/147301/files#diff-036a7470d5307f13c9a6a51c3a65dd014f00ca02f476c545488cd856bea9bcf2L1313
            if str(torch.__version__).split("+", maxsplit=1)[0] <= "2.6.0":
                # Reduce learning rate when loss has stopped decrease
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # pylint: disable=E1123
                    self.optimizer,
                    mode="min",
                    factor=0.5,
                    patience=10,
                    verbose=True,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0.00001,
                    eps=1e-08,
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=0.5,
                    patience=10,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0.00001,
                    eps=1e-08,
                )
        elif scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(optimizer=self.optimizer)

        self.fitted = False

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def _prepare_dataloaders(self, dataset: DatasetH, reweighter: Optional[Reweighter] = None) -> Dict[str, DataLoader]:
        """
        Prepare PyTorch DataLoaders for training and validation segments.

        Parameters
        ----------
        dataset : DatasetH
            The Qlib dataset containing 'train' and 'valid' segments.
        reweighter : Reweighter, optional
            Reweighter instance to calculate sample weights.

        Returns
        -------
        Dict[str, DataLoader]
            A dictionary mapping segment names ('train', 'valid') to their respective DataLoaders.
        """
        dataloaders = {}
        self.processed_index = {}  # Store index for metric calculation if needed

        for segment in ["train", "valid"]:
            if segment not in dataset.segments:
                if segment == "valid":
                    self.logger.warning(
                        "Validation segment not found in dataset.")
                continue

            self.logger.info(f"Preparing data for segment: {segment}...")
            # Use dataset.prepare to get features and labels DataFrame
            data_key = self.valid_key if segment == "valid" else DataHandlerLP.DK_L
            df = dataset.prepare(
                segment, col_set=["feature", "label"], data_key=data_key)

            features = df["feature"].values
            # Ensure label shape is (N, 1)
            labels = df["label"].values.reshape(-1, 1)
            index = df.index  # Store index for metrics

            # Calculate sample weights
            if reweighter is None:
                weights = np.ones_like(labels, dtype=np.float32)
            elif isinstance(reweighter, Reweighter):
                weights = reweighter.reweight(
                    df).values.reshape(-1, 1).astype(np.float32)
            else:
                raise ValueError("Unsupported reweighter type.")

            # Create PyTorchDataset
            torch_dataset = PyTorchDataset(features, labels, weights, index)
            self.processed_index[segment] = index  # Store index

            # Create DataLoader
            dataloaders[segment] = DataLoader(
                dataset=torch_dataset,
                batch_size=self.batch_size,
                shuffle=(segment == "train"),  # Shuffle only training data
                num_workers=self.num_workers,
                pin_memory=self.pin_memory and self.use_gpu,
                # Drop last incomplete batch for training only
                drop_last=(segment == "train"),
            )
            self.logger.info(
                f"Segment '{segment}' loaded. Samples: {len(torch_dataset)}, Batches: {len(dataloaders[segment])}")

            # Clean up large DataFrame to save memory
            del df, features, labels, weights
            gc.collect()

        return dataloaders

    def fit(
        self,
        dataset: DatasetH,
        evals_result: Dict[str, List] = defaultdict(list),
        verbose=True,
        save_path=None,
        reweighter=None,
    ):
        # Setup save path
        save_path = get_or_create_path(save_path)

        # Prepare DataLoaders
        dataloaders = self._prepare_dataloaders(dataset, reweighter)
        train_loader = dataloaders.get("train", None)
        valid_loader = dataloaders.get("valid", None)

        if train_loader is None:
            raise ValueError(
                "Training data loader could not be created. Check 'train' segment in dataset.")

        has_valid = valid_loader is not None
        stop_steps = 0  # Counter for early stopping
        # train_loss = 0
        # best_loss = np.inf
        best_val_loss = np.inf  # Track best validation loss

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(1, self.max_steps + 1):  # step represents an epoch
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()

            # === Training Phase ===
            self.model.train()  # Set model to training mode
            train_loss_meter = AverageMeter()

            for i, (x_batch, y_batch, w_batch) in enumerate(train_loader):
                # Move batch data to the target device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = w_batch.to(self.device)

                # Forward pass
                preds = self.model(x_batch)
                # Calculate loss
                loss = self.get_loss(preds, w_batch, y_batch, self.loss_type)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update training loss tracker
                train_loss_meter.update(loss.item(), n=x_batch.size(0))

            # Log average training loss for the epoch
            avg_train_loss = train_loss_meter.avg
            R.log_metrics(train_loss=avg_train_loss, step=step)
            evals_result["train"].append(
                avg_train_loss)  # Store epoch train loss

            # === Validation Phase ===
            # Evaluate every `eval_steps` epochs or on the last epoch
            if step % self.eval_steps == 0 or step == self.max_steps:
                if has_valid:
                    self.model.eval()  # Set model to evaluation mode
                    val_loss_meter = AverageMeter()
                    all_preds_val = []
                    all_labels_val = []

                    with torch.no_grad():  # Disable gradient calculation for validation
                        for x_batch_val, y_batch_val, w_batch_val in valid_loader:
                            x_batch_val = x_batch_val.to(self.device)
                            y_batch_val = y_batch_val.to(self.device)
                            w_batch_val = w_batch_val.to(self.device)

                            # Predict
                            preds_val = self.model(x_batch_val)
                            # Calculate validation loss for the batch
                            loss_val_batch = self.get_loss(
                                preds_val, w_batch_val, y_batch_val, self.loss_type)
                            val_loss_meter.update(
                                loss_val_batch.item(), n=x_batch_val.size(0))

                            # Collect predictions and labels (move to CPU for metric calculation)
                            all_preds_val.append(preds_val.cpu())
                            all_labels_val.append(y_batch_val.cpu())
                            # all_weights_val.append(w_batch_val.cpu())

                    # Concatenate results from all validation batches
                    all_preds_val = torch.cat(all_preds_val, dim=0)
                    all_labels_val = torch.cat(all_labels_val, dim=0)
                    # all_weights_val = torch.cat(all_weights_val, dim=0) # If needed

                    avg_val_loss = val_loss_meter.avg
                    R.log_metrics(val_loss=avg_val_loss, step=step)
                    # Store epoch validation loss
                    evals_result["valid"].append(avg_val_loss)

                    # Calculate validation metric (e.g., ICLoss) using collected results
                    # Assumes get_metric works with CPU tensors and requires the original index
                    metric_val = self.get_metric(
                        all_preds_val, all_labels_val, self.processed_index["valid"]
                    ).item()
                    R.log_metrics(val_metric=metric_val, step=step)

                    # Optional: Calculate and log training metric
                    metric_train = np.nan
                    if self.eval_train_metric:
                        # This requires predicting on the entire training set again, which can be slow.
                        # Consider doing this less frequently or using the epoch's average loss as a proxy.
                        self.logger.warning(
                            "Calculating train metric requires full pass over training data, can be slow.")
                        train_preds_np = self.predict_on_loader(
                            train_loader, return_cpu=True)
                        train_preds_tensor = torch.from_numpy(train_preds_np)

                        try:
                            metric_train = self.get_metric(
                                train_preds_tensor, train_loader.dataset.labels, self.processed_index["train"]).item()
                            R.log_metrics(train_metric=metric_train, step=step)
                        except Exception as e:
                            self.logger.error(
                                f"Failed to calculate train metric: {e}")

                    # Print progress
                    if verbose:
                        self.logger.info(
                            f"[Epoch {step}/{self.max_steps}]: train_loss={avg_train_loss:.6f}, "
                            f"val_loss={avg_val_loss:.6f}, val_metric={metric_val:.6f}, "
                            f"train_metric={metric_train:.6f}"
                        )

                    # Early stopping and checkpointing
                    stop_steps += 1  # Increment counter
                    if avg_val_loss < best_val_loss:
                        if verbose:
                            self.logger.info(
                                f"\tValidation loss improved ({best_val_loss:.6f} --> {avg_val_loss:.6f}). Saving model to {save_path}"
                            )
                        best_val_loss = avg_val_loss
                        self.best_step = step
                        R.log_metrics(best_step=self.best_step, step=step)
                        stop_steps = 0  # Reset counter
                        # Save the best model state
                        torch.save(self.model.state_dict(), save_path)
                    elif stop_steps >= self.early_stop_rounds:
                        if verbose:
                            self.logger.info(
                                f"Early stopping triggered after {stop_steps} epochs without improvement.")
                        break  # Exit training loop

                    # Step the scheduler based on validation loss
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            # Pass validation loss
                            self.scheduler.step(metrics=avg_val_loss)
                        else:
                            auto_filter_kwargs(self.scheduler.step, warning=False)(
                                epoch=step)  # Step based on epoch for others

                else:  # No validation set
                    if verbose:
                        self.logger.info(
                            f"[Epoch {step}/{self.max_steps}]: train_loss={avg_train_loss:.6f}")
                    # Step scheduler if it's not based on validation metrics (e.g., StepLR)
                    if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        auto_filter_kwargs(
                            self.scheduler.step, warning=False)(epoch=step)

                # Log learning rate after potential scheduler step
                R.log_metrics(lr=self.get_lr(), step=step)

        if has_valid:
            # restore the optimal parameters after training
            self.model.load_state_dict(
                torch.load(save_path, map_location=self.device))
        if self.use_gpu:
            torch.cuda.empty_cache()  # === Training End ===
        # Load the best model if early stopping occurred and a checkpoint exists
        if has_valid and os.path.exists(save_path) and self.best_step > 0:
            self.logger.info(
                f"Training finished. Loading best model from epoch {self.best_step} saved at {save_path}")
            try:
                self.model.load_state_dict(torch.load(
                    save_path, map_location=self.device))
            except Exception as e:
                self.logger.error(
                    f"Failed to load best model checkpoint: {e}. Using model from last step.")
        elif has_valid:
            self.logger.info(
                "Training finished. No checkpoint found or validation did not improve. Using model from last step.")
        else:
            self.logger.info(
                "Training finished without validation. Using model from last step.")
            # Save the final model if no validation was performed
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"Model from last step saved to {save_path}")

        # Clean up GPU memory cache if used
        if self.use_gpu:
            torch.cuda.empty_cache()

    def get_lr(self):
        """Get the current learning rate from the optimizer."""
        assert len(self.optimizer.param_groups) == 1
        return self.optimizer.param_groups[0]["lr"]

    def get_loss(self, pred: torch.Tensor, w: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
        """
        Calculate the weighted loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions (raw logits for binary, values for mse). Shape (batch_size, 1) or (batch_size,).
        w : torch.Tensor
            Sample weights. Shape (batch_size, 1) or (batch_size,).
        target : torch.Tensor
            True labels. Shape (batch_size, 1) or (batch_size,).
        loss_type : str
            Type of loss ('mse' or 'binary').

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the mean weighted loss for the batch.
        """
        # Ensure inputs are flattened or have compatible shapes (N, 1) or (N,)
        pred = pred.reshape(-1)
        w = w.reshape(-1)
        target = target.reshape(-1)

        if loss_type == "mse":
            # Calculate weighted mean squared error
            # Use reduction='none' to get per-sample loss, then apply weights
            loss_func = nn.MSELoss(reduction='none')
            unweighted_loss = loss_func(pred, target)
            weighted_loss = unweighted_loss * w
            return weighted_loss.mean()  # Return the mean of weighted losses
        elif loss_type == "binary":
            # Use BCEWithLogitsLoss which combines Sigmoid and BCELoss for numerical stability
            # The 'weight' argument in BCEWithLogitsLoss handles per-sample weighting correctly
            loss_func = nn.BCEWithLogitsLoss(weight=w, reduction='mean')
            return loss_func(pred, target)
        else:
            raise NotImplementedError(f"loss {loss_type} is not supported!")

    def get_metric(self, pred: torch.Tensor, target: torch.Tensor, index: pd.Index) -> torch.Tensor:
        """
        Calculate the evaluation metric (e.g., using ICLoss).
        Assumes pred and target are CPU tensors.

        Parameters
        ----------
        pred : torch.Tensor
             Model predictions (on CPU).
        target : torch.Tensor
             True labels (on CPU).
        index : pd.Index
             Pandas index corresponding to predictions/labels (datetime, instrument).

        Returns
        -------
        torch.Tensor
             Scalar tensor of the calculated metric (e.g., negative IC).
        """
        # Ensure tensors are on CPU if the metric function requires it
        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()

        return -ICLoss()(pred_cpu, target_cpu, index)  # pylint: disable=E1130

    def predict_on_loader(self, data_loader: DataLoader, return_cpu: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """Internal helper to predict using a DataLoader."""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _, _ in data_loader:  # Labels and weights are not needed for prediction
                x_batch = x_batch.to(self.device)
                batch_preds = self.model(x_batch).detach()
                if return_cpu:
                    all_preds.append(batch_preds.cpu())
                else:
                    all_preds.append(batch_preds)

        preds = torch.cat(all_preds, dim=0)
        if return_cpu:
            return preds.numpy().reshape(-1)  # Return flattened numpy array
        else:
            # Return flattened tensor on original device
            return preds.reshape(-1)

    def _nn_predict(self, data, return_cpu=True):
        """Reusing predicting NN.
        Scenarios
        1) test inference (data may come from CPU and expect the output data is on CPU)
        2) evaluation on training (data may come from GPU)
        """
        if not isinstance(data, torch.Tensor):
            if isinstance(data, pd.DataFrame):
                data = data.values
            data = torch.Tensor(data)
        data = data.to(self.device)
        preds = []
        self.model.eval()
        with torch.no_grad():
            batch_size = 8096
            for i in range(0, len(data), batch_size):
                x = data[i: i + batch_size]
                preds.append(self.model(
                    x.to(self.device)).detach().reshape(-1))
        if return_cpu:
            preds = np.concatenate([pr.cpu().numpy() for pr in preds])
        else:
            preds = torch.cat(preds, axis=0)
        return preds

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test_pd = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self._nn_predict(x_test_pd)
        return pd.Series(preds.reshape(-1), index=x_test_pd.index)

    def save(self, filename, **kwargs):
        with save_multiple_parts_file(filename) as model_dir:
            model_path = os.path.join(model_dir, os.path.split(model_dir)[-1])
            # Save model
            torch.save(self.model.state_dict(), model_path)

    def load(self, buffer, **kwargs):
        with unpack_archive_with_buffer(buffer) as model_dir:
            # Get model name
            _model_name = os.path.splitext(list(filter(lambda x: x.startswith("model.bin"), os.listdir(model_dir)))[0])[
                0
            ]
            _model_path = os.path.join(model_dir, _model_name)
            # Load model
            self.model.load_state_dict(torch.load(
                _model_path, map_location=self.device))
        self.fitted = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        layers=(256,),
        act="LeakyReLU",
        dropout_input: float = 0.05,
        dropout_hidden: float = 0.05,
    ):
        """
        Initialize the network layers.

        Parameters
        ----------
        input_dim : int
            Dimension of the input features.
        output_dim : int, optional
            Dimension of the output layer, by default 1.
        layers : tuple, optional
            Tuple defining the number of units in each hidden layer, by default (256,).
        act : str, optional
            Activation function name ('LeakyReLU', 'ReLU', 'SiLU', 'Tanh', etc.), by default "LeakyReLU".
            Must match a class name in torch.nn.
        dropout_input : float, optional
            Dropout rate applied to the input layer, by default 0.05.
        dropout_hidden : float, optional
            Dropout rate applied after activation in hidden layers, by default 0.05.
        """
        super(Net, self).__init__()

        dnn_layers = []  # List to hold all network layers/modules

        # Input Dropout
        if dropout_input > 0:
            dnn_layers.append(nn.Dropout(dropout_input))

        # Dynamically create hidden layers
        current_dim = input_dim
        for i, hidden_units in enumerate(layers):
            dnn_layers.append(nn.Linear(current_dim, hidden_units))
            dnn_layers.append(nn.BatchNorm1d(hidden_units))

            # Add activation function
            try:
                # Dynamically get activation class from torch.nn
                activation_class = getattr(nn, act)
                # Handle activations like LeakyReLU that need arguments vs those that don't
                if act == "LeakyReLU":
                    dnn_layers.append(activation_class(
                        negative_slope=0.1, inplace=False))
                else:
                    dnn_layers.append(activation_class(inplace=False if hasattr(
                        activation_class, 'inplace') else None))  # Check if inplace is supported
            except AttributeError:
                raise NotImplementedError(
                    f"Activation function '{act}' not found in torch.nn")
            except TypeError:  # Handle activations without inplace argument cleanly
                dnn_layers.append(activation_class())

            # Add hidden layer dropout
            if dropout_hidden > 0:
                dnn_layers.append(nn.Dropout(dropout_hidden))

            current_dim = hidden_units  # Update dimension for the next layer

        # Output layer
        dnn_layers.append(nn.Linear(current_dim, output_dim))

        # Combine all layers into a Sequential module for simplicity in forward pass
        self.dnn_layers_seq = nn.Sequential(*dnn_layers)

        # Initialize weights
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnn_layers_seq(x)
