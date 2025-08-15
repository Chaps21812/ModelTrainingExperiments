import torch

def check_loss_for_nans(loss_dict, step=None):
    """
    Checks if any losses in the loss_dict contain NaNs or Infs.
    Prints which ones have problems.

    Args:
        loss_dict (dict): e.g., {"loss_classifier": tensor(...), "loss_box_reg": tensor(...), ...}
        step (int, optional): Training step/epoch for logging context.
    """
    for name, loss_val in loss_dict.items():
        if not torch.isfinite(loss_val):
            raise ValueError(f"NaN or Inf detected in {name} at step {step}: {loss_val.item()}")