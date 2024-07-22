import criterionFile

def Loss(labels_x, x_hat, cfg_loss):
    """
    Computes the loss for the predicted source signals given the target labels.

    Args:
        labels_x (torch.Tensor): Target source signals.
        x_hat    (torch.Tensor): Predicted source signals.
        cfg_loss       (object): Configuration object containing loss parameters.

    Returns:
        loss (torch.Tensor): The computed loss value.
    """

    # Parameters
    norm = cfg_loss.norm

    # Loss functions
    criterion_L1 = criterionFile.criterionL1

    # Calculate L1 loss
    loss_s_L1 = criterion_L1(x_hat.float(), labels_x.float(), norm)

    # Define the loss function
    loss = (loss_s_L1) * 10000

    return loss
