import torch

def timeKLDivLoss(output, target):
    """
    """
    batch_size, _, _, grid_cells = output.size()

    running_kld = 0

    # assign the probs
    for b in range(batch_size):
        for g in range(grid_cells):
            # get time series
            output_series = output[b, 0, :, g]
            target_series = target[b, 0, :, g]
            # get probs
            output_probs = torch.histc(output_series, bins=2000, min=-1, max=+1) / len(output_series)
            target_probs = torch.histc(target_series, bins=2000, min=-1, max=+1) / len(target_series)
            # replace zero values
            output_probs[output_probs == 0] = 0.0001
            target_probs[target_probs == 0] = 0.0001
            # calculate kl divergence
            running_kld += (target_probs * (target_probs.log() - output_probs.log())).sum()

    return running_kld / (batch_size * grid_cells)
