import torch


def collate_trajectory_batch(batch):
    collated = {}
    first_item = batch[0]
    for key, value in first_item.items():
        if torch.is_tensor(value):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            collated[key] = [item[key] for item in batch]
    return collated
