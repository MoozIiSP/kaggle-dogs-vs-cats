import torch

# NOTE: Generate Test Set and save it into the disk


def testset_collate_fn(batch):
    images, meta = zip(*batch)
    return torch.stack(images), meta