from multiprocessing import cpu_count

from torch.utils.data import DataLoader


def compute_stats(dataset):
    n = len(dataset) // 1000
    loader = DataLoader(
        dataset,
        batch_size=n,
        num_workers=cpu_count())
    mean, std, total = 0., 0., 0
    for batch, _ in iter(loader):
        image = batch.squeeze()
        mean += image.mean().item()
        std += image.std().item()
        total += 1
    mean /= total
    std /= total
    print(mean, std)
