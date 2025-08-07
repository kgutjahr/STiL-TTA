import torch

# only extrapolate datapoints from the same class
def extrapolate(x: torch.Tensor, lambda_: float, y: torch.Tensor) -> torch.Tensor:
    x_l = x[:len(y)]
    index_map = {}

    for idx, val in enumerate(y.tolist()):
        if val in index_map:
            index_map[val].append(idx)
        else:
            index_map[val] = [idx]

    # Filter to keep only duplicates
    duplicates = {k: v for k, v in index_map.items() if len(v) > 1}
    
    for _, dup in duplicates.items():
        same_class = x[dup]
        # get nearest neighbors
        if len(dup) > 2:
            dist = torch.cdist(same_class, same_class, p=2)
            # Set diagonal to large value so self-distance is ignored
            dist.fill_diagonal_(float('inf'))
            # Get index of the nearest neighbor for each sample
            nn_indices = torch.argmin(dist, dim=1)
    
    return x

def random_noise(x: torch.Tensor, rate: float, min_range: float, max_range: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    
    sample_size = x.size()[1:]
    batch_size = x.size()[0]
    random_indices = torch.randperm(batch_size, generator=gen)[:int(batch_size*rate)]
    for i in random_indices:
        # addition
        rand_sample_add = (min_range - max_range) * torch.rand(sample_size, generator=gen) + max_range
        x[i] = x[i] + rand_sample_add
        # multiplication
        rand_sample_mul = (min_range - max_range) * torch.rand(sample_size, generator=gen) + max_range
        x[i] = x[i] * rand_sample_mul
    return x

def mixstyle(x: torch.Tensor, rate: float) -> torch.Tensor:
    result = x + rate
    print(result)
    return result