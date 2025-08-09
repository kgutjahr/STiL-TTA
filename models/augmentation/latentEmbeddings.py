import torch
import random

def get_same_class_points(x: torch.Tensor, y: torch.Tensor) -> dict:
    index_map = {}

    for idx, val in enumerate(y.tolist()):
        if val in index_map:
            index_map[val].append(idx)
        else:
            index_map[val] = [idx]
    return index_map

# only extrapolate datapoints from the same class
def extrapolate(x: torch.Tensor, lambda_: float, y: torch.Tensor, rate: float, sample_randomly: bool) -> torch.Tensor:
    if not sample_randomly:
        index_map = get_same_class_points(x=x, y=y)
        # Filter to keep only duplicates
        candidates = [v for _, v in index_map.items() if len(v) > 1]
        
        pair_list = []
        for can in candidates:
            same_class = x[can]
            # get nearest neighbors
            if len(can) > 2:
                dist = torch.cdist(same_class, same_class, p=2)
                # Set diagonal to large value so self-distance is ignored
                dist.fill_diagonal_(float('inf'))
                # Get index of the nearest neighbor for each sample
                nn_indices = torch.argmin(dist, dim=1)
                new_pairs = [[can[i], can[d]] for i, d in enumerate(nn_indices)]
                pair_list.extend(new_pairs)
            else:
                pair_list.append(can)
        
    else:
        x_len = torch.tensor(list(range(0, len(x))))
        shuffled = x_len[torch.randperm(len(x))]
        pairs = shuffled.view(-1, 2)
        pair_list = pairs.tolist()
    print(pair_list)
    
    # only extrapolate certain amount of pairs
    num_to_keep = int(len(pair_list) * rate)

    # Randomly select 30% of the pairs
    kept_pairs = random.sample(pair_list, num_to_keep)
    
    # actually extrapolate the tensors
    z = x.clone()
    for pair in kept_pairs:
        x_i = x[pair[0]]
        x_j = x[pair[1]]
        z[pair[0]] = (x_i - x_j) * lambda_ + x_i

    return z

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

if __name__ == "__main__":
    x = torch.rand([512, 4])
    y = torch.randint(low=0, high=286, size=(64,))
    y[0] = 50
    y[51] = 50
    y[28] = 50
    
    z = extrapolate(x=x, y=y, lambda_=0.5, rate=0.5, sample_randomly=True)
    print(x == z)