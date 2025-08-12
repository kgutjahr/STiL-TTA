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
def extrapolate(x: torch.Tensor, lambda_: float, y: torch.Tensor, rate: float, sample_randomly: bool, seed: float) -> torch.Tensor:
    if rate == 0.0:
        return x

    rng = random.Random(seed)
    
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
        add_pairs = [p[::-1] for p in pair_list]
        pair_list.extend(add_pairs)

    if len(pair_list) < 0:
        return x
    
    # only extrapolate certain amount of pairs
    num_to_keep = int(len(pair_list) * rate)
    num_to_keep = max(num_to_keep, 1)

    z = x.clone()
    # Randomly select sample of the pairs
    kept_pairs = rng.sample(pair_list, num_to_keep)

    # actually extrapolate the tensors
    pairs = torch.tensor(kept_pairs)

    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]

    x_i = x[i_indices]
    x_j = x[j_indices]

    z[i_indices] = (x_i - x_j) * lambda_ + x_i
    return z

def random_noise(x: torch.Tensor, rate: float, min_range: float, max_range: float, seed: int) -> torch.Tensor:
    if rate == 0.0:
        return x
    gen = torch.Generator().manual_seed(seed)
    
    sample_size = x.size()[1:]
    batch_size = x.size()[0]
    num_to_keep = int(batch_size * rate)
    num_to_keep = max(num_to_keep, 1)
    random_indices = torch.randperm(batch_size, generator=gen)[:num_to_keep]
    z = x.clone()
    for i in random_indices:
        # addition
        rand_sample_add = (min_range - max_range) * torch.rand(sample_size, generator=gen) + max_range
        z[i] = x[i] + rand_sample_add
        # multiplication
        rand_sample_mul = (min_range - max_range) * torch.rand(sample_size, generator=gen) + max_range
        z[i] = x[i] * rand_sample_mul
    return z

def mixstyle(x: torch.Tensor, rate: float, alpha: float, seed: float) -> torch.Tensor:
    if rate == 0.0:
        return x

    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    
    x_prime = x[torch.randperm(x.size()[0], generator=gen)]
    batch_size = x.size(0)
    num_selected = int(batch_size * rate)
    selected_indices = torch.randperm(batch_size, generator=gen)[:num_selected]

    # Extract selected samples
    x_sel = x[selected_indices]
    x_prime_sel = x_prime[selected_indices]

    # Compute std and mean for each selected sample
    with torch.no_grad():
        x_std, x_mean = torch.std_mean(x_sel, dim=tuple(range(1, x_sel.ndim)), unbiased=False)
        x_prime_std, x_prime_mean = torch.std_mean(x_prime_sel, dim=tuple(range(1, x_prime_sel.ndim)), unbiased=False)

    # Per-sample betavariate values
    lambdas = torch.tensor(
        [rng.betavariate(alpha, alpha) for _ in range(num_selected)],
        dtype=x.dtype, device=x.device
    )

    # Compute gamma_mix and beta_mix
    gamma_mix = lambdas * x_std + (1 - lambdas) * x_prime_std
    beta_mix = lambdas * x_mean + (1 - lambdas) * x_prime_mean

    # Reshape for broadcasting
    gamma_mix = gamma_mix.view(-1, *([1] * (x_sel.ndim - 1)))
    beta_mix = beta_mix.view(-1, *([1] * (x_sel.ndim - 1)))
    x_std = x_std.view(-1, *([1] * (x_sel.ndim - 1)))
    x_mean = x_mean.view(-1, *([1] * (x_sel.ndim - 1)))

    # Apply transformation all at once
    z = x.clone()
    z[selected_indices] = gamma_mix * ((x_sel - x_mean) / x_std) + beta_mix
    return z

if __name__ == "__main__":
    x = torch.rand([4, 4])
    y = torch.randint(low=0, high=286, size=(x.size()[0],))
    y[1] = 24
    y[2] = 24
    print(y)
    
    z = extrapolate(x=x, lambda_=0.5, y=y, rate=0.75, sample_randomly=False, seed=2022)
    print(get_augment_rate(A=x, B=z))