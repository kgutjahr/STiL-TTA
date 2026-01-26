import torch
import random
from itertools import permutations, combinations

def get_same_class_points(y: torch.Tensor) -> dict:
    index_map = {}

    for idx, val in enumerate(y.tolist()):
        if val in index_map:
            index_map[val].append(idx)
        else:
            index_map[val] = [idx]
    return index_map

def sample_unique_first(data, seed=None):
    if seed is not None:
        random.seed(seed)
    # group by the first element
    groups = {}
    for t in data:
        groups.setdefault(t[0], []).append(t)
    # randomly choose one permutation from each group
    result = [random.choice(v) for v in groups.values()]
    return result

def get_idx_pairs(x: torch.Tensor, y: torch.Tensor, sample_randomly: bool, seed: int, rate: float) -> list:
    pair_list = []
    if not sample_randomly:
        index_map = get_same_class_points(y=y)
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
        gen = torch.Generator(device=x.device).manual_seed(seed)
        x_len = torch.tensor(list(range(0, len(x))))
        shuffled = x_len[torch.randperm(len(x), generator=gen)]
        pairs = shuffled.view(-1, 2)
        pair_list = pairs.tolist()
        add_pairs = [p[::-1] for p in pair_list]
        pair_list.extend(add_pairs)
    
    rng = random.Random(seed)
    pair_list = sample_unique_first(data=pair_list, seed=seed)
    # only extrapolate certain amount of pairs
    num_to_keep = int(len(pair_list) * rate)

    # Randomly select sample of the pairs
    return rng.sample(pair_list, num_to_keep)
    
def get_idx_triples(batch_size: int, y: torch.Tensor, sample_randomly: bool, seed: int, rate: float) -> list:
    triple_list = []
    if not sample_randomly:
        index_map = get_same_class_points(y=y)
        candidates = [v for _, v in index_map.items() if len(v) > 1]
        filtered_cand = [c for c in candidates if len(c) > 2]
        triple_list.extend([tuple(p) for inner in filtered_cand for p in permutations(inner, 3)])
    else:
        gen = torch.Generator().manual_seed(seed)
        x_len = torch.tensor(list(range(0, batch_size)))
        shuffled = x_len[torch.randperm(batch_size, generator=gen)].tolist()
        triple_list = list(combinations(shuffled, 3))
        
    kept_triples = sample_unique_first(data=triple_list, seed=seed)
    
    # only extrapolate certain amount of pairs
    num_to_keep = int(len(kept_triples) * rate)

    # Randomly select sample of the pairs
    rng = random.Random(seed)
    return rng.sample(kept_triples, num_to_keep)

def get_random_idx(batch_size: int, rate: float, seed: int):
    gen = torch.Generator().manual_seed(seed)
    num_selected = int(batch_size * rate)
    return torch.randperm(batch_size, generator=gen)[:num_selected].tolist()

# only extrapolate datapoints from the same class
def extrapolate(x: torch.Tensor, lambda_: float, y: torch.Tensor, rate: float, sample_randomly: bool, seed: int, idx = None) -> torch.Tensor:
    if rate == 0.0:
        return x
    pair_list = idx if idx is not None else get_idx_pairs(x=x, y=y, sample_randomly=sample_randomly, seed=seed, rate=rate)
    
    if len(pair_list) == 0:
        return x

    z = x.clone()
    # actually extrapolate the tensors
    pairs = torch.tensor(pair_list)

    i_indices = pairs[:, 0]
    j_indices = pairs[:, 1]

    x_i = x[i_indices]
    x_j = x[j_indices]

    z[i_indices] = (x_i - x_j) * lambda_ + x_i
    return z

def random_noise(x: torch.Tensor, rate: float, min_range: float, max_range: float, seed: int, idx = None) -> torch.Tensor:
    if rate == 0.0:
        return x
    gen = torch.Generator(device=x.device).manual_seed(seed)
    
    batch_std = x.std(dim=0, unbiased=False)
    
    sample_size = x.size()[1:]
    random_indices = idx if idx is not None else get_random_idx(batch_size=x.size()[0], rate=rate, seed=seed)
    z = x.clone()
    for i in random_indices:
        # addition
        rand_sample_add = torch.randn(size=sample_size, generator=gen, device=x.device) * batch_std
        z[i] = x[i] + rand_sample_add
    return z

def mixstyle(x: torch.Tensor, rate: float, alpha: float, seed: int, idx = None) -> torch.Tensor:
    if rate == 0.0:
        return x

    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    
    x_prime = x[torch.randperm(x.size()[0], generator=gen)]
    selected_indices = idx if idx is not None else get_random_idx(batch_size=x.size()[0], rate=rate, seed=seed)

    # Extract selected samples
    x_sel = x[selected_indices]
    x_prime_sel = x_prime[selected_indices]

    # Compute std and mean for each selected sample
    with torch.no_grad():
        x_std, x_mean = torch.std_mean(x_sel, dim=tuple(range(1, x_sel.ndim)), unbiased=False)
        x_prime_std, x_prime_mean = torch.std_mean(x_prime_sel, dim=tuple(range(1, x_prime_sel.ndim)), unbiased=False)

    # Per-sample betavariate values
    lambdas = torch.tensor(
        [rng.betavariate(alpha, alpha) for _ in range(len(selected_indices))],
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

def linear_delta(x: torch.Tensor, y: torch.Tensor, sample_randomly: bool, rate: float, seed: int, idx = None) -> torch.Tensor:
    if rate == 0.0:
        return x
    
    triple_list = idx if idx is not None else get_idx_triples(batch_size=x.size()[0], y=y, sample_randomly=sample_randomly, seed=seed, rate=rate)
    
    if len(triple_list) == 0:
        return x
    
    z = x.clone()
    # actually extrapolate the tensors
    triples = torch.tensor(triple_list)

    i_indices = triples[:, 0]
    j_indices = triples[:, 1]
    k_indices = triples[:, 2]

    x_i = x[i_indices]
    x_j = x[j_indices]
    x_k = x[k_indices]

    z[k_indices] = (x_i - x_j) + x_k
    return z

if __name__ == "__main__":
    #from AugmentSummarizer import AugmentSummarizer
    x = torch.rand([20, 4])
    a = torch.rand([20, 4])
    b = torch.rand([20, 4])
    y = torch.randint(low=0, high=286, size=(x.size()[0],))
    y[1] = 24
    y[2] = 24
    y[3] = 24
    #
    y[6] = 17
    y[9] = 17
    y[16] = 17
    
    #aug = AugmentSummarizer()
    
    pairs = get_random_idx(batch_size=x.size()[0], rate=0.35, seed=2022)
    
    z = random_noise(x=x, rate=0.1, min_range=-1.0, max_range=1.0, seed=2022)
    #aug.register_rate(modality="image", A=x, B=z)
    u = random_noise(x=a, rate=0.1, min_range=-1.0, max_range=1.0, seed=2022)
    #aug.register_rate(modality="tabular", A=a, B=u)
    v = random_noise(x=b, rate=0.1, min_range=-1.0, max_range=1.0, seed=2022)
    #aug.register_rate(modality="multimodal", A=b, B=v)
    
    #print(aug.summarize())
    
    assert len(z) == len(y)
    diff = (z != x)
    rows_with_diff = diff.any(dim=1)
    print(rows_with_diff)
    
    assert len(u) == len(y)
    diff = (u != a)
    rows_with_diff = diff.any(dim=1)
    print(rows_with_diff)
    
    assert len(v) == len(y)
    diff = (v != b)
    rows_with_diff = diff.any(dim=1)
    print(rows_with_diff)