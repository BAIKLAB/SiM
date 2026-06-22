from tqdm import tqdm

import torch

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset


def compute_V(features, k):
    if len(features.shape) != 2:
        raise ValueError(f"Input tensor must have shape (batch_size, feature_dim), but {features.shape}.")

    mean = features.mean(dim=0)
    centered_features = features - mean
    _U, _S, Vt = torch.linalg.svd(centered_features, full_matrices=False)
    D = Vt.shape[1]
    r = max(1, int(round(D * k)))
    r = min(r, Vt.shape[0])
    return mean, Vt[:r]


def project(x, mean, Vk):
    xc = x - mean
    return (xc @ Vk.T) @ Vk


def residual_norm(x, mean, Vk):
    xc = x - mean
    proj = project(x, mean, Vk)
    res = xc - proj
    return torch.linalg.norm(res, ord=2, dim=1)


def get_manifolds(
    args,
    source_dataset_name,
    device,
    base_encoder,
):
    features = []
    dataset = get_dataset(
        f"{source_dataset_name}Val",
        base_encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = get_dataloader(dataset, is_train=True, args=args)
    n_pool = len(dataset.train_dataset)

    if isinstance(args.n_samples, float):
        n_samples = round(n_pool * args.n_samples)
    else:
        n_samples = args.n_samples
    print(f"n_samples: {n_samples}")

    with torch.no_grad():
        n = 0
        with tqdm(dataloader, unit=f"batch({args.batch_size})") as tepoch:
            for data in tepoch:
                tepoch.set_description(source_dataset_name)

                data = maybe_dictionarize(data)
                x = data["images"].to(device)

                if n >= n_samples:
                    break

                if n_samples - n < args.batch_size:
                    x = x[:n_samples - n, ...]

                feature = base_encoder(x)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                features.append(feature)

                n += x.size(0)

        features = torch.cat(features)
        mean, V = compute_V(features, args.k)
        return {"mean": mean, "V": V}
