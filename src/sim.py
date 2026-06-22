import argparse
from collections import OrderedDict
import copy
import json
import os
import pickle

from tqdm import tqdm

import torch
from torch import nn

from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.heads import get_classification_head, get_original_classification_head
from src.merging_utils import (
    build_tsv_state_dict,
    emr_merging,
    tall_mask,
    tsv_c,
)
from src.modeling import ImageEncoder
from src.routing_utils import get_manifolds, residual_norm
from src.sim_utils import set_random_seed


def parse_arguments_for_merge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        required=True,
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-ckpt-dir",
        type=str,
        required=True,
        help="The root directory for the encoder checkpoint.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=8,
        choices=[8, 14, 20],
        help="Number of tasks to merge.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of calibration set samples used to estimate each task subspace.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize final image features before routing/classification.",
    )
    parser.add_argument(
        "--tallmask-setting",
        action="store_true",
        help="Use fine-tuned checkpoints prepared for the TALL-mask setting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--merging-method",
        type=str,
        default="emr",
        choices=["emr", "tm_ta", "tm_ties", "tsv"],
        help="Subspace/mask-based merging method to use.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.1,
        help="Subspace rank ratio used for task subspace estimation.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory for caching zero-shot classification heads.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if parsed_args.save is None:
        parsed_args.save = os.path.join(parsed_args.model_ckpt_dir, parsed_args.model)

    return parsed_args


def get_task_names(num_tasks):
    if num_tasks == 8:
        return ["Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN"]
    if num_tasks == 14:
        return [
            "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
            "CIFAR100", "STL10", "Flowers102", "OxfordIIITPet", "PCAM", "FER2013",
        ]
    if num_tasks == 20:
        return [
            "Cars", "DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SUN397", "SVHN",
            "CIFAR100", "STL10", "Flowers102", "OxfordIIITPet", "PCAM", "FER2013",
            "EMNIST", "CIFAR10", "Food101", "FashionMNIST", "RenderedSST2", "KMNIST",
        ]
    raise NotImplementedError


def load_finetuned_state_dicts(args, datasets):
    load_model_paths = []
    for source_dataset_name in datasets:
        ds_name = f"{source_dataset_name}Val" if args.tallmask_setting else source_dataset_name
        ckpt_name = "nonlinear_finetuned.pt" if args.tallmask_setting else "finetuned.pt"
        load_model_path = os.path.join(args.model_ckpt_dir, args.model, ds_name, ckpt_name)
        print(f"loading a checkpoint from {load_model_path}")
        load_model_paths.append(load_model_path)

    ft_state_dicts = []
    for model_path in load_model_paths:
        try:
            state_dict = torch.load(model_path, map_location="cpu")
        except RuntimeError:
            with open(model_path, "rb") as f:
                state_dict = pickle.load(f)
        if isinstance(state_dict, nn.Module):
            state_dict = copy.deepcopy(state_dict.state_dict())
        ft_state_dicts.append(state_dict)
    return ft_state_dicts


def build_task_vectors(pre_state_dict, ft_state_dicts):
    task_vectors = []
    for ft_state_dict in ft_state_dicts:
        task_vector = OrderedDict()
        for name in pre_state_dict.keys():
            task_vector[name] = ft_state_dict[name] - pre_state_dict[name]
        task_vectors.append(task_vector)
    return task_vectors


def build_classification_heads(args, datasets, device):
    heads = {}
    for dataset_name in datasets:
        args.dataset_name = f"{dataset_name}Val" if args.tallmask_setting else dataset_name
        if args.tallmask_setting:
            heads[dataset_name] = get_classification_head(args, dataset_name)
        else:
            heads[dataset_name] = get_original_classification_head(args)
        heads[dataset_name] = heads[dataset_name].to(device)
        heads[dataset_name].eval()
    return heads


def build_memory_bank(args, datasets, base_model, merged_state_dict, device, save_params_path):
    memory_bank = {}
    base_model.load_state_dict(merged_state_dict)
    for source_dataset_name in datasets:
        print("(merged) Generating memory bank of", source_dataset_name)
        memory_bank[source_dataset_name] = get_manifolds(
            args, source_dataset_name, device, base_model
        )
    torch.save(memory_bank, save_params_path)
    return memory_bank


def load_or_build_memory_bank(args, datasets, base_model, merged_state_dict, device, save_params_path):
    if not os.path.exists(save_params_path):
        return build_memory_bank(args, datasets, base_model, merged_state_dict, device, save_params_path)

    memory_bank = torch.load(save_params_path, map_location=device)
    print("Cached memory bank has been uploaded.")
    missing = [name for name in datasets if name not in memory_bank]
    if missing:
        print(
            f"[WARN] Cached memory bank keys mismatch. Missing keys={missing}. "
            f"Rebuilding memory bank at {save_params_path}."
        )
        return build_memory_bank(args, datasets, base_model, merged_state_dict, device, save_params_path)
    return memory_bank


def make_experiment_name(args):
    exp_name = f"exp_{args.model}_{args.num_tasks}"
    if args.n_samples != 128:
        exp_name += f"_ns{args.n_samples}"
    if args.batch_size != 128:
        exp_name += f"_bs{args.batch_size}"
    if args.k != 0.1:
        exp_name += f"_k{args.k}"
    return exp_name


def merge_over_two_ckpts(args):
    datasets = get_task_names(args.num_tasks)
    device = args.device

    base_model = ImageEncoder(args)
    base_model.eval()

    pre_state_dict = copy.deepcopy(base_model.state_dict())
    merged_state_dict = copy.deepcopy(base_model.state_dict())
    ft_state_dicts = load_finetuned_state_dicts(args, datasets)
    task_vectors = build_task_vectors(pre_state_dict, ft_state_dicts)
    del ft_state_dicts
    if args.merging_method == "emr":
        vector_unified, masks, rescalers = emr_merging(task_vectors)
    elif args.merging_method in ["tm_ta", "tm_ties"]:
        merged_tv, masks = tall_mask(args, task_vectors, pre_state_dict, 0.6)
    elif args.merging_method == "tsv":
        tsv_svd_dict = tsv_c(task_vectors, num_datasets=len(datasets))
    else:
        raise NotImplementedError
    del task_vectors
    torch.cuda.empty_cache()

    heads = build_classification_heads(args, datasets, device)
    base_model = base_model.to(device)

    exp_name = make_experiment_name(args)
    save_params_dir = os.path.join("params", exp_name, args.model, f"t{args.num_tasks}")
    os.makedirs(save_params_dir, exist_ok=True)
    fn_bank = "memory_bank"
    if args.tallmask_setting:
        fn_bank += "_tallmask"
    save_params_path = os.path.join(save_params_dir, f"{fn_bank}.pt")
    memory_bank = load_or_build_memory_bank(
        args, datasets, base_model, merged_state_dict, device, save_params_path
    )

    total = []
    for target_dataset_name in datasets:
        print("Evaluating on", target_dataset_name)

        result_variant = "tallmask" if args.tallmask_setting else "standard"
        save_result_dir = os.path.join(
            "results",
            result_variant,
            args.model,
            f"t{args.num_tasks}_{target_dataset_name}_{args.merging_method}",
            exp_name,
        )
        os.makedirs(save_result_dir, exist_ok=True)
        save_result_path = os.path.join(save_result_dir, f"{args.model}.json")

        dataset = get_dataset(
            target_dataset_name,
            base_model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        dataloader = get_dataloader(dataset, is_train=False, args=args)

        with torch.no_grad():
            correct = 0.0
            cls_last_correct = 0.0
            n = 0
            with tqdm(dataloader, unit=f"batch({args.batch_size})") as tepoch:
                for data in tepoch:
                    tepoch.set_description(target_dataset_name)

                    data = maybe_dictionarize(data)
                    x = data["images"].to(device)
                    y = data["labels"].to(device)
                    target_task_id = torch.zeros(x.size(0), device=device)
                    target_task_id[:] = datasets.index(target_dataset_name)

                    base_model.load_state_dict(merged_state_dict)
                    last_feature = base_model(x)
                    if args.normalize:
                        last_feature = last_feature / last_feature.norm(dim=-1, keepdim=True)

                    ts_logits = torch.zeros(x.size(0), len(datasets), device=device)
                    for task_id, dataset_name in enumerate(datasets):
                        ts_logit = residual_norm(
                            last_feature,
                            mean=memory_bank[dataset_name]["mean"],
                            Vk=memory_bank[dataset_name]["V"],
                        )
                        ts_logits[:, task_id] = ts_logit
                    ts_pred = ts_logits.argmin(dim=1, keepdim=True)

                    cls_last_correct += ts_pred.eq(target_task_id.view_as(ts_pred)).sum().item()

                    pred = torch.zeros(x.size(0), dtype=torch.long, device=device)
                    for task_id in range(len(datasets)):
                        grouped_x = x[ts_pred.squeeze(1) == task_id]
                        if grouped_x.size(0) == 0:
                            continue

                        if args.merging_method == "emr":
                            dyn_merged_params = OrderedDict()
                            for name in pre_state_dict.keys():
                                dyn_merged_params[name] = (
                                    pre_state_dict[name].to(device)
                                    + vector_unified[name].to(device)
                                    * masks[name][task_id].to(device)
                                    * rescalers[task_id].to(device)
                                )
                            base_model.load_state_dict(dyn_merged_params)
                        elif args.merging_method in ["tm_ta", "tm_ties"]:
                            dyn_merged_params = OrderedDict()
                            for name in pre_state_dict.keys():
                                dyn_merged_params[name] = (
                                    pre_state_dict[name].to(device)
                                    + merged_tv[name].to(device)
                                    * masks[name][task_id].to(device)
                                )
                            base_model.load_state_dict(dyn_merged_params)
                        elif args.merging_method == "tsv":
                            base_model.load_state_dict(
                                build_tsv_state_dict(
                                    task_id,
                                    tsv_svd_dict,
                                    pre_state_dict,
                                    device,
                                )
                            )
                        else:
                            raise NotImplementedError

                        last_feature = base_model(grouped_x)
                        if args.normalize:
                            last_feature = last_feature / last_feature.norm(dim=-1, keepdim=True)
                        logits = heads[target_dataset_name](last_feature)
                        pred[ts_pred.squeeze(1) == task_id] = logits.argmax(dim=1)

                    correct += pred.eq(y.view_as(pred)).sum().item()
                    n += y.size(0)
                    tepoch.set_postfix(acc=correct / n, cls_last=cls_last_correct / n)

            top1 = correct / n
            cls_top1_last = cls_last_correct / n
            total.append(top1)

        print(
            f"{target_dataset_name} Top-1 accuracy: {top1:.4f} "
            f"CLS last: {cls_top1_last:.4f}"
        )

        results = {
            "top1": top1,
            "cls_last": cls_top1_last,
        }
        with open(save_result_path, "w") as f:
            json.dump(results, f)
        print(f"Results saved to {save_result_path}.")

    print(f"mean: {sum(total) / len(total)}")


if __name__ == "__main__":
    args = parse_arguments_for_merge()
    set_random_seed(seed=args.seed)
    merge_over_two_ckpts(args)
