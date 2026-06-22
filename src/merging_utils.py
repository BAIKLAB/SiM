from collections import OrderedDict

import torch

from src.sim_utils import state_dict_to_vector, vector_to_state_dict


def compute_svd_and_compress(key, matrix, sv_reduction):
    u, s, v = torch.linalg.svd(matrix, full_matrices=False)
    reduced_index_s = int(s.shape[0] * sv_reduction)
    return key, u[:, :reduced_index_s], s[:reduced_index_s], v[:reduced_index_s, :]


def compress_tv(task_vectors, sv_reduction):
    with torch.no_grad():
        svd_dict = {}
        for task_key, task_vector in enumerate(task_vectors):
            svd_dict[task_key] = {}
            layer_dict = task_vector.vector if hasattr(task_vector, "vector") else task_vector
            for key, layer in layer_dict.items():
                if len(layer.shape) == 2:
                    _, u, s, v = compute_svd_and_compress(key, layer, sv_reduction)
                    svd_dict[task_key][key] = {"u": u, "s": s, "v": v}
                else:
                    svd_dict[task_key][key] = {"dim1": layer}
        return svd_dict


def tsv_c(task_vectors, num_datasets):
    return compress_tv(task_vectors, 1 / num_datasets)


def build_tsv_state_dict(task_id, svd_dict, pre_state_dict, device):
    task_svd = svd_dict[task_id]
    new_state_dict = OrderedDict()
    for key in pre_state_dict.keys():
        if "u" in task_svd[key]:
            delta = (
                task_svd[key]["u"]
                @ torch.diag_embed(task_svd[key]["s"])
                @ task_svd[key]["v"]
            ).to(device)
        else:
            delta = task_svd[key]["dim1"].to(device)
        new_state_dict[key] = pre_state_dict[key].to(device) + delta
    return new_state_dict


def generate_task_masks(
    tv_flat_checks,
    flat_ptm,
    tv=None,
    tall_mask_lambda=1.0,
):
    if tv is None:
        tv = tv_flat_checks.sum(0)

    tv_flat_checks = state_dict_to_vector(tv_flat_checks)
    tv = state_dict_to_vector(tv)
    mask = tv_flat_checks.abs() > (tv - tv_flat_checks).abs() * tall_mask_lambda
    return vector_to_state_dict(mask.float(), flat_ptm)


def topk_values_mask(matrix, K=0.7, return_mask=False):
    if K > 1:
        K /= 100
    original_shape = matrix.shape
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)

    _n, d = matrix.shape
    k = d - int(d * K)
    kth_values, _ = matrix.abs().kthvalue(k, dim=1, keepdim=True)
    mask = matrix.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == matrix.squeeze().shape else mask
    if return_mask:
        return matrix * final_mask, final_mask.float().mean(dim=1), final_mask
    return matrix * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(tensor):
    sign_to_mult = torch.sign(tensor.sum(dim=0))
    return resolve_zero_signs(sign_to_mult, "majority")


def disjoint_merge(tensor, merge_func, sign_to_mult):
    merge_func = merge_func.split("-")[-1]
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, tensor > 0, tensor < 0
        )
        selected_entries = tensor * rows_to_keep
    else:
        selected_entries = tensor * (tensor != 0)

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        return torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    if merge_func == "sum":
        return torch.sum(selected_entries, dim=0)
    if merge_func == "max":
        out = selected_entries.abs().max(dim=0)[0]
        return out * sign_to_mult
    raise ValueError(f"Merge method {merge_func} is not defined.")


def ties_merging(flat_task_checks, reset_thresh=None, merge_func=""):
    updated_checks, *_ = topk_values_mask(flat_task_checks.clone(), K=reset_thresh)
    final_signs = resolve_sign(updated_checks)
    return disjoint_merge(updated_checks, merge_func, final_signs)


def tall_mask(args, task_vectors, pre_state_dict, lamb):
    if args.merging_method == "tm_ta":
        merged_tv = {}
        for task_vector in task_vectors:
            for key, value in task_vector.items():
                if key not in merged_tv:
                    merged_tv[key] = value.clone()
                else:
                    merged_tv[key] += value
        sum_param_ties = merged_tv
    elif args.merging_method == "tm_ties":
        tv_flat_checks = torch.vstack([state_dict_to_vector(tv) for tv in task_vectors])
        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=20,
            merge_func="dis-sum",
        )
        sum_param_ties = vector_to_state_dict(merged_tv, pre_state_dict)
    else:
        raise NotImplementedError

    masks = {}
    for task_id, task_vector in enumerate(task_vectors):
        lamb_m = lamb[task_id] if isinstance(lamb, (list, tuple)) else lamb
        mask = generate_task_masks(task_vector, pre_state_dict, sum_param_ties, lamb_m)
        for name, value in mask.items():
            if task_id == 0:
                masks[name] = [value]
            else:
                masks[name].append(value)
    return sum_param_ties, masks


def emr_merging(task_vectors):
    device = next(iter(task_vectors[0].values())).device
    sum_param = {}
    n2p = []
    for task_vector in task_vectors:
        n2p.append(task_vector)
        for name in task_vector:
            if name not in sum_param:
                sum_param[name] = []
            sum_param[name].append(task_vector[name])
    sum_param = {key: torch.stack(value, 0).mean(0) for key, value in sum_param.items()}

    vector_unified = {}
    scales = torch.zeros(len(task_vectors), device=device)
    masks = {}
    for name in sum_param:
        masks[name] = []
        flag = (sum_param[name] > 0) * 2 - 1
        param_max = torch.zeros_like(n2p[0][name])
        for task_id, task_vector in enumerate(task_vectors):
            param = task_vector[name]
            mask = (param * flag) > 0
            masks[name].append(mask)
            param_abs = torch.abs(mask * param)
            param_max = torch.where(param_abs > param_max, param_abs, param_max)
            scales[task_id] += torch.mean(torch.abs(param))
        vector_unified[name] = param_max * flag

    new_scales = torch.zeros(len(task_vectors), device=device)
    for task_id in range(len(task_vectors)):
        for name in vector_unified:
            p = vector_unified[name] * masks[name][task_id]
            new_scales[task_id] += torch.mean(torch.abs(p))
    rescalers = scales / new_scales

    return vector_unified, masks, rescalers
