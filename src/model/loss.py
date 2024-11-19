import torch


def calc_mean_std(feat, eps=1e-5):
    """
    Calculate the channel-wise mean and standard deviation for a 4D tensor.
    """
    assert feat.ndim == 4, "Input should be a 4D tensor"
    mean = feat.mean(dim=(2, 3), keepdim=True)
    std = feat.std(dim=(2, 3), keepdim=True) + eps
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
    """
    Perform adaptive instance normalization (AdaIN) between content and style features.
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], \
        "Content and style features must have the same batch and channel dimensions"

    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean


def _calc_feat_flatten_mean_std(feat):
    """
    Compute the flattened feature, mean, and standard deviation for a 3D tensor.
    """
    assert feat.ndim == 3, "Input should be a 3D tensor (C, H, W)"
    feat_flatten = feat.view(feat.size(0), -1)  # Flatten spatial dimensions
    mean = feat_flatten.mean(dim=1, keepdim=True)
    std = feat_flatten.std(dim=1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(matrix):
    """
    Compute the square root of a positive semi-definite matrix using singular value decomposition (SVD).
    """
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U @ torch.diag(S.sqrt()) @ Vh


def coral(source, target):
    """
    Perform Correlation Alignment (CORAL) between source and target feature maps.
    """
    assert source.ndim == 3 and target.ndim == 3, \
        "Source and target should be 3D tensors (C, H, W)"
    assert source.size(0) == target.size(0) == 3, \
        "Both source and target must have 3 channels"

    # Flatten and normalize source
    source_flatten, source_mean, source_std = _calc_feat_flatten_mean_std(source)
    source_norm = (source_flatten - source_mean) / source_std
    source_cov_eye = source_norm @ source_norm.t() + torch.eye(source.size(0), device=source.device)

    # Flatten and normalize target
    target_flatten, target_mean, target_std = _calc_feat_flatten_mean_std(target)
    target_norm = (target_flatten - target_mean) / target_std
    target_cov_eye = target_norm @ target_norm.t() + torch.eye(target.size(0), device=target.device)

    # Transfer source covariance to match target
    transfer = _mat_sqrt(target_cov_eye) @ torch.linalg.inv(_mat_sqrt(source_cov_eye)) @ source_norm
    transferred_source = transfer * target_std + target_mean

    return transferred_source.view_as(source)
