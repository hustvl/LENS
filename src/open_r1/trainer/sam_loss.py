import torch
import torch.nn.functional as F

def dice_loss(inputs, targets, num_objects, loss_on_multimask=False, ignore_index=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        Dice loss tensor
    """
    inputs = inputs.sigmoid()
    # if ignore_index is not None:
    #     valid_mask = targets != ignore_index
    if loss_on_multimask:
        # inputs and targets are [N, M, H, W] where M corresponds to multiple predicted masks
        assert inputs.dim() == 4 and targets.dim() == 4
        # flatten spatial dimension while keeping multimask channel dimension
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def sigmoid_bce_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
    ignore_index=None
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_objects: Number of objects in the batch
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        loss_on_multimask: True if multimask prediction is enabled
    Returns:
        focal loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if loss_on_multimask:
        # loss is [N, M, H, W] where M corresponds to multiple predicted masks
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects  # average over spatial dims
    return loss.mean(1).sum() / num_objects

def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch
        loss_on_multimask: True if multimask prediction is enabled
        use_l1_loss: Whether to use L1 loss is used instead of MSE loss
    Returns:
        IoU loss tensor
    """
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious.float(), actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious.float(), actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects

def calculate_boundary_iou(mask1, mask2):
    assert mask1.dim() == 4 and mask2.dim() == 4
    assert mask1.shape == mask2.shape
    assert mask1.size(1)==1

    mask2 = mask2.to(mask1.device)
    
    # 扩充掩码边界以抵消卷积过程中边界效应
    boundary1 = mask2boundary(mask1)
    boundary2 = mask2boundary(mask2)

    # 转换为二值边界(mask>0得到边界)
    boundary1 = (boundary1 > 0).squeeze(0)
    boundary2 = (boundary2 > 0).squeeze(0)

    # 计算交集和并集
    boundary_intersection = boundary1 & boundary2
    boundary_union = boundary1 | boundary2
    
    # 计算交集和并集的面积
    intersection_area = boundary_intersection.sum().float()
    union_area = boundary_union.sum().float()
    
    # 如果并集面积为0，返回特殊值
    if union_area == 0:
        return torch.tensor(1.0 if intersection_area == 0 else 0.0)
    
    # 计算BIoU
    biou = intersection_area / union_area
    return biou, boundary1, boundary2

def mask2boundary(bin_mask):
    assert bin_mask.max()<=1
    assert bin_mask.min()>=0
    bin_mask = bin_mask.float()
    boudary_width=min(bin_mask.shape[-2:])//20
    if boudary_width%2 ==0:
        boudary_width+=1
    inner_area = 1-F.max_pool2d(1-bin_mask, boudary_width, 1, boudary_width//2)
    boundary_area = (bin_mask - inner_area)>0.5
    return boundary_area
