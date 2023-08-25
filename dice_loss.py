import torch
from torch.autograd import Function
from hausdorff import hausdorff_distance


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def avg_iou_score(input, target):
    assert input.shape == target.shape
    assert input.shape[1] == 1
    num = input.shape[0]
    score = 0.0
    for i in range(num):
        score += get_iou(input[i,0,:,:], target[i,0,:,:])
    return score / num

def get_iou(input, target):
    assert input.shape == target.shape
    input = input.long()
    target = target.long()
    inter = torch.sum(input & target)
    sec = torch.sum(input | target)
    num_min = 1e-6
    return (inter + num_min) / (sec + num_min)

def hd95_score(input, target):
    assert input.shape == target.shape
    assert input.shape[1] == 1
    num = input.shape[0]
    score = 0.0
    for i in range(num):
        input_item = input[i,0,:,:].cpu().numpy()
        target_item = target[i,0,:,:].cpu().numpy()
        score += hausdorff_distance(input_item, target_item)
    return score / num