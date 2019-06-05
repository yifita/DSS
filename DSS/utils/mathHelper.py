import torch
import math
import unittest


def dot(tensor1, tensor2, dim=0):
    """
    computes the dot product along a dimension dim
    """
    elem = tensor1.mul(tensor2)
    return elem.sum(dim)


def div(tensor1, tensor2, dim=0):
    """
    tensor1[j,:] / tensor2[j] for i-th dimension dim
    """
    tensor2 = tensor2.unsqueeze(dim)
    return torch.div(tensor1, tensor2)


def mm(t1, t2):
    """
    broadcasted matrix multiplication
    """
    return torch.matmul(t1, t2)
    assert(t1.device == t2.device)
    R = torch.empty(t1.size(), device=t1.device)
    R[:, 0, 0] = t1[:, 0, 0]*t2[:, 0, 0] + t1[:, 0, 1]*t2[:, 1, 0]
    R[:, 0, 1] = t1[:, 0, 0]*t2[:, 0, 1] + t1[:, 0, 1]*t2[:, 1, 1]
    R[:, 1, 0] = t1[:, 1, 0]*t2[:, 0, 0] + t1[:, 1, 1]*t2[:, 1, 0]
    R[:, 1, 1] = t1[:, 1, 0]*t2[:, 0, 1] + t1[:, 1, 1]*t2[:, 1, 1]
    return R


def mul(tensor1, tensor2, dim=0):
    """
    tensor1[j,:] * tensor2[j] for i-th dimension
    """
    tensor2 = tensor2.unsqueeze(dim)
    return torch.mul(tensor1, tensor2)


def normalize(tensor, dim=-1):
    return div(tensor, torch.norm(tensor, dim=dim)+1e-8, dim)

def det22(tensor):
    """
    computes the determinant of a list of 2x2 matrices
    """
    return tensor[:, 0, 0]*tensor[:, 1, 1] - tensor[:, 0, 1]*tensor[:, 1, 0]


def inverse22(t):
    """
    computes the inverses of a list of 2x2 matrices
    """
    # return t.inverse()
    # Here the special case implementation seems faster
    detInv = 1/det22(t)
    R = torch.empty(t.size(), device=t.device)
    R[:, 0, 0] = t[:, 1, 1] * detInv
    R[:, 0, 1] = -t[:, 0, 1] * detInv
    R[:, 1, 0] = -t[:, 1, 0] * detInv
    R[:, 1, 1] = t[:, 0, 0] * detInv
    return R


def inverse33(A):
    """
    Computes the inverses of a list of 3x3 matrices
    """
    return A.inverse()


if __name__ == '__main__':
    unittest.main()
