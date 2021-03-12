import torch
from collections import namedtuple

__all__ = ['BaseGenerator', 'PointModel']


class BaseGenerator(object):
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    def generate_meshes(self, *args, **kwargs):
        return []

    def generate_pointclouds(self, *args, **kwargs):
        return []

    def generate_images(self, *args, **kwargs):
        return []

from .point_modeling import Model as PointModel
