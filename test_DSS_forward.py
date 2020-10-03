import torch
import unittest
from fvcore.common.benchmark import benchmark

from DSS.training.trainer import Trainer
from DSS.models import PointModel
from DSS.utils import tolerating_collate

import config


class TestDSSForward(unittest.TestCase):
  @staticmethod
  def nn(cfg_fname):
    cfg = config.load_config(cfg_fname, 'configs/default.yaml')
    train_dataset = config.create_dataset(cfg.data, mode='train')
    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=cfg['training']['batch_size'],
      collate_fn=tolerating_collate
    )
    device = torch.device('cuda')
    model = config.create_model(cfg, dataset=train_dataset, device=device)
    cameras = train_dataset.get_cameras()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=cfg['training']['batch_size'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, cfg['training']['scheduler_milestones'],
      gamma=cfg['training']['scheduler_gamma'], last_epoch=-1)
    generator = config.create_generator(cfg, model, device=device)
    trainer = config.create_trainer(
      cfg, model, optimizer, scheduler, generator, device=device)
    it = -1
    trainer.model.train()
    data_list = []
    cameras_list = []
    for batch in train_loader:
      data, cameras_cuda = trainer.process_data_dict(batch, cameras) 
      data_list.append(data)
      cameras_list.append(cameras_cuda)
      break
    torch.cuda.synchronize()

    def output():
      with torch.no_grad():
        for data, cameras_cuda in zip(data_list, cameras_list):
          trainer.compute_loss(data, cameras, it=0)
      torch.cuda.synchronize()
    
    return output

def bm_frnn():
  benchmark(TestDSSForward.nn, "frnn", [{"cfg_fname":"configs/donut_dss_frnn.yml"}], num_iters=5, warmup_iters=1)
  benchmark(TestDSSForward.nn, "knn", [{"cfg_fname":"configs/donut_dss_knn.yml"}], num_iters=5, warmup_iters=1)

if __name__ == "__main__":
  bm_frnn()




        

    
    


