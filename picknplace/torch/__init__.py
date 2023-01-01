from picknplace.torch.td3.train import train as train_td3
from picknplace.torch.td3.optimize import optimize_hyparams as optimize_td3
from picknplace.torch.ddpg.train import train as train_ddpg
from picknplace.torch.ddpg.optimize import optimize_hyparams as optimize_ddpg
from picknplace.torch.ddpg_rnd.train import train as train_ddpg_rnd
from picknplace.torch.ddpg_rnd.optimize import optimize_hyparams as optimize_ddpg_rnd


TRAIN_FNS = {
    "td3": train_td3,
    "ddpg": train_ddpg,
    "ddpg_rnd": train_ddpg_rnd
}


OPTIMIZE_FNS = {
    "td3": optimize_td3,
    "ddpg": optimize_ddpg,
    "ddpg_rnd": optimize_ddpg_rnd
}
