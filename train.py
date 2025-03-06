"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import pdb
import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.joinpath(
#         'diffusion_policy','config'))
# )
@hydra.main(
    version_base=None,
    config_path='diffusion_policy/config',
    config_name='train_diffusion_unet_lowdim_workspace.yaml'
)
def main(cfg: DictConfig):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    print("cfg", cfg)
    OmegaConf.resolve(cfg)


    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
