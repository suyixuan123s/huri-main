import copy
import os
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import wandb
from policy.learner import Learner
from policy.actor import Actor, Eval
from policy.reanalyzer import Reanalyzer
import torch.multiprocessing as mp
from utils import SharedState
from huri.learning.utils import select_device
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, RackStatePlot
from huri.learning.method.APEX_DQN.replay import SharedReplayBuffer, SharedPERBuffer, HERSharedReplayBuffer, \
    SharedHERPERBuffer
import huri.core.file_sys as fs
from nn.ddqn import DuelingDQNCNN4, DuelingDQNCNN2, DuelingDQNTransformer, DDQN

# wandb server start
# os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'
os.environ['WANDB_API_KEY'] = 'local-6a592a197806eea7595235186fe87f4f0861b314'


def create_env(rack_sz, num_tube_class, seed=888, scheduler=None, toggle_curriculum=False, toggle_goal_fixed=False):
    observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(tuple(rack_sz))
    return RackArrangementEnv(rack_size=tuple(rack_sz),
                              num_classes=num_tube_class,
                              observation_space_dim=observation_space_dim,
                              action_space_dim=action_space_dim,
                              is_curriculum_lr=toggle_curriculum,
                              is_goalpattern_fixed=toggle_goal_fixed,
                              scheduler=scheduler,
                              seed=seed)


def create_agent(obs_dim, act_dim, n_classes, device):
    # return DuelingDQNCNN4(obs_dim=obs_dim, action_dim=act_dim, num_classes=n_classes).to(device)

    return DDQN(num_actions=act_dim,
                input_shape=(2, *obs_dim[1:]),
                num_filters=128,
                num_res_block=19).to(device)


def actor_init(aid: int,
               env: RackArrangementEnv,
               net,
               cfg,
               shared_net,
               shared_state,
               replay_buffer,
               logger=None,
               toggle_debug=False):
    return Actor(actor_id=aid,
                 net=net,
                 env=env.copy(),
                 cfg=cfg,
                 shared_net=shared_net,
                 shared_state=shared_state.get_dict(),
                 replay_buffer=replay_buffer,
                 logger=logger,
                 toggle_visual=toggle_debug)


def reanalyzer_init():
    pass


@hydra.main(config_path='params', config_name='20230517', version_base='1.3')
def main(cfg):
    # logging
    # wandb bug: fix(sdk): fix timeout in offline mode when trying to attach to an existing run
    # https://github.com/wandb/wandb/pull/5144
    save_path = fs.workdir_learning / "run" / f"dqn_debug"
    run = wandb.init(
        dir=str(save_path),
        # Set the project where this run will be logged
        project="dqn_learn_20230517",
        # Track hyperparameters and run metadata
        config=cfg)
    print(f"{'-' * 20}Start{'-' * 20}")
    print(f'Num actor: {cfg["num_actor"]}, ')
    # shared info / replay buffer
    shared_state = SharedState()
    shared_state['state_level'] = cfg['env']['init_state_level']
    shared_state['class_level'] = cfg['env']['init_class_level']
    shared_state['trajectory_list'] = []
    # replay buffer
    replay_buffer = SharedHERPERBuffer(obs_dim=list(cfg['env']['rack_sz']),
                                       size=int(cfg['rl']['replay_sz']),
                                       batch_size=int(cfg['rl']['batch_sz']),
                                       device=cfg['rl']['device'],
                                       beta_decay_steps=int(cfg['rl']['beta_decay_step']))
    # create env
    env_meta = create_env(rack_sz=cfg['env']['rack_sz'],
                          num_tube_class=cfg['env']['num_tube_class'],
                          seed=cfg['env']['seed'],
                          toggle_curriculum=cfg['env']['toggle_curriculum'],
                          toggle_goal_fixed=cfg['env']['toggle_goal_fixed'])
    env_meta.scheduler.state_level = cfg['env']['init_state_level']
    env_meta.scheduler.class_level = cfg['env']['init_class_level']
    # env_meta.set_goal_pattern(np.array([[0, 0, 0],
    #                                     [0, 1, 0],
    #                                     [1, 0, 0]]))
    net = create_agent(env_meta.observation_space_dim,
                       env_meta.action_space_dim,
                       n_classes=env_meta.num_classes,
                       device=cfg['rl']['device'])
    shared_net = copy.deepcopy(net)
    shared_net.eval()
    shared_net.share_memory()
    # actor
    actor_procs = []
    for i in range(cfg['num_actor']):
        env = env_meta.copy()
        env.set_seed(i * cfg['env']['seed'])
        actor_proc = actor_init(aid=i,
                                env=env.copy(),
                                net=copy.deepcopy(net),
                                cfg=cfg['rl'],
                                shared_net=shared_net,
                                shared_state=shared_state,
                                replay_buffer=replay_buffer,
                                logger=run,
                                toggle_debug=cfg['toggle_actor_debug'])
        actor_proc.start()
        actor_procs.append(actor_proc)
    # reanalyzer
    reanlyz_procs = []
    for i in range(cfg['num_reanalyzer']):
        env = env_meta.copy()
        env.set_seed(2000 + i * cfg['env']['seed'])
        reanlyz_proc = Reanalyzer(uid=i,
                                  env=env.copy(),
                                  shared_state=shared_state.get_dict(),
                                  replay_buffer=replay_buffer,
                                  toggle_visual=cfg['toggle_reanalyzer_debug'],
                                  toggle_her_replay=cfg['reanalyzer']['toggle_her_replay'],
                                  logger=run)
        reanlyz_proc.start()
        reanlyz_procs.append(reanlyz_proc)
    # eval
    eval_proc = Eval(net=copy.deepcopy(net),
                     env=env_meta.copy(),
                     cfg=cfg['eval'],
                     shared_net=shared_net,
                     shared_state=shared_state.get_dict(),
                     save_path=save_path,
                     logger=run)
    eval_proc.start()
    # Learner init
    # it has a bug: spawn will make neural network parameters loss!
    net = create_agent(env_meta.observation_space_dim,
                       env_meta.action_space_dim,
                       n_classes=env_meta.num_classes,
                       device=cfg['rl']['device'])
    learner = Learner(env_meta.copy(),
                      net,
                      shared_net,
                      shared_state,
                      replay_buffer,
                      cfg['rl'],
                      save_path=save_path, )
    learner.train(1000000000000000000000, min_replay_sz=1000)


if __name__ == "__main__":
    main()
    while True:
        time.sleep(1)
