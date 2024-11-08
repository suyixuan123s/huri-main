import numpy as np
import collections
import torch
from tqdm import tqdm
import logging
import multiprocessing as mp
from huri.learning.env.arrangement_planning_rack.env import RackState, RackArrangementEnv
import huri.learning.method.model_based.net as huri_net
import huri.core.file_sys as fs
import datetime

ACTION_SPACE_DIM = 2500
huri_net.ACTION_SPACE_DIM = ACTION_SPACE_DIM
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


class UCTNode():
    env = None
    lr = None

    def __init__(self, state: RackState, move, reward=0, parent=None):
        # node
        self.state = state  # state s
        self.parent = parent

        # edge resulting to the node
        self.reward = reward
        self.move = move  # action index
        self.is_expanded = False

        # children node information
        self.children = {}
        self.child_priors = np.zeros([ACTION_SPACE_DIM], dtype=np.float32)
        self.child_Q = np.zeros([ACTION_SPACE_DIM], dtype=np.float32)
        self.child_number_visits = np.zeros([ACTION_SPACE_DIM], dtype=np.float32)
        self.action_idxes = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def V_s(self):
        """
        V(s): V_s of the current state
        V(s) = argmax_a'{Q(s,a')}
        """
        return np.max(self.child_Q[self.action_idxes])

    def set_V(self, v):
        """
        Q(s,a) = r + yV(s), s is the current state, a is the action
        :param v : children state V(s)
        """
        self.parent.child_Q[self.move] = self.lr * v + self.reward

    def set_V_end(self):
        """
        Q(s,a) = r, if done
        """
        self.parent.child_Q[self.move] = self.reward

    def child_U(self, c_puct=5):
        return c_puct * np.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if len(self.action_idxes) != 0:
            bestmove = self.child_Q + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q + self.child_U())
        return bestmove

    def select_leaf(self):
        current = self
        # print('root', end='')
        while current.is_expanded:
            best_move = current.best_child()
            # print(f"->{best_move}", end='')
            current = current.maybe_add_child(best_move)
        # print()
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs]  # select only legal moves entries in child_priors array
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], \
                                                                                             dtype=np.float32) + 192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        """
        To figure out 1. feasible actions for the current leaf node
                      2. move probabilities for each feasible action
        """
        self.is_expanded = True
        action_inds = self.state.feasible_action_set
        c_p = child_priors
        if len(action_inds) == 0:
            self.is_expanded = False
        self.action_idxes = action_inds
        _mask = np.zeros_like(c_p)
        _mask[action_inds] = 1
        c_p = _mask * c_p
        if self.parent.parent == None:  # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_inds, c_p)
        # TODO should we make the sum of the probability to 1
        self.child_priors = c_p

    def maybe_add_child(self, move):
        if move not in self.children:
            self.env.reset_state(self.state)
            state_moved, reward, is_finished, _ = self.env.step(move)
            self.children[move] = UCTNode(state=state_moved, move=move, reward=reward, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float, is_done: bool = False):
        """
        Backpropagate the 1. N: visit times
                          2. W: total values
        """
        current = self
        current.number_visits += 1
        if not is_done:
            current.set_V(value_estimate)
        else:
            current.set_V_end()
        # print(f'r:{current.reward}, m: {current.move}', end='->')
        # try:
        #     print(current.parent.child_Q[[current.parent.action_idxes]])
        # except:
        #     pass
        current = current.parent
        while current.parent is not None:
            current.number_visits += 1
            current.set_V(current.V_s)
            # print(f'r:{current.reward}, m: {current.move}, v: {current.V_s}', end='->')
            # try:
            #     print(current.parent.child_Q[[current.parent.action_idxes]])
            # except:
            #     pass
            current = current.parent
        # print()
        # print("-" * 20)


class DummyNode(object):
    def __init__(self, lr):
        self.parent = None
        self.child_Q = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.lr = lr

    def set_Q(self, a, child_v, rew):
        """
        Q(s,a) = r + yV(s), s is the current state, a is the action
        :param a : action
        :param child_v : children state V(s)
        :param rew: reward of action a
        """
        self.child_Q[a] = self.lr * child_v + rew


def UCT_search(state: RackState, iter_num: int, net, env: RackArrangementEnv, lr=.98):
    root = UCTNode(state, move=None, parent=DummyNode(lr))
    UCTNode.lr = lr
    UCTNode.env = env
    for i in range(iter_num):
        leaf = root.select_leaf()
        encoded_s = leaf.state
        encoded_g = env.goal_pattern.state
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        encoded_g = torch.from_numpy(encoded_g).float().cuda()
        encoded_sg = torch.zeros((2, encoded_s.shape[0], encoded_s.shape[1])).cuda()
        encoded_sg[0, :, :] = encoded_s
        encoded_sg[1, :, :] = encoded_g
        child_priors, value_estimate = net(encoded_sg)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        if env.is_finished(leaf.state) or len(leaf.state.feasible_action_set) == 0:  # if somebody won or draw
            leaf.backup(value_estimate, is_done=True)
            continue
        leaf.expand(child_priors)  # need to make sure valid moves
        leaf.backup(value_estimate)
    return root


def get_policy(root, temp=1):
    # policy = np.zeros([7], dtype=np.float32)
    # for idx in np.where(root.child_number_visits!=0)[0]:
    #    policy[idx] = ((root.child_number_visits[idx])**(1/temp))/sum(root.child_number_visits**(1/temp))
    return ((root.child_number_visits) ** (1 / temp)) / sum(root.child_number_visits ** (1 / temp))


def MCTS(env: RackArrangementEnv, rangenet, lr=.98, train_num=100, start_idx=0, cpu=0, iteration=0):
    logger.info("[CPU: %d]: Starting MCTS searching..." % cpu)

    # create the folder to store the data generated by MCTS
    if not (fs.workdir_learning / f"datasets/iter_{iteration}").is_dir():
        if not (fs.workdir_learning / "datasets").is_dir():
            (fs.workdir_learning / "datasets").mkdir()
        (fs.workdir_learning / f"datasets/iter_{iteration}").mkdir()

    # generate $train_num data
    for idxx in tqdm(range(start_idx, train_num + start_idx)):
        logger.info("[CPU: %d]: Search %d" % (cpu, idxx))
        env_state = env.reset()
        dataset = []  # to get state, policy, value for neural network training
        value = [0]
        move_count = 0
        while len(env_state.feasible_action_set) > 0 and not env.is_finished(env_state):
            # simulation
            rack_state = env_state.state.copy()
            env_cp = env.copy()
            root = UCT_search(env_state, 1000, rangenet, env_cp, lr=lr)
            del env_cp
            policy = get_policy(root)
            # print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idxx), policy)
            # Do the step
            env_state, r, is_finished, _ = env.step(env._np_random.choice(np.arange(ACTION_SPACE_DIM), p=policy))
            dataset.append([rack_state, policy])
            print(f"[Iteration: %d CPU: {cpu}]: Search {idxx} CURRENT BOARD:\n{env_state}, GOAL:\n{env.goal_pattern}")
            value.append(r)
            if is_finished:
                dataset.append([env_state.state, policy])
            move_count += 1
        print(f"VALUE IS {value}")
        dataset_p = []
        goal_pattern = env.goal_pattern
        value_np = np.array(value)
        for idx, data in enumerate(dataset):
            if idx == len(dataset) - 1:
                dataset_p.append([*data, 0])
            else:
                dataset_p.append([*data, np.sum(value_np[idx:])])
        del dataset
        fs.dump_pickle([goal_pattern, dataset_p], fs.workdir_learning / f"datasets/iter_{iteration}" /
                       f"dataset_iter{iteration}_cpu{cpu}_{idxx}_{datetime.datetime.today().strftime('%Y-%m-%d')}")


def run_MCTS(args, start_idx=0, iteration=0):
    net_to_play = "%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    net = huri_net.RangeNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()

    # init environment
    rack_size = (5, 10)
    num_classes = 2
    observation_space_dim = (num_classes * 2, *rack_size)
    action_space_dim = np.prod(rack_size) ** 2
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True)

    if args.MCTS_num_processes > 1:
        logger.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn", force=True)
        net.share_memory()
        net.eval()

        # save the model file
        current_net_filename = fs.workdir_learning / "network" / "model_data" / f"{net_to_play}"
        if current_net_filename.is_file():
            checkpoint = torch.load(str(current_net_filename))
            net.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded {current_net_filename} model.")
        else:
            torch.save({'state_dict': net.state_dict()}, str(current_net_filename))
            logger.info("Initialized model.")

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info(
                "Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes

        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=MCTS,
                               args=(
                                   env.copy(), net, start_idx, args.lr, args.num_games_per_MCTS_process, start_idx, i,
                                   iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("Finished multi-process MCTS!")

    elif args.MCTS_num_processes == 1:
        logger.info("Preparing model for MCTS...")
        net.eval()

        current_net_filename = fs.workdir_learning / "network" / "model_data" / f"{net_to_play}"

        if current_net_filename.is_file():
            checkpoint = torch.load(str(current_net_filename))
            net.load_state_dict(checkpoint['state_dict'])
            logger.info(f"Loaded {current_net_filename} model.")
        else:
            torch.save({'state_dict': net.state_dict()}, str(current_net_filename))
            logger.info("Initialized model.")

        with torch.no_grad():
            MCTS(env, net, lr=args.lr, train_num=args.num_games_per_MCTS_process, start_idx=start_idx, cpu=0,
                 iteration=iteration)
        logger.info("Finished MCTS!")


if __name__ == "__main__":
    pass
