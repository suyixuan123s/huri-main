import torch
import numpy as np
from typing import List, Tuple, Mapping, Union, Text, Any
from huri.learning.method.AlphaZero.mcts import Node, parallel_uct_search, uct_search


def create_mcts_player(
        network: torch.nn.Module,
        device: torch.device,
        num_simulations: int,
        num_parallel: int,
        root_noise: bool = False,
        deterministic: bool = False,
):
    """Give a network and device, returns a 'act' function to act on the specific environment."""

    @torch.no_grad()
    def eval_func(state_tensor: np.ndarray, batched: bool = False) -> Tuple[np.ndarray, Union[float, np.ndarray]]:
        """Give a game state tensor, returns the action probabilities
        and estimated winning probability from current player's perspective."""

        if not batched:
            state_tensor = state_tensor[None, ...]

        state = torch.from_numpy(state_tensor).to(device=device, dtype=torch.float32, non_blocking=True)
        output = network(state)
        pi_prob = torch.softmax(output.pi_logits, dim=-1).cpu().numpy()
        value = torch.detach(output.value).cpu().numpy()

        if not batched:
            # Remove batch dimensions
            pi_prob = np.squeeze(pi_prob, axis=0)
            value = np.squeeze(value, axis=0)

            # Convert value into float.
            value = value.item()

        return (pi_prob, value)

    def act(
            env: 'RackArrangementEnv',
            root_node: Node,
            c_puct_base: float,
            c_puct_init: float,
            temperature: float,
    ):
        if num_parallel > 1:
            return parallel_uct_search(
                env=env,
                eval_func=eval_func,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                temperature=temperature,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=root_noise,
                deterministic=deterministic,
            )
        else:
            return uct_search(
                env=env,
                eval_func=eval_func,
                root_node=root_node,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                temperature=temperature,
                num_simulations=num_simulations,
                root_noise=root_noise,
                deterministic=deterministic,
            )

    return act


if __name__ == '__main__':
    pass
