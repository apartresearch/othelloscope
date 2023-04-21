# Import libraries for reading and writing files
import os
import sys
from typing import Tuple

# Import libraries for image processing
import numpy as np

# Import stuff
import torch
from torch import Tensor
from torch.nn import functional
import einops
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from othelloscope import html, calculations

import transformer_lens.utils as utils
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)

torch.set_grad_enabled(False)


OTHELLO_ROOT = Path(".")
# Import othello util functions from file mechanistic_interpretability/mech_interp_othello_utils.py
sys.path.append(str(OTHELLO_ROOT))
sys.path.append(str(OTHELLO_ROOT / "mechanistic_interpretability"))
from mech_interp_othello_utils import (
    OthelloBoardState,
)

USE_CUDA: bool = torch.cuda.is_available()
DEVICE: str = "cuda" if USE_CUDA else "cpu"


def to_device(object):
    if USE_CUDA:
        return object.cuda()
    else:
        return object.cpu()


# Creates a one dimensional tensor of size `num_classes` with the specified indices set to 1.0
def one_hot(list_of_ints: list[int], num_classes: int = 64) -> torch.Tensor:
    out = torch.zeros((num_classes,), dtype=torch.float32)
    out[list_of_ints] = 1.0
    return out

def state_stack_to_one_hot(state_stack: Tensor) -> Tensor:
    """Convert a state stack to one hot encoding.

    Parameters
    ----------
    state_stack : Tensor
        The state stack.

    Returns
    -------
    Tensor
        The one hot encoding.
    """
    one_hot = torch.zeros(
        state_stack.shape[0],  # num games
        state_stack.shape[1],  # num moves
        8,  # rows
        8,  # cols
        3,  # the two options
        device=state_stack.device,
        dtype=torch.int,
    )
    one_hot[..., 0] = state_stack == 0  # empty
    one_hot[..., 1] = state_stack == -1  # white
    one_hot[..., 2] = state_stack == 1  # black

    return one_hot


def main():
    """Main function."""

    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model: HookedTransformer = to_device(HookedTransformer(cfg))

    sd = utils.download_file_from_hf(
        "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
    )
    model.load_state_dict(sd)

    # Sequences of actions taken in game
    # Shape: (num_games, length_of_game)
    board_seqs_int = torch.tensor(
        np.load(OTHELLO_ROOT / "board_seqs_int_small.npy"), dtype=torch.long
    )
    # Shape: (num_games, length_of_game)
    board_seqs_string = torch.tensor(
        np.load(OTHELLO_ROOT / "board_seqs_string_small.npy"), dtype=torch.long
    )
    print("board_seqs_int:", board_seqs_int[0])
    print("board_seqs_string:", board_seqs_string[0])

    num_games, length_of_game = board_seqs_int.shape
    print(
        "Number of games:",
        num_games,
    )
    print("Length of game:", length_of_game)

    num_games = 50
    focus_games_int = board_seqs_int[:num_games]
    focus_games_string = board_seqs_string[:num_games]

    focus_states = np.zeros((num_games, 60, 8, 8), dtype=np.float32)
    focus_valid_moves = torch.zeros((num_games, 60, 64), dtype=torch.float32)
    for i in range(num_games):
        board = OthelloBoardState()
        for j in range(60):
            board.umpire(focus_games_string[i, j].item())
            focus_states[i, j] = board.state
            focus_valid_moves[i, j] = one_hot(board.get_valid_moves())
    print("focus states:", focus_states.shape)
    print("focus_valid_moves", focus_valid_moves.shape)

    focus_logits, focus_cache = model.run_with_cache(to_device(focus_games_int[:, :-1]))
    print("focus cache post:", focus_cache["post", 0].shape)

    full_linear_probe = torch.load(
        OTHELLO_ROOT / "main_linear_probe.pth", map_location=DEVICE
    )
    rows = 8

    cols = 8
    options = 3
    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2
    linear_probe = torch.zeros(cfg.d_model, rows, cols, options, device=DEVICE)
    linear_probe[..., blank_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 0]
        + full_linear_probe[white_to_play_index, ..., 0]
    )
    linear_probe[..., their_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 1]
        + full_linear_probe[white_to_play_index, ..., 2]
    )
    linear_probe[..., my_index] = 0.5 * (
        full_linear_probe[black_to_play_index, ..., 2]
        + full_linear_probe[white_to_play_index, ..., 1]
    )

    print("Computing accuracy over 50 games")
    # We first convert the board states to be in terms of my (+1) and their (-1)
    alternating = np.array(
        [-1 if i % 2 == 0 else 1 for i in range(focus_games_int.shape[1])]
    )
    flipped_focus_states = focus_states * alternating[None, :, None, None]

    # We now convert to one hot
    focus_states_flipped_one_hot = state_stack_to_one_hot(
        to_device(torch.tensor(flipped_focus_states))
    )

    # Take the argmax
    focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1)
    probe_out = einops.einsum(
        to_device(focus_cache["resid_post", 6]),
        linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    probe_out_value = probe_out.argmax(dim=-1)

    correct_middle_answers = (
        to_device(probe_out_value) == focus_states_flipped_value[:, :-1]
    )[:, 5:-5]
    accuracies = einops.reduce(
        correct_middle_answers.float(), "game move row col -> row col", "mean"
    )
    print("Accuracy over 50 games:", accuracies.mean().item())

    blank_probe = (
        linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
    )
    my_probe = linear_probe[..., 2] - linear_probe[..., 1]

    game_index = 0

    print("\nACTIVATE NEURON REPRESENTATION\n")
    # Scale the probes down to be unit norm per feature
    blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
    my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
    print("my_probe:", my_probe.shape)
    # Set the center blank probes to 0, since they're never blank so the probe is meaningless
    blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    layer = 5
    neuron = 1393
    w_in = model.blocks[layer].mlp.W_in[:, neuron].detach()
    w_in /= w_in.norm()
    w_out = model.blocks[layer].mlp.W_out[neuron, :].detach()
    w_out /= w_out.norm()

    U, S, Vh = torch.svd(
        torch.cat(
            [my_probe.reshape(cfg.d_model, 64), blank_probe.reshape(cfg.d_model, 64)],
            dim=1,
        )
    )
    # Remove the final four dimensions of U, as the 4 center cells are never blank and so the blank probe is meaningless there
    probe_space_basis = U[:, :-4]

    print(
        "Fraction of input weights in probe basis:",
        (w_in @ probe_space_basis).norm().item() ** 2,
    )
    print(
        "Fraction of output weights in probe basis:",
        (w_out @ probe_space_basis).norm().item() ** 2,
    )

    # Calculate all heatmaps
    heatmaps_blank, heatmaps_my = calculations.calculate_heatmaps(
        model, blank_probe_normalised, my_probe_normalised
    )

    attributions = calculations.calculate_logit_attributions(model)

    print(type(heatmaps_my))
    print(heatmaps_my.shape)

    # Calculate heatmap standard deviations for each neuron
    heatmaps_my_sd = heatmaps_my.detach().std(dim=(2, 3))

    heatmaps_my_sd = heatmaps_my_sd.detach().cpu().numpy()

    # Sort neuron indices by standard deviation
    print("Sorting neurons by standard deviation...")
    variance_ranks = []
    variance_sorted_neurons = []
    for heatmap_my_sd in heatmaps_my_sd:
        neuron_indices_my = list(enumerate(heatmap_my_sd))
        neuron_indices_my.sort(
            reverse=True,
            key=lambda x: x[1],
        )
        variance_sorted_neurons.append([x[0] for x in neuron_indices_my])

        layer_variance_ranks = np.zeros(len(neuron_indices_my), dtype=np.int32)
        for neuron_index, (rank, _) in enumerate(neuron_indices_my):
            layer_variance_ranks[neuron_index] = rank

        assert len([x for x in layer_variance_ranks if x == 0]) == 1
        variance_ranks.append(layer_variance_ranks)

    output_path = "othelloscope/output"

    html.generate_main_index(output_path, variance_sorted_neurons)

    print("Generating neuron pages...")
    # Generate file for each neuron.
    html.generate_neuron_pages(
        output_path,
        heatmaps_blank,
        heatmaps_my,
        attributions,
        variance_ranks,
        focus_cache,
        board_seqs_int,
    )


# Make an 8x8 html table from a numpy array
# table = generate_activation_table(3, 1123)

# # Read the template file
# template = generate_from_template(
#     "othelloscope/index.html",
#     3,
#     1123,
#     "Doe",
# )

# # Create a folder if it doesn't exist
# if not os.path.exists("othelloscope/test"):
#     os.makedirs("othelloscope/test")

# # Write the generated file
# with open("othelloscope/test/test.html", "w") as f:
#     f.write(template)


if __name__ == "__main__":
    main()
