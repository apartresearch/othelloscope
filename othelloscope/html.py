import os

from torch import Tensor
import numpy as np
import einops

from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)


from othelloscope.mech_interp_othello_utils import (
    int_to_label,
)

from othelloscope import templating


def generate_neuron_path(layer_index: int, neuron_index: int):
    """Generate the path to the neuron.

    Parameters
    ----------
    layer : int
        The layer of the neuron.
    neuron : int
        The index of the neuron.

    Returns
    -------
    str
        The path to the neuron.
    """
    return "othelloscope/L{0}/N{1}".format(layer_index, neuron_index)


def generate_activation_table(heatmap: Tensor) -> str:
    """Generate an activation table.

    Parameters
    ----------
    layer : int
        The layer of the neuron.
    neuron : int
        The index of the neuron.

    Returns
    -------
    str
        The generated activation table.
    """
    # Convert heatmap to numpy array
    heatmap = np.array(heatmap.detach().cpu())
    othello_board = np.array(
        [
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            ["1", "2", "3", "4", "5", "6", "7", "8"],
        ]
    )

    # Create a table
    table = "<table>"

    # Loop through the rows
    for row in range(heatmap.shape[0]):
        table += "<tr>"

        # Loop through the columns
        for col in range(heatmap.shape[1]):
            table += "<td title={0}>{1}</td>".format(
                heatmap[row, col], othello_board[0, row] + othello_board[1, col]
            )

        table += "</tr>"

    table += "</table>"

    return table


def generate_neuron_pages(
    heatmaps_blank: Tensor,
    heatmaps_my: Tensor,
    attributions: Tensor,
    variance_ranks: list[list[int]],
    focus_cache: dict[str, Tensor],
    board_seqs_int: Tensor,
):
    """Generates pages for all neurons based on precomputed heatmaps.

    Parameters
    ----------
    model : HookedTransformer
        The model.
    layer : int
        The layer to generate the probe for.
    focus_cache : dict
        The focus cache.
    blank_probe_normalised : Tensor
        The normalised blank probe.
    my_probe_normalised : Tensor
        The normalised my probe.

    Returns
    -------
    None
    """

    for layer_index, (
        heatmaps_blank,
        heatmaps_my,
        attributions,
        layer_variance_ranks,
    ) in enumerate(zip(heatmaps_blank, heatmaps_my, attributions, variance_ranks)):
        print(f"Generating pages for neurons in layer {layer_index}")
        generate_neuron_pages_for_layer(
            layer_index,
            heatmaps_blank,
            heatmaps_my,
            attributions,
            layer_variance_ranks,
            focus_cache,
            board_seqs_int,
        )


def generate_neuron_pages_for_layer(
    layer_index: int,
    heatmaps_blank: Tensor,
    heatmaps_my: Tensor,
    attributions: Tensor,
    ranks: list[int],
    focus_cache: dict[str, Tensor],
    board_seqs_int: Tensor,
):
    for neuron_index, (heatmap_blank, heatmap_my, attributions, rank) in enumerate(
        zip(heatmaps_blank, heatmaps_my, attributions, ranks)
    ):
        games = top_50_games(layer_index, neuron_index, focus_cache, board_seqs_int)
        generate_page(
            layer_index,
            neuron_index,
            rank,
            heatmap_blank,
            heatmap_my,
            attributions,
            games,
        )


# Generate the page for a specific neuron
def generate_page(
    layer_index: int,
    neuron_index: int,
    rank: int,
    heatmap_blank: Tensor,
    heatmap_my: Tensor,
    attributions: Tensor,
    games: str,
):
    """Generate a page."""

    # Get the path to the neuron
    path = generate_neuron_path(layer_index, neuron_index)

    # Create a folder if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Read the template file
    template = templating.generate_from_template(
        "othelloscope/template.html",
        (
            f"<a href='../../L{layer_index}/N{neuron_index - 1}/index.html'>Previous neuron</a> - "
            if neuron_index > 0
            else (
                f"<a href='../../L{layer_index-1}/N{2047}'>Previous layer</a> - "
                if layer_index > 0
                else ""
            )
        )
        + (
            f"<a href='../../L{layer_index}/N{neuron_index + 1}/index.html'>Next</a>"
            if neuron_index < 2047
            else (
                f"<a href='../../L{layer_index+1}/N0'>Next layer</a>"
                if layer_index < 7
                else ""
            )
        ),
        layer_index,
        neuron_index,
        rank,
        generate_activation_table(heatmap_blank),
        generate_activation_table(heatmap_my),
        generate_activation_table(attributions),
        games,
    )

    # Write the generated file
    with open(path + "/index.html", "w") as f:
        f.write(template)


def top_50_games(
    layer_index: int,
    neuron_index: int,
    focus_cache: dict[str, Tensor],
    board_seqs_int: Tensor,
) -> str:
    """Takes top 50 games and visualizes them in a grid with visualization of the board state when hovering along with the neuron activation for that specific game."""
    neuron_acts = focus_cache["post", layer_index, "mlp"][:, :, neuron_index]
    num_games = 50
    focus_games_int = board_seqs_int[:num_games]

    # Generate a table where y = game, x = move, and the value is the move from focus_games_string
    moves = []
    for game in focus_games_int:
        moves.append([int_to_label(move) for move in game])

    # Return HTML table with the title = activation of the neuron
    # Create a table
    table = "<table class='games'>"

    # Loop through the rows
    table += "<tr><td class='game_step_id'></td>"
    for col in range(neuron_acts.shape[1]):
        table += "<td class='game_step_id'>{0}</td>".format(col + 1)
    table += "</tr>"
    for row in range(neuron_acts.shape[0]):
        table += "<tr>"
        table += "<td class='game_id'>{0}</td>".format(row + 1)

        # Loop through the columns
        for col in range(neuron_acts.shape[1]):
            table += "<td class='game_step' title={0}>{1}</td>".format(
                neuron_acts[row, col], moves[row][col]
            )

        table += "</tr>"

    table += "</table>"

    return table

def generate_main_index(variance_sorted_neurons: list[list[int]]):
    out_path = "othelloscope/index.html"

    # Read the template file
    file = templating.generate_from_template(
        "othelloscope/index_template.html",
        ranked_neuron_table(variance_sorted_neurons),
    )

    # Write the generated file
    with open(out_path, "w") as f:
        f.write(file)

def ranked_neuron_table(variance_sorted_neurons: list[list[int]]) -> str:
    """Generate a table of ranked neurons."""

    table = "<table class='neurons'>"
    layer_header_strings = "".join(
        [f"<th>Layer {layer_index}</th>" for layer_index in range(8)]
    )
    table += f"<tr><th></th>{layer_header_strings}</tr>"
    for rank in range(len(variance_sorted_neurons[0])):
        table += f"<tr><th>{rank}</th>"
        table += "".join(
            [
                f"<td><a href='L{layer_index}/N{variance_sorted_neurons[layer_index][rank]}/index.html'>{variance_sorted_neurons[layer_index][rank]}</a></td>"
                for layer_index in range(8)
            ]
        )
        table += "</tr>"
    table += "</table>"
    return table