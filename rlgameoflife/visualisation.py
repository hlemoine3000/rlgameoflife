import json
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


from rlgameoflife import entities


ENTITY_COLOR_DICT = {
    entities.EntityType.ITEM: [0, 1, 0],
    entities.EntityType.CREATURE: [0, 0, 1],
}


def load_simulation_history(history_dir_path: str) -> dict:
    entities_dict = {}
    entities_file_list = os.listdir(history_dir_path)
    entities_file_list = [
        os.path.join(history_dir_path, filepath)
        for filepath in entities_file_list
        if filepath.endswith(".npy")
    ]
    for entity_filepath in entities_file_list:
        history_loader = entities.EntityHistoryLoader()
        history_loader.load(entity_filepath)
        name = os.path.basename(entity_filepath)[:-4]
        entities_dict[name] = history_loader.history
    return entities_dict


def visualize_simulation(simulation_dir_path: str):

    # Load simulation history
    history_dir_path = os.path.join(simulation_dir_path, "history")
    entities_dict = load_simulation_history(history_dir_path)
    num_entities = len(entities_dict)

    # Load simulation parameters
    parameters_filepath = os.path.join(simulation_dir_path, "parameters.json")
    with open(parameters_filepath, "r") as parameters_file:
        parameters_dict = json.load(parameters_file)
    total_ticks = parameters_dict["total_ticks"]

    entities_position_np = np.zeros((num_entities, total_ticks * 2 + 2))
    colors_np = np.zeros((num_entities, 3))
    position_mask = np.tile(
        np.array([False, True, True, False, False, False, False]), total_ticks + 1
    )
    for entity_idx, entity_np in enumerate(entities_dict.values()):
        flatten_entity_np = entity_np.flatten()
        entities_position_np[entity_idx, :] = flatten_entity_np[position_mask]
        colors_np[entity_idx, :] = ENTITY_COLOR_DICT[entity_np[0][5]]

    # Set up the figure and axis
    fig, ax = plt.subplots()
    scat = ax.scatter(
        entities_position_np[:, 0], entities_position_np[:, 1], s=200, c=colors_np
    )
    ax.set_xlim(-1, parameters_dict["boundaries"]["x"])
    ax.set_ylim(-1, parameters_dict["boundaries"]["y"])

    # Function to update the animation
    def update(frame):
        tick_idx = 2 * frame
        scat.set_offsets(entities_position_np[:, tick_idx : tick_idx + 2])
        return (scat,)

    # Set up the animation
    anim = animation.FuncAnimation(
        fig, update, frames=total_ticks, interval=20, blit=True
    )

    # Save animation to video
    video_filepath = os.path.join(simulation_dir_path, "simulation.gif")
    anim.save(video_filepath, writer="PillowWriter", fps=60)
