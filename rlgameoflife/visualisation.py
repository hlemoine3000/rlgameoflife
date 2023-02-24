import logging
import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt


from rlgameoflife import entities


ENTITY_COLOR_DICT = {
    entities.EntityType.ITEM: [0, 1, 0],
    entities.EntityType.CREATURE: [0, 0, 1],
}


class Visualizer:
    def __init__(self, simulation_dir_path: str) -> None:
        self._logger = logging.getLogger(__class__.__name__)
        self._simulation_dir_path = simulation_dir_path
        self._entities_history_loader = entities.EntitiesHistoryLoader(
            self._simulation_dir_path
        )
        self._entities_history_loader.load(
            os.path.join(simulation_dir_path, "entities_history.npz")
        )

    def make_gif(self):
        history_dict = self._entities_history_loader.get_timed_history()
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            ax.set_xlim([0, 1000])  # Change as needed
            ax.set_ylim([0, 1000])  # Change as needed
            ax.set_aspect("equal", "box")
            ax.set_title(f"Itereation: {frame}")

            for entity_name, entity_data in history_dict[frame].items():
                pos = entity_data["position"]
                if entity_data["type"] == entities.EntityType.ITEM.value:
                    color = "blue"
                elif entity_data["type"] == entities.EntityType.CREATURE.value:
                    color = "red"
                else:
                    color = "black"
                ax.plot(pos[0], pos[1], marker="o", markersize=5, color=color)
                ax.annotate(entity_name, pos)

        save_start = time.time()
        anim = animation.FuncAnimation(
            fig, update, frames=list(history_dict.keys()), interval=1
        )
        writervideo = animation.FFMpegWriter(fps=120)
        anim.save(
            os.path.join(self._simulation_dir_path, "entities_history.mp4"),
            writer=writervideo,
        )
        save_end = time.time()
        self._logger.info("Video saved in %d s", save_end - save_start)
