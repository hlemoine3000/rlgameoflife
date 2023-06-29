import logging
import os
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt


from rlgameoflife import entities


ENTITY_COLOR_DICT = {
    entities.EntityType.FOOD: [0, 1, 0],
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

    def make_video(self):
        self._logger.info("Create video of simulation %s", self._simulation_dir_path)
        history_dict, boundaries = self._entities_history_loader.get_timed_history()
        fig, ax = plt.subplots()
        
        xlim = [boundaries[0] - 20., boundaries[2] + 20]
        ylim = [boundaries[1] - 20., boundaries[3] + 20]
        dir_line_size = (xlim[1] - xlim[0]) / 20.

        def update(frame):
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", "box")
            ax.set_title(f"Itereation: {frame}")

            for entity_name, entity_data in history_dict[frame].items():
                pos = entity_data["position"]
                dir = entity_data["direction"]
                if entity_data["type"] == entities.EntityType.FOOD.value:
                    color = "blue"
                elif entity_data["type"] == entities.EntityType.CREATURE.value:
                    color = "red"
                else:
                    color = "black"
                ax.plot(pos[0], pos[1], marker="o", markersize=5, color=color)
                if entity_data["type"] == entities.EntityType.CREATURE.value:
                    ax.plot([pos[0], pos[0] + dir[0]* dir_line_size], [pos[1], pos[1] + dir[1] * dir_line_size], 'k-', lw=1)
                ax.annotate(entity_name, pos)

        save_start = time.time()
        anim = animation.FuncAnimation(
            fig, update, frames=list(history_dict.keys())
        )
        writervideo = animation.ImageMagickWriter(fps=240)
        anim.save(
            os.path.join(self._simulation_dir_path, "entities_history.mp4"),
            writer=writervideo,
        )
        save_end = time.time()
        self._logger.info("Video saved in %d s", save_end - save_start)
