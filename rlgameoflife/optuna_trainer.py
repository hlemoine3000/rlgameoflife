import logging

import optuna

from rlgameoflife import agent


class OptunaAgentTrainer:
    def __init__(self):
        self._logger = logging.getLogger(__class__.__name__)
        self.study = optuna.create_study(
            direction="maximize", study_name="OptunaAgentTrainer"
        )

    def objective(self, trial: optuna.Trial):
        agent_parameters = agent.AgentTrainerParameters(
            batch_size=trial.suggest_int("batch_size", 64, 252, log=True),
            lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            num_episodes=trial.suggest_int("num_episodes", 60, 100, log=True),
            max_steps_per_episode=trial.suggest_int(
                "max_steps_per_episode", 200, 600, log=True
            ),
            replay_memory_size=trial.suggest_int("replay_memory_size", 1000, 10000),
            eval_each_n_episode=0,
        )
        agent_trainer = agent.AgentTrainer(agent_parameters)
        return agent_trainer.train(save_final_eval=False)

    def optimize(self):
        self.study.optimize(self.objective, n_trials=10, timeout=3600)

        pruned_trials = self.study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
        )
        complete_trials = self.study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
        )

        self._logger.info("Study statistics: ")
        self._logger.info("  Number of finished trials: ", len(self.study.trials))
        self._logger.info("  Number of pruned trials: ", len(pruned_trials))
        self._logger.info("  Number of complete trials: ", len(complete_trials))

        self._logger.info("Best trial:")
        trial = self.study.best_trial

        self._logger.info("  Value: ", trial.value)

        self._logger.info("  Params: ")
        for key, value in trial.params.items():
            self._logger.info("    {}: {}".format(key, value))
