"""
This script starts a training session.
"""

# Questo fa schifo, ma ci serve per fare in modo che pylance cooperi e ci dia i suggerimenti corretti
# Sembra essere necessario solo per gli script.

from dql_multirotor_landing.double_q_learning import DoubleQLearningAgent
from dql_multirotor_landing.trainer import Trainer

if __name__ == "__main__":
    curriculum_steps = 4
    agent = DoubleQLearningAgent(curriculum_steps)
    trainer = Trainer(agent)

    if True:
        # Inizializza il training environment,
        # trainer.curriculum_training()
        trainer.curriculum_training()

    else:
        print(
            "\033[91mSelected parameters do not allow starting the training. ABORT..."
        )
