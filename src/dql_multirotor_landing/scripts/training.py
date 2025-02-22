"""
This script starts a training session.
"""

# Questo fa schifo, ma ci serve per fare in modo che pylance cooperi e ci dia i suggerimenti corretti
# Sembra essere necessario solo per gli script.

import rospy

from dql_multirotor_landing.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(max_num_episodes=10)
    rospy.init_node("landing_simulation_gym_node")
    if True:
        # Inizializza il training environment,
        # trainer.curriculum_training()
        trainer.curriculum_training()

    else:
        print(
            "\033[91mSelected parameters do not allow starting the training. ABORT..."
        )
    rospy.signal_shutdown("Training ended sucessfully")
