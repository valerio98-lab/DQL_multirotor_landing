"""
This script starts a training session.
"""

# Questo fa schifo, ma ci serve per fare in modo che pylance cooperi e ci dia i suggerimenti corretti
# Sembra essere necessario solo per gli script.

import rospy

from dql_multirotor_landing.trainer import Trainer

INITIAL_CURRICULUM_STEP = 0
if __name__ == "__main__":
    rospy.init_node("training_node")
    trainer = Trainer(
        max_num_episodes=20,
        initial_curriculum_step=INITIAL_CURRICULUM_STEP,
        successive_successful_episodes=10,
    )
    trainer.curriculum_training()
    rospy.signal_shutdown("Training ended sucessfully")
