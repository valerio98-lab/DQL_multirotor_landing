"""
This script starts a training session.
"""

import rospy

from dql_multirotor_landing.trainer import Trainer

if __name__ == "__main__":
    rospy.init_node("training_node")
    trainer = Trainer.load()
    trainer.curriculum_training()
    rospy.signal_shutdown("Training ended sucessfully")
