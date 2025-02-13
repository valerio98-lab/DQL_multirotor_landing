#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import numpy as np
from geometry_msgs.msg import Pose, TransformStamped
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from std_msgs.msg import Float64, Bool 
from training_q_learning.parameters import Parameters
from training_q_learning.srv import ResetRandomSeed, ResetRandomSeedResponse

class MovingPlatformNode(object):
    def __init__(self):

        print("New instance of MovingPlatformNode")
        # Parametri per la traiettoria
        self.parameters = Parameters()
        self.trajectory_type = rospy.get_param("moving_platform_node/trajectory_type", "straight")
        self.trajectory_speed = float(rospy.get_param("moving_platform_node/trajectory_speed", "1"))
        self.trajectory_frequency = float(rospy.get_param("moving_platform_node/trajectory_frequency", "100"))
        self.trajectory_scale_factor = float(rospy.get_param("moving_platform_node/trajectory_scale_factor", "1"))
        self.trajectory_start_position = rospy.get_param("moving_platform_node/trajectory_start_position", {'x':0,'y':0,'z':0})
        self.trajectory_start_orientation = rospy.get_param("moving_platform_node/trajectory_start_orientation", {'phi':0,'theta':0,'psi':0})
        self.trajectory_radius = float(rospy.get_param("moving_platform_node/trajectory_radius", "10"))
        self.trajectory_radius_vertical = float(rospy.get_param("moving_platform_node/trajectory_radius_vertical", "10"))
        self.trajectory_speed_vertical = float(rospy.get_param("moving_platform_node/trajectory_speed_vertical", "1"))
        self.trajectory_type_vertical = rospy.get_param("moving_platform_node/trajectory_type_vertical", "straight")
        self.trajectory_radius_lateral = float(rospy.get_param("moving_platform_node/trajectory_radius_lateral", "10"))
        self.trajectory_speed_lateral = float(rospy.get_param("moving_platform_node/trajectory_speed_lateral", "1"))
        
        for k, v in self.trajectory_start_position.items():
            self.trajectory_start_position[k] = float(v)
        for k, v in self.trajectory_start_orientation.items():
            self.trajectory_start_orientation[k] = float(v)
        
        # Stato corrente della piattaforma
        self.x = self.trajectory_start_position['x']
        self.y = self.trajectory_start_position['y']
        self.z = self.trajectory_start_position['z']
        # Angoli di Eulero della piattaforma
        self.phi = self.trajectory_start_orientation['phi']  
        self.theta = self.trajectory_start_orientation['theta']
        self.psi = self.trajectory_start_orientation['psi']
        # Velocità della piattaforma lungo gli assi (rispettivamente x,y,z)
        self.u = 0  
        self.v = 0 
        self.w = 0 
        self.t = 0
        self.delta_t = 1.0 / self.trajectory_frequency
        
        # Publisher per la traiettoria
        self.pose_pub = rospy.Publisher('moving_platform/commanded/pose', Pose, queue_size=3)
        self.gazebo_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=3)
        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
            self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        except rospy.ROSException:
            rospy.logwarn("Servizio /gazebo/set_model_state non disponibile al momento. Procedo senza aggiornare Gazebo.")
            self.set_state_service = None

        
        # Sottoscrittori per eventuali aggiornamenti dei parametri dinamici
        self.trajectory_speed_sub = rospy.Subscriber('moving_platform/setpoints/trajectory_speed', Float64, self.read_trajectory_speed)
        self.trajectory_radius_sub = rospy.Subscriber('moving_platform/setpoints/trajectory_radius', Float64, self.read_trajectory_radius)
        self.trajectory_speed_lateral_sub = rospy.Subscriber('moving_platform/setpoints/trajectory_speed_lateral', Float64, self.read_trajectory_speed_lateral)
        self.trajectory_radius_lateral_sub = rospy.Subscriber('moving_platform/setpoints/trajectory_radius_lateral', Float64, self.read_trajectory_radius_lateral)
        drone_name = rospy.get_param("command_moving_platform_trajectories_node/drone_name", "hummingbird")
        reset_topic = "/" + drone_name + "/training/reset_simulation"
        self.reset_sub = rospy.Subscriber(reset_topic, Bool, self.read_reset)
        self.reset_service = rospy.Service('/moving_platform/reset_random_seed', ResetRandomSeed, self.reset_random_seed)
        
        # Trasform broadcaster
        self.br = tf2_ros.TransformBroadcaster()
        # Sottoscrittore per ricevere lo stato attuale dei modelli in Gazebo
        self.gazebo_state_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.read_model_state)
        
        # Imposta un timer per il calcolo e la pubblicazione della traiettoria
        self.trajectory_timer = rospy.Timer(rospy.Duration(self.delta_t), self.trajectory_callback)
        
        rospy.loginfo("MovingPlatformNode inizializzato.")

    # --- Metodi per la gestione dei parametri dinamici ---
    def read_trajectory_speed(self, msg):
        self.trajectory_speed = msg.data

    def read_trajectory_radius(self, msg):
        self.trajectory_radius = msg.data

    def read_trajectory_speed_lateral(self, msg):
        self.trajectory_speed_lateral = msg.data

    def read_trajectory_radius_lateral(self, msg):
        self.trajectory_radius_lateral = msg.data

    def read_reset(self, msg):
        if msg.data:
            # Reset di t a un valore casuale entro un intervallo definito
            self.t = np.random.uniform(0, self.parameters.rl_parameters.max_num_timesteps_episode * self.parameters.rl_parameters.running_step_time)

    def reset_random_seed(self, req):
        seed = None if req.seed == 'None' else int(req.seed)
        rospy.loginfo("Set seed for random initial values to: %s", seed)
        np.random.seed(seed)
        return ResetRandomSeedResponse()

    # --- Calcolo della traiettoria ---
    def compute_trajectory(self):
        # Calcola la traiettoria orizzontale in base al tipo specificato
        if self.trajectory_type == "circle":
            omega = self.trajectory_speed / self.trajectory_radius
            self.x = self.trajectory_radius * np.cos(omega * self.t) + self.trajectory_start_position["x"]
            self.y = self.trajectory_radius * np.sin(omega * self.t) + self.trajectory_start_position["y"]
            self.u = -self.trajectory_radius * omega * np.sin(omega * self.t)
            self.v = self.trajectory_radius * omega * np.cos(omega * self.t)
        elif self.trajectory_type == "straight":
            self.x = self.x + self.trajectory_speed * self.delta_t
            self.u = self.trajectory_speed
            self.v = 0
        elif self.trajectory_type == "rectilinear_periodic_straight":
            omega = self.trajectory_speed / self.trajectory_radius
            omega_lateral = self.trajectory_speed_lateral / self.trajectory_radius_lateral
            self.x = self.trajectory_radius * np.sin(omega * self.t) + self.trajectory_start_position["x"]
            self.y = self.trajectory_radius_lateral * np.sin(omega_lateral * self.t) + self.trajectory_start_position["y"]
            self.u = self.trajectory_radius * omega * np.cos(omega * self.t)
            self.v = self.trajectory_radius_lateral * omega_lateral * np.cos(omega_lateral * self.t)
        else:
            rospy.logerr("Tipo di traiettoria orizzontale non riconosciuto: %s", self.trajectory_type)
            return

        # Calcola la traiettoria verticale
        if self.trajectory_type_vertical == "rectilinear_periodic_straight":
            omega_vertical = self.trajectory_speed_vertical / self.trajectory_radius_vertical
            self.z = self.trajectory_radius_vertical * np.sin(omega_vertical * self.t) + self.trajectory_start_position["z"]
            self.w = self.trajectory_radius_vertical * omega_vertical * np.cos(omega_vertical * self.t)
        elif self.trajectory_type_vertical == "straight":
            self.z = self.z + self.trajectory_speed_vertical * self.delta_t
            self.w = self.trajectory_speed_vertical
        else:
            rospy.logerr("Tipo di traiettoria verticale non riconosciuto: %s", self.trajectory_type_vertical)
            return

        self.t += self.delta_t

    def publish_trajectory(self):
        # Prepara e pubblica il messaggio Pose per il comando
        pose_msg = Pose()
        pose_msg.position.x = self.x
        pose_msg.position.y = self.y
        pose_msg.position.z = self.z
        quat = tf.transformations.quaternion_from_euler(self.phi, self.theta, self.psi)
        pose_msg.orientation.x = quat[0]
        pose_msg.orientation.y = quat[1]
        pose_msg.orientation.z = quat[2]
        pose_msg.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # Prepara e pubblica il messaggio ModelState per Gazebo
        gazebo_msg = ModelState()
        gazebo_msg.model_name = 'moving_platform'
        gazebo_msg.reference_frame = 'ground_plane'
        gazebo_msg.pose = pose_msg
        gazebo_msg.twist.linear.x = self.u
        gazebo_msg.twist.linear.y = self.v
        gazebo_msg.twist.linear.z = self.w
        self.gazebo_pose_pub.publish(gazebo_msg)
        try:
            self.set_state_service(gazebo_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Chiamata al servizio set_model_state fallita: %s", e)

    def trajectory_callback(self, event):
        self.compute_trajectory()
        self.publish_trajectory()

    # --- Gestione dei TF dalla lettura dello stato Gazebo ---
    def read_model_state(self, msg):
        try:
            idx = msg.name.index("moving_platform")
        except ValueError:
            return  # se il modello non è presente, esce
        pose = msg.pose[idx]
        self.publish_transforms(pose)

    def publish_transforms(self, pose):
        # Trasformazione: world -> world_link (identità)
        t1 = TransformStamped()
        t1.header.stamp = rospy.Time.now()
        t1.header.frame_id = "world"
        t1.child_frame_id = "world_link"
        t1.transform.translation.x = 0
        t1.transform.translation.y = 0
        t1.transform.translation.z = 0
        t1.transform.rotation.x = 0
        t1.transform.rotation.y = 0
        t1.transform.rotation.z = 0
        t1.transform.rotation.w = 1
        self.br.sendTransform(t1)

        # Trasformazione: world_link -> moving_platform/base_link
        t2 = TransformStamped()
        t2.header.stamp = rospy.Time.now()
        t2.header.frame_id = "world_link"
        t2.child_frame_id = "moving_platform/base_link"
        t2.transform.translation.x = pose.position.x
        t2.transform.translation.y = pose.position.y
        t2.transform.translation.z = pose.position.z
        t2.transform.rotation = pose.orientation
        self.br.sendTransform(t2)

if __name__ == '__main__':
    rospy.init_node('moving_platform_node', anonymous=False)
    node = MovingPlatformNode()
    rospy.spin()
