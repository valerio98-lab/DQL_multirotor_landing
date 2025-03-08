/*
 * Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
 * Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
 * Copyright 2015-2018 PX4 Pro Development Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <rotors_gazebo_plugins/gazebo_mavlink_interface.h>

namespace gazebo
{

  // Set global reference point
  // Zurich Irchel Park: 47.397742, 8.545594, 488m
  // Seattle downtown (15 deg declination): 47.592182, -122.316031, 86m
  // Moscow downtown: 55.753395, 37.625427, 155m

  // Zurich Irchel Park
  static const double kLatZurich_rad = 47.397742 * M_PI / 180; // rad
  static const double kLonZurich_rad = 8.545594 * M_PI / 180;  // rad
  static const double kAltZurich_m = 488.0;                    // meters
  static const float kEarthRadius_m = 6353000;                 // m

  GZ_REGISTER_MODEL_PLUGIN(GazeboMavlinkInterface);

  GazeboMavlinkInterface::~GazeboMavlinkInterface()
  {
    updateConnection_->~Connection();
  }

  void GazeboMavlinkInterface::Load(
      physics::ModelPtr _model, sdf::ElementPtr _sdf)
  {
    // Store the pointer to the model.
    model_ = _model;

    world_ = model_->GetWorld();

    const char *env_alt = std::getenv("PX4_HOME_ALT");
    if (env_alt)
    {
      gzmsg << "Home altitude is set to " << env_alt << ".\n";
      alt_home = std::stod(env_alt);
    }

    namespace_.clear();
    if (_sdf->HasElement("robotNamespace"))
    {
      namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
    }
    else
    {
      gzerr << "[gazebo_mavlink_interface] Please specify a robotNamespace.\n";
    }

    if (_sdf->HasElement("protocol_version"))
    {
      protocol_version_ = _sdf->GetElement("protocol_version")->Get<float>();
    }

    node_handle_ = transport::NodePtr(new transport::Node());
    node_handle_->Init();

    getSdfParam<std::string>(
        _sdf, "motorSpeedCommandPubTopic", motor_velocity_reference_pub_topic_,
        motor_velocity_reference_pub_topic_);
    gzdbg << "motorSpeedCommandPubTopic = \""
          << motor_velocity_reference_pub_topic_ << "\"." << std::endl;
    getSdfParam<std::string>(
        _sdf, "lidarSubTopic", lidar_sub_topic_, lidar_sub_topic_);
    getSdfParam<std::string>(
        _sdf, "opticalFlowSubTopic", opticalFlow_sub_topic_,
        opticalFlow_sub_topic_);

    // set input_reference_ from inputs.control
    input_reference_.resize(kNOutMax);
    joints_.resize(kNOutMax);
    pids_.resize(kNOutMax);
    for (int i = 0; i < kNOutMax; ++i)
    {
      pids_[i].Init(0, 0, 0, 0, 0, 0, 0);
      input_reference_[i] = 0;
    }

    if (_sdf->HasElement("control_channels"))
    {
      sdf::ElementPtr control_channels = _sdf->GetElement("control_channels");
      sdf::ElementPtr channel = control_channels->GetElement("channel");
      while (channel)
      {
        if (channel->HasElement("input_index"))
        {
          int index = channel->Get<int>("input_index");
          if (index < kNOutMax)
          {
            input_offset_[index] = channel->Get<double>("input_offset");
            input_scaling_[index] = channel->Get<double>("input_scaling");
            zero_position_disarmed_[index] =
                channel->Get<double>("zero_position_disarmed");
            zero_position_armed_[index] =
                channel->Get<double>("zero_position_armed");
            if (channel->HasElement("joint_control_type"))
            {
              joint_control_type_[index] =
                  channel->Get<std::string>("joint_control_type");
            }
            else
            {
              gzwarn << "joint_control_type[" << index
                     << "] not specified, using velocity.\n";
              joint_control_type_[index] = "velocity";
            }

            // start gz transport node handle
            if (joint_control_type_[index] == "position_gztopic")
            {
              // setup publisher handle to topic
              if (channel->HasElement("gztopic"))
                gztopic_[index] = "~/" + model_->GetName() +
                                  channel->Get<std::string>("gztopic");
              else
                gztopic_[index] =
                    "control_position_gztopic_" + std::to_string(index);

              joint_control_pub_[index] =
                  node_handle_->Advertise<gazebo::msgs::Any>(gztopic_[index]);
            }

            if (channel->HasElement("joint_name"))
            {
              std::string joint_name = channel->Get<std::string>("joint_name");
              joints_[index] = model_->GetJoint(joint_name);
              if (joints_[index] == nullptr)
              {
                gzwarn << "joint [" << joint_name << "] not found for channel["
                       << index << "] no joint control for this channel.\n";
              }
              else
              {
                gzdbg << "joint [" << joint_name << "] found for channel["
                      << index << "] joint control active for this channel.\n";
              }
            }
            else
            {
              gzdbg << "<joint_name> not found for channel[" << index
                    << "] no joint control will be performed for this channel.\n";
            }

            // setup joint control pid to control joint
            if (channel->HasElement("joint_control_pid"))
            {
              sdf::ElementPtr pid = channel->GetElement("joint_control_pid");
              double p = 0;
              if (pid->HasElement("p"))
                p = pid->Get<double>("p");
              double i = 0;
              if (pid->HasElement("i"))
                i = pid->Get<double>("i");
              double d = 0;
              if (pid->HasElement("d"))
                d = pid->Get<double>("d");
              double iMax = 0;
              if (pid->HasElement("iMax"))
                iMax = pid->Get<double>("iMax");
              double iMin = 0;
              if (pid->HasElement("iMin"))
                iMin = pid->Get<double>("iMin");
              double cmdMax = 0;
              if (pid->HasElement("cmdMax"))
                cmdMax = pid->Get<double>("cmdMax");
              double cmdMin = 0;
              if (pid->HasElement("cmdMin"))
                cmdMin = pid->Get<double>("cmdMin");
              pids_[index].Init(p, i, d, iMax, iMin, cmdMax, cmdMin);
            }
          }
          else
          {
            gzerr << "input_index[" << index << "] out of range, not parsing.\n";
          }
        }
        else
        {
          gzerr << "no input_index, not parsing.\n";
          break;
        }
        channel = channel->GetNextElement("channel");
      }
    }

    // Listen to the update event. This event is broadcast every
    // sim_ulation iteration.
    updateConnection_ = event::Events::ConnectWorldUpdateBegin(
        boost::bind(&GazeboMavlinkInterface::OnUpdate, this, _1));

    // Publish gazebo's motor_speed message
    motor_velocity_reference_pub_ =
        node_handle_->Advertise<gz_mav_msgs::CommandMotorSpeed>(
            "~/" + model_->GetName() + motor_velocity_reference_pub_topic_, 1);

    _rotor_count = 5;

    last_time_ = world_->SimTime();
    gravity_W_ = world_->Gravity();

    if (_sdf->HasElement("hil_state_level"))
    {
      hil_mode_ = _sdf->GetElement("hil_mode")->Get<bool>();
    }

    if (_sdf->HasElement("hil_state_level"))
    {
      hil_state_level_ = _sdf->GetElement("hil_state_level")->Get<bool>();
    }

    // Get serial params
    if (_sdf->HasElement("serialEnabled"))
    {
      serial_enabled_ = _sdf->GetElement("serialEnabled")->Get<bool>();
    }

    if (serial_enabled_)
    {
      // Set up serial interface
      if (_sdf->HasElement("serialDevice"))
      {
        device_ = _sdf->GetElement("serialDevice")->Get<std::string>();
      }

      if (_sdf->HasElement("baudRate"))
      {
        baudrate_ = _sdf->GetElement("baudRate")->Get<int>();
      }
      io_service_.post(std::bind(&GazeboMavlinkInterface::do_read, this));

      // run io_service for async io
      io_thread_ = std::thread([this]()
                               { io_service_.run(); });
      open();
    }

    // Create socket
    // udp socket data
    mavlink_addr_ = htonl(INADDR_ANY);
    if (_sdf->HasElement("mavlink_addr"))
    {
      std::string mavlink_addr =
          _sdf->GetElement("mavlink_addr")->Get<std::string>();
      if (mavlink_addr != "INADDR_ANY")
      {
        mavlink_addr_ = inet_addr(mavlink_addr.c_str());
        if (mavlink_addr_ == INADDR_NONE)
        {
          fprintf(stderr, "invalid mavlink_addr \"%s\"\n", mavlink_addr.c_str());
          return;
        }
      }
    }
    if (_sdf->HasElement("mavlink_udp_port"))
    {
      mavlink_udp_port_ = _sdf->GetElement("mavlink_udp_port")->Get<int>();
    }

    auto worldName = world_->Name();
    model_param(worldName, model_->GetName(), "mavlink_udp_port",
                mavlink_udp_port_);

    qgc_addr_ = htonl(INADDR_ANY);
    if (_sdf->HasElement("qgc_addr"))
    {
      std::string qgc_addr = _sdf->GetElement("qgc_addr")->Get<std::string>();
      if (qgc_addr != "INADDR_ANY")
      {
        qgc_addr_ = inet_addr(qgc_addr.c_str());
        if (qgc_addr_ == INADDR_NONE)
        {
          fprintf(stderr, "invalid qgc_addr \"%s\"\n", qgc_addr.c_str());
          return;
        }
      }
    }
    if (_sdf->HasElement("qgc_udp_port"))
    {
      qgc_udp_port_ = _sdf->GetElement("qgc_udp_port")->Get<int>();
    }

    // try to setup udp socket for communcation
    if ((_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
      printf("create socket failed\n");
      return;
    }

    memset((char *)&myaddr_, 0, sizeof(myaddr_));
    myaddr_.sin_family = AF_INET;
    srcaddr_.sin_family = AF_INET;

    if (serial_enabled_)
    {
      // gcs link
      myaddr_.sin_addr.s_addr = mavlink_addr_;
      myaddr_.sin_port = htons(mavlink_udp_port_);
      srcaddr_.sin_addr.s_addr = qgc_addr_;
      srcaddr_.sin_port = htons(qgc_udp_port_);
    }

    else
    {
      myaddr_.sin_addr.s_addr = htonl(INADDR_ANY);
      // Let the OS pick the port
      myaddr_.sin_port = htons(0);
      srcaddr_.sin_addr.s_addr = mavlink_addr_;
      srcaddr_.sin_port = htons(mavlink_udp_port_);
    }

    addrlen_ = sizeof(srcaddr_);

    if (bind(_fd, (struct sockaddr *)&myaddr_, sizeof(myaddr_)) < 0)
    {
      printf("bind failed\n");
      return;
    }

    fds[0].fd = _fd;
    fds[0].events = POLLIN;

    mavlink_status_t *chan_state = mavlink_get_channel_status(MAVLINK_COMM_0);

    // set the Mavlink protocol version to use on the link
    if (protocol_version_ == 2.0)
    {
      chan_state->flags &= ~(MAVLINK_STATUS_FLAG_OUT_MAVLINK1);
      gzmsg << "Using MAVLink protocol v2.0\n";
    }
    else if (protocol_version_ == 1.0)
    {
      chan_state->flags |= MAVLINK_STATUS_FLAG_OUT_MAVLINK1;
      gzmsg << "Using MAVLink protocol v1.0\n";
    }
    else
    {
      gzerr << "Unkown protocol version! Using v" << protocol_version_
            << "by default \n";
    }
  }

  // This gets called by the world update start event.
  void GazeboMavlinkInterface::OnUpdate(const common::UpdateInfo & /*_info*/)
  {
    common::Time current_time = world_->SimTime();
    double dt = (current_time - last_time_).Double();

    pollForMAVLinkMessages(dt, 1000);

    handle_control(dt);

    if (received_first_reference_)
    {
      gz_mav_msgs::CommandMotorSpeed turning_velocities_msg;

      for (int i = 0; i < input_reference_.size(); i++)
      {
        if (last_actuator_time_ == 0 ||
            (current_time - last_actuator_time_).Double() > 0.2)
        {
          turning_velocities_msg.add_motor_speed(0);
        }
        else
        {
          turning_velocities_msg.add_motor_speed(input_reference_[i]);
        }
      }
      // TODO Add timestamp and Header
      // turning_velocities_msg->header.stamp.sec = current_time.sec;
      // turning_velocities_msg->header.stamp.nsec = current_time.nsec;

      motor_velocity_reference_pub_->Publish(turning_velocities_msg);
    }

    last_time_ = current_time;
  }

  void GazeboMavlinkInterface::send_mavlink_message(
      const mavlink_message_t *message, const int destination_port)
  {
    if (serial_enabled_ && destination_port == 0)
    {
      assert(message != nullptr);
      if (!is_open())
      {
        gzerr << "Serial port closed! \n";
        return;
      }

      {
        lock_guard lock(mutex_);

        if (tx_q_.size() >= MAX_TXQ_SIZE)
        {
          //         gzwarn << "TX queue overflow. \n";
        }
        tx_q_.emplace_back(message);
      }
      io_service_.post(std::bind(&GazeboMavlinkInterface::do_write, this, true));
    }

    else
    {
      uint8_t buffer[MAVLINK_MAX_PACKET_LEN];
      int packetlen = mavlink_msg_to_send_buffer(buffer, message);

      struct sockaddr_in dest_addr;
      memcpy(&dest_addr, &srcaddr_, sizeof(srcaddr_));

      if (destination_port != 0)
      {
        dest_addr.sin_port = htons(destination_port);
      }

      ssize_t len = sendto(
          _fd, buffer, packetlen, 0, (struct sockaddr *)&srcaddr_,
          sizeof(srcaddr_));

      if (len <= 0)
      {
        printf("Failed sending mavlink message\n");
      }
    }
  }
