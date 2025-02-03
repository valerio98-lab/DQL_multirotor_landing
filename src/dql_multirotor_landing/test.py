import torch
import pytest
from dql_multirotor_landing.parameters import Parameters
from dql_multirotor_landing.environment.mdp import (
    StateSpace,
    DiscreteState,
    ContinuousState,
    INCREASING,
    DECREASING,
    NOTHING,
)


@pytest.fixture
def parameters():
    """Fixture for initializing the Parameters instance."""
    return Parameters()


@pytest.fixture
def state_space():
    """Fixture for initializing the StateSpace instance."""
    return StateSpace(goal_pos=0.5, p_lim=3, goal_vel=0.2, v_lim=2, goal_acc=0.0, a_lim=1)


# ------------------------
# TESTING PARAMETERS CLASS
# ------------------------
# def test_parameters_initialization(parameters):
#     """Test whether Parameters are initialized correctly."""
#     assert parameters.p_max > 0, "p_max should be greater than 0"
#     assert parameters.v_max > 0, "v_max should be greater than 0"
#     assert parameters.a_max > 0, "a_max should be greater than 0"
#     assert parameters.angle_max is not None, "angle_max should be initialized"
#     assert parameters.w_p < 0, "w_p should be negative to reward approaching the goal"
#     assert parameters.r_success > 0, "r_success should be positive"
#     assert parameters.r_failure < 0, "r_failure should be negative"


# ------------------------
# TESTING ACTION SPACE
# ------------------------
def test_discretized_action_space(state_space):
    """Test whether action space is correctly discretized."""
    assert len(state_space.parameters.angle_values) > 0, "Angles set should not be empty"
    assert torch.all(
        state_space.parameters.angle_values <= state_space.parameters.angle_max
    ), "Angles should not exceed angle_max"


def test_get_discrete_action(state_space):
    """Test whether discrete action selection works correctly."""
    current_angle = torch.tensor(0.0)

    angle_increase_idx, angle_i = state_space._get_discrete_angle(current_angle, INCREASING)
    angle_decrease_idx, angle_d = state_space._get_discrete_angle(current_angle, DECREASING)
    angle_nothing_idx, angle_n = state_space._get_discrete_angle(current_angle, NOTHING)

    assert angle_increase_idx >= 0, "INCREASING should return a valid action index"
    assert angle_increase_idx < 7, "INCREASING should return a valid action index (upper bound)"
    assert angle_decrease_idx >= 0, "DECREASING should return a valid action index"
    assert angle_decrease_idx < 7, "DECREASING should return a valid action index (upper bound)"
    assert angle_nothing_idx >= 0, "NOTHING should return a valid action index"
    assert angle_nothing_idx < 7, "NOTHING should return a valid action index (upper bound)"

    print(angle_i, angle_d, angle_n)


# ------------------------
# TESTING STATE SPACE
# ------------------------
def test_d_f(state_space):
    """Test whether the d_f function correctly categorizes states."""

    state = state_space.d_f(torch.tensor([0.3, 0.2, 0.1]), 0.1, 1)

    assert state in [0, 1, 2], "Should be in range [0, 2]"  # 0: far, 1: medium, 2: close


def test_get_discretized_state(state_space):
    """Test whether the get_discretized_state function correctly maps continuous states to discrete ones."""
    obs = ContinuousState(
        relative_position=torch.tensor([0.3, 0.2, 0.1]),
        relative_velocity=torch.tensor([0.2, 0.5, 2]),
        relative_acceleration=torch.tensor([0.1, 1, 2]),
        pitch_angle=15,
    )

    discrete_state = state_space.get_discretized_state(state=obs)

    assert isinstance(discrete_state, DiscreteState), "Should return an instance of DiscreteState"
    assert 0 <= discrete_state.position <= 2, "Discrete position should be in range [0, 2]"
    assert 0 <= discrete_state.velocity <= 2, "Discrete velocity should be in range [0, 2]"
    assert 0 <= discrete_state.acceleration <= 2, "Discrete acceleration should be in range [0, 2]"


def test_get_reward(state_space):
    """Test whether the get_reward function correctly computes rewards."""
    current_relative_pos = torch.tensor([0.3, 0.2, 0.2])
    current_relative_vel = torch.tensor([0.3, 0.2, 0.2])
    last_relative_pos = torch.tensor([0.4, 0.3, 0.2])
    last_relative_vel = torch.tensor([0.4, 0.3, 0.2])

    current_obs = ContinuousState(
        relative_position=current_relative_pos,
        relative_velocity=current_relative_vel,
        relative_acceleration=torch.tensor([0.1, 1, 2]),
        pitch_angle=20,
    )

    last_obs = ContinuousState(
        relative_position=last_relative_pos,
        relative_velocity=last_relative_vel,
        relative_acceleration=torch.tensor([0.1, 1, 2]),
        pitch_angle=15.7,
    )

    state_space._set_last_state(last_obs)

    reward = state_space.get_reward(current_continuous_state=current_obs)

    print(reward)
    assert isinstance(reward, float), "Reward should be a float"
    assert reward < 0, "Reward should be negative when moving away from the goal"
