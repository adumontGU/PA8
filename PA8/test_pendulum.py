##############################################
# Programmer: Adrien Dumont
# Class: ENSC 201-01, Fall 2026
# Programming Assignment #7
# 11/17/26
#
# Description: Unit tests for pendulum.py. Tests the single-step physics
# update, equilibrium-crossing detection, threshold checking, swing
# counting, and the full simulation's stopping behavior using small inputs
# with hand-calculated expected results. Run with: pytest -v
#
# Floating-point results are compared with np.isclose(), because floats are
# not stored exactly and == can fail on values that are effectively equal.
#
# Notes: Written with help from Claude (Anthropic AI assistant).
##############################################
import math

import numpy as np

from pendulum import (compute_angular_acceleration, step_pendulum,
                      is_equilibrium_crossing, is_threshold_met,
                      count_swings, simulate_pendulum, REASON_THRESHOLD,
                      REASON_TIME)


def make_parameters(initial_angle_degrees=30.0, length=1.0, gravity=9.81,
                    friction_coefficient=0.999, time_step=0.01,
                    total_time=10.0, angular_velocity_threshold=0.02):
    """Builds a parameter dictionary for simulate_pendulum() tests.

    Args:
        initial_angle_degrees (float): starting angle in degrees
        length (float): pendulum length in meters
        gravity (float): gravitational acceleration in m/s^2
        friction_coefficient (float): per-step velocity decay factor
        time_step (float): time step in seconds
        total_time (float): maximum simulation time in seconds
        angular_velocity_threshold (float): stopping threshold in rad/s

    Returns:
        dict: parameter name -> value, in the same form as read_parameters()
    """
    parameters = {
        "initial_angle_degrees": initial_angle_degrees,
        "length": length,
        "gravity": gravity,
        "friction_coefficient": friction_coefficient,
        "time_step": time_step,
        "total_time": total_time,
        "angular_velocity_threshold": angular_velocity_threshold,
    }
    return parameters


# ---------- compute_angular_acceleration() ----------

def test_angular_acceleration_at_equilibrium_is_zero():
    """Hanging straight down there is no restoring acceleration."""
    assert np.isclose(compute_angular_acceleration(0.0, 9.81, 1.0), 0.0)


def test_angular_acceleration_at_horizontal():
    """At 90 degrees, sin = 1, so acceleration = -g / L."""
    acceleration = compute_angular_acceleration(math.pi / 2, 9.81, 2.0)
    assert np.isclose(acceleration, -4.905)


# ---------- step_pendulum() ----------

def test_step_small_positive_angle_from_rest():
    """angle 0.1 rad, at rest, no damping, L = 1, dt = 0.01:
    accel = -9.81 * sin(0.1) = -0.979366
    next velocity = 0 + (-0.979366)(0.01) = -0.00979366
    next angle = 0.1 + (-0.00979366)(0.01) = 0.09990206
    """
    next_angle, next_velocity = step_pendulum(0.1, 0.0, 9.81, 1.0, 1.0,
                                              0.01)
    assert np.isclose(next_velocity, -0.00979366)
    assert np.isclose(next_angle, 0.09990206)


def test_step_small_negative_angle_from_rest():
    """Mirror of the positive case: the pendulum accelerates to the right
    (positive velocity) and the angle moves toward zero.
    """
    next_angle, next_velocity = step_pendulum(-0.1, 0.0, 9.81, 1.0, 1.0,
                                              0.01)
    assert np.isclose(next_velocity, 0.00979366)
    assert np.isclose(next_angle, -0.09990206)


def test_step_nonzero_velocity_at_equilibrium():
    """angle 0 so accel = 0; only friction acts on the velocity:
    next velocity = 1.0 * 0.9 = 0.9
    next angle = 0 + 0.9 * 0.1 = 0.09
    """
    next_angle, next_velocity = step_pendulum(0.0, 1.0, 9.81, 1.0, 0.9,
                                              0.1)
    assert np.isclose(next_velocity, 0.9)
    assert np.isclose(next_angle, 0.09)


def test_step_nonzero_angle_and_velocity_with_damping():
    """angle 0.2, velocity 0.5, L = 2, friction 0.99, dt = 0.05:
    accel = -(9.81 / 2) * sin(0.2) = -0.974473
    next velocity = 0.5 * 0.99 + (-0.974473)(0.05) = 0.44627635
    next angle = 0.2 + 0.44627635 * 0.05 = 0.22231382
    """
    next_angle, next_velocity = step_pendulum(0.2, 0.5, 9.81, 2.0, 0.99,
                                              0.05)
    assert np.isclose(next_velocity, 0.44627635)
    assert np.isclose(next_angle, 0.22231382)


def test_step_at_rest_at_equilibrium_stays_put():
    """A pendulum hanging still at the bottom should not move."""
    next_angle, next_velocity = step_pendulum(0.0, 0.0, 9.81, 1.0, 0.999,
                                              0.01)
    assert next_angle == 0.0
    assert next_velocity == 0.0


# ---------- is_equilibrium_crossing() ----------

def test_crossing_positive_to_negative():
    assert is_equilibrium_crossing(0.01, -0.02) is True


def test_crossing_negative_to_positive():
    assert is_equilibrium_crossing(-0.01, 0.02) is True


def test_no_crossing_same_sign():
    assert is_equilibrium_crossing(0.3, 0.2) is False
    assert is_equilibrium_crossing(-0.3, -0.2) is False


def test_no_crossing_when_angle_is_exactly_zero():
    """A product of zero is not < 0, so it does not count as a crossing."""
    assert is_equilibrium_crossing(0.0, -0.1) is False
    assert is_equilibrium_crossing(0.1, 0.0) is False


# ---------- is_threshold_met() ----------

def test_threshold_met_below():
    assert is_threshold_met(0.01, 0.02) is True


def test_threshold_met_negative_velocity_uses_magnitude():
    """Moving left (negative velocity) slowly still meets the threshold."""
    assert is_threshold_met(-0.01, 0.02) is True


def test_threshold_met_exactly_equal():
    """The rule is "less than or equal to" the threshold."""
    assert is_threshold_met(0.02, 0.02) is True
    assert is_threshold_met(-0.02, 0.02) is True


def test_threshold_not_met_above():
    assert is_threshold_met(0.5, 0.02) is False
    assert is_threshold_met(-0.5, 0.02) is False


# ---------- count_swings() ----------

def test_count_swings_zero_crossings():
    assert count_swings(0) == 0


def test_count_swings_one_crossing_is_not_a_complete_swing():
    assert count_swings(1) == 0


def test_count_swings_even_crossings():
    assert count_swings(2) == 1
    assert count_swings(10) == 5


def test_count_swings_odd_crossings_rounds_down():
    assert count_swings(7) == 3


# ---------- simulate_pendulum() ----------

def test_simulation_starts_at_initial_state():
    """The first recorded state is time 0, the initial angle in radians,
    and zero angular velocity (released from rest).
    """
    results = simulate_pendulum(make_parameters(initial_angle_degrees=30.0))
    assert results["times"][0] == 0.0
    assert np.isclose(results["angles"][0], math.pi / 6)
    assert results["angular_velocities"][0] == 0.0


def test_simulation_first_step_moves_left():
    """Released to the right, gravity pulls the pendulum left, so the
    first angular velocity is negative and the angle decreases.
    """
    results = simulate_pendulum(make_parameters())
    assert results["angular_velocities"][1] < 0
    assert results["angles"][1] < results["angles"][0]


def test_simulation_stops_at_total_time():
    """With too little time to reach the threshold, the simulation runs to
    total_time: 0.5 s / 0.01 s = 50 steps, plus the initial state = 51.
    """
    results = simulate_pendulum(make_parameters(total_time=0.5))
    assert len(results["times"]) == 51
    assert np.isclose(results["final_time"], 0.5)
    assert results["threshold_time"] is None
    assert results["stop_reason"] == REASON_TIME


def test_simulation_stops_at_first_crossing_with_large_threshold():
    """A threshold larger than any possible speed is met at the very first
    equilibrium crossing (a quarter period, about 0.5 s for L = 1 m).
    """
    results = simulate_pendulum(
        make_parameters(angular_velocity_threshold=100.0))
    assert results["crossings"] == 1
    assert results["swings"] == 0
    assert results["stop_reason"] == REASON_THRESHOLD
    assert results["threshold_time"] == results["final_time"]
    assert 0.4 < results["final_time"] < 0.6
    # the last two recorded angles are on opposite sides of equilibrium
    assert results["angles"][-2] > 0
    assert results["angles"][-1] < 0


def test_simulation_stops_only_at_a_crossing():
    """The threshold is never checked at turning points, so when the
    simulation stops for the threshold, the last step must be a crossing.
    """
    results = simulate_pendulum(make_parameters(total_time=200.0))
    assert results["stop_reason"] == REASON_THRESHOLD
    assert is_equilibrium_crossing(results["angles"][-2],
                                   results["angles"][-1])
    assert abs(results["angular_velocities"][-1]) <= 0.02


def test_simulation_undamped_swing_count():
    """With no damping (friction 1.0), the small-angle period of a 1 m
    pendulum is T = 2 * pi * sqrt(1 / 9.81) = 2.006 s. Crossings happen at
    T/4, 3T/4, 5T/4, ..., and the 10th one is at 19T/4 = 9.53 s (the 11th
    would be 10.53 s), so 10 s gives 10 crossings and 5 complete swings.
    """
    results = simulate_pendulum(
        make_parameters(initial_angle_degrees=5.0, friction_coefficient=1.0,
                        total_time=10.0))
    assert results["crossings"] == 10
    assert results["swings"] == 5
    assert results["stop_reason"] == REASON_TIME


def test_simulation_at_rest_never_crosses():
    """A pendulum starting straight down at rest never moves or crosses."""
    results = simulate_pendulum(
        make_parameters(initial_angle_degrees=0.0, total_time=1.0))
    assert results["crossings"] == 0
    assert results["swings"] == 0
    assert results["threshold_time"] is None
    assert max(results["angles"]) == 0.0
