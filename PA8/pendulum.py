##############################################
# Programmer: Adrien Dumont
# Class: ENSC 201-01, Fall 2026
# Programming Assignment #8
# 11/24/26
#
# Description: Damped simple pendulum simulation. Reads simulation
# parameters from pendulum_parameters.txt, steps the pendulum forward in
# time with semi-implicit Euler integration, counts equilibrium crossings
# and complete swings, stops when the angular velocity at an equilibrium
# crossing drops to the threshold (or when total_time is reached), and
# writes pendulum_trajectory.csv and pendulum_summary.txt. PA8 adds a
# plotting helper and a table helper that the analysis notebook
# (pa8_pendulum_analysis.ipynb) imports to present the baseline results.
#
# Notes: Written with help from Claude (Anthropic AI assistant). Physics
# model and threshold/swing rules come from the PA7 specification.
##############################################
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# build paths from this script's folder so the program works no matter
# which directory it is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMETERS_FILENAME = os.path.join(SCRIPT_DIR, "pendulum_parameters.txt")
TRAJECTORY_FILENAME = os.path.join(SCRIPT_DIR, "pendulum_trajectory.csv")
SUMMARY_FILENAME = os.path.join(SCRIPT_DIR, "pendulum_summary.txt")

# stopping reasons reported in the summary
REASON_THRESHOLD = "angular velocity at an equilibrium crossing reached " \
    "the threshold"
REASON_TIME = "total_time was reached before the threshold"

# custom color palette: one color per role, reused in every chart
COLORS = {
    "angle": "#1b7f79",
    "velocity": "#6a4c93",
    "phase": "#d1495b",
    "equilibrium": "#3d3d3d",
    "threshold": "#e09f3e",
    "crossing": "#00798c",
    "stop": "#9e2a2b",
    "start": "#2a9d8f",
    "end": "#9e2a2b",
}


def read_parameters(filename):
    """Reads simulation parameters from a text file into a dictionary.

    Args:
        filename (str): name of the parameter file to read

    Returns:
        dict: parameter name (str) -> parameter value (float)

    Notes:
        Each line holds one name-value pair separated by "=", ":" or ",".
        Blank lines, lines starting with "#", and lines whose value is not
        a number (such as a header row) are skipped.
    """
    parameters = {}

    infile = open(filename, "r", encoding="utf-8-sig")
    for line in infile:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue

        # split on the first separator found in the line
        for separator in ["=", ":", ","]:
            if separator in line:
                name, value = line.split(separator, 1)
                try:
                    parameters[name.strip()] = float(value.strip())
                except ValueError:
                    pass  # not a numeric line (e.g., a header), skip it
                break
    infile.close()

    return parameters


def compute_angular_acceleration(angle, gravity, length):
    """Computes the pendulum's angular acceleration at a given angle.

    Args:
        angle (float): current angle from vertical in radians
        gravity (float): gravitational acceleration in m/s^2
        length (float): pendulum length in meters

    Returns:
        float: angular acceleration in rad/s^2

    Notes:
        Pre-conditions:
            - length > 0
    """
    angular_acceleration = -(gravity / length) * math.sin(angle)
    return angular_acceleration


def step_pendulum(angle, angular_velocity, gravity, length,
                  friction_coefficient, time_step):
    """Advances the pendulum state by one time step (semi-implicit Euler).

    Args:
        angle (float): current angle in radians
        angular_velocity (float): current angular velocity in rad/s
        gravity (float): gravitational acceleration in m/s^2
        length (float): pendulum length in meters
        friction_coefficient (float): unitless decay factor applied to the
            angular velocity each step (closer to 1 = less damping)
        time_step (float): time step in seconds

    Returns:
        float: the next angle in radians
        float: the next angular velocity in rad/s

    Notes:
        The next angle is calculated with the NEXT angular velocity
        (semi-implicit Euler), which keeps the simulation stable.
    """
    angular_acceleration = compute_angular_acceleration(angle, gravity,
                                                        length)
    next_angular_velocity = (angular_velocity * friction_coefficient
                             + angular_acceleration * time_step)
    next_angle = angle + next_angular_velocity * time_step
    return next_angle, next_angular_velocity


def is_equilibrium_crossing(previous_angle, current_angle):
    """Checks whether the pendulum crossed equilibrium (angle changed sign).

    Args:
        previous_angle (float): angle at the previous step in radians
        current_angle (float): angle at the current step in radians

    Returns:
        bool: True if the angle changed sign, False otherwise

    Notes:
        A product of exactly zero (an angle sitting at 0) is not a crossing.
    """
    return previous_angle * current_angle < 0


def is_threshold_met(angular_velocity, threshold):
    """Checks whether the angular velocity magnitude is at/below a threshold.

    Args:
        angular_velocity (float): angular velocity in rad/s
        threshold (float): minimum angular velocity threshold in rad/s

    Returns:
        bool: True if abs(angular_velocity) <= threshold, False otherwise
    """
    return abs(angular_velocity) <= threshold


def count_swings(crossings):
    """Counts complete swings from the number of equilibrium crossings.

    Args:
        crossings (int): number of equilibrium crossings

    Returns:
        int: number of complete swings (two crossings per swing)

    Notes:
        Pre-conditions:
            - crossings >= 0
    """
    return crossings // 2


def simulate_pendulum(parameters):
    """Runs the damped pendulum time-stepping simulation.

    Args:
        parameters (dict): simulation parameters with the keys
            initial_angle_degrees, length, gravity, friction_coefficient,
            time_step, total_time, angular_velocity_threshold

    Returns:
        dict: simulation results with the keys
            times (list of float): time of each recorded state in seconds
            angles (list of float): angle of each state in radians
            angular_velocities (list of float): angular velocity of each
                state in rad/s
            crossings (int): number of equilibrium crossings
            swings (int): number of complete swings
            threshold_time (float or None): time the threshold was reached,
                or None if it was not reached
            final_time (float): time of the last recorded state
            stop_reason (str): why the simulation stopped

    Notes:
        The pendulum starts from rest at the initial angle. The threshold
        is only checked at equilibrium crossings, because the angular
        velocity is naturally near zero at every turning point.
    """
    gravity = parameters["gravity"]
    length = parameters["length"]
    friction_coefficient = parameters["friction_coefficient"]
    time_step = parameters["time_step"]
    total_time = parameters["total_time"]
    threshold = parameters["angular_velocity_threshold"]

    # initial state: released from rest, angle converted to radians
    angle = math.radians(parameters["initial_angle_degrees"])
    angular_velocity = 0.0
    times = [0.0]
    angles = [angle]
    angular_velocities = [angular_velocity]

    crossings = 0
    threshold_time = None
    stop_reason = REASON_TIME

    # compute the step count up front so repeated float additions of
    # time_step cannot add or drop a step at the end
    num_steps = round(total_time / time_step)

    for step in range(1, num_steps + 1):
        previous_angle = angle
        angle, angular_velocity = step_pendulum(angle, angular_velocity,
                                                gravity, length,
                                                friction_coefficient,
                                                time_step)
        time = step * time_step

        # record this step
        times.append(time)
        angles.append(angle)
        angular_velocities.append(angular_velocity)

        # only check the threshold when the pendulum crosses equilibrium
        if is_equilibrium_crossing(previous_angle, angle):
            crossings += 1
            if is_threshold_met(angular_velocity, threshold):
                threshold_time = time
                stop_reason = REASON_THRESHOLD
                break

    results = {
        "times": times,
        "angles": angles,
        "angular_velocities": angular_velocities,
        "crossings": crossings,
        "swings": count_swings(crossings),
        "threshold_time": threshold_time,
        "final_time": times[-1],
        "stop_reason": stop_reason,
    }
    return results


def write_trajectory(filename, times, angles, angular_velocities):
    """Writes the time-series trajectory data to a CSV file.

    Args:
        filename (str): name of the CSV file to write
        times (list of float): time of each state in seconds
        angles (list of float): angle of each state in radians
        angular_velocities (list of float): angular velocity of each state
            in rad/s

    Notes:
        Post-conditions:
            - the file is overwritten with a header row and one row per
              recorded state
    """
    outfile = open(filename, "w")
    outfile.write("time_s,angle_rad,angular_velocity_rad_s\n")
    for i in range(len(times)):
        outfile.write(f"{times[i]:.2f},{angles[i]:.6f},"
                      f"{angular_velocities[i]:.6f}\n")
    outfile.close()


def write_summary(filename, parameters, results):
    """Writes a human-readable simulation summary report to a text file.

    Args:
        filename (str): name of the text file to write
        parameters (dict): the simulation parameters that were used
        results (dict): the results returned by simulate_pendulum()

    Notes:
        Post-conditions:
            - the file is overwritten with the parameter values, time to
              threshold, complete swings, final time, and stopping reason
    """
    if results["threshold_time"] is None:
        threshold_text = "not reached"
    else:
        threshold_text = f"{results['threshold_time']:.2f} s"

    outfile = open(filename, "w")
    outfile.write("Damped Pendulum Simulation Summary\n")
    outfile.write("==================================\n")
    outfile.write("Input parameters:\n")
    for name in parameters:
        outfile.write(f"  {name}: {parameters[name]}\n")
    outfile.write("\nResults:\n")
    outfile.write(f"  Time to threshold: {threshold_text}\n")
    outfile.write(f"  Equilibrium crossings: {results['crossings']}\n")
    outfile.write(f"  Complete swings: {results['swings']}\n")
    outfile.write(f"  Final simulation time: {results['final_time']:.2f} s\n")
    outfile.write(f"  Stopping reason: {results['stop_reason']}\n")
    outfile.close()


def find_crossing_indices(angles):
    """Finds the index of the state just after each equilibrium crossing.

    Args:
        angles (list of float): angle of each state in radians

    Returns:
        numpy.ndarray: indices i where angles[i - 1] and angles[i] have
            opposite signs (the same rule as is_equilibrium_crossing())
    """
    angles = np.array(angles)
    return np.where(angles[:-1] * angles[1:] < 0)[0] + 1


def plot_baseline(parameters, results):
    """Builds the three-chart baseline figure (angle, velocity, phase space).

    Args:
        parameters (dict): the simulation parameters that were used
        results (dict): the results returned by simulate_pendulum()

    Returns:
        matplotlib.figure.Figure: the figure with charts (a) angle vs. time,
            (b) angular velocity vs. time, and (c) phase space

    Notes:
        Angles are plotted in degrees; angular velocities are plotted in
        rad/s, the same units as the threshold.
    """
    times = np.array(results["times"])
    angles_degrees = np.degrees(np.array(results["angles"]))
    angular_velocities = np.array(results["angular_velocities"])
    threshold = parameters["angular_velocity_threshold"]
    crossing_indices = find_crossing_indices(results["angles"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("Baseline Damped Pendulum Motion", fontsize=16,
                 fontweight="bold")

    # (a) angle vs. time with the zero-equilibrium reference line
    ax = axes[0]
    ax.plot(times, angles_degrees, color=COLORS["angle"], linewidth=1.0,
            label="Angle")
    ax.axhline(0, color=COLORS["equilibrium"], linestyle="--", linewidth=1,
               label="Equilibrium (0°)")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle from vertical (degrees)", fontsize=11)
    ax.set_title("(a) Angle vs. Time", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # (b) angular velocity vs. time with threshold, crossings, and stop time
    ax = axes[1]
    ax.plot(times, angular_velocities, color=COLORS["velocity"],
            linewidth=1.0, label="Angular velocity")
    ax.axhline(threshold, color=COLORS["threshold"], linestyle="--",
               linewidth=1.5, label=f"±Threshold ({threshold:g} rad/s)")
    ax.axhline(-threshold, color=COLORS["threshold"], linestyle="--",
               linewidth=1.5)
    ax.plot(times[crossing_indices], angular_velocities[crossing_indices],
            "o", color=COLORS["crossing"], markersize=3,
            label="Equilibrium crossings")
    if results["threshold_time"] is not None:
        ax.axvline(results["threshold_time"], color=COLORS["stop"],
                   linestyle=":", linewidth=2,
                   label=f"Threshold reached "
                         f"({results['threshold_time']:.2f} s)")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angular velocity (rad/s)", fontsize=11)
    ax.set_title("(b) Angular Velocity vs. Time", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # (c) phase space: angular velocity vs. angle with start and end marked
    ax = axes[2]
    ax.plot(angles_degrees, angular_velocities, color=COLORS["phase"],
            linewidth=0.7, label="Trajectory")
    ax.plot(angles_degrees[0], angular_velocities[0], "o",
            color=COLORS["start"], markersize=9, label="Start")
    ax.plot(angles_degrees[-1], angular_velocities[-1], "s",
            color=COLORS["end"], markersize=9, label="End")
    ax.annotate(f"Start: {angles_degrees[0]:.0f}°, released from rest",
                xy=(angles_degrees[0], angular_velocities[0]),
                xytext=(-200, 70), textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=COLORS["start"]))
    ax.annotate(f"End: t = {results['final_time']:.2f} s",
                xy=(angles_degrees[-1], angular_velocities[-1]),
                xytext=(40, -60), textcoords="offset points", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=COLORS["end"]))
    ax.set_xlabel("Angle from vertical (degrees)", fontsize=11)
    ax.set_ylabel("Angular velocity (rad/s)", fontsize=11)
    ax.set_title("(c) Phase Space", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def show_summary_table(parameters, results):
    """Displays a Markdown table of baseline parameters and result metrics.

    Args:
        parameters (dict): the simulation parameters that were used
        results (dict): the results returned by simulate_pendulum()

    Notes:
        Every value comes from parameters or results; nothing is typed in
        by hand. A threshold that was not reached shows "not reached".
    """
    if results["threshold_time"] is None:
        threshold_text = "not reached"
    else:
        threshold_text = f"{results['threshold_time']:.2f}"

    max_angle_degrees = np.max(np.abs(np.degrees(results["angles"])))
    peak_angular_speed = np.max(np.abs(results["angular_velocities"]))
    final_angular_velocity = results["angular_velocities"][-1]

    table = f"""
| Parameter or Metric | Value | Units |
|---|---|---|
| Initial angle | {parameters['initial_angle_degrees']:g} | degrees |
| Pendulum length | {parameters['length']:g} | m |
| Gravitational acceleration | {parameters['gravity']:g} | m/s² |
| Friction coefficient (per-step decay) | \
{parameters['friction_coefficient']:g} | unitless |
| Time step | {parameters['time_step']:g} | s |
| Maximum simulation time | {parameters['total_time']:g} | s |
| Angular velocity threshold | \
{parameters['angular_velocity_threshold']:g} | rad/s |
| **Time to threshold** | **{threshold_text}** | **s** |
| **Equilibrium crossings** | **{results['crossings']}** | **count** |
| **Completed swings** | **{results['swings']}** | **count** |
| **Final time** | **{results['final_time']:.2f}** | **s** |
| **Stop reason** | **{results['stop_reason']}** | |
| **Maximum angle magnitude** | **{max_angle_degrees:.2f}** | **degrees** |
| **Peak angular speed** | **{peak_angular_speed:.4f}** | **rad/s** |
| **Final angular velocity** | **{final_angular_velocity:.4f}** | \
**rad/s** |
"""
    display(Markdown(table))


def main():
    """Coordinates the workflow: read parameters, simulate, write outputs.
    """
    parameters = read_parameters(PARAMETERS_FILENAME)
    results = simulate_pendulum(parameters)

    write_trajectory(TRAJECTORY_FILENAME, results["times"],
                     results["angles"], results["angular_velocities"])
    write_summary(SUMMARY_FILENAME, parameters, results)

    print(f"Simulation stopped at {results['final_time']:.2f} s "
          f"({results['stop_reason']}).")
    print(f"Wrote {TRAJECTORY_FILENAME}")
    print(f"Wrote {SUMMARY_FILENAME}")


if __name__ == "__main__":
    main()
