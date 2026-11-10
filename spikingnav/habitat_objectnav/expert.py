"""Online ObjectNav expert from goal viewpoints.

HM3D ObjectNav episodes do not store action traces (``shortest_paths`` is
null). Each goal does store viewpoints: navigable poses where the object is
visible. The expert walks the geodesic shortest path to the nearest viewpoint
and emits ``stop`` inside the success radius. It never uses look up / down.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

FOLLOWER_ACTION_NAMES = ("stop", "move_forward", "turn_left", "turn_right")
DEFAULT_GOAL_RADIUS = 0.1
DEMO_ACTION_NAMES = {
    "STOP": "stop",
    "MOVE_FORWARD": "move_forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
    "LOOK_UP": "look_up",
    "LOOK_DOWN": "look_down",
}


def demo_action_index(replay: Sequence, elapsed_steps: int, action_names: Sequence[str]) -> Tuple[int, bool]:
    """Action label from a Habitat-Web ``reference_replay``.

    The stored trace starts with a sentinel ``STOP`` at the spawn pose.
    Step ``elapsed_steps`` is supervised by the next real action.
    """
    if not replay:
        return 0, False
    offset = 0
    first = replay[0]["action"] if isinstance(replay[0], dict) else replay[0]
    if str(first).upper() == "STOP":
        offset = 1
    index = int(elapsed_steps) + offset
    if index < 0 or index >= len(replay):
        return list(action_names).index("stop"), False
    raw = replay[index]["action"] if isinstance(replay[index], dict) else replay[index]
    name = DEMO_ACTION_NAMES.get(str(raw).upper(), str(raw).lower())
    if name not in action_names:
        return list(action_names).index("stop"), False
    return list(action_names).index(name), True


def map_follower_action(action: int, action_names: Sequence[str]) -> int:
    """Map a Habitat-Sim greedy-follower id onto the task action index."""
    name = FOLLOWER_ACTION_NAMES[int(action)]
    return list(action_names).index(name)


def dagger_beta(
    steps: int,
    total_steps: int,
    beta_start: float,
    beta_end: float,
    decay_steps: int,
) -> float:
    """Linear mixture weight. 1 executes the expert, 0 executes the student."""
    horizon = decay_steps if decay_steps > 0 else total_steps
    if horizon <= 0:
        return float(beta_end)
    progress = min(1.0, max(0.0, float(steps) / float(horizon)))
    return float(beta_start + (beta_end - beta_start) * progress)


def viewpoint_positions(episode) -> List[np.ndarray]:
    points: List[np.ndarray] = []
    for goal in getattr(episode, "goals", None) or []:
        for view in getattr(goal, "view_points", None) or []:
            state = view.agent_state if hasattr(view, "agent_state") else view["agent_state"]
            position = state["position"] if isinstance(state, dict) else state.position
            points.append(np.asarray(position, dtype=np.float32))
    if points:
        return points
    for goal in getattr(episode, "goals", None) or []:
        position = goal["position"] if isinstance(goal, dict) else getattr(goal, "position", None)
        if position is not None:
            points.append(np.asarray(position, dtype=np.float32))
    return points


def _action_names(task) -> List[str]:
    if hasattr(task, "_action_keys"):
        return list(task._action_keys)
    names = []
    index = 0
    while True:
        try:
            names.append(task.get_action_name(index))
        except (ValueError, IndexError):
            break
        index += 1
    return names


def _nearest_viewpoint(sim, episode, points: Sequence[np.ndarray]):
    agent = np.asarray(sim.get_agent_state().position, dtype=np.float32)
    distance = float(sim.geodesic_distance(agent, points, episode))
    if not np.isfinite(distance):
        return None, distance
    path = getattr(episode, "_shortest_path_cache", None)
    index = int(getattr(path, "closest_end_point_index", 0))
    if index < 0 or index >= len(points):
        return None, distance
    return points[index], distance


def query_shortest_path_expert(rl_env, follower=None) -> Tuple[int, bool, object]:
    """Expert action for the current state of a Habitat ``RLEnv``.

    Returns ``(action_index, valid, follower)``. ``valid`` is false when no
    viewpoint is reachable; the action is then ``stop`` and should be ignored
    by the imitation loss.
    """
    habitat_env = rl_env.habitat_env
    sim = habitat_env.sim
    episode = habitat_env.current_episode
    names = _action_names(habitat_env.task)
    stop = map_follower_action(0, names)
    points = viewpoint_positions(episode)
    if not points:
        return stop, False, follower

    goal, distance = _nearest_viewpoint(sim, episode, points)
    if goal is None:
        return stop, False, follower

    try:
        radius = float(rl_env.config.task.measurements.success.success_distance)
    except Exception:
        radius = DEFAULT_GOAL_RADIUS
    if distance <= radius:
        return stop, True, follower

    if follower is None or getattr(follower, "_goal_radius", None) != radius:
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

        follower = ShortestPathFollower(
            sim,
            goal_radius=radius,
            return_one_hot=False,
            stop_on_error=False,
        )
    try:
        sim_action = follower.get_next_action(goal)
        if sim_action is None:
            return stop, False, follower
        return map_follower_action(int(sim_action), names), True, follower
    except Exception:
        return stop, False, follower
