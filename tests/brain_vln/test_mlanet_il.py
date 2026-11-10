"""MLANet EnvDrop / DAgger schedule and the inflection-weighted loss."""

import torch

from spikingnav.brain_vln.mlanet_il import (
    DAGGER_EPOCHS,
    DAGGER_ITERATIONS,
    DAGGER_P,
    DAGGER_UPDATE_SIZE,
    ENVDROP_EPOCHS,
    ENVDROP_P,
    ENVDROP_UPDATE_SIZE,
    INFLECTION_COEF,
    dagger_beta,
    inflection_weights,
    partition_episodes,
    select_dagger_episodes,
    shard_dagger_demos,
    weighted_action_loss,
)


def test_published_schedule_matches_the_yaml_files():
    assert ENVDROP_EPOCHS == 45
    assert ENVDROP_P == 1.0
    assert ENVDROP_UPDATE_SIZE == 157232
    assert DAGGER_EPOCHS == 4
    assert DAGGER_ITERATIONS == 10
    assert DAGGER_UPDATE_SIZE == 5000
    assert DAGGER_P == 0.5
    assert INFLECTION_COEF == 3.2


def test_dagger_beta_decays_only_after_a_loaded_checkpoint():
    assert dagger_beta(1.0, 0) == 1.0
    assert dagger_beta(1.0, 1) == 1.0
    assert dagger_beta(0.5, 1) == 0.5
    assert dagger_beta(0.5, 2) == 0.25
    assert dagger_beta(0.5, 10) == 0.5**10
    assert dagger_beta(0.0, 3) == 0.0
    assert dagger_beta(1.5, 4) == 0.5


def test_inflection_weight_marks_the_first_step_and_action_changes():
    actions = torch.tensor([1, 1, 2, 2])
    weights = inflection_weights(actions, 3.2)
    assert torch.allclose(weights, torch.tensor([3.2, 1.0, 3.2, 1.0]))


def test_weighted_loss_is_the_mean_of_per_trajectory_means():
    logits = torch.zeros(2, 2, 4)
    logits[0, 0, 1] = 10.0
    logits[1, 0, 1] = 10.0
    logits[0, 1, 0] = 10.0
    actions = torch.tensor([[1, 0], [1, 1]])
    weights = torch.tensor([[3.2, 1.0], [1.0, 0.0]])
    loss = weighted_action_loss(logits, actions, weights)
    first = torch.nn.functional.cross_entropy(logits[:, 0], actions[:, 0], reduction="none")
    first = (first * weights[:, 0]).sum() / weights[:, 0].sum()
    second = torch.nn.functional.cross_entropy(logits[:1, 1], actions[:1, 1], reduction="none")
    expected = (first + second) / 2
    assert torch.allclose(loss, expected)


def test_dagger_shards_partition_scenes_without_dropping_episodes():
    demos = [{"episode_id": str(index), "scene_id": f"scene-{index % 7}"} for index in range(40)]
    chosen = select_dagger_episodes(demos, iteration=3, seed=0, limit=30)
    assert len(chosen) == 30
    covered = []
    scene_shards = {}
    for shard in range(4):
        part = shard_dagger_demos(chosen, 4, shard)
        covered.extend(part)
        for demo in part:
            scene_shards.setdefault(demo["scene_id"], set()).add(shard)
    assert sorted((demo["scene_id"], demo["episode_id"]) for demo in covered) == sorted(
        (demo["scene_id"], demo["episode_id"]) for demo in chosen
    )
    assert all(len(shards) == 1 for shards in scene_shards.values())


def test_episode_parts_cover_a_shard_without_overlap():
    demos = [{"episode_id": str(index)} for index in range(10)]
    assert partition_episodes(demos, 0, 1) == demos
    covered = []
    for part in range(4):
        covered.extend(partition_episodes(demos, part, 4))
    assert sorted(demo["episode_id"] for demo in covered) == [demo["episode_id"] for demo in demos]
    assert len(covered) == len({demo["episode_id"] for demo in covered})
