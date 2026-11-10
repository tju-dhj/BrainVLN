from types import SimpleNamespace
import pytest
from spikingnav.habitat_objectnav.episode_selection import validate_episode_keys,resolve_selected_episodes


def test_repeated_raw_hm3d_ids_are_rejected():
    with pytest.raises(ValueError,match='Duplicate evaluation'):
        validate_episode_keys([{'content_scene':'a','episode_id':'0','category':'chair'},
                               {'content_scene':'a','episode_id':'0','category':'bed'}])
    validate_episode_keys([{'content_scene':'a','episode_id':'0'},{'content_scene':'b','episode_id':'0'}])


def test_loaded_identity_category_and_pose_are_checked():
    episode=SimpleNamespace(episode_id='3',object_category='chair',start_position=[1,2,3],start_rotation=[0,0,0,1])
    row={'content_scene':'a','episode_id':'3','category':'chair','start_position':[1,2,3]}
    assert resolve_selected_episodes([episode],[row],'a')==[episode]
    with pytest.raises(ValueError,match='category mismatch'):
        resolve_selected_episodes([episode],[{**row,'category':'bed'}],'a')
    with pytest.raises(ValueError,match='start state mismatch'):
        resolve_selected_episodes([episode],[{**row,'start_position':[2,2,3]}],'a')
    with pytest.raises(ValueError,match='Duplicate loaded'):
        resolve_selected_episodes([episode,episode],[row],'a')
