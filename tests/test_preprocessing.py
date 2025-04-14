import polars as pl
from polars.testing import assert_frame_equal
from src.preprocessing import add_action, add_event, add_closest,align_status_rewards, add_reward


def test_add_action(sample_game, sample_action):
    assert_frame_equal(add_action(sample_game),sample_action), "Actions are not correctly added"


def test_add_event(sample_action, sample_event):
    assert_frame_equal(add_event(sample_action), sample_event)

def test_add_closest(sample_event, sample_closest):
    assert_frame_equal(add_closest(sample_event), sample_closest)    

def test_align_status_rewards(sample_closest, sample_align_status_rewards):
    assert_frame_equal(align_status_rewards(sample_closest), sample_align_status_rewards)

def test_add_reward():
    rewards = {"0": 3, "W": 5, "R": 7, "G": 9, "S": 11}
    df = pl.DataFrame({"event":["0","W","R","G","S"]})
    print(df.schema)
    df_final=pl.DataFrame({"event":["0","W","R","G","S"],"rewards":[3,5,7,9,11]}, schema={'event': pl.String, "rewards":pl.Int32})
    assert_frame_equal(add_reward(df, rewards),df_final)



