import pytest
import polars as pl


@pytest.fixture
def sample_game():
    game = pl.read_csv("tests/testlogs.csv")
    return game

@pytest.fixture
def sample_action():
    game = pl.read_csv("tests/test_action.csv")
    return game


@pytest.fixture
def sample_event():
    game = pl.read_csv("tests/test_event.csv")
    return game


@pytest.fixture
def sample_closest():
    game = pl.read_csv("tests/test_closest.csv")
    return game

@pytest.fixture
def sample_align_status_rewards():
    game = pl.read_csv("tests/test_align_status_rewards.csv")
    return game