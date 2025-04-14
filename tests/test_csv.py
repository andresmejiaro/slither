# tests/test_csv.py
def test_sample_game(sample_game):
    # Check that the DataFrame is not empty.
    assert sample_game.height > 0, "The DataFrame should not be empty"
    
    # Check for an expected column, e.g. "game_id".
    assert "game_id" in sample_game.columns, "Missing 'game_id' column in the DataFrame"
    
    # Optionally, you can add other checks, for example:
    # assert sample_game.width == expected_width, "Unexpected number of columns"