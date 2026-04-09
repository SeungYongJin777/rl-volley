from .actions import direction_name
from .actions import ACTION_NAMES
from .constants import BALL_TOUCHING_GROUND_Y_COORD
from .constants import GROUND_HALF_WIDTH
from .constants import GROUND_WIDTH
from .constants import PLAYER_TOUCHING_GROUND_Y_COORD


PLAYER_X_BUCKETS = 6
PLAYER_Y_BUCKETS = 4
BALL_X_BUCKETS = 12
BALL_Y_BUCKETS = 8
BALL_VELOCITY_BUCKETS = 5

KEY_PLAYER_X_BUCKETS = 5
KEY_BALL_Y_BUCKETS = 6
KEY_BALL_X_VELOCITY_BUCKETS = 3
KEY_BALL_Y_VELOCITY_BUCKETS = 4
KEY_LANDING_BUCKETS = 6
KEY_DISTANCE_BUCKETS = 6


def bucketize(value, minimum_value, maximum_value, bucket_count):
    if value <= minimum_value:
        return 0
    if value >= maximum_value:
        return bucket_count - 1

    span = maximum_value - minimum_value + 1
    scaled = (value - minimum_value) * bucket_count
    bucket = int(scaled / span)
    if bucket >= bucket_count:
        return bucket_count - 1
    if bucket < 0:
        return 0
    return bucket


def normalize_player_state(raw_state):
    if raw_state in (1, 2):
        return "jump"
    if raw_state in (3, 4):
        return "dive"
    if raw_state in (5, 6):
        return "end"
    return "normal"


def rebucket(bucket_value, old_bucket_count, new_bucket_count):
    if new_bucket_count <= 1:
        return 0

    bucket = int(bucket_value)
    if bucket <= 0:
        return 0
    if bucket >= old_bucket_count - 1:
        return new_bucket_count - 1

    scaled = bucket * new_bucket_count
    new_bucket = int(scaled / old_bucket_count)
    if new_bucket >= new_bucket_count:
        return new_bucket_count - 1
    if new_bucket < 0:
        return 0
    return new_bucket


def player_state_code(state_name):
    if state_name == "jump":
        return "j"
    if state_name == "dive":
        return "d"
    if state_name == "end":
        return "e"
    return "n"


def _get_perspective_values(engine, perspective_player_id, direction_memory, last_action_names, scores):
    if perspective_player_id == 0:
        self_player = engine.players[0]
        opponent_player = engine.players[1]
        self_x = self_player.x
        opponent_x = opponent_player.x
        ball_x = engine.ball.x
        ball_velocity_x = engine.ball.x_velocity
        expected_landing_x = engine.ball.expected_landing_point_x
        self_score = scores["player1"]
        opponent_score = scores["player2"]
    else:
        self_player = engine.players[1]
        opponent_player = engine.players[0]
        self_x = GROUND_WIDTH - self_player.x
        opponent_x = GROUND_WIDTH - opponent_player.x
        ball_x = GROUND_WIDTH - engine.ball.x
        ball_velocity_x = -engine.ball.x_velocity
        expected_landing_x = GROUND_WIDTH - engine.ball.expected_landing_point_x
        self_score = scores["player2"]
        opponent_score = scores["player1"]

    self_last_action = str(last_action_names[perspective_player_id] or "")
    opponent_last_action = str(last_action_names[1 - perspective_player_id] or "")

    self_spike_used = self_player.state == 2 or self_last_action.startswith("spike")
    opponent_spike_used = (
        opponent_player.state == 2
        or opponent_last_action.startswith("spike")
    )

    return {
        "self_player": self_player,
        "opponent_player": opponent_player,
        "self_x": self_x,
        "opponent_x": opponent_x,
        "ball_x": ball_x,
        "ball_velocity_x": ball_velocity_x,
        "expected_landing_x": expected_landing_x,
        "self_score": self_score,
        "opponent_score": opponent_score,
        "self_spike_used": self_spike_used,
        "opponent_spike_used": opponent_spike_used,
        "self_action_name": self_last_action or "idle",
        "opponent_action_name": opponent_last_action or "idle",
        "self_direction": direction_memory[perspective_player_id],
        "opponent_direction": direction_memory[1 - perspective_player_id],
    }


def _build_player_raw(player, x_value, direction_value, spike_used, action_name):
    return {
        "x": int(x_value),
        "y": int(player.y),
        "boldness": int(player.computer_boldness),
        "x_bucket": bucketize(int(x_value), 0, GROUND_WIDTH - 1, PLAYER_X_BUCKETS),
        "y_bucket": bucketize(
            int(player.y), 0, PLAYER_TOUCHING_GROUND_Y_COORD, PLAYER_Y_BUCKETS
        ),
        "state": normalize_player_state(player.state),
        "direction": direction_name(direction_value),
        "spike_used": int(spike_used),
        "action_name": str(action_name),
    }


def _build_ball_raw(x_value, y_value, x_velocity, y_velocity, expected_landing_x):
    ball_side = "self"
    if int(x_value) > GROUND_HALF_WIDTH:
        ball_side = "opponent"

    # raw state 에는 실제 위치와 실제 속도를 함께 넣는다.
    return {
        "x": int(x_value),
        "y": int(y_value),
        "x_velocity": int(x_velocity),
        "y_velocity": int(y_velocity),
        "x_bucket": bucketize(int(x_value), 0, GROUND_WIDTH - 1, BALL_X_BUCKETS),
        "y_bucket": bucketize(
            int(y_value), 0, BALL_TOUCHING_GROUND_Y_COORD, BALL_Y_BUCKETS
        ),
        "x_velocity_bucket": bucketize(int(x_velocity), -20, 20, BALL_VELOCITY_BUCKETS),
        "y_velocity_bucket": bucketize(int(y_velocity), -20, 20, BALL_VELOCITY_BUCKETS),
        "expected_landing_x": int(expected_landing_x),
        "expected_landing_bucket": bucketize(
            int(expected_landing_x), 0, GROUND_WIDTH - 1, BALL_X_BUCKETS
        ),
        "side": ball_side,
    }


def _build_training_player_raw(player, x_value, direction_value, spike_used, action_name):
    return {
        "x": int(x_value),
        "y": int(player.y),
        "boldness": int(player.computer_boldness),
        "x_bucket": bucketize(int(x_value), 0, GROUND_WIDTH - 1, PLAYER_X_BUCKETS),
        "y_bucket": bucketize(
            int(player.y), 0, PLAYER_TOUCHING_GROUND_Y_COORD, PLAYER_Y_BUCKETS
        ),
        "state": normalize_player_state(player.state),
        "direction": direction_name(direction_value),
        "spike_used": int(spike_used),
        "action_name": str(action_name),
    }


def _build_training_ball_raw(x_value, y_value, x_velocity, y_velocity, expected_landing_x):
    ball_side = "self"
    if int(x_value) > GROUND_HALF_WIDTH:
        ball_side = "opponent"

    return {
        "x": int(x_value),
        "y": int(y_value),
        "x_velocity": int(x_velocity),
        "y_velocity": int(y_velocity),
        "x_bucket": bucketize(int(x_value), 0, GROUND_WIDTH - 1, BALL_X_BUCKETS),
        "y_bucket": bucketize(
            int(y_value), 0, BALL_TOUCHING_GROUND_Y_COORD, BALL_Y_BUCKETS
        ),
        "x_velocity_bucket": bucketize(int(x_velocity), -20, 20, BALL_VELOCITY_BUCKETS),
        "y_velocity_bucket": bucketize(int(y_velocity), -20, 20, BALL_VELOCITY_BUCKETS),
        "expected_landing_x": int(expected_landing_x),
        "side": ball_side,
    }


def serialize_state(raw_state):
    self_state = raw_state["self"]
    opponent_state = raw_state["opponent"]
    ball_state = raw_state["ball"]
    landing_x = int(ball_state["expected_landing_x"])
    self_x = int(self_state["x"])
    opponent_x = int(opponent_state["x"])

    landing_bucket = bucketize(landing_x, 0, GROUND_WIDTH - 1, KEY_LANDING_BUCKETS)
    landing_distance_bucket = bucketize(
        abs(self_x - landing_x),
        0,
        GROUND_WIDTH // 2,
        KEY_DISTANCE_BUCKETS,
    )
    opponent_gap_bucket = bucketize(
        abs(opponent_x - landing_x),
        0,
        GROUND_WIDTH // 2,
        KEY_DISTANCE_BUCKETS,
    )
    ball_side_code = "s"
    if ball_state["side"] == "opponent":
        ball_side_code = "o"

    pieces = [
        f"sx{rebucket(self_state['x_bucket'], PLAYER_X_BUCKETS, KEY_PLAYER_X_BUCKETS)}",
        f"ss{player_state_code(self_state['state'])}",
        f"ox{rebucket(opponent_state['x_bucket'], PLAYER_X_BUCKETS, KEY_PLAYER_X_BUCKETS)}",
        f"by{rebucket(ball_state['y_bucket'], BALL_Y_BUCKETS, KEY_BALL_Y_BUCKETS)}",
        f"bvx{rebucket(ball_state['x_velocity_bucket'], BALL_VELOCITY_BUCKETS, KEY_BALL_X_VELOCITY_BUCKETS)}",
        f"bvy{rebucket(ball_state['y_velocity_bucket'], BALL_VELOCITY_BUCKETS, KEY_BALL_Y_VELOCITY_BUCKETS)}",
        f"lx{landing_bucket}",
        f"ld{landing_distance_bucket}",
        f"og{opponent_gap_bucket}",
        f"bs{ball_side_code}",
    ]
    return "|".join(pieces)


def build_state_view(
    engine,
    perspective_player_id,
    direction_memory,
    last_action_names,
    scores,
    rally_done,
    match_done,
    rally_step_count,
):
    perspective = _get_perspective_values(
        engine,
        perspective_player_id,
        direction_memory,
        last_action_names,
        scores,
    )

    raw_state = {
        "self": _build_player_raw(
            perspective["self_player"],
            perspective["self_x"],
            perspective["self_direction"],
            perspective["self_spike_used"],
            perspective["self_action_name"],
        ),
        "opponent": _build_player_raw(
            perspective["opponent_player"],
            perspective["opponent_x"],
            perspective["opponent_direction"],
            perspective["opponent_spike_used"],
            perspective["opponent_action_name"],
        ),
        "ball": _build_ball_raw(
            perspective["ball_x"],
            engine.ball.y,
            perspective["ball_velocity_x"],
            engine.ball.y_velocity,
            perspective["expected_landing_x"],
        ),
        "score": {
            "self": int(perspective["self_score"]),
            "opponent": int(perspective["opponent_score"]),
        },
        "rally_step": int(rally_step_count),
        "rally_done": bool(rally_done),
        "match_done": bool(match_done),
    }

    return {"key": serialize_state(raw_state), "raw": raw_state}


def build_training_state_bundle(
    engine,
    perspective_player_id,
    direction_memory,
    last_action_names,
    scores,
    rally_step_count,
):
    perspective = _get_perspective_values(
        engine,
        perspective_player_id,
        direction_memory,
        last_action_names,
        scores,
    )

    raw_state = {
        "self": _build_training_player_raw(
            perspective["self_player"],
            perspective["self_x"],
            perspective["self_direction"],
            perspective["self_spike_used"],
            perspective["self_action_name"],
        ),
        "opponent": _build_training_player_raw(
            perspective["opponent_player"],
            perspective["opponent_x"],
            perspective["opponent_direction"],
            perspective["opponent_spike_used"],
            perspective["opponent_action_name"],
        ),
        "ball": _build_training_ball_raw(
            perspective["ball_x"],
            engine.ball.y,
            perspective["ball_velocity_x"],
            engine.ball.y_velocity,
            perspective["expected_landing_x"],
        ),
        "rally_step": int(rally_step_count),
    }

    return {"raw": raw_state}
