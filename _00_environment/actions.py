from .input import UserInput
from .constants import PLAYER_TOUCHING_GROUND_Y_COORD


IDLE_ACTION_NAME = "idle"
ACTION_NAMES = (
    "forward",
    "backward",
    "jump",
    "jump_forward",
    "jump_backward",
    "dive_forward",
    "dive_backward",
    "spike_soft_up",
    "spike_soft_flat",
    "spike_soft_down",
    "spike_fast_up",
    "spike_fast_flat",
    "spike_fast_down",
)
ACTION_INDEX_BY_NAME = {}

for index, action_name in enumerate(ACTION_NAMES):
    ACTION_INDEX_BY_NAME[action_name] = index


def action_value(value):
    if value is None:
        return 0.0
    return float(value)


def normalize_action_name(action_value):
    if isinstance(action_value, int):
        if action_value < 0 or action_value >= len(ACTION_NAMES):
            raise ValueError("action index is out of range")
        return ACTION_NAMES[action_value]

    action_name = str(action_value).strip().lower()
    if action_name == IDLE_ACTION_NAME:
        return IDLE_ACTION_NAME
    if action_name not in ACTION_INDEX_BY_NAME:
        raise ValueError(f"unknown action: {action_value}")
    return action_name


def build_action_materials(action_source=None):
    materials = {}
    for action_name in ACTION_NAMES:
        materials[action_name] = 0.0

    if action_source is None:
        return materials

    if isinstance(action_source, dict):
        for action_name in ACTION_NAMES:
            materials[action_name] = action_value(action_source.get(action_name, 0.0))
        return materials

    if isinstance(action_source, (list, tuple)):
        for index, action_name in enumerate(ACTION_NAMES):
            if index < len(action_source):
                materials[action_name] = action_value(action_source[index])
        return materials

    action_name = normalize_action_name(action_source)
    if action_name in ACTION_INDEX_BY_NAME:
        materials[action_name] = 1.0
    return materials


def build_action_vector(action_source=None):
    vector = [0.0] * len(ACTION_NAMES)

    if action_source is None:
        return vector

    if isinstance(action_source, dict):
        for index, action_name in enumerate(ACTION_NAMES):
            vector[index] = action_value(action_source.get(action_name, 0.0))
        return vector

    if isinstance(action_source, (list, tuple)):
        for index in range(len(ACTION_NAMES)):
            if index < len(action_source):
                vector[index] = action_value(action_source[index])
        return vector

    action_name = normalize_action_name(action_source)
    if action_name == IDLE_ACTION_NAME:
        return vector

    if action_name in ACTION_INDEX_BY_NAME:
        vector[ACTION_INDEX_BY_NAME[action_name]] = 1.0
    return vector


def apply_action_mask(action_source=None, action_mask=None):
    vector = build_action_vector(action_source)
    if action_mask is None:
        return vector

    masked = [0.0] * len(ACTION_NAMES)
    for index in range(len(ACTION_NAMES)):
        mask_value = 1.0
        if index < len(action_mask):
            mask_value = action_value(action_mask[index])
        masked[index] = vector[index] * mask_value
    return masked


def select_action_name(action_source=None):
    vector = build_action_vector(action_source)
    selected_action = IDLE_ACTION_NAME
    selected_value = 0.0

    for index, action_name in enumerate(ACTION_NAMES):
        if vector[index] > selected_value:
            selected_action = action_name
            selected_value = vector[index]

    return selected_action, vector


def relative_to_actual_x(player_id, relative_x):
    if player_id == 0:
        return relative_x
    return -relative_x


def actual_to_relative_x(player_id, actual_x):
    if player_id == 0:
        return actual_x
    return -actual_x


def direction_name(relative_x):
    if relative_x >= 0:
        return "forward"
    return "backward"


def _relative_direction_toward_ball(player_id, player, ball, default_direction):
    actual_x = 0
    if ball.x > player.x:
        actual_x = 1
    elif ball.x < player.x:
        actual_x = -1

    if actual_x == 0:
        return default_direction
    return actual_to_relative_x(player_id, actual_x)


def _relative_direction_name(relative_x):
    if relative_x > 0:
        return "forward"
    if relative_x < 0:
        return "backward"
    return "neutral"


def _vertical_action_name(y_direction):
    if y_direction < 0:
        return "up"
    if y_direction > 0:
        return "down"
    return "flat"


def describe_user_input(player_id, user_input, player=None):
    if (
        user_input.x_direction == 0
        and user_input.y_direction == 0
        and user_input.power_hit == 0
    ):
        return "idle"

    relative_x = actual_to_relative_x(player_id, user_input.x_direction)

    if user_input.power_hit == 1:
        player_state = 0
        player_y = PLAYER_TOUCHING_GROUND_Y_COORD
        if player is not None:
            player_state = int(player.state)
            player_y = int(player.y)

        # Match the engine order:
        # 1. Grounded jump input is applied first.
        # 2. Power-hit is then resolved from the updated state.
        if (
            player_state == 0
            and user_input.y_direction == -1
            and player_y == PLAYER_TOUCHING_GROUND_Y_COORD
        ):
            player_state = 1

        if player_state == 0:
            if relative_x > 0:
                return "dive_forward"
            if relative_x < 0:
                return "dive_backward"
            return "idle"

        speed_name = "soft"
        if relative_x != 0:
            speed_name = "fast"
        vertical_name = _vertical_action_name(user_input.y_direction)
        return f"spike_{speed_name}_{vertical_name}"

    if user_input.y_direction == -1:
        if relative_x > 0:
            return "jump_forward"
        if relative_x < 0:
            return "jump_backward"
        return "jump"

    if relative_x >= 0:
        return "forward"
    return "backward"


def build_user_input(action_name, player_id, player, opponent, ball, default_direction):
    normalized_action = normalize_action_name(action_name)
    relative_x = 0
    y_direction = 0
    power_hit = 0

    if normalized_action == "idle":
        relative_x = 0
    elif normalized_action == "forward":
        relative_x = 1
    elif normalized_action == "backward":
        relative_x = -1
    elif normalized_action == "jump":
        y_direction = -1
    elif normalized_action == "jump_forward":
        relative_x = 1
        y_direction = -1
    elif normalized_action == "jump_backward":
        relative_x = -1
        y_direction = -1
    elif normalized_action == "dive_forward":
        relative_x = 1
        power_hit = 1
    elif normalized_action == "dive_backward":
        relative_x = -1
        power_hit = 1
    elif normalized_action.startswith("spike_"):
        if player.state in (1, 2):
            if normalized_action.startswith("spike_fast_"):
                relative_x = _relative_direction_toward_ball(
                    player_id, player, ball, default_direction
                )
                if relative_x == 0:
                    relative_x = default_direction
            else:
                relative_x = 0

            if normalized_action.endswith("_up"):
                y_direction = -1
            elif normalized_action.endswith("_down"):
                y_direction = 1
            else:
                y_direction = 0

            power_hit = 1
        else:
            normalized_action = "idle"

    user_input = UserInput()
    user_input.x_direction = relative_to_actual_x(player_id, relative_x)
    user_input.y_direction = y_direction
    user_input.power_hit = power_hit
    return user_input, normalized_action
