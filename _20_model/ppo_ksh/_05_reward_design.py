# Import Required Internal Libraries
from _00_environment.constants import GROUND_HALF_WIDTH


def clamp(value, minimum_value, maximum_value):
    return max(minimum_value, min(maximum_value, float(value)))


def select_mat_for_reward(materials):
    self_position = materials["self_position"]
    opponent_position = materials["opponent_position"]
    ball_position = materials["ball_position"]

    self_action_name = str(materials["self_action_name"])
    opponent_action_name = str(materials["opponent_action_name"])

    rally_total_frames_until_point = float(
        materials["rally_total_frames_until_point"]
    )

    point_scored = int(materials["point_result"]["scored"])
    point_lost = int(materials["point_result"]["lost"])

    self_spike_used = int(self_action_name.startswith("spike_"))
    self_dive_used = int(self_action_name.startswith("dive_"))

    opponent_dive_used = int(opponent_action_name.startswith("dive_"))
    opponent_spike_used = int(opponent_action_name.startswith("spike_"))

    match_won = int(materials["match_result"]["won"] > 0.5)

    self_net_distance = abs(float(self_position[0]) - float(GROUND_HALF_WIDTH))
    opponent_net_distance = abs(float(opponent_position[0]) - float(GROUND_HALF_WIDTH))

    selected_materials = {
        "self_position": self_position,
        "opponent_position": opponent_position,
        "ball_position": ball_position,
        "self_action_name": self_action_name,
        "opponent_action_name": opponent_action_name,
        "self_net_distance": self_net_distance,
        "opponent_net_distance": opponent_net_distance,
        "point_scored": point_scored,
        "point_lost": point_lost,
        "self_spike_used": self_spike_used,
        "self_dive_used": self_dive_used,
        "opponent_dive_used": opponent_dive_used,
        "opponent_spike_used": opponent_spike_used,
        "match_won": match_won,
        "rally_total_frames_until_point": rally_total_frames_until_point,
    }
    return selected_materials


def calculate_reward(materials):
    mat = select_mat_for_reward(materials)

    # 1. 최종 목표 중심 보상
    SCALE_POINT_SCORE_REWARD = 40.0
    SCALE_POINT_LOST_PENALTY = 40.0
    SCALE_MATCH_WIN_BONUS = 20.0

    # 2. 아주 작은 shaping
    SCALE_SELF_SPIKE_BONUS = 0.15
    SCALE_SELF_DIVE_BONUS = 0.05
    SCALE_OPPONENT_SPIKE_PENALTY = 0.05

    # 3. 랠리 보상은 "득점했을 때만" 아주 작게
    SCALE_RALLY_FRAME_ON_SCORE = 0.005
    SCALE_RALLY_FRAME_MAX = 1.0

    reward = 0.0

    # 최종 성과
    reward += SCALE_POINT_SCORE_REWARD * mat["point_scored"]
    reward -= SCALE_POINT_LOST_PENALTY * mat["point_lost"]
    reward += SCALE_MATCH_WIN_BONUS * mat["match_won"]

    # 행동 shaping: 아주 약하게만
    # 공격 시도를 장려하지만, reward를 지배하지 못하게 작게 둔다
    reward += SCALE_SELF_SPIKE_BONUS * mat["self_spike_used"]
    reward += SCALE_SELF_DIVE_BONUS * mat["self_dive_used"]

    # 상대 공격이 자주 나온 상황은 약한 불리함으로 반영
    reward -= SCALE_OPPONENT_SPIKE_PENALTY * mat["opponent_spike_used"]

    # 랠리 보상은 이긴 랠리에만 아주 소폭
    if mat["point_scored"] > 0.5:
        reward += min(
            mat["rally_total_frames_until_point"] * SCALE_RALLY_FRAME_ON_SCORE,
            SCALE_RALLY_FRAME_MAX,
        )

    # reward 폭 제한
    reward = clamp(reward, -50.0, 50.0)
    return reward