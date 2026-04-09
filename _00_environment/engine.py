import random

import numpy as np

from .constants import BALL_RADIUS
from .constants import BALL_TOUCHING_GROUND_Y_COORD
from .constants import GROUND_HALF_WIDTH
from .constants import GROUND_WIDTH
from .constants import NET_PILLAR_HALF_WIDTH
from .constants import NET_PILLAR_TOP_BOTTOM_Y_COORD
from .constants import NET_PILLAR_TOP_TOP_Y_COORD
from .constants import PLAYER_HALF_LENGTH
from .constants import PLAYER_LENGTH
from .constants import PLAYER_TOUCHING_GROUND_Y_COORD
from .input import UserInput


class Engine:
    __slots__ = ["players", "ball", "viewer", "more_random"]

    def __init__(self, is_player1_computer, is_player2_computer, more_random):
        self.players = (
            Player(False, is_player1_computer),
            Player(True, is_player2_computer),
        )
        self.ball = Ball(False)
        self.viewer = None
        self.more_random = more_random

    def step(self, user_inputs, refresh_expected_landing=True):
        is_ball_touching_ground = self._process_collision_between_ball_and_world_and_set_ball_position()

        # Player movement does not change the ball, so one prediction per step is enough.
        if refresh_expected_landing:
            self._calculate_expected_landing_point_x_for(self.ball)

        for player_id in range(2):
            self._process_player_movement_and_set_player_position(player_id, user_inputs[player_id])

        for player_id in range(2):
            is_happening = self._is_collision_between_ball_and_player_happening(player_id)
            player = self.players[player_id]
            if is_happening:
                if not player.is_collision_with_ball_happening:
                    self._process_collision_between_ball_and_player(player_id, user_inputs[player_id])
                    player.is_collision_with_ball_happening = True
            else:
                player.is_collision_with_ball_happening = False

        return is_ball_touching_ground

    def reset(self, is_player2_serve):
        self.players[0].reset()
        self.players[1].reset()
        self.ball.reset(is_player2_serve, self.more_random)

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _is_collision_between_ball_and_player_happening(self, player_id):
        player = self.players[player_id]
        return (
            abs(self.ball.x - player.x) <= PLAYER_HALF_LENGTH
            and abs(self.ball.y - player.y) <= PLAYER_HALF_LENGTH
        )

    def _process_collision_between_ball_and_world_and_set_ball_position(self):
        self.ball.previous_previous_x = self.ball.previous_x
        self.ball.previous_previous_y = self.ball.previous_y
        self.ball.previous_x = self.ball.x
        self.ball.previous_y = self.ball.y

        self.ball.fine_rotation = (self.ball.fine_rotation + self.ball.x_velocity // 2) % 50
        self.ball.rotation = self.ball.fine_rotation // 10

        future_ball_x = self.ball.x + self.ball.x_velocity
        if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
            self.ball.x_velocity = -self.ball.x_velocity

        future_ball_y = self.ball.y + self.ball.y_velocity
        if future_ball_y < 0:
            self.ball.y_velocity = 1

        if abs(self.ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and self.ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
            if self.ball.y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
                if self.ball.y_velocity > 0:
                    self.ball.y_velocity = -self.ball.y_velocity
            else:
                if self.ball.x < GROUND_HALF_WIDTH:
                    self.ball.x_velocity = -abs(self.ball.x_velocity)
                else:
                    self.ball.x_velocity = abs(self.ball.x_velocity)

        future_ball_y = self.ball.y + self.ball.y_velocity
        if future_ball_y > BALL_TOUCHING_GROUND_Y_COORD:
            self.ball.y = BALL_TOUCHING_GROUND_Y_COORD
            self.ball.y_velocity = -self.ball.y_velocity
            self.ball.punch_effect_x = self.ball.x
            self.ball.punch_effect_y = BALL_TOUCHING_GROUND_Y_COORD + BALL_RADIUS
            self.ball.punch_effect_radius = BALL_RADIUS
            return True

        self.ball.y = future_ball_y
        self.ball.x = self.ball.x + self.ball.x_velocity
        self.ball.y_velocity += 1
        return False

    def _process_player_movement_and_set_player_position(self, player_id, user_input):
        player = self.players[player_id]

        if player.state == 4:
            player.lying_down_duration_left -= 1
            if player.lying_down_duration_left < -1:
                player.state = 0
            return

        player_velocity_x = 0
        if player.state < 5:
            if player.state < 3:
                player_velocity_x = user_input.x_direction * 6
            else:
                player_velocity_x = player.diving_direction * 8

        future_player_x = player.x + player_velocity_x
        player.x = future_player_x

        if not player.is_player2:
            if future_player_x < PLAYER_HALF_LENGTH:
                player.x = PLAYER_HALF_LENGTH
            elif future_player_x > GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH:
                player.x = GROUND_HALF_WIDTH - PLAYER_HALF_LENGTH
        else:
            if future_player_x < GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH:
                player.x = GROUND_HALF_WIDTH + PLAYER_HALF_LENGTH
            elif future_player_x > GROUND_WIDTH - PLAYER_HALF_LENGTH:
                player.x = GROUND_WIDTH - PLAYER_HALF_LENGTH

        if (
            player.state < 3
            and user_input.y_direction == -1
            and player.y == PLAYER_TOUCHING_GROUND_Y_COORD
        ):
            player.y_velocity = -16
            player.state = 1
            player.frame_number = 0

        future_player_y = player.y + player.y_velocity
        player.y = future_player_y

        if future_player_y < PLAYER_TOUCHING_GROUND_Y_COORD:
            player.y_velocity += 1
        elif future_player_y > PLAYER_TOUCHING_GROUND_Y_COORD:
            player.y_velocity = 0
            player.y = PLAYER_TOUCHING_GROUND_Y_COORD
            player.frame_number = 0

            if player.state == 3:
                player.state = 4
                player.frame_number = 0
                player.lying_down_duration_left = 3
            else:
                player.state = 0

        if user_input.power_hit == 1:
            if player.state == 1:
                player.delay_before_next_frame = 5
                player.frame_number = 0
                player.state = 2
            elif player.state == 0 and user_input.x_direction != 0:
                player.state = 3
                player.frame_number = 0
                player.diving_direction = user_input.x_direction
                player.y_velocity = -5

        if player.state == 1:
            player.frame_number = (player.frame_number + 1) % 3
        elif player.state == 2:
            if player.delay_before_next_frame < 1:
                player.frame_number += 1
                if player.frame_number > 4:
                    player.frame_number = 0
                    player.state = 1
            else:
                player.delay_before_next_frame += 1
        elif player.state == 0:
            player.delay_before_next_frame += 1
            if player.delay_before_next_frame > 3:
                player.delay_before_next_frame = 0
                future_frame_number = player.frame_number + player.normal_status_arm_swing_direction
                if future_frame_number < 0 or future_frame_number > 4:
                    player.normal_status_arm_swing_direction *= -1
                player.frame_number += player.normal_status_arm_swing_direction

        if player.game_ended:
            if player.state == 0:
                if player.is_winner:
                    player.state = 5
                else:
                    player.state = 6
                player.delay_before_next_frame = 0
                player.frame_number = 0

            self._process_game_end_frame_for(player_id)

    def _process_collision_between_ball_and_player(self, player_id, user_input):
        player = self.players[player_id]

        if self.ball.x < player.x:
            self.ball.x_velocity = -abs(self.ball.x - player.x) // 3
        elif self.ball.x > player.x:
            self.ball.x_velocity = abs(self.ball.x - player.x) // 3

        if self.ball.x_velocity == 0:
            self.ball.x_velocity = random.randint(-1, +1)

        ball_abs_y_velocity = abs(self.ball.y_velocity)
        self.ball.y_velocity = -ball_abs_y_velocity

        if ball_abs_y_velocity < 15:
            self.ball.y_velocity = -15

        if player.state == 2:
            if self.ball.x < GROUND_HALF_WIDTH:
                self.ball.x_velocity = (abs(user_input.x_direction) + 1) * 10
            else:
                self.ball.x_velocity = -(abs(user_input.x_direction) + 1) * 10

            self.ball.punch_effect_x = self.ball.x
            self.ball.punch_effect_y = self.ball.y
            self.ball.y_velocity = abs(self.ball.y_velocity) * user_input.y_direction * 2
            self.ball.punch_effect_radius = BALL_RADIUS
            self.ball.is_power_hit = True
        else:
            self.ball.is_power_hit = False

    def _process_game_end_frame_for(self, player_id):
        player = self.players[player_id]
        if player.game_ended and player.frame_number < 4:
            player.delay_before_next_frame += 1
            if player.delay_before_next_frame > 4:
                player.delay_before_next_frame = 0
                player.frame_number += 1

    def create_viewer(self, render_mode):
        if render_mode in (None, "log"):
            self.viewer = None
            return

        from .viewer import Viewer

        self.viewer = Viewer(self)
        if render_mode == "human":
            self.viewer.init_screen()

    def render(self, mode):
        if self.viewer is None:
            return None
        if mode == "human":
            self.viewer.render()
            return None
        return self.viewer.get_screen_rgb_array()

    def update_expected_landing_point(self):
        self._calculate_expected_landing_point_x_for(self.ball)

    def _expected_landing_point_x_when_power_hit(self, user_input_x_direction, user_input_y_direction, ball):
        copy_ball = CopyBall(ball.x, ball.y, ball.x_velocity, ball.y_velocity)

        if copy_ball.x < GROUND_HALF_WIDTH:
            copy_ball.x_velocity = (abs(user_input_x_direction) + 1) * 10
        else:
            copy_ball.x_velocity = -(abs(user_input_x_direction) + 1) * 10

        copy_ball.y_velocity = abs(copy_ball.y_velocity) * user_input_y_direction * 2
        loop_counter = 0

        while True:
            loop_counter += 1

            future_ball_x = copy_ball.x + copy_ball.x_velocity
            if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
                copy_ball.x_velocity = -copy_ball.x_velocity

            if copy_ball.y + copy_ball.y_velocity < 0:
                copy_ball.y_velocity = 1

            if abs(copy_ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
                if copy_ball.y <= NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    if copy_ball.y_velocity > 0:
                        copy_ball.y_velocity = -copy_ball.y_velocity
                else:
                    if copy_ball.x < GROUND_HALF_WIDTH:
                        copy_ball.x_velocity = -abs(copy_ball.x_velocity)
                    else:
                        copy_ball.x_velocity = abs(copy_ball.x_velocity)

            copy_ball.y = copy_ball.y + copy_ball.y_velocity
            if copy_ball.y > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= 1000:
                return copy_ball.x

            copy_ball.x = copy_ball.x + copy_ball.x_velocity
            copy_ball.y_velocity += 1

    def _decide_whether_input_power_hit(self, player, ball, the_other_player, user_input):
        if random.randrange(0, 2) == 0:
            y_directions = range(-1, 2)
        else:
            y_directions = range(1, -2, -1)

        for x_direction in range(1, -1, -1):
            for y_direction in y_directions:
                expected_landing_point_x = self._expected_landing_point_x_when_power_hit(
                    x_direction, y_direction, ball
                )
                is_enemy_court = (
                    expected_landing_point_x <= int(player.is_player2) * GROUND_HALF_WIDTH
                    or expected_landing_point_x >= int(player.is_player2) * GROUND_WIDTH + GROUND_HALF_WIDTH
                )
                far_from_opponent = abs(expected_landing_point_x - the_other_player.x) > PLAYER_LENGTH
                if is_enemy_court and far_from_opponent:
                    user_input.x_direction = x_direction
                    user_input.y_direction = y_direction
                    return True
        return False

    def let_computer_decide_user_input(self, player_id):
        ball = self.ball
        player = self.players[player_id]
        the_other_player = self.players[1 - player_id]
        user_input = UserInput()

        virtual_expected_landing_point_x = ball.expected_landing_point_x

        if abs(ball.x - player.x) > 100 and abs(ball.x_velocity) < player.computer_boldness + 5:
            left_boundary = int(player.is_player2) * GROUND_HALF_WIDTH
            is_outside_court = (
                ball.expected_landing_point_x <= left_boundary
                or ball.expected_landing_point_x >= GROUND_WIDTH + GROUND_HALF_WIDTH
            )
            if is_outside_court and player.computer_where_to_stand_by == 0:
                virtual_expected_landing_point_x = left_boundary + GROUND_HALF_WIDTH // 2

        if abs(virtual_expected_landing_point_x - player.x) > player.computer_boldness + 8:
            user_input.x_direction = 1 if player.x < virtual_expected_landing_point_x else -1
        elif random.randrange(0, 20) == 0:
            player.computer_where_to_stand_by = random.randrange(0, 2)

        if player.state == 0:
            should_jump = (
                abs(ball.x_velocity) < player.computer_boldness + 3
                and abs(ball.x - player.x) < PLAYER_HALF_LENGTH
                and ball.y > -36
                and ball.y < 10 * player.computer_boldness + 84
                and ball.y_velocity > 0
            )
            if should_jump:
                user_input.y_direction = -1

            left_boundary = int(player.is_player2) * GROUND_HALF_WIDTH
            right_boundary = (int(player.is_player2) + 1) * GROUND_HALF_WIDTH
            should_dive = (
                ball.expected_landing_point_x > left_boundary
                and ball.expected_landing_point_x < right_boundary
                and abs(ball.x - player.x) > player.computer_boldness * 5 + PLAYER_LENGTH
                and ball.x > left_boundary
                and ball.x < right_boundary
                and ball.y > 174
            )
            if should_dive:
                user_input.power_hit = 1
                user_input.x_direction = 1 if player.x < ball.x else -1

        elif player.state == 1 or player.state == 2:
            if abs(ball.x - player.x) > 8:
                user_input.x_direction = 1 if player.x < ball.x else -1

            if abs(ball.x - player.x) < 48 and abs(ball.y - player.y) < 48:
                will_input_power_hit = self._decide_whether_input_power_hit(
                    player, ball, the_other_player, user_input
                )
                if will_input_power_hit:
                    user_input.power_hit = 1
                    if abs(the_other_player.x - player.x) < 80 and user_input.y_direction != -1:
                        user_input.y_direction = -1

        return user_input

    def _calculate_expected_landing_point_x_for(self, ball):
        copy_ball = CopyBall(ball.x, ball.y, ball.x_velocity, ball.y_velocity)
        loop_counter = 0

        while True:
            loop_counter += 1

            future_ball_x = copy_ball.x_velocity + copy_ball.x
            if future_ball_x < BALL_RADIUS or future_ball_x > GROUND_WIDTH:
                copy_ball.x_velocity = -copy_ball.x_velocity

            if copy_ball.y + copy_ball.y_velocity < 0:
                copy_ball.y_velocity = 1

            if abs(copy_ball.x - GROUND_HALF_WIDTH) < NET_PILLAR_HALF_WIDTH and copy_ball.y > NET_PILLAR_TOP_TOP_Y_COORD:
                if copy_ball.y < NET_PILLAR_TOP_BOTTOM_Y_COORD:
                    if copy_ball.y_velocity > 0:
                        copy_ball.y_velocity = -copy_ball.y_velocity
                else:
                    if copy_ball.x < GROUND_HALF_WIDTH:
                        copy_ball.x_velocity = -abs(copy_ball.x_velocity)
                    else:
                        copy_ball.x_velocity = abs(copy_ball.x_velocity)

            copy_ball.y = copy_ball.y + copy_ball.y_velocity
            if copy_ball.y > BALL_TOUCHING_GROUND_Y_COORD or loop_counter >= 1000:
                break

            copy_ball.x = copy_ball.x + copy_ball.x_velocity
            copy_ball.y_velocity += 1

        ball.expected_landing_point_x = copy_ball.x

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class Player:
    __slots__ = [
        "is_player2",
        "is_computer",
        "diving_direction",
        "lying_down_duration_left",
        "is_winner",
        "game_ended",
        "computer_where_to_stand_by",
        "x",
        "y",
        "y_velocity",
        "is_collision_with_ball_happening",
        "state",
        "frame_number",
        "normal_status_arm_swing_direction",
        "delay_before_next_frame",
        "computer_boldness",
    ]

    def __init__(self, is_player2, is_computer):
        self.is_player2 = is_player2
        self.is_computer = is_computer
        self.diving_direction = 0
        self.lying_down_duration_left = -1
        self.is_winner = False
        self.game_ended = False
        self.computer_where_to_stand_by = 0
        self.reset()

    def reset(self):
        self.x = 36 if not self.is_player2 else GROUND_WIDTH - 36
        self.y = PLAYER_TOUCHING_GROUND_Y_COORD
        self.y_velocity = 0
        self.diving_direction = 0
        self.lying_down_duration_left = -1
        self.is_winner = False
        self.game_ended = False
        self.computer_where_to_stand_by = 0
        self.is_collision_with_ball_happening = False
        self.state = 0
        self.frame_number = 0
        self.normal_status_arm_swing_direction = 1
        self.delay_before_next_frame = 0
        self.computer_boldness = random.randrange(0, 5)


class Ball:
    __slots__ = [
        "expected_landing_point_x",
        "rotation",
        "fine_rotation",
        "punch_effect_x",
        "punch_effect_y",
        "previous_x",
        "previous_previous_x",
        "previous_y",
        "previous_previous_y",
        "x",
        "y",
        "x_velocity",
        "y_velocity",
        "punch_effect_radius",
        "is_power_hit",
    ]

    def __init__(self, is_player2_serve):
        self.reset(is_player2_serve, False)
        self.expected_landing_point_x = 0
        self.rotation = 0
        self.fine_rotation = 0
        self.punch_effect_x = 0
        self.punch_effect_y = 0
        self.previous_x = 0
        self.previous_previous_x = 0
        self.previous_y = 0
        self.previous_previous_y = 0

    def reset(self, is_player2_serve, more_random):
        self.x = 56 if not is_player2_serve else GROUND_WIDTH - 56
        self.y = 0
        self.x_velocity = 0

        if more_random:
            self.x = GROUND_HALF_WIDTH
            self.x_velocity = np.random.randint(low=-20, high=20)
            self.y_velocity = np.random.randint(low=-10, high=0)
        else:
            self.y_velocity = 1

        self.punch_effect_radius = 0
        self.is_power_hit = False


class CopyBall:
    __slots__ = ["x", "y", "x_velocity", "y_velocity"]

    def __init__(self, x, y, x_velocity, y_velocity):
        self.x = x
        self.y = y
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity
