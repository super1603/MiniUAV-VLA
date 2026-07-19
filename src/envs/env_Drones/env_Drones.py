import itertools
import numpy as np


ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
    4: (0, 0),   # stay
}


class Drones(object):
    def __init__(self, pos, view_range):
        self.pos = list(pos)
        self.view_range = view_range


class Human(object):
    def __init__(self, pos):
        self.pos = list(pos)


class EnvDrones(object):
    def __init__(
        self,
        map_size,
        drone_num,
        view_range,
        tree_num,
        human_num,
        episode_limit=100,
        wall_prob=0.01,
        seed=None,
        human_stay_prob=0.2,
        human_random_move_prob=0.2,
        reward_new_target=5.0,
        reward_new_elimination=15.0,
        reward_view_overlap_penalty=-0.02,
        reward_collision_penalty=-0.2,
        reward_step_penalty=-0.01,
        reward_success=20.0,
        reward_timeout=0.0,
        reward_approach_coef=0.05,
        drone_start_mode="random",
        allow_drone_through_wall=False,
        allow_drone_through_tree=False,
        attack_range=None,
        mission_mode="eliminate",
    ):
        self.map_size = map_size
        self.drone_num = drone_num
        self.view_range = view_range
        self.tree_num = tree_num
        self.human_num = human_num
        self.episode_limit = episode_limit
        self.wall_prob = wall_prob

        self.human_stay_prob = human_stay_prob
        self.human_random_move_prob = human_random_move_prob

        self.reward_new_target = reward_new_target
        self.reward_new_elimination = reward_new_elimination
        self.reward_view_overlap_penalty = reward_view_overlap_penalty
        self.reward_collision_penalty = reward_collision_penalty
        self.reward_step_penalty = reward_step_penalty
        self.reward_success = reward_success
        self.reward_timeout = reward_timeout
        self.reward_approach_coef = reward_approach_coef

        self.allow_drone_through_wall = allow_drone_through_wall
        self.allow_drone_through_tree = allow_drone_through_tree

        self.drone_start_mode = drone_start_mode
        self.start_pos = [self.map_size - 1, self.map_size - 1]

        if attack_range is None:
            attack_range = max(1, int(view_range) // 2)
        self.attack_range = int(attack_range)

        if mission_mode not in ("detect", "eliminate", "both"):
            raise ValueError(
                f"mission_mode must be one of 'detect'/'eliminate'/'both', got {mission_mode}"
            )
        self.mission_mode = mission_mode

        self._prev_min_distance_to_unfinished = None
        self._seed = None
        self._rng = None

        self.land_mark_map = None
        self.human_list = []
        self.drone_list = []
        self.discovered_humans = np.zeros(self.human_num, dtype=np.float32)
        self.eliminated_humans = np.zeros(self.human_num, dtype=np.float32)
        self.step_count = 0
        self.last_drone_actions = [4 for _ in range(self.drone_num)]

        self.seed(seed)
        self.reset(seed=seed)

    def seed(self, seed=None):
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        return self._seed

    def _in_bounds(self, x, y):
        return 0 <= x < self.map_size and 0 <= y < self.map_size

    def _is_blocked_for_drone(self, x, y):
        if not self._in_bounds(x, y):
            return True
        tile = int(self.land_mark_map[x, y])
        if tile == 1 and not self.allow_drone_through_wall:
            return True
        if tile == 2 and not self.allow_drone_through_tree:
            return True
        return False

    def _is_blocked_for_human(self, x, y):
        if not self._in_bounds(x, y):
            return True
        return int(self.land_mark_map[x, y]) != 0

    def _sample_free_cell(self, occupied):
        while True:
            x = int(self._rng.integers(0, self.map_size))
            y = int(self._rng.integers(0, self.map_size))
            if self.land_mark_map[x, y] == 0 and (x, y) not in occupied:
                return [x, y]

    def _init_landmark_map(self):
        min_free_cells = self.human_num + self.drone_num
        while True:
            self.land_mark_map = np.zeros((self.map_size, self.map_size), dtype=np.int8)
            wall_mask = self._rng.random((self.map_size, self.map_size)) < self.wall_prob
            self.land_mark_map[wall_mask] = 1

            free_cells = list(zip(*np.where(self.land_mark_map == 0)))
            if len(free_cells) < min_free_cells:
                continue

            num_tree = min(self.tree_num, len(free_cells) - min_free_cells)
            if num_tree > 0:
                tree_indices = self._rng.choice(len(free_cells), size=num_tree, replace=False)
                for index in np.atleast_1d(tree_indices):
                    x, y = free_cells[int(index)]
                    self.land_mark_map[x, y] = 2

            remaining_free = int(np.sum(self.land_mark_map == 0))
            if remaining_free >= min_free_cells:
                break

    def _init_humans(self):
        self.human_list = []
        occupied = set()
        for _ in range(self.human_num):
            pos = self._sample_free_cell(occupied)
            occupied.add((pos[0], pos[1]))
            self.human_list.append(Human(pos))

    def _init_drones_with_view(self, view_range):
        self.drone_list = []
        occupied = {(human.pos[0], human.pos[1]) for human in self.human_list}
        for _ in range(self.drone_num):
            if self.drone_start_mode == "corner":
                start_x, start_y = self.start_pos
                if (
                    self._in_bounds(start_x, start_y)
                    and self.land_mark_map[start_x, start_y] == 0
                    and (start_x, start_y) not in occupied
                ):
                    pos = [start_x, start_y]
                else:
                    pos = self._sample_free_cell(occupied)
            else:
                pos = self._sample_free_cell(occupied)
            occupied.add((pos[0], pos[1]))
            self.drone_list.append(Drones(pos, view_range))

    def get_full_obs(self):
        obs = np.ones((self.map_size, self.map_size, 3), dtype=np.float32)
        wall_mask = self.land_mark_map == 1
        tree_mask = self.land_mark_map == 2

        obs[wall_mask] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs[tree_mask] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for idx, human in enumerate(self.human_list):
            if self.eliminated_humans[idx] > 0.5:
                continue
            obs[human.pos[0], human.pos[1]] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return obs

    def _drone_visible_cells(self, drone):
        radius = drone.view_range
        cells = set()
        for i in range(2 * radius - 1):
            for j in range(2 * radius - 1):
                x = i + drone.pos[0] - radius + 1
                y = j + drone.pos[1] - radius + 1
                if (radius - 1 - i) ** 2 + (radius - 1 - j) ** 2 > radius * radius:
                    continue
                if self._in_bounds(x, y):
                    cells.add((x, y))
        return cells

    def get_drone_obs(self, drone):
        obs_size = 2 * drone.view_range - 1
        obs = np.ones((obs_size, obs_size, 3), dtype=np.float32)
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1

                if (drone.view_range - 1 - i) ** 2 + (drone.view_range - 1 - j) ** 2 > drone.view_range * drone.view_range:
                    obs[i, j] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                    continue

                if not self._in_bounds(x, y):
                    obs[i, j] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
                    continue

                if self.land_mark_map[x, y] == 1:
                    obs[i, j] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                elif self.land_mark_map[x, y] == 2:
                    obs[i, j] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                for h_idx, human in enumerate(self.human_list):
                    if self.eliminated_humans[h_idx] > 0.5:
                        continue
                    if human.pos[0] == x and human.pos[1] == y:
                        obs[i, j] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                        break
        return obs

    def get_joint_obs(self):
        obs = np.full((self.map_size, self.map_size, 3), 0.5, dtype=np.float32)
        for drone in self.drone_list:
            local_obs = self.get_drone_obs(drone)
            size = local_obs.shape[0]
            for i in range(size):
                for j in range(size):
                    x = i + drone.pos[0] - drone.view_range + 1
                    y = j + drone.pos[1] - drone.view_range + 1
                    if not self._in_bounds(x, y):
                        continue
                    pix = local_obs[i, j]
                    if pix[0] == 0.5 and pix[1] == 0.5 and pix[2] == 0.5:
                        continue
                    obs[x, y] = pix
        return obs

    def rand_reset_drone_pos(self):
        occupied = {(human.pos[0], human.pos[1]) for human in self.human_list}
        for drone in self.drone_list:
            drone.pos = self._sample_free_cell(occupied)
            occupied.add((drone.pos[0], drone.pos[1]))

    def _try_move(self, pos, action, blocked_fn):
        if action not in ACTION_TO_DELTA:
            return list(pos), True

        dx, dy = ACTION_TO_DELTA[action]
        if dx == 0 and dy == 0:
            return list(pos), False

        nx = pos[0] + dx
        ny = pos[1] + dy
        if blocked_fn(nx, ny):
            return list(pos), True
        return [nx, ny], False

    def drone_step(self, drone_act_list):
        if len(drone_act_list) != self.drone_num:
            raise ValueError("drone_act_list length must match drone_num")

        original_positions = [list(drone.pos) for drone in self.drone_list]
        proposed_positions = []
        invalid_count = 0
        for drone, action in zip(self.drone_list, drone_act_list):
            candidate_pos, invalid = self._try_move(
                drone.pos,
                int(action),
                self._is_blocked_for_drone,
            )
            proposed_positions.append(candidate_pos)
            invalid_count += int(invalid)

        collision_flags = [False for _ in range(self.drone_num)]
        for i, j in itertools.combinations(range(self.drone_num), 2):
            if proposed_positions[i] == proposed_positions[j]:
                collision_flags[i] = True
                collision_flags[j] = True
            elif (
                proposed_positions[i] == original_positions[j]
                and proposed_positions[j] == original_positions[i]
                and proposed_positions[i] != original_positions[i]
            ):
                collision_flags[i] = True
                collision_flags[j] = True

        for idx, drone in enumerate(self.drone_list):
            if collision_flags[idx]:
                drone.pos = original_positions[idx]
            else:
                drone.pos = proposed_positions[idx]

        return {
            "invalid_actions": invalid_count,
            "drone_collisions": int(sum(collision_flags)),
        }

    def human_step(self, human_act_list):
        if len(human_act_list) != self.human_num:
            raise ValueError("human_act_list length must match human_num")
        occupied = set()
        original_positions = [tuple(h.pos) for h in self.human_list]
        new_positions = []
        for human, action in zip(self.human_list, human_act_list):
            candidate, _ = self._try_move(human.pos, int(action), self._is_blocked_for_human)
            new_positions.append(candidate)
        for idx, candidate in enumerate(new_positions):
            key = (candidate[0], candidate[1])
            if key in occupied:
                new_positions[idx] = list(original_positions[idx])
            else:
                occupied.add(key)
        for human, new_pos in zip(self.human_list, new_positions):
            human.pos = list(new_pos)

    def _choose_human_action(self, human):
        legal_actions = []
        for action in ACTION_TO_DELTA:
            candidate_pos, invalid = self._try_move(human.pos, action, self._is_blocked_for_human)
            if not invalid:
                legal_actions.append((action, candidate_pos))

        if not legal_actions:
            return 4

        if self._rng.random() < self.human_stay_prob:
            return 4

        visible_drones = [
            drone.pos for drone in self.drone_list if self._point_in_drone_view(drone, human.pos)
        ]

        if not visible_drones or self._rng.random() < self.human_random_move_prob:
            return int(legal_actions[int(self._rng.integers(0, len(legal_actions)))][0])

        best_score = None
        best_actions = []
        for action, candidate_pos in legal_actions:
            score = min(
                abs(candidate_pos[0] - drone_pos[0]) + abs(candidate_pos[1] - drone_pos[1])
                for drone_pos in visible_drones
            )
            if best_score is None or score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return int(best_actions[int(self._rng.integers(0, len(best_actions)))])

    def _point_in_drone_view(self, drone, point):
        dx = point[0] - drone.pos[0]
        dy = point[1] - drone.pos[1]
        return dx * dx + dy * dy <= drone.view_range * drone.view_range

    def _sample_human_actions(self):
        actions = []
        for idx, human in enumerate(self.human_list):
            if self.eliminated_humans[idx] > 0.5:
                actions.append(4)
            else:
                actions.append(self._choose_human_action(human))
        return actions

    def _update_discovered_humans(self):
        visible_cells = [self._drone_visible_cells(drone) for drone in self.drone_list]
        newly_found = 0
        for idx, human in enumerate(self.human_list):
            if self.eliminated_humans[idx] > 0.5:
                continue
            human_cell = (human.pos[0], human.pos[1])
            seen = any(human_cell in cells for cells in visible_cells)
            if seen and self.discovered_humans[idx] < 0.5:
                self.discovered_humans[idx] = 1.0
                newly_found += 1
        return newly_found, visible_cells

    def _update_eliminated_humans(self):
        newly_eliminated = 0
        attack_sq = self.attack_range * self.attack_range
        for idx, human in enumerate(self.human_list):
            if self.eliminated_humans[idx] > 0.5:
                continue
            for drone in self.drone_list:
                dx = human.pos[0] - drone.pos[0]
                dy = human.pos[1] - drone.pos[1]
                if dx * dx + dy * dy <= attack_sq:
                    self.eliminated_humans[idx] = 1.0
                    if self.discovered_humans[idx] < 0.5:
                        self.discovered_humans[idx] = 1.0
                    newly_eliminated += 1
                    break
        return newly_eliminated

    def _compute_view_overlap(self, visible_cells):
        overlap_score = 0.0
        for i, j in itertools.combinations(range(len(visible_cells)), 2):
            cell_i = visible_cells[i]
            cell_j = visible_cells[j]
            if not cell_i or not cell_j:
                continue
            overlap = len(cell_i.intersection(cell_j))
            overlap_score += overlap / max(1, min(len(cell_i), len(cell_j)))
        return overlap_score

    def _compose_reward(
        self,
        newly_found,
        newly_eliminated,
        overlap_score,
        collision_events,
        terminated,
        truncated,
        approach_delta,
    ):
        reward = self.reward_step_penalty
        reward += self.reward_new_target * newly_found
        reward += self.reward_new_elimination * newly_eliminated
        reward += self.reward_view_overlap_penalty * overlap_score
        reward += self.reward_collision_penalty * collision_events
        reward += self.reward_approach_coef * approach_delta

        if terminated:
            reward += self.reward_success
        elif truncated:
            reward += self.reward_timeout
        return float(reward)

    def _active_human_mask(self):
        """Mask of humans that still need to be handled based on mission_mode."""
        if self.mission_mode == "detect":
            return self.discovered_humans < 0.5
        if self.mission_mode == "eliminate":
            return self.eliminated_humans < 0.5
        return (self.discovered_humans < 0.5) | (self.eliminated_humans < 0.5)

    def _min_distance_to_unfinished(self):
        mask = self._active_human_mask()
        unfinished = [
            human.pos for idx, human in enumerate(self.human_list) if mask[idx]
        ]
        if not unfinished or not self.drone_list:
            return 0.0
        best = None
        for human_pos in unfinished:
            for drone in self.drone_list:
                dist = abs(drone.pos[0] - human_pos[0]) + abs(drone.pos[1] - human_pos[1])
                if best is None or dist < best:
                    best = dist
        return float(best if best is not None else 0.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        options = options or {}
        view_range = int(options.get("view_range", self.view_range))
        self.view_range = view_range

        self._init_landmark_map()
        self._init_humans()
        self._init_drones_with_view(view_range=view_range)
        self.discovered_humans = np.zeros(self.human_num, dtype=np.float32)
        self.eliminated_humans = np.zeros(self.human_num, dtype=np.float32)
        self.step_count = 0
        self.last_drone_actions = [4 for _ in range(self.drone_num)]
        newly_found, _ = self._update_discovered_humans()
        newly_eliminated = self._update_eliminated_humans()
        self._prev_min_distance_to_unfinished = self._min_distance_to_unfinished()

        info = {
            "newly_found": newly_found,
            "newly_eliminated": newly_eliminated,
            "targets_found": int(np.sum(self.discovered_humans)),
            "targets_eliminated": int(np.sum(self.eliminated_humans)),
            "targets_total": self.human_num,
            "step_count": self.step_count,
            "episode_limit": False,
            "detect_success": bool(np.all(self.discovered_humans > 0.5)),
            "elimination_success": bool(np.all(self.eliminated_humans > 0.5)),
        }
        return_joint_obs = bool(options.get("return_joint_obs", True))
        if return_joint_obs:
            return self.get_joint_obs(), info
        return None, info

    def step(self, human_act_list, drone_act_list, return_joint_obs=True):
        if drone_act_list is None:
            raise ValueError("drone_act_list cannot be None")

        drone_act_list = [int(a) for a in drone_act_list]
        if human_act_list is None:
            human_act_list = self._sample_human_actions()
        else:
            human_act_list = [int(a) for a in human_act_list]

        drone_step_info = self.drone_step(drone_act_list)
        self.human_step(human_act_list)
        self.step_count += 1
        self.last_drone_actions = list(drone_act_list)

        newly_found, visible_cells = self._update_discovered_humans()
        newly_eliminated = self._update_eliminated_humans()
        overlap_score = self._compute_view_overlap(visible_cells)

        all_detected = bool(np.all(self.discovered_humans > 0.5))
        all_eliminated = bool(np.all(self.eliminated_humans > 0.5))
        hit_episode_limit = self.step_count >= self.episode_limit

        if self.mission_mode == "detect":
            terminated = all_detected
        elif self.mission_mode == "eliminate":
            terminated = all_eliminated
        else:  # both
            terminated = all_detected and all_eliminated
        truncated = hit_episode_limit and not terminated

        collision_events = (
            drone_step_info["invalid_actions"] + drone_step_info["drone_collisions"]
        )

        current_min_dist = self._min_distance_to_unfinished()
        if (
            self._prev_min_distance_to_unfinished is None
            or newly_found > 0
            or newly_eliminated > 0
        ):
            approach_delta = 0.0
        else:
            approach_delta = float(
                self._prev_min_distance_to_unfinished - current_min_dist
            )
        self._prev_min_distance_to_unfinished = current_min_dist

        reward = self._compose_reward(
            newly_found=newly_found,
            newly_eliminated=newly_eliminated,
            overlap_score=overlap_score,
            collision_events=collision_events,
            terminated=terminated,
            truncated=truncated,
            approach_delta=approach_delta,
        )

        info = {
            "newly_found": newly_found,
            "newly_eliminated": newly_eliminated,
            "targets_found": int(np.sum(self.discovered_humans)),
            "targets_eliminated": int(np.sum(self.eliminated_humans)),
            "targets_total": self.human_num,
            "view_overlap": overlap_score,
            "invalid_actions": drone_step_info["invalid_actions"],
            "drone_collisions": drone_step_info["drone_collisions"],
            "step_count": self.step_count,
            "episode_limit": bool(truncated),
            "mission_success": bool(terminated),
            "detect_success": all_detected,
            "elimination_success": all_eliminated,
        }
        if return_joint_obs:
            return self.get_joint_obs(), reward, terminated, truncated, info
        return None, reward, terminated, truncated, info