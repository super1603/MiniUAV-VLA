import numpy as np

from .env_Drones.env_Drones import ACTION_TO_DELTA, EnvDrones
from .multiagentenv import MultiAgentEnv


class DronesWrapper(MultiAgentEnv):
    def __init__(
        self,
        map_size,
        drone_num,
        view_range,
        tree_num,
        human_num,
        episode_limit,
        seed,
        common_reward,
        reward_scalarisation,
        obs_num_closest_humans=3,
        obs_num_closest_obstacles=6,
        obs_include_teammates=True,
        obs_only_undiscovered_humans=True,
        obs_obstacles_within_view_only=True,
        **kwargs,
    ):
        self.n_agents = int(drone_num)
        self.episode_limit = int(episode_limit)
        self.n_actions = 5
        self.common_reward = bool(common_reward)
        self.reward_scalarisation = reward_scalarisation

        self.obs_num_closest_humans = int(obs_num_closest_humans)
        self.obs_num_closest_obstacles = int(obs_num_closest_obstacles)
        self.obs_include_teammates = bool(obs_include_teammates)
        self.obs_only_undiscovered_humans = bool(obs_only_undiscovered_humans)
        self.obs_obstacles_within_view_only = bool(obs_obstacles_within_view_only)

        self._seed = seed
        self._last_actions = [4 for _ in range(self.n_agents)]
        self._obs = None
        self._state = None
        self._info = {}
        self._wall_coords_cache = None
        self._tree_coords_cache = None
        self._wall_layer_cache = None
        self._tree_layer_cache = None

        self.env = EnvDrones(
            map_size=map_size,
            drone_num=drone_num,
            view_range=view_range,
            tree_num=tree_num,
            human_num=human_num,
            episode_limit=episode_limit,
            seed=seed,
            **kwargs,
        )

        self._update_cache()

    def _normalise_coord(self, v):
        if self.env.map_size <= 1:
            return 0.0
        return float(v) / float(self.env.map_size - 1)

    def _normalise_delta(self, dv):
        return float(dv) / float(max(1, self.env.map_size - 1))

    def _is_visible(self, drone_pos, target_pos):
        dx = target_pos[0] - drone_pos[0]
        dy = target_pos[1] - drone_pos[1]
        return dx * dx + dy * dy <= self.env.view_range * self.env.view_range

    def _get_local_occupancy(self, drone_pos, self_idx):
        radius = self.env.view_range
        size = 2 * radius - 1
        occupancy = np.zeros((size, size, 4), dtype=np.float32)
        human_pos_set = {
            (h.pos[0], h.pos[1])
            for idx, h in enumerate(self.env.human_list)
            if self.env.eliminated_humans[idx] < 0.5
        }
        teammate_pos_set = {
            (d.pos[0], d.pos[1])
            for i, d in enumerate(self.env.drone_list)
            if i != self_idx
        }

        for i in range(size):
            for j in range(size):
                x = i + drone_pos[0] - radius + 1
                y = j + drone_pos[1] - radius + 1

                if not (0 <= x < self.env.map_size and 0 <= y < self.env.map_size):
                    occupancy[i, j, 0] = 1.0
                    continue

                tile = int(self.env.land_mark_map[x, y])
                if tile == 1:
                    occupancy[i, j, 0] = 1.0
                elif tile == 2:
                    occupancy[i, j, 1] = 1.0
                if (x, y) in human_pos_set:
                    occupancy[i, j, 2] = 1.0
                if (x, y) in teammate_pos_set:
                    occupancy[i, j, 3] = 1.0
        return occupancy.reshape(-1)

    def _build_agent_obs(self, agent_id):
        drone = self.env.drone_list[agent_id]
        drone_pos = drone.pos
        obs = []

        obs.extend([self._normalise_coord(drone_pos[0]), self._normalise_coord(drone_pos[1])])

        if self.obs_include_teammates:
            teammate_features = []
            for idx, other in enumerate(self.env.drone_list):
                if idx == agent_id:
                    continue
                dx = other.pos[0] - drone_pos[0]
                dy = other.pos[1] - drone_pos[1]
                dist = abs(dx) + abs(dy)
                teammate_features.append((dist, dx, dy))
            teammate_features.sort(key=lambda x: x[0])
            for _, dx, dy in teammate_features:
                obs.extend([self._normalise_delta(dx), self._normalise_delta(dy)])
            while len(teammate_features) < self.n_agents - 1:
                obs.extend([0.0, 0.0])
                teammate_features.append(None)

        human_features = []
        for idx, human in enumerate(self.env.human_list):
            discovered = float(self.env.discovered_humans[idx])
            eliminated = float(self.env.eliminated_humans[idx])
            if eliminated >= 0.5:
                continue
            if self.obs_only_undiscovered_humans and discovered >= 0.5:
                continue
            dx = human.pos[0] - drone_pos[0]
            dy = human.pos[1] - drone_pos[1]
            dist = abs(dx) + abs(dy)
            human_features.append((dist, idx, dx, dy, discovered))
        human_features.sort(key=lambda x: x[0])

        for k in range(self.obs_num_closest_humans):
            if k < len(human_features):
                _, h_idx, dx, dy, discovered = human_features[k]
                visible = 1.0 if self._is_visible(drone_pos, self.env.human_list[h_idx].pos) else 0.0
                obs.extend(
                    [self._normalise_delta(dx), self._normalise_delta(dy), visible, discovered]
                )
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        if self.obs_obstacles_within_view_only:
            radius = self.env.view_range
            x0 = max(0, drone_pos[0] - radius + 1)
            x1 = min(self.env.map_size, drone_pos[0] + radius)
            y0 = max(0, drone_pos[1] - radius + 1)
            y1 = min(self.env.map_size, drone_pos[1] + radius)
            local_map = self.env.land_mark_map[x0:x1, y0:y1]
            local_walls = np.argwhere(local_map == 1)
            local_trees = np.argwhere(local_map == 2)
            wall_iter = [(int(x) + x0, int(y) + y0) for x, y in local_walls]
            tree_iter = [(int(x) + x0, int(y) + y0) for x, y in local_trees]
        else:
            wall_iter = self._wall_coords_cache
            tree_iter = self._tree_coords_cache

        obstacle_cells = []
        for x, y in wall_iter:
            dx = x - drone_pos[0]
            dy = y - drone_pos[1]
            obstacle_cells.append((abs(dx) + abs(dy), dx, dy, 1.0, 0.0))
        for x, y in tree_iter:
            dx = x - drone_pos[0]
            dy = y - drone_pos[1]
            obstacle_cells.append((abs(dx) + abs(dy), dx, dy, 0.0, 1.0))
        obstacle_cells.sort(key=lambda x: x[0])

        for k in range(self.obs_num_closest_obstacles):
            if k < len(obstacle_cells):
                _, dx, dy, is_wall, is_tree = obstacle_cells[k]
                obs.extend([self._normalise_delta(dx), self._normalise_delta(dy), is_wall, is_tree])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        obs.extend(self._get_local_occupancy(drone_pos, agent_id).tolist())

        last_action = [0.0 for _ in range(self.n_actions)]
        last_action[self._last_actions[agent_id]] = 1.0
        obs.extend(last_action)

        obs.extend(self.env.discovered_humans.astype(np.float32).tolist())
        obs.extend(self.env.eliminated_humans.astype(np.float32).tolist())

        time_ratio = float(self.env.step_count) / float(max(1, self.episode_limit))
        remain_ratio = 1.0 - time_ratio
        obs.extend([time_ratio, remain_ratio])

        return np.array(obs, dtype=np.float32)

    def _build_state(self):
        state = []

        for drone in self.env.drone_list:
            state.extend([self._normalise_coord(drone.pos[0]), self._normalise_coord(drone.pos[1])])

        for human in self.env.human_list:
            state.extend([self._normalise_coord(human.pos[0]), self._normalise_coord(human.pos[1])])

        state.extend(self._wall_layer_cache)
        state.extend(self._tree_layer_cache)

        state.extend(self.env.discovered_humans.astype(np.float32).tolist())
        state.extend(self.env.eliminated_humans.astype(np.float32).tolist())

        time_ratio = float(self.env.step_count) / float(max(1, self.episode_limit))
        remain_ratio = float(max(0, self.episode_limit - self.env.step_count)) / float(max(1, self.episode_limit))
        state.extend([time_ratio, remain_ratio])
        return np.array(state, dtype=np.float32)

    def _rebuild_map_cache(self):
        wall_coords = np.argwhere(self.env.land_mark_map == 1)
        tree_coords = np.argwhere(self.env.land_mark_map == 2)
        self._wall_coords_cache = [(int(x), int(y)) for x, y in wall_coords]
        self._tree_coords_cache = [(int(x), int(y)) for x, y in tree_coords]
        self._wall_layer_cache = (
            (self.env.land_mark_map == 1).astype(np.float32).reshape(-1).tolist()
        )
        self._tree_layer_cache = (
            (self.env.land_mark_map == 2).astype(np.float32).reshape(-1).tolist()
        )

    def _update_cache(self):
        if self._wall_layer_cache is None:
            self._rebuild_map_cache()
        self._obs = [self._build_agent_obs(i) for i in range(self.n_agents)]
        self._state = self._build_state()

    def step(self, actions):
        actions = [int(a) for a in actions]
        _, reward, terminated, truncated, info = self.env.step(
            None, actions, return_joint_obs=False
        )
        self._last_actions = list(actions)
        self._info = info
        self._update_cache()

        if self.common_reward:
            wrapped_reward = float(reward)
        else:
            if self.reward_scalarisation == "mean":
                per_agent_reward = float(reward) / float(max(1, self.n_agents))
            else:
                per_agent_reward = float(reward)
            wrapped_reward = np.array(
                [per_agent_reward for _ in range(self.n_agents)],
                dtype=np.float32,
            )
        return self._obs, wrapped_reward, terminated, truncated, info

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return int(self._obs[0].shape[0])

    def get_state(self):
        return self._state

    def get_state_size(self):
        return int(self._state.shape[0])

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        drone = self.env.drone_list[agent_id]
        avail = [0 for _ in range(self.n_actions)]
        for action in range(self.n_actions):
            dx, dy = ACTION_TO_DELTA[action]
            nx = drone.pos[0] + dx
            ny = drone.pos[1] + dy
            if action == 4:
                avail[action] = 1
            elif self.env._is_blocked_for_drone(nx, ny):
                avail[action] = 0
            else:
                avail[action] = 1
        return avail

    def get_total_actions(self):
        return self.n_actions

    def reset(self, seed=None, options=None):
        reset_options = {} if options is None else dict(options)
        reset_options["return_joint_obs"] = False
        _, info = self.env.reset(seed=seed, options=reset_options)
        self._last_actions = [4 for _ in range(self.n_agents)]
        self._info = info
        self._rebuild_map_cache()
        self._update_cache()
        return self._obs, info

    def render(self):
        return self.env.get_joint_obs()

    def close(self):
        return None

    def seed(self, seed=None):
        self._seed = seed
        return self.env.seed(seed)

    def save_replay(self):
        return None

    def get_stats(self):
        return {}
