import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, Union
import random


class RubiksCubeEnv(gym.Env):
    """
    A Gymnasium environment for the Rubik's Cube puzzle.

    This environment allows for n x n Rubik's Cubes, with the standard 3x3x3 as the default.
    The state is represented as a 6x(n*n) array where each row corresponds to a face of the cube
    and each element represents the colour of a specific cubelet.

    Attributes:
        n (int): The dimension of the cube (n x n x n).
        action_space (spaces.Discrete): The space of possible actions.
        observation_space (spaces.Box): The space of possible observations.
        state (np.ndarray): The current state of the cube.
        max_steps (int): Maximum number of steps per episode.
        current_step (int): Current step count in the current episode.
        reward_type (str): Type of reward function to use. Options:
                         - "sparse": Only reward for solving the cube (1) or no reward (0)
                         - "dense": Reward based on number of correctly placed cubelets
                         - "distance": Reward based on distance to solved state
        state_type (str): Type of state representation to use. Options:
                        - "flat": Original 6x(n*n) array representation
                        - "onehot": One-hot encoded representation
                        - "corner_edge": Corner and edge orientation representation
        step_penalty (float): Penalty for each step taken (to encourage faster solutions)
        truncation_penalty (float): Penalty when episode is truncated due to max steps
    """

    # Define the mapping of colours to numbers
    # 0: White, 1: Yellow, 2: Blue, 3: Green, 4: Red, 5: Orange
    COLOURS = {"WHITE": 0, "YELLOW": 1, "BLUE": 2, "GREEN": 3, "RED": 4, "ORANGE": 5}

    # Define the face indices for easier reference
    FACES = {
        "UP": 0,  # White
        "DOWN": 1,  # Yellow
        "FRONT": 2,  # Blue
        "BACK": 3,  # Green
        "RIGHT": 4,  # Red
        "LEFT": 5,  # Orange
    }

    def __init__(
        self,
        n: int = 3,
        max_steps: int = 100,
        reward_type: str = "dense",
        state_type: str = "flat",
        extended_action_space: bool = False,
        step_penalty: float = 0.1,
        truncation_penalty: float = 10.0,
    ):
        """
        Initialise the Rubik's Cube environment.

        Args:
            n (int): The dimension of the cube (n x n x n).
            max_steps (int): Maximum number of steps per episode.
            reward_type (str): Type of reward function to use. Options:
                             - "sparse": Only reward for solving the cube (1) or no reward (0)
                             - "dense": Reward based on number of correctly placed cubelets
                             - "distance": Reward based on distance to solved state
            state_type (str): Type of state representation to use. Options:
                            - "flat": Original 6x(n*n) array representation
                            - "onehot": One-hot encoded representation
                            - "corner_edge": Corner and edge orientation representation
            extended_action_space (bool): Whether to use the extended action space with inner layer moves
                                        and double turns. If True:
                                        - For n=3: 18 actions (6 faces × 3 rotations)
                                        - For n>3: 6 + 6*(n-2) faces × 3 rotations
            step_penalty (float): Penalty for each step taken (to encourage faster solutions)
            truncation_penalty (float): Penalty when episode is truncated due to max steps
        """
        self.n = n
        self.max_steps = max_steps
        self.current_step = 0
        self.reward_type = reward_type
        self.state_type = state_type
        self.extended_action_space = extended_action_space
        self.step_penalty = step_penalty
        self.truncation_penalty = truncation_penalty

        # Define action space
        if extended_action_space:
            if n == 3:
                # For 3x3x3: 6 faces × 3 rotations (90° CW, 90° CCW, 180°)
                self.num_actions = 18
            else:
                # For larger cubes: 6 outer faces + (n-2) inner layers per axis × 3 axes × 3 rotations
                self.num_actions = (6 + 6 * (n - 2)) * 3
        else:
            # Original action space: 6 faces × 2 rotations (CW, CCW)
            self.num_actions = 12

        self.action_space = spaces.Discrete(self.num_actions)

        # Define observation space based on state type
        if state_type == "flat":
            self.observation_space = spaces.Box(
                low=0, high=5, shape=(6, n * n), dtype=np.int8
            )
        elif state_type == "onehot":
            # One-hot encoding: each cubelet is represented by a 6-dimensional one-hot vector
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(6, n * n, 6), dtype=np.int8
            )
        elif state_type == "corner_edge":
            # For 3x3x3 cube:
            # - 8 corners with 3 orientations each
            # - 12 edges with 2 orientations each
            # - 6 face centers with 1 orientation each
            if n != 3:
                raise ValueError(
                    "Corner/edge representation only supported for 3x3x3 cubes"
                )
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(8 * 3 + 12 * 2 + 6,), dtype=np.int8
            )
        else:
            raise ValueError(f"Unknown state type: {state_type}")

        # The state representation will be a 6x(n*n) array where:
        # - Each row corresponds to a face (UP, DOWN, FRONT, BACK, RIGHT, LEFT)
        # - Each element represents the colour of a specific cubelet
        self.state = self._init_solved_state()

    def _init_solved_state(self) -> np.ndarray:
        """
        Initialise the state of a solved Rubik's Cube.

        Returns:
            np.ndarray: A 6x(n*n) array representing the solved state.
        """
        state = np.zeros((6, self.n * self.n), dtype=np.int8)

        # Fill each face with its corresponding colour
        for face_idx in range(6):
            state[face_idx, :] = face_idx

        return state

    def _convert_to_onehot(self, state: np.ndarray) -> np.ndarray:
        """
        Convert the flat state representation to one-hot encoding.

        Args:
            state (np.ndarray): The flat state representation.

        Returns:
            np.ndarray: The one-hot encoded state.
        """
        n = self.n
        onehot = np.zeros((6, n * n, 6), dtype=np.int8)

        for face in range(6):
            for pos in range(n * n):
                colour = state[face, pos]
                onehot[face, pos, colour] = 1

        return onehot

    def _convert_to_corner_edge(self, state: np.ndarray) -> np.ndarray:
        """
        Convert the flat state representation to corner/edge orientation representation.
        Only supported for 3x3x3 cubes.

        Args:
            state (np.ndarray): The flat state representation.

        Returns:
            np.ndarray: The corner/edge orientation state.
        """
        if self.n != 3:
            raise ValueError(
                "Corner/edge representation only supported for 3x3x3 cubes"
            )

        # Define corner and edge positions
        corners = [
            # UP face corners
            (0, 0),
            (0, 2),
            (0, 6),
            (0, 8),
            # DOWN face corners
            (1, 0),
            (1, 2),
            (1, 6),
            (1, 8),
        ]

        edges = [
            # UP face edges
            (0, 1),
            (0, 3),
            (0, 5),
            (0, 7),
            # Middle layer edges
            (2, 1),
            (2, 3),
            (2, 5),
            (2, 7),
            # DOWN face edges
            (1, 1),
            (1, 3),
            (1, 5),
            (1, 7),
        ]

        # Initialize the output array
        # 8 corners * 3 orientations + 12 edges * 2 orientations + 6 face centers = 54 total
        output = np.zeros(54, dtype=np.int8)

        # Encode corners (indices 0-23)
        for i, (face, pos) in enumerate(corners):
            colour = state[face, pos]
            # Each corner has 3 possible orientations
            base_idx = i * 3
            # Ensure we don't exceed array bounds
            if base_idx + 2 < 24:  # 8 corners * 3 orientations = 24
                output[base_idx + (colour % 3)] = 1

        # Encode edges (indices 24-47)
        for i, (face, pos) in enumerate(edges):
            colour = state[face, pos]
            # Each edge has 2 possible orientations
            base_idx = 24 + (i * 2)  # Start after corners
            # Ensure we don't exceed array bounds
            if base_idx + 1 < 48:  # 24 + (12 edges * 2 orientations) = 48
                output[base_idx + (colour % 2)] = 1

        # Encode face centers (indices 48-53)
        for i in range(6):
            # Center pieces are fixed and always match their face
            center_idx = 48 + i  # Start after edges
            # Ensure we don't exceed array bounds
            if center_idx < 54:  # 48 + 6 centers = 54
                output[center_idx] = 1

        return output

    def get_state(self) -> np.ndarray:
        """
        Get the current state in the specified representation.

        Returns:
            np.ndarray: The current state in the specified representation.
        """
        if self.state_type == "flat":
            return self.state
        elif self.state_type == "onehot":
            return self._convert_to_onehot(self.state)
        elif self.state_type == "corner_edge":
            return self._convert_to_corner_edge(self.state)
        else:
            raise ValueError(f"Unknown state type: {self.state_type}")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state and return the initial observation.

        Args:
            seed (Optional[int]): The seed for the random number generator.
            options (Optional[dict]): Additional options for customising the reset.
                                     If 'scramble' is in options and set to True,
                                     the cube will be scrambled.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Tuple containing initial state and info dict.
        """
        super().reset(seed=seed)
        self.current_step = 0

        # Start with a solved cube
        self.state = self._init_solved_state()

        # Scramble the cube if specified in options
        scramble_moves = 0
        if options and "scramble" in options and options["scramble"]:
            scramble_moves = options.get("scramble_moves", 20)
            self._scramble(scramble_moves)

        info = {
            "is_solved": self._is_solved(),
            "scramble_moves": scramble_moves,
            "state_type": self.state_type,
        }

        # Return the state in the specified representation
        return self.get_state(), info

    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the current state and reward type.

        Returns:
            float: The calculated reward value.
        """
        if self._is_solved():
            return 1.0  # Maximum reward for solving the cube

        if self.reward_type == "sparse":
            return 0.0  # No reward until solved

        elif self.reward_type == "dense":
            # Count the number of correctly placed cubelets
            total_cubelets = 6 * self.n * self.n
            correct_cubelets = 0

            for face_idx in range(6):
                face_values = self.state[face_idx]
                correct_colour = face_idx
                correct_cubelets += np.sum(face_values == correct_colour)

            # Normalize the reward between 0 and 1
            return correct_cubelets / total_cubelets

        elif self.reward_type == "distance":
            # Calculate the minimum number of moves needed to solve the cube
            # This is a simplified version that counts the number of faces
            # that are not completely solved
            unsolved_faces = 0
            for face_idx in range(6):
                face_values = self.state[face_idx]
                if not np.all(face_values == face_values[0]):
                    unsolved_faces += 1

            # Normalize the reward between 0 and 1
            # More unsolved faces = lower reward
            return 1.0 - (unsolved_faces / 6.0)

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment by applying an action.

        Args:
            action (int): The action to take.
            For standard action space (extended_action_space=False):
                - Actions 0-5: Clockwise turns of faces 0-5
                - Actions 6-11: Counter-clockwise turns of faces 0-5
            For extended action space (extended_action_space=True):
                For 3x3x3:
                    - Actions [0-5]: Face turns 90° clockwise
                    - Actions [6-11]: Face turns 90° counter-clockwise
                    - Actions [12-17]: Face turns 180°
                For larger cubes:
                    - Actions [0-5]: Outer face turns 90° clockwise
                    - Actions [6-11]: Outer face turns 90° counter-clockwise
                    - Actions [12-17]: Outer face turns 180°
                    - Actions [18+]: Inner layer turns (grouped by axis)

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                - The new state of the environment
                - The reward achieved by taking the action
                - Whether the episode is terminated (cube is solved)
                - Whether the episode is truncated (max steps reached)
                - Additional info about the step
        """
        self.current_step += 1

        # Apply the move
        self._apply_move(action)

        # Check if the cube is solved
        solved = self._is_solved()

        # Calculate reward based on the current state and reward type
        reward = self._calculate_reward()

        # Apply step penalty to encourage faster solutions
        reward -= self.step_penalty

        # Determine if episode is terminated or truncated
        terminated = solved
        truncated = self.current_step >= self.max_steps

        # Apply truncation penalty when episode is truncated
        if truncated:
            reward -= self.truncation_penalty

        info = {
            "is_solved": solved,
            "steps": self.current_step,
            "reward_type": self.reward_type,
            "state_type": self.state_type,
            "truncated": truncated,
        }

        # Return the state in the specified representation
        return self.get_state(), reward, terminated, truncated, info

    def _apply_move(self, action: int) -> None:
        """
        Apply a move to the current state of the cube.

        Args:
            action (int): The action to take.
            For standard action space (extended_action_space=False):
                - Actions 0-5: Clockwise turns of faces 0-5
                - Actions 6-11: Counter-clockwise turns of faces 0-5
            For extended action space (extended_action_space=True):
                For 3x3x3:
                    - Actions [0-5]: Face turns 90° clockwise
                    - Actions [6-11]: Face turns 90° counter-clockwise
                    - Actions [12-17]: Face turns 180°
                For larger cubes:
                    - Actions [0-5]: Outer face turns 90° clockwise
                    - Actions [6-11]: Outer face turns 90° counter-clockwise
                    - Actions [12-17]: Outer face turns 180°
                    - Actions [18+]: Inner layer turns (grouped by axis)
        """
        if not self.extended_action_space:
            # Original action space
            face = action % 6
            direction = 1 if action < 6 else -1
            self._rotate_face(face, direction, layer=0)
        else:
            if self.n == 3:
                # 3x3x3 with extended actions
                face = action % 6
                if action < 12:
                    # 90-degree turns
                    direction = 1 if action < 6 else -1
                    self._rotate_face(face, direction, layer=0)
                else:
                    # 180-degree turns
                    self._rotate_face(face, 1, layer=0)
                    self._rotate_face(face, 1, layer=0)
            else:
                # Larger cubes with inner layer moves
                total_faces = 6 + 6 * (self.n - 2)  # Outer faces + inner layers
                action_group = action // total_faces  # 0: CW, 1: CCW, 2: 180°
                action_within_group = action % total_faces

                if action_within_group < 6:
                    # Outer face moves
                    face = action_within_group
                    layer = 0
                else:
                    # Inner layer moves
                    inner_action = action_within_group - 6
                    axis = inner_action // (self.n - 2)  # 0: LR, 1: UD, 2: FB
                    layer = (inner_action % (self.n - 2)) + 1  # Layer number (1 to n-2)
                    face = axis * 2  # Convert axis to face index

                if action_group == 0:
                    # 90° clockwise
                    self._rotate_face(face, 1, layer)
                elif action_group == 1:
                    # 90° counter-clockwise
                    self._rotate_face(face, -1, layer)
                else:
                    # 180° turn
                    self._rotate_face(face, 1, layer)
                    self._rotate_face(face, 1, layer)

    def _rotate_face(self, face: int, direction: int, layer: int = 0) -> None:
        """
        Rotate a face or layer of the cube and update the adjacent faces.

        Args:
            face (int): The index of the face to rotate (0-5).
            direction (int): 1 for clockwise, -1 for counter-clockwise.
            layer (int): Layer to rotate (0 for outer face, 1 to n-2 for inner layers).
        """
        n = self.n

        if layer == 0:
            # Rotate the outer face itself
            face_2d = self.state[face].reshape(n, n)
            if direction == 1:  # Clockwise
                face_2d = np.rot90(face_2d, k=3)
            else:  # Counter-clockwise
                face_2d = np.rot90(face_2d, k=1)
            self.state[face] = face_2d.flatten()

        # Helper functions for getting and setting rows/columns
        def get_row(face_idx, row_idx):
            start_idx = row_idx * n
            return self.state[face_idx][start_idx : start_idx + n].copy()

        def set_row(face_idx, row_idx, values):
            start_idx = row_idx * n
            self.state[face_idx][start_idx : start_idx + n] = values

        def get_col(face_idx, col_idx):
            return self.state[face_idx][col_idx::n].copy()

        def set_col(face_idx, col_idx, values):
            self.state[face_idx][col_idx::n] = values

        def reverse(arr):
            return arr[::-1]

        # Define which rows/columns to update based on the layer
        if face == self.FACES["UP"]:
            row_to_update = layer
            front_row = get_row(self.FACES["FRONT"], row_to_update)
            right_row = get_row(self.FACES["RIGHT"], row_to_update)
            back_row = get_row(self.FACES["BACK"], row_to_update)
            left_row = get_row(self.FACES["LEFT"], row_to_update)

            if direction == 1:
                set_row(self.FACES["LEFT"], row_to_update, front_row)
                set_row(self.FACES["FRONT"], row_to_update, right_row)
                set_row(self.FACES["RIGHT"], row_to_update, back_row)
                set_row(self.FACES["BACK"], row_to_update, left_row)
            else:
                set_row(self.FACES["RIGHT"], row_to_update, front_row)
                set_row(self.FACES["BACK"], row_to_update, right_row)
                set_row(self.FACES["LEFT"], row_to_update, back_row)
                set_row(self.FACES["FRONT"], row_to_update, left_row)

        elif face == self.FACES["DOWN"]:
            row_to_update = n - 1 - layer
            front_row = get_row(self.FACES["FRONT"], row_to_update)
            right_row = get_row(self.FACES["RIGHT"], row_to_update)
            back_row = get_row(self.FACES["BACK"], row_to_update)
            left_row = get_row(self.FACES["LEFT"], row_to_update)

            if direction == 1:
                set_row(self.FACES["RIGHT"], row_to_update, front_row)
                set_row(self.FACES["BACK"], row_to_update, right_row)
                set_row(self.FACES["LEFT"], row_to_update, back_row)
                set_row(self.FACES["FRONT"], row_to_update, left_row)
            else:
                set_row(self.FACES["LEFT"], row_to_update, front_row)
                set_row(self.FACES["FRONT"], row_to_update, right_row)
                set_row(self.FACES["RIGHT"], row_to_update, back_row)
                set_row(self.FACES["BACK"], row_to_update, left_row)

        elif face == self.FACES["FRONT"]:
            col_to_update = layer
            up_row = get_row(self.FACES["UP"], n - 1 - col_to_update)
            right_col = get_col(self.FACES["RIGHT"], col_to_update)
            down_row = get_row(self.FACES["DOWN"], col_to_update)
            left_col = get_col(self.FACES["LEFT"], n - 1 - col_to_update)

            if direction == 1:
                set_col(self.FACES["RIGHT"], col_to_update, up_row)
                set_row(self.FACES["DOWN"], col_to_update, reverse(right_col))
                set_col(self.FACES["LEFT"], n - 1 - col_to_update, down_row)
                set_row(self.FACES["UP"], n - 1 - col_to_update, reverse(left_col))
            else:
                set_col(self.FACES["LEFT"], n - 1 - col_to_update, reverse(up_row))
                set_row(self.FACES["UP"], n - 1 - col_to_update, right_col)
                set_col(self.FACES["RIGHT"], col_to_update, reverse(down_row))
                set_row(self.FACES["DOWN"], col_to_update, left_col)

        elif face == self.FACES["BACK"]:
            col_to_update = n - 1 - layer
            up_row = get_row(self.FACES["UP"], col_to_update)
            left_col = get_col(self.FACES["LEFT"], col_to_update)
            down_row = get_row(self.FACES["DOWN"], n - 1 - col_to_update)
            right_col = get_col(self.FACES["RIGHT"], n - 1 - col_to_update)

            if direction == 1:
                set_col(self.FACES["LEFT"], col_to_update, up_row)
                set_row(self.FACES["DOWN"], n - 1 - col_to_update, reverse(left_col))
                set_col(self.FACES["RIGHT"], n - 1 - col_to_update, down_row)
                set_row(self.FACES["UP"], col_to_update, reverse(right_col))
            else:
                set_col(self.FACES["RIGHT"], n - 1 - col_to_update, reverse(up_row))
                set_row(self.FACES["UP"], col_to_update, left_col)
                set_col(self.FACES["LEFT"], col_to_update, reverse(down_row))
                set_row(self.FACES["DOWN"], n - 1 - col_to_update, right_col)

        elif face == self.FACES["RIGHT"]:
            col_to_update = n - 1 - layer
            up_col = get_col(self.FACES["UP"], col_to_update)
            back_col = get_col(self.FACES["BACK"], col_to_update)
            down_col = get_col(self.FACES["DOWN"], col_to_update)
            front_col = get_col(self.FACES["FRONT"], col_to_update)

            if direction == 1:
                set_col(self.FACES["FRONT"], col_to_update, up_col)
                set_col(self.FACES["DOWN"], col_to_update, front_col)
                set_col(self.FACES["BACK"], col_to_update, reverse(down_col))
                set_col(self.FACES["UP"], col_to_update, reverse(back_col))
            else:
                set_col(self.FACES["BACK"], col_to_update, reverse(up_col))
                set_col(self.FACES["UP"], col_to_update, front_col)
                set_col(self.FACES["FRONT"], col_to_update, down_col)
                set_col(self.FACES["DOWN"], col_to_update, reverse(back_col))

        elif face == self.FACES["LEFT"]:
            col_to_update = layer
            up_col = get_col(self.FACES["UP"], col_to_update)
            front_col = get_col(self.FACES["FRONT"], col_to_update)
            down_col = get_col(self.FACES["DOWN"], col_to_update)
            back_col = get_col(self.FACES["BACK"], n - 1 - col_to_update)

            if direction == 1:
                set_col(self.FACES["BACK"], n - 1 - col_to_update, reverse(up_col))
                set_col(self.FACES["UP"], col_to_update, front_col)
                set_col(self.FACES["FRONT"], col_to_update, down_col)
                set_col(self.FACES["DOWN"], col_to_update, reverse(back_col))
            else:
                set_col(self.FACES["FRONT"], col_to_update, up_col)
                set_col(self.FACES["DOWN"], col_to_update, front_col)
                set_col(self.FACES["BACK"], n - 1 - col_to_update, reverse(down_col))
                set_col(self.FACES["UP"], col_to_update, reverse(back_col))

    def _is_solved(self) -> bool:
        """
        Check if the Rubik's Cube is solved.

        Returns:
            bool: True if the cube is solved, False otherwise.
        """
        # A cube is solved if each face has a single colour
        # For each face, check if all elements have the same value
        for face_idx in range(6):
            face_values = self.state[face_idx]
            if not np.all(face_values == face_values[0]):
                return False
        return True

    def _scramble(self, num_moves: int = 20) -> None:
        """
        Scramble the cube by applying a series of random moves.

        Args:
            num_moves (int): The number of random moves to apply.
        """
        for _ in range(num_moves):
            action = self.action_space.sample()
            self._apply_move(action)

    def render(self, mode: str = "ansi") -> Optional[str]:
        """
        Render the cube state.

        Args:
            mode (str): The rendering mode. Options:
                       - "ansi": ASCII representation (default)
                       - "3d": 3D visualization using OpenGL

        Returns:
            Optional[str]: A string representation of the cube if mode is 'ansi'.
        """
        if mode == "ansi":
            n = self.n
            output = []

            # Display the cube in a flat layout:
            #     U
            #   L F R B
            #     D

            # Helper function to get 2D representation of a face
            def get_face_2d(face_idx):
                return self.state[face_idx].reshape(n, n)

            # Define colour codes
            colour_codes = {
                0: "W",  # White
                1: "Y",  # Yellow
                2: "B",  # Blue
                3: "G",  # Green
                4: "R",  # Red
                5: "O",  # Orange
            }

            # UP face
            up_face = get_face_2d(self.FACES["UP"])
            for i in range(n):
                row = " " * (n * 2) + " "
                for j in range(n):
                    row += colour_codes[up_face[i, j]] + " "
                output.append(row)
            output.append("")

            # MIDDLE faces (LEFT, FRONT, RIGHT, BACK)
            for i in range(n):
                row = ""
                for face_idx in [
                    self.FACES["LEFT"],
                    self.FACES["FRONT"],
                    self.FACES["RIGHT"],
                    self.FACES["BACK"],
                ]:
                    face = get_face_2d(face_idx)
                    for j in range(n):
                        row += colour_codes[face[i, j]] + " "
                    row += " "
                output.append(row)
            output.append("")

            # DOWN face
            down_face = get_face_2d(self.FACES["DOWN"])
            for i in range(n):
                row = " " * (n * 2) + " "
                for j in range(n):
                    row += colour_codes[down_face[i, j]] + " "
                output.append(row)

            return "\n".join(output)

        elif mode == "3d":
            try:
                from rubixRL.visualization.cube_3d import visualize_cube

                visualize_cube(self.state, self.n)
                return None
            except ImportError:
                print(
                    "3D visualization requires PyOpenGL and Pygame. Please install them with:"
                )
                print("pip install PyOpenGL Pygame")
                return None
        else:
            raise NotImplementedError(f"Render mode {mode} is not supported.")


# Example usage and testing
if __name__ == "__main__":
    # Create a 3x3 Rubik's Cube environment
    env = RubiksCubeEnv(n=3)

    # Reset the environment with scrambling
    state, info = env.reset(options={"scramble": True, "scramble_moves": 10})

    # Display the initial state
    print("Initial state after scrambling:")
    print(env.render())
    print(f"Is solved: {info['is_solved']}")

    # Take some random actions
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        print(f"\nAction {i+1}: {action}")
        print(env.render())
        print(f"Reward: {reward}, Is solved: {info['is_solved']}")

        if terminated or truncated:
            break
