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

    def __init__(self, n: int = 3, max_steps: int = 100):
        """
        Initialise the Rubik's Cube environment.

        Args:
            n (int): The dimension of the cube (n x n x n).
            max_steps (int): Maximum number of steps per episode.
        """
        self.n = n
        self.max_steps = max_steps
        self.current_step = 0

        # Define action space
        # For a standard 3x3 cube, there are 12 basic moves (6 faces * 2 directions)
        # For n > 3, consider inner layer moves as well, but we'll keep it simple
        # with just the 12 basic moves for demonstration purposes
        self.action_space = spaces.Discrete(12)

        # Define observation space
        # The state is a 6x(n*n) array representing the 6 faces of the cube
        # Each element takes a value from 0 to 5, representing the colours
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(6, n * n), dtype=np.int8
        )

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

        info = {"is_solved": self._is_solved(), "scramble_moves": scramble_moves}

        return self.state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment by applying an action.

        Args:
            action (int): The action to take, a number between 0 and 11.
                         Actions 0-5 represent clockwise turns of faces 0-5.
                         Actions 6-11 represent counter-clockwise turns of faces 0-5.

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

        # Calculate reward (sparse reward: 1 if solved, 0 otherwise)
        reward = 1.0 if solved else 0.0

        # Determine if episode is terminated or truncated
        terminated = solved
        truncated = self.current_step >= self.max_steps

        info = {"is_solved": solved, "steps": self.current_step}

        return self.state, reward, terminated, truncated, info

    def _apply_move(self, action: int) -> None:
        """
        Apply a move to the current state of the cube.

        Args:
            action (int): The action to take, a number between 0 and 11.
                         Actions 0-5 represent clockwise turns of faces 0-5.
                         Actions 6-11 represent counter-clockwise turns of faces 0-5.
        """
        face = action % 6
        direction = 1 if action < 6 else -1  # 1 for clockwise, -1 for counter-clockwise

        # For simplicity, we'll use a helper method that applies a rotation to a face
        # and updates the adjacent faces accordingly
        self._rotate_face(face, direction)

    def _rotate_face(self, face: int, direction: int) -> None:
        """
        Rotate a face of the cube and update the adjacent faces.

        This method handles the complex logic of rotating a face and updating
        the affected cubelets on adjacent faces.

        Args:
            face (int): The index of the face to rotate (0-5).
            direction (int): 1 for clockwise, -1 for counter-clockwise.
        """
        n = self.n

        # First, rotate the face itself
        # Convert the 1D slice of the face to a 2D array for easier rotation
        face_2d = self.state[face].reshape(n, n)

        # Rotate the 2D face array
        if direction == 1:  # Clockwise
            face_2d = np.rot90(face_2d, k=3)  # Rotate 270 degrees (same as -90 degrees)
        else:  # Counter-clockwise
            face_2d = np.rot90(face_2d, k=1)  # Rotate 90 degrees

        # Update the state with the rotated face
        self.state[face] = face_2d.flatten()

        # Now, update the adjacent faces
        # For each face rotation, we need to update the appropriate rows/columns
        # of the adjacent faces

        # Helper functions to get and set rows/columns in the flattened state
        def get_row(face_idx, row_idx):
            """Get a row from a face in the flattened state."""
            start_idx = row_idx * n
            return self.state[face_idx][start_idx : start_idx + n].copy()

        def set_row(face_idx, row_idx, values):
            """Set a row in a face in the flattened state."""
            start_idx = row_idx * n
            self.state[face_idx][start_idx : start_idx + n] = values

        def get_col(face_idx, col_idx):
            """Get a column from a face in the flattened state."""
            return self.state[face_idx][col_idx::n].copy()

        def set_col(face_idx, col_idx, values):
            """Set a column in a face in the flattened state."""
            self.state[face_idx][col_idx::n] = values

        # Helper function to reverse an array
        def reverse(arr):
            """Return a reversed copy of the array."""
            return arr[::-1]

        # UP face rotation (White)
        if face == self.FACES["UP"]:
            # Affected faces: FRONT, RIGHT, BACK, LEFT (top row of each)
            front_row = get_row(self.FACES["FRONT"], 0)
            right_row = get_row(self.FACES["RIGHT"], 0)
            back_row = get_row(self.FACES["BACK"], 0)
            left_row = get_row(self.FACES["LEFT"], 0)

            if direction == 1:  # Clockwise
                # FRONT -> LEFT, RIGHT -> FRONT, BACK -> RIGHT, LEFT -> BACK
                set_row(self.FACES["LEFT"], 0, front_row)
                set_row(self.FACES["FRONT"], 0, right_row)
                set_row(self.FACES["RIGHT"], 0, back_row)
                set_row(self.FACES["BACK"], 0, left_row)
            else:  # Counter-clockwise
                # FRONT -> RIGHT, RIGHT -> BACK, BACK -> LEFT, LEFT -> FRONT
                set_row(self.FACES["RIGHT"], 0, front_row)
                set_row(self.FACES["BACK"], 0, right_row)
                set_row(self.FACES["LEFT"], 0, back_row)
                set_row(self.FACES["FRONT"], 0, left_row)

        # DOWN face rotation (Yellow)
        elif face == self.FACES["DOWN"]:
            # Affected faces: FRONT, RIGHT, BACK, LEFT (bottom row of each)
            front_row = get_row(self.FACES["FRONT"], n - 1)
            right_row = get_row(self.FACES["RIGHT"], n - 1)
            back_row = get_row(self.FACES["BACK"], n - 1)
            left_row = get_row(self.FACES["LEFT"], n - 1)

            if direction == 1:  # Clockwise
                # FRONT -> RIGHT, RIGHT -> BACK, BACK -> LEFT, LEFT -> FRONT
                set_row(self.FACES["RIGHT"], n - 1, front_row)
                set_row(self.FACES["BACK"], n - 1, right_row)
                set_row(self.FACES["LEFT"], n - 1, back_row)
                set_row(self.FACES["FRONT"], n - 1, left_row)
            else:  # Counter-clockwise
                # FRONT -> LEFT, RIGHT -> FRONT, BACK -> RIGHT, LEFT -> BACK
                set_row(self.FACES["LEFT"], n - 1, front_row)
                set_row(self.FACES["FRONT"], n - 1, right_row)
                set_row(self.FACES["RIGHT"], n - 1, back_row)
                set_row(self.FACES["BACK"], n - 1, left_row)

        # FRONT face rotation (Blue)
        elif face == self.FACES["FRONT"]:
            # Affected faces: UP (bottom row), RIGHT (left column), DOWN (top row), LEFT (right column)
            up_row = get_row(self.FACES["UP"], n - 1)
            right_col = get_col(self.FACES["RIGHT"], 0)
            down_row = get_row(self.FACES["DOWN"], 0)
            left_col = get_col(self.FACES["LEFT"], n - 1)

            if direction == 1:  # Clockwise
                # UP -> RIGHT, RIGHT -> DOWN, DOWN -> LEFT, LEFT -> UP
                # Note: Some rotations need row/column reversal to maintain orientation
                set_col(self.FACES["RIGHT"], 0, up_row)
                set_row(self.FACES["DOWN"], 0, reverse(right_col))
                set_col(self.FACES["LEFT"], n - 1, down_row)
                set_row(self.FACES["UP"], n - 1, reverse(left_col))
            else:  # Counter-clockwise
                # UP -> LEFT, RIGHT -> UP, DOWN -> RIGHT, LEFT -> DOWN
                set_col(self.FACES["LEFT"], n - 1, reverse(up_row))
                set_row(self.FACES["UP"], n - 1, right_col)
                set_col(self.FACES["RIGHT"], 0, reverse(down_row))
                set_row(self.FACES["DOWN"], 0, left_col)

        # BACK face rotation (Green)
        elif face == self.FACES["BACK"]:
            # Affected faces: UP (top row), LEFT (left column), DOWN (bottom row), RIGHT (right column)
            up_row = get_row(self.FACES["UP"], 0)
            left_col = get_col(self.FACES["LEFT"], 0)
            down_row = get_row(self.FACES["DOWN"], n - 1)
            right_col = get_col(self.FACES["RIGHT"], n - 1)

            if direction == 1:  # Clockwise
                # UP -> LEFT, LEFT -> DOWN, DOWN -> RIGHT, RIGHT -> UP
                set_col(self.FACES["LEFT"], 0, up_row)
                set_row(self.FACES["DOWN"], n - 1, reverse(left_col))
                set_col(self.FACES["RIGHT"], n - 1, down_row)
                set_row(self.FACES["UP"], 0, reverse(right_col))
            else:  # Counter-clockwise
                # UP -> RIGHT, LEFT -> UP, DOWN -> LEFT, RIGHT -> DOWN
                set_col(self.FACES["RIGHT"], n - 1, reverse(up_row))
                set_row(self.FACES["UP"], 0, left_col)
                set_col(self.FACES["LEFT"], 0, reverse(down_row))
                set_row(self.FACES["DOWN"], n - 1, right_col)

        # RIGHT face rotation (Red)
        elif face == self.FACES["RIGHT"]:
            # Affected faces: UP (right column), BACK (left column), DOWN (right column), FRONT (right column)
            up_col = get_col(self.FACES["UP"], n - 1)
            back_col = get_col(self.FACES["BACK"], 0)
            down_col = get_col(self.FACES["DOWN"], n - 1)
            front_col = get_col(self.FACES["FRONT"], n - 1)

            if direction == 1:  # Clockwise
                # UP -> FRONT, FRONT -> DOWN, DOWN -> BACK, BACK -> UP
                set_col(self.FACES["FRONT"], n - 1, up_col)
                set_col(self.FACES["DOWN"], n - 1, front_col)
                set_col(self.FACES["BACK"], 0, reverse(down_col))
                set_col(self.FACES["UP"], n - 1, reverse(back_col))
            else:  # Counter-clockwise
                # UP -> BACK, FRONT -> UP, DOWN -> FRONT, BACK -> DOWN
                set_col(self.FACES["BACK"], 0, reverse(up_col))
                set_col(self.FACES["UP"], n - 1, front_col)
                set_col(self.FACES["FRONT"], n - 1, down_col)
                set_col(self.FACES["DOWN"], n - 1, reverse(back_col))

        # LEFT face rotation (Orange)
        elif face == self.FACES["LEFT"]:
            # Affected faces: UP (left column), FRONT (left column), DOWN (left column), BACK (right column)
            up_col = get_col(self.FACES["UP"], 0)
            front_col = get_col(self.FACES["FRONT"], 0)
            down_col = get_col(self.FACES["DOWN"], 0)
            back_col = get_col(self.FACES["BACK"], n - 1)

            if direction == 1:  # Clockwise
                # UP -> BACK, FRONT -> UP, DOWN -> FRONT, BACK -> DOWN
                set_col(self.FACES["BACK"], n - 1, reverse(up_col))
                set_col(self.FACES["UP"], 0, front_col)
                set_col(self.FACES["FRONT"], 0, down_col)
                set_col(self.FACES["DOWN"], 0, reverse(back_col))
            else:  # Counter-clockwise
                # UP -> FRONT, FRONT -> DOWN, DOWN -> BACK, BACK -> UP
                set_col(self.FACES["FRONT"], 0, up_col)
                set_col(self.FACES["DOWN"], 0, front_col)
                set_col(self.FACES["BACK"], n - 1, reverse(down_col))
                set_col(self.FACES["UP"], 0, reverse(back_col))

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
            mode (str): The rendering mode. Only 'ansi' mode is supported for now.

        Returns:
            Optional[str]: A string representation of the cube if mode is 'ansi'.
        """
        if mode != "ansi":
            raise NotImplementedError(f"Render mode {mode} is not supported.")

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
