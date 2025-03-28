import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from typing import Tuple, List, Optional
import math


class Cube3D:
    """3D visualization of a Rubik's Cube using PyOpenGL."""

    # Define colors for the cube faces
    COLORS = {
        0: (1.0, 1.0, 1.0),  # White
        1: (1.0, 1.0, 0.0),  # Yellow
        2: (0.0, 0.0, 1.0),  # Blue
        3: (0.0, 1.0, 0.0),  # Green
        4: (1.0, 0.0, 0.0),  # Red
        5: (1.0, 0.5, 0.0),  # Orange
    }

    def __init__(self, n: int = 3, size: float = 1.0):
        """
        Initialize the 3D cube visualization.

        Args:
            n (int): The dimension of the cube (n x n x n)
            size (float): The size of the cube in 3D space
        """
        self.n = n
        self.size = size
        self.cubelet_size = size / n

        # Initialize Pygame and OpenGL
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        # Set up the perspective
        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        # Initialize rotation angles
        self.rot_x = 0
        self.rot_y = 0

        # Initialize cube state
        self.state = np.zeros((6, n * n), dtype=np.int8)
        for face in range(6):
            self.state[face, :] = face

    def update_state(self, state: np.ndarray) -> None:
        """
        Update the cube state for visualization.

        Args:
            state (np.ndarray): The new state of the cube
        """
        self.state = state

    def draw_cubelet(
        self, x: float, y: float, z: float, color: Tuple[float, float, float]
    ) -> None:
        """
        Draw a single cubelet at the specified position.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            z (float): Z coordinate
            color (Tuple[float, float, float]): RGB color tuple
        """
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3fv(color)

        # Draw the cubelet faces
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(-self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)

        # Back face
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            -self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2)
        glVertex3f(
            self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )

        # Top face
        glVertex3f(
            -self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(-self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2)

        # Bottom face
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2
        )

        # Right face
        glVertex3f(
            self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2)

        # Left face
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2
        )
        glVertex3f(-self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(
            -self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glEnd()

        # Draw black lines between cubelets
        glColor3f(0.0, 0.0, 0.0)  # Black color
        glLineWidth(1.0)  # Thin lines
        glBegin(GL_LINES)

        # Front face edges
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(-self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(-self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2
        )

        # Back face edges
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2)
        glVertex3f(
            -self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            -self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )

        # Connecting edges
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(
            -self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2
        )
        glVertex3f(
            self.cubelet_size / 2, -self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(self.cubelet_size / 2, -self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2)
        glVertex3f(self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glVertex3f(
            -self.cubelet_size / 2, self.cubelet_size / 2, -self.cubelet_size / 2
        )
        glVertex3f(-self.cubelet_size / 2, self.cubelet_size / 2, self.cubelet_size / 2)
        glEnd()

        glPopMatrix()

    def draw_cube(self) -> None:
        """Draw the entire Rubik's Cube."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # Apply manual rotation based on user input
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        # Draw all cubelets
        for face in range(6):
            for i in range(self.n):
                for j in range(self.n):
                    color = self.COLORS[self.state[face, i * self.n + j]]

                    # Calculate position based on face and indices
                    if face == 0:  # Up
                        x = (j - self.n / 2 + 0.5) * self.cubelet_size
                        y = self.size / 2
                        z = (i - self.n / 2 + 0.5) * self.cubelet_size
                    elif face == 1:  # Down
                        x = (j - self.n / 2 + 0.5) * self.cubelet_size
                        y = -self.size / 2
                        z = (i - self.n / 2 + 0.5) * self.cubelet_size
                    elif face == 2:  # Front
                        x = (j - self.n / 2 + 0.5) * self.cubelet_size
                        y = (i - self.n / 2 + 0.5) * self.cubelet_size
                        z = self.size / 2
                    elif face == 3:  # Back
                        x = (j - self.n / 2 + 0.5) * self.cubelet_size
                        y = (i - self.n / 2 + 0.5) * self.cubelet_size
                        z = -self.size / 2
                    elif face == 4:  # Right
                        x = self.size / 2
                        y = (i - self.n / 2 + 0.5) * self.cubelet_size
                        z = (j - self.n / 2 + 0.5) * self.cubelet_size
                    else:  # Left
                        x = -self.size / 2
                        y = (i - self.n / 2 + 0.5) * self.cubelet_size
                        z = (j - self.n / 2 + 0.5) * self.cubelet_size

                    self.draw_cubelet(x, y, z, color)

        pygame.display.flip()

    def handle_events(self) -> bool:
        """
        Handle Pygame events.

        Returns:
            bool: True if the window should close, False otherwise
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
                elif event.key == pygame.K_LEFT:
                    self.rot_y -= 5
                elif event.key == pygame.K_RIGHT:
                    self.rot_y += 5
                elif event.key == pygame.K_UP:
                    self.rot_x -= 5
                elif event.key == pygame.K_DOWN:
                    self.rot_x += 5
            elif event.type == pygame.KEYUP:
                # Reset rotation when key is released
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    self.rot_y = 0
                elif event.key in [pygame.K_UP, pygame.K_DOWN]:
                    self.rot_x = 0
        return False

    def close(self) -> None:
        """Close the visualization window."""
        pygame.quit()


def visualize_cube(state: np.ndarray, n: int = 3) -> None:
    """
    Visualize a Rubik's Cube state in 3D.

    Args:
        state (np.ndarray): The state of the cube to visualize
        n (int): The dimension of the cube
    """
    cube = Cube3D(n=n)
    cube.update_state(state)

    running = True
    while running:
        cube.draw_cube()
        running = not cube.handle_events()
        pygame.time.wait(10)

    cube.close()
