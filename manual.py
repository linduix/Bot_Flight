from Drones.manualDrone import Drone
import numpy as np
import pygame as pg
import sys


def target_gen(w1, w, h1, h):
    w = np.random.randint(w1, high=w)
    h = np.random.randint(h1, high=h)
    return [w, h]


# Initialize Pygame
pg.init()

# Set up the screena
WIDTH, HEIGHT = 800, 600
screen = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Pygame Boilerplate")
WORLDSCALE = 20
dt = 16 / 1000

# drone and target setup
drone = Drone()
# Target Creation
target_amt: int = 4
targets = []
offset = np.random.randint(0, 4)
for j in range(offset, target_amt+offset):
    # alternate left and right
    right = j % 2 == 0
    up = j % 4 < 2
    xmin, xmax = WIDTH/WORLDSCALE * (0.5 if right else 0.1), WIDTH/WORLDSCALE * (0.9 if right else 0.5)
    ymin, ymax = HEIGHT/WORLDSCALE * (0.5 if up else 0.1), HEIGHT/WORLDSCALE * (0.9 if up else 0.5)
    targets.append(target_gen(xmin, xmax, ymin, ymax))
drone.set_targets(targets)

drone_sprite = pg.Surface([20, 20])
drone_sprite.fill('black')
pg.draw.rect(drone_sprite, 'white', [1, 1, 18, 18])
pg.draw.circle(drone_sprite, 'red', (10, 2), 2)

# Main loop
rotate = 0.5
thrust = False
running = True
while running:
    # Event handling
    for event in pg.event.get():
        match event.type:
            case pg.QUIT:
                running = False
            case pg.KEYDOWN:
                match event.key:
                    case pg.K_w:
                        thrust = True
                match event.key:
                    case pg.K_a:
                        rotate = 1
                match event.key:
                    case pg.K_d:
                        rotate = 0
            case pg.KEYUP:
                match event.key:
                    case pg.K_w:
                        thrust = False
                match event.key:
                    case pg.K_a:
                        rotate = 0.5
                match event.key:
                    case pg.K_d:
                        rotate = 0.5

    # Clear the screen
    screen.fill('black')

    # Update Sim
    if thrust:
        drone.output = 1.
    else:
        drone.output = 0.

    drone.rotate(rotate, dt)
    # drone.process(dt)
    drone.step(dt)

    if drone.done:
        running = False

    # Calculations
    center = drone.pos.copy() * WORLDSCALE
    center[1] = HEIGHT - center[1]

    # Draw here
    drone_sprite.fill('black')
    pg.draw.rect(drone_sprite, (255 * drone.output, 255 * (1 - drone.output), 50), [1, 1, 18, 18])
    pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
    rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
    sprite_rect = rotated_sprite.get_rect(center=center)
    screen.blit(rotated_sprite, sprite_rect)

    if not drone.done:
        target_pos = drone.targets[drone.target_ix] * WORLDSCALE
    target_pos[1] = HEIGHT - target_pos[1]
    pg.draw.circle(screen, 'red',  target_pos, 3)

    # Update the display
    pg.display.flip()

    # Cap the frame rate
    dt = pg.time.Clock().tick(60) / 1000

# Quit Pygame
print(drone.score)
pg.quit()
sys.exit()
