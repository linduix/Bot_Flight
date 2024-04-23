from Drones.droneV1 import AiDrone
import numpy as np
import pygame as pg
import json
import sys
from Levels import level2

# Initialize Pygame
pg.init()

# Set up the screen
WIDTH, HEIGHT = 1600, 900
screen = pg.display.set_mode((WIDTH, HEIGHT), pg.SRCALPHA)
pg.display.set_caption("Pygame Boilerplate")
WORLDSCALE = 20

LEVEL = 5
SHUFFLE = False

# drone and target setup
path = f'data/level{LEVEL}.json' if not SHUFFLE else 'data/shuffle.json'
with open(path, 'r') as f:
    data = json.load(f)
    best_genome = data['best']

drone = AiDrone([20, 15], genome=best_genome)

drone_sprite = pg.Surface([20, 20])
drone_sprite.fill('black')
pg.draw.rect(drone_sprite, 'white', [1, 1, 18, 18])
pg.draw.circle(drone_sprite, 'red', (10, 2), 2)


def l1_sim():
    dt = 16 / 1000
    target = np.array([(WIDTH / 2) // WORLDSCALE, (HEIGHT / 2) // WORLDSCALE])  # Middle of screen

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        # Update Sim
        drone.process(target)
        drone.update(dt)
        if drone.done:
            running = False

        # Calculations
        center = drone.pos.copy() * WORLDSCALE
        center[1] = HEIGHT - center[1]

        # Draw here
        drone_sprite.fill('black')
        pg.draw.rect(drone_sprite, (255 * drone.thrust_output, 255 * (1 - drone.thrust_output), 50), [1, 1, 18, 18])
        pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
        rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
        sprite_rect = rotated_sprite.get_rect(center=center)
        screen.blit(rotated_sprite, sprite_rect)

        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        draw_nn()

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000


def l2_sim():
    dt = 16 / 1000
    target = np.random.rand(1, 2)[0] * np.array([WIDTH // WORLDSCALE, HEIGHT // WORLDSCALE])

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        # Update Sim
        drone.process(target)
        drone.update(dt)

        dist = np.linalg.norm(drone.pos - target)
        if dist < 0.5:
            drone.touch_time += 1
            if drone.touch_time > 5:
                drone.completed += 1
                drone.touch_time = 0
        elif dist > 100:
            drone.crash = True
            drone.done = True
            print('crash')
            return

        if drone.completed == 1:
            target = np.random.rand(1, 2)[0] * np.array([WIDTH // WORLDSCALE, HEIGHT // WORLDSCALE])
            drone.completed = 0

        # Calculations
        center = drone.pos.copy() * WORLDSCALE
        center[1] = HEIGHT - center[1]

        # Draw here
        drone_sprite.fill('black')
        pg.draw.rect(drone_sprite, (255 * drone.thrust_output, 255 * (1 - drone.thrust_output), 50), [1, 1, 18, 18])
        pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
        rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
        sprite_rect = rotated_sprite.get_rect(center=center)
        screen.blit(rotated_sprite, sprite_rect)

        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        draw_nn()

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000


def l3_sim():
    t_time = 0
    offset = 0
    dt = 16 / 1000

    # Constant Rotating Target
    direction = 1 if np.random.random() < 0.5 else -1
    speed = np.random.random() / 2 + 0.25
    speeds = np.array([speed, 1 - speed])
    speeds = speeds * direction

    x, y = speeds * t_time
    target = np.array([np.sin(x), np.cos(x)]) / 2 + 0.5
    target = target * np.array([WIDTH, HEIGHT]) / WORLDSCALE

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        # Calc Target
        if t_time//5 % 2 == 0:
            x, y = speeds * offset
            target = np.array([np.sin(x), np.cos(y)]) / 2 + 0.5
            target = target * np.array([WIDTH, HEIGHT]) / WORLDSCALE
            offset += dt * 1.5

        # Update Sim
        drone.process(target)
        drone.update(dt)
        if drone.done:
            return

        if np.linalg.norm(drone.pos - target) > 100:
            drone.crash = True
            drone.done = True
            print('crash')

        # Calculations
        center = drone.pos.copy() * WORLDSCALE
        center[1] = HEIGHT - center[1]

        # Draw here
        drone_sprite.fill('black')
        pg.draw.rect(drone_sprite, (255 * drone.thrust_output, 255 * (1 - drone.thrust_output), 50), [1, 1, 18, 18])
        pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
        rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
        sprite_rect = rotated_sprite.get_rect(center=center)
        screen.blit(rotated_sprite, sprite_rect)

        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        draw_nn()

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000
        t_time += dt


def manual_sim():
    t_time = 0
    dt = 16 / 1000
    pg.mouse.set_visible(False)

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        # Get mousepos
        mouse_pos = np.array(pg.mouse.get_pos())
        mouse_pos[1] = HEIGHT - mouse_pos[1]
        target = mouse_pos/WORLDSCALE

        # Update Sim
        drone.process(target)
        drone.update(dt)
        if drone.done:
            return

        if np.linalg.norm(drone.pos - target) > 100:
            drone.crash = True
            drone.done = True
            print('crash')

        # Calculations
        center = drone.pos.copy() * WORLDSCALE
        center[1] = HEIGHT - center[1]

        # Draw here
        drone_sprite.fill('black')
        pg.draw.rect(drone_sprite, (255 * drone.thrust_output, 255 * (1 - drone.thrust_output), 50), [1, 1, 18, 18])
        pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
        rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
        sprite_rect = rotated_sprite.get_rect(center=center)
        screen.blit(rotated_sprite, sprite_rect)

        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        draw_nn()

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000
        t_time += dt


def draw_nn():
    shape = drone.brain.shape
    width = 200
    gap = 80

    for i, layer_amt in enumerate(shape):
        xpos = gap * (i+1) - (gap-30)
        for ix in range(layer_amt):
            ypos = HEIGHT + (width / (layer_amt + 1)) * (ix + 1) - width

            # Draw weight lines:
            intesity = 10
            if i < len(shape) - 1:
                xpos2 = gap * (i + 2) - (gap-30)
                for jx in range(shape[i + 1]):
                    layer2_amt = shape[i + 1]
                    ypos2 = HEIGHT + (width / (layer2_amt + 1)) * (jx + 1) - width

                    activation = drone.brain.weight_activations[i][jx, ix]
                    weights = drone.brain.layers[i].weights
                    line_size = np.abs(weights[jx, ix]) * 3 / np.abs(weights[jx, ix]).max()
                    if activation >= 0:
                        color = (intesity, (255-intesity)*activation+intesity, intesity)
                    else:
                        color = ((255-intesity)*-activation+intesity, intesity, intesity)

                    pg.draw.line(screen, color, [xpos, ypos], [xpos2, ypos2], width=int(line_size))

            # Draw Node activations
            activation = drone.brain.node_activations[i][ix, 0]
            if activation >= 0:
                color = (intesity, (255 - intesity) * activation + intesity, intesity)
            else:
                color = ((255 - intesity) * -activation + intesity, intesity, intesity)

            pg.draw.circle(screen, color, (xpos, ypos), 5)


# manual_sim()
l2_sim()

# Quit Pygame
pg.quit()
sys.exit()
