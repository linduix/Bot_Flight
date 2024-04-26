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
drone_font = pg.font.Font(None, 24)
position_font = pg.font.Font(None, 46)

LEVELS = ['level1', 'level2', 'level3', 'level4', 'level5', 'shuffle']
SHUFFLE = False
if not SHUFFLE:
    LEVELS = LEVELS[:-1]

# drone setup
drones: dict[str, AiDrone] = {}
for level in LEVELS:
    with open(f'data/{level}.json', 'r') as f:
        data = json.load(f)
        best_genome: dict = data['best']
    _drone: AiDrone = AiDrone([WIDTH/(WORLDSCALE * 2), HEIGHT/(WORLDSCALE * 2)], genome=best_genome)
    drones[level] = _drone


drone_sprite = pg.Surface([20, 20])
drone_sprite.fill('black')
NAMES = ['Lv1', 'Lv2', 'Lv3', 'Lv4', 'Lv5', 'Shuffle']
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
pg.draw.rect(drone_sprite, 'white', [1, 1, 18, 18])
pg.draw.circle(drone_sprite, 'red', (10, 2), 2)


def get_target() -> np.ndarray:
    return np.random.rand(1, 2)[0] * np.array([WIDTH // WORLDSCALE, HEIGHT // WORLDSCALE])


def main():
    dt = 16 / 1000
    target: np.ndarray = get_target()
    new_target = False

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    return

        # Clear the screen
        screen.fill('black')

        # Update Sim
        for i, name in enumerate(LEVELS):
            drone: AiDrone = drones[name]
            color: str = COLORS[i]
            name: pg.surface = drone_font.render(NAMES[i], True, color)

            if drone.done:
                continue

            drone.process(target)
            drone.update(dt)

            dist: float = np.linalg.norm(drone.pos - target)
            if dist < 0.5:
                drone.touch_time += 1
                if drone.touch_time > 5:
                    drone.completed += 1
                    drone.touch_time = 0
                    new_target = True
            elif dist > 100:
                drone.crash = True
                drone.done = True

            # Calculations
            center: np.ndarray = drone.pos.copy() * WORLDSCALE
            center[1] = HEIGHT - center[1]

            # Draw here
            # drone_sprite.fill('black')
            pg.draw.rect(drone_sprite, color, [1, 1, 18, 18])
            pg.draw.circle(drone_sprite, 'red', (10, 2), 2)
            rotated_sprite = pg.transform.rotate(drone_sprite, np.degrees(drone.angle))
            rotated_sprite.set_colorkey((0, 0, 0))
            sprite_rect = rotated_sprite.get_rect(center=center)
            screen.blit(rotated_sprite, sprite_rect)
            name_rect = name.get_rect(center=center-[0, 25])
            screen.blit(name, name_rect)

        # Target Move
        if new_target:
            target = get_target()
            new_target = False

        # Target draw
        target_pos = target * WORLDSCALE
        target_pos[1] = HEIGHT - target_pos[1]
        pg.draw.circle(screen, 'red', target_pos, 3)

        # Get Leader
        leader: AiDrone = max(drones.values(), key=lambda x: x.completed)
        for name, drone in drones.items():
            if drone.completed == 10:
                print(f'Winner: {name.capitalize()}')
                return

        # Print positions
        leader_name = 'Na'
        for i, level in enumerate(LEVELS):
            ypos = 50*(i+1) - 30
            name = NAMES[i]
            drone = drones[level]
            completed = drone.completed if not drone.crash else 'Crashed'
            color = 'white' if not drone.crash else 'red'
            if drones[level] == leader:
                leader_name = name
                color = 'yellow'

            text = position_font.render(f'{name}: {completed}', True, color)
            screen.blit(text, [20, ypos])

        draw_nn(leader)
        nn_text = position_font.render(f'{leader_name} Brain', True, 'yellow')
        screen.blit(nn_text, [30, HEIGHT - 275])

        # Update the display
        pg.display.flip()

        # Cap the frame rate
        dt = pg.time.Clock().tick(60) / 1000


def draw_nn(drone):
    shape = drone.brain.shape
    width = 250
    gap = 100

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
                    line_size = np.abs(weights[jx, ix]) * 4 / np.abs(weights[jx, ix]).max()
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

            pg.draw.circle(screen, color, (xpos, ypos), 8)


main()

# Quit Pygame
pg.quit()
sys.exit()
