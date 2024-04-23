import numpy as np
from Drones.droneV1 import AiDrone

WIDTH, HEIGHT = 800, 600
WORLDSCALE = 20
TARGETS = 1

target = np.array([(WIDTH / 2) // WORLDSCALE, (HEIGHT / 2) // WORLDSCALE])  # Middle of screen


def _score(drone: AiDrone, dt):
    distance = np.linalg.norm(target - drone.pos)
    drone.score += dt / (distance + 1)

    if distance > 40:
        drone.crash = True
        drone.done = True


def run_level(drones: list[AiDrone]):
    # time setup
    dt = 32 / 1000
    t_time = 0
    time_threshold = 30

    while True:
        # Update sim
        for drone in drones:
            # Skip if done
            if drone.done:
                continue

            # Update sim
            drone.process(target)
            drone.update(dt)
            drone.survived += dt

            # Score drone
            _score(drone, dt)

        # Time step
        t_time += dt

        # break if done or time thresh reached
        if all([drone.done for drone in drones]) or (t_time >= time_threshold):
            break

    return drones


def run_level_pg(drones: list[AiDrone], SCREEN):
    import pygame as pg

    DRONE_SPRITE = pg.Surface([20, 20])

    # time setup
    dt = 32 / 1000
    t_time = 0
    time_threshold = 30

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    pg.quit()
                    raise KeyboardInterrupt

        # Clear the screen
        SCREEN.fill('black')

        # Update sim
        for drone in drones:
            # Skip if done
            if drone.done:
                continue

            # Draw target
            target_pos = target * WORLDSCALE
            target_pos[1] = HEIGHT - target_pos[1]
            pg.draw.circle(SCREEN, 'red', target_pos, 3)

            # Update sim
            drone.process(target)
            drone.update(dt)

            # Position calculations
            center = drone.pos.copy() * WORLDSCALE
            center[1] = HEIGHT - center[1]

            # Draw Drone
            DRONE_SPRITE.fill('black')
            pg.draw.rect(DRONE_SPRITE, (255 * drone.thrust_output, 255 * (1 - drone.thrust_output), 50), [1, 1, 18, 18])
            pg.draw.circle(DRONE_SPRITE, 'red', (10, 2), 2)
            rotated_sprite = pg.transform.rotate(DRONE_SPRITE, np.degrees(drone.angle))
            sprite_rect = rotated_sprite.get_rect(center=center)
            SCREEN.blit(rotated_sprite, sprite_rect)

            # Score drone
            _score(drone, dt)

        # Update the display
        pg.display.flip()

        # Time step
        t_time += dt

        # break if done or time thresh reached
        if all([drone.done for drone in drones]) or (t_time >= time_threshold):
            break

    return drones
