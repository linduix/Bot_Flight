import numpy as np
from Drones.droneV1 import AiDrone

WIDTH, HEIGHT = 1600, 900
WORLDSCALE = 20
MAX_DIST = np.sqrt(np.square(np.array([WIDTH, HEIGHT])/WORLDSCALE).sum())
TARGETS = 1


def _score(drone: AiDrone, target: np.ndarray, dt):
    distance = np.linalg.norm(target - drone.pos)

    # Calculate allignment: if it's going towards target or not (1 to -1)
    targvec = target - drone.pos
    movvec = drone.pos - drone.old_pos
    alignment = max(0.5, np.dot(targvec, movvec) / np.linalg.norm(targvec)**2)
    alignment = 1 if distance < 5 else alignment

    if distance < 0.1:
        drone.score += dt * 4 / (distance + 1)
        drone.touch_time += dt
        if drone.touch_time > 15:
            drone.completed = 1
            drone.completion_time = 0
            drone.done = True
    elif distance > MAX_DIST:
        drone.crash = True
        drone.done = True
    else:
        drone.score += dt * alignment / np.square(distance + 1)
        drone.completion_time += dt


def run_level(drones: list[AiDrone]):
    # time setup
    dt = 32 / 1000
    t_time = 0
    offset = 0
    time_threshold = 60

    # Constant Rotating Target
    direction = 1 if np.random.random() < 0.5 else -1
    speed = max(np.random.random(), 0.5)
    speeds = np.array([speed, 1 - speed])
    speeds = speeds * direction

    x, y = speeds * offset
    target = np.array([np.sin(x), np.cos(y)]) / 2 + 0.5
    target = target * np.array([WIDTH, HEIGHT]) / WORLDSCALE

    while True:
        # Calc Target
        if t_time//5 % 2 == 0:
            x, y = speeds * offset
            target = np.array([np.sin(x), np.cos(y)]) / 2 + 0.5
            target = target * np.array([WIDTH, HEIGHT]) / WORLDSCALE
            offset += dt * 0.5

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
            _score(drone, target, dt)

        # Time step
        t_time += dt

        # break if done or time thresh reached
        if all([drone.done for drone in drones]) or (t_time >= time_threshold):
            break

    # Finalize scores
    for drone in drones:
        if drone.completed != TARGETS:
            points = 10 - np.linalg.norm(target - drone.pos)
            drone.score += max(0, points / (drone.completion_time+1))
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / drone.survived

    return drones


def run_level_pg(drones: list[AiDrone], SCREEN):
    import pygame as pg

    DRONE_SPRITE = pg.Surface([20, 20])

    # time setup
    dt = 32 / 1000
    t_time = 0
    offset = 0
    time_threshold = 60

    # Constant Rotating Target
    direction = 1 if np.random.random() < 0.5 else -1
    speed = max(np.random.random(), 0.5)
    speeds = np.array([speed, 1 - speed])
    speeds = speeds * direction

    x, y = speeds * offset
    target = np.array([np.sin(x), np.cos(y)]) / 2 + 0.5
    target = target * np.array([WIDTH, HEIGHT]) / WORLDSCALE

    while True:
        # Event handling
        for event in pg.event.get():
            match event.type:
                case pg.QUIT:
                    pg.quit()
                    raise KeyboardInterrupt

        # Clear the screen
        SCREEN.fill('black')

        # Calc Target
        if t_time//5 % 2 == 0:
            x, y = speeds * offset
            target = np.array([np.sin(x), np.cos(y)]) / 2 + 0.5
            target = target * np.array([WIDTH, HEIGHT]) / WORLDSCALE
            offset += dt * 0.5

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
            rotated_sprite.set_colorkey((0, 0, 0))
            sprite_rect = rotated_sprite.get_rect(center=center)
            SCREEN.blit(rotated_sprite, sprite_rect)

            # Score drone
            _score(drone, target, dt)

        # Update the display
        pg.display.flip()

        # Time step
        t_time += dt

        # break if done or time thresh reached
        if all([drone.done for drone in drones]) or (t_time >= time_threshold):
            break

    # Finalize scores
    for drone in drones:
        if drone.completed != TARGETS:
            points = 10 - np.linalg.norm(target - drone.pos)
            drone.score += max(0, points / (drone.completion_time+1))
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / drone.survived

    return drones
