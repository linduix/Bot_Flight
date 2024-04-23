import numpy as np
from Drones.droneV1 import AiDrone

WIDTH, HEIGHT = 800, 600
WORLDSCALE = 20
TARGETS = 15


class _Target(object):
    def __init__(self, old: np.ndarray):
        ratio = np.random.rand(1, 2)[0]
        self.pos: np.ndarray = np.array([(WIDTH//WORLDSCALE), (HEIGHT//WORLDSCALE)]) * ratio

        self.points: float = np.linalg.norm(self.pos - old)


def _score(drone: AiDrone, target: _Target, dt):
    distance = np.linalg.norm(target.pos - drone.pos)
    drone.score += dt/((distance + 1) * (drone.completion_time + 1))

    if distance < 0.2:
        drone.touch_time += dt
        if drone.touch_time > 1:
            drone.score += target.points  # points based on speed

            drone.completed += 1
            drone.completion_time = 0
    elif distance > 50:
        drone.crash = True
        drone.done = True
    else:
        drone.touch_time = 0  # reset touch timer if outside range
        drone.completion_time += dt


def run_level(drones: list[AiDrone]):
    # time setup
    dt = 32 / 1000
    t_time = 0
    time_threshold = 60

    # Target Generation
    targets: list[_Target] = []
    old: np.ndarray = np.array([WIDTH/(2*WORLDSCALE), HEIGHT/(2*WORLDSCALE)])
    for i in range(TARGETS):
        new: _Target = _Target(old)
        targets.append(new)
        old: np.ndarray = new.pos

    while True:
        # Update sim
        for drone in drones:
            # Skip if done
            if drone.done:
                continue

            # Get target
            target = targets[drone.completed]

            # Update sim
            drone.process(target.pos)
            drone.update(dt)
            drone.survived += dt

            # Score drone
            _score(drone, target, dt)

            # Check for completion
            if drone.completed == TARGETS:
                drone.done = True

        # Time step
        t_time += dt

        # break if done or time thresh reached
        if all([drone.done for drone in drones]) or (t_time >= time_threshold):
            break

    # Finalize scores
    for drone in drones:
        if not drone.done:
            target = targets[drone.completed]

            diff = np.linalg.norm(target.pos - drone.pos)
            points = max(target.points - diff, 0)

            drone.score += points / (drone.completion_time + 1)
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / (drone.survived + 1)

        drone.score *= drone.completed/TARGETS

    return drones


def run_level_pg(drones: list[AiDrone], SCREEN):
    import pygame as pg

    DRONE_SPRITE = pg.Surface([20, 20])

    # time setup
    dt = 32 / 1000
    t_time = 0
    time_threshold = 60

    # Target Generation
    targets: list[_Target] = []
    old: np.ndarray = np.array([WIDTH/(2*WORLDSCALE), HEIGHT/(2*WORLDSCALE)])
    for i in range(TARGETS):
        new: _Target = _Target(old)
        targets.append(new)
        old: np.ndarray = new.pos

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

            # Get target
            target = targets[drone.completed]

            # Draw target
            target_pos = target.pos * WORLDSCALE
            target_pos[1] = HEIGHT - target_pos[1]
            pg.draw.circle(SCREEN, 'red', target_pos, 3)

            # Update sim
            drone.process(target.pos)
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
            _score(drone, target, dt)

            # Check for completion
            if drone.completed == TARGETS:
                drone.done = True

        # Update the display
        pg.display.flip()

        # Time step
        t_time += dt

        # break if done or time thresh reached
        if all([drone.done for drone in drones]) or (t_time >= time_threshold):
            break

    # Finalize scores
    for drone in drones:
        if not drone.done:
            target = targets[drone.completed]

            diff = np.linalg.norm(target.pos - drone.pos)
            points = max(target.points - diff, 0)

            drone.score += points / (drone.completion_time + 1)
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / (drone.survived + 1)

        drone.score *= drone.completed / TARGETS

    return drones
