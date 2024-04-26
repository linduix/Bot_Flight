import numpy as np
from Drones.droneV1 import AiDrone

WIDTH, HEIGHT = 1600, 900
WORLDSCALE = 20
MAX_DIST = np.sqrt(np.square(np.array([WIDTH, HEIGHT])/WORLDSCALE).sum())
TARGETS = 12


class _Target(object):
    def __init__(self, yaxis: bool):
        position = np.random.random()  # float [0, 1)
        if yaxis:
            self.pos = np.array([(WIDTH//WORLDSCALE)/2, (HEIGHT//WORLDSCALE)*position])
        else:
            self.pos = np.array([(WIDTH//WORLDSCALE)*position, (HEIGHT//WORLDSCALE)/2])

        # self.points = np.linalg.norm(self.pos - np.array([(WIDTH//WORLDSCALE)/2, (HEIGHT//WORLDSCALE)/2]))
        self.points = 10


def _score(drone: AiDrone, target: _Target, dt):
    distance = np.linalg.norm(target.pos - drone.pos)
    drone.score += dt / (distance + 1)

    # Check for touching
    if distance < 0.5:
        drone.touch_time += dt
    else:
        drone.completion_time += dt

    # Check for completion
    if drone.touch_time > 2:
        # Give points
        drone.score += target.points

        # Increment target
        drone.completed += 1

        # Reset time variables
        drone.touch_time = 0
        drone.completion_time = 0

    # Check for crash
    if distance > MAX_DIST:
        drone.crash = True
        drone.done = True


def run_level(drones: list[AiDrone]):
    # Time setup
    dt = 32 / 1000
    t_time = 0
    time_threshold = 60

    # Target Generation
    # yaxis: bool = np.random.random() < HEIGHT / sum((WIDTH, HEIGHT))
    yaxis: bool = False
    targets: list[_Target] = [_Target(yaxis) for _ in range(TARGETS)]

    while True:
        # Update sim
        for drone in drones:
            # Skip if done
            if drone.done or drone.crash:
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
        if drone.completed != TARGETS:
            final_targ = targets[drone.completed]
            points = final_targ.points - np.linalg.norm(final_targ.pos - drone.pos)
            drone.score += max(0, points / (drone.completion_time+1))
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
    # yaxis: bool = np.random.random() < HEIGHT / sum((WIDTH, HEIGHT))
    yaxis: bool = False
    targets: list[_Target] = [_Target(yaxis) for _ in range(TARGETS)]

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
            rotated_sprite.set_colorkey((0, 0, 0))
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
        if all([drone.done or drone.crash for drone in drones]) or (t_time >= time_threshold):
            break

    # Finalize scores
    for drone in drones:
        if drone.completed != TARGETS:
            final_targ = targets[drone.completed]
            points = final_targ.points - np.linalg.norm(final_targ.pos - drone.pos)
            drone.score += max(0, points / (drone.completion_time+1))
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / (drone.survived + 1)
        drone.score *= drone.completed/TARGETS

    return drones
