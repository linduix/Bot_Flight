import numpy as np
from Drones.droneV1 import AiDrone

WIDTH, HEIGHT = 800, 600
WORLDSCALE = 20
TARGETS = 15


class _Target(object):
    def __init__(self, yaxis: bool, border: int):
        border = np.clip(border, 0, 1)  # one axis will be on the border
        ratio = np.random.random()  # other axis will be random postion

        if yaxis:
            self.pos = np.array([(WIDTH//WORLDSCALE)*ratio, (HEIGHT//WORLDSCALE)*border])
        else:
            self.pos = np.array([(WIDTH//WORLDSCALE)*border, (HEIGHT//WORLDSCALE)*ratio])


def _score(drone: AiDrone, target: _Target, dt):
    distance = np.linalg.norm(target.pos - drone.pos)

    if distance < 0.2:
        drone.touch_time += dt
        if drone.touch_time > 1:
            drone.score += 10*drone.completed / drone.completion_time  # points based on speed

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
    yaxis: bool = np.random.random() < 0.5
    targets: list[_Target] = []
    for i in range(TARGETS):
        targets.append(_Target(yaxis, i % 2))

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
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / drone.survived

    # # Finalize scores
    # for drone in drones:
    #     avg_vel: float = np.mean(np.abs(np.array(drone.velocities)))
    #     drone.score += drone.completed * avg_vel/10
    #     if drone.crash:
    #         drone.score /= 2
    #     elif drone.done:
    #         drone.score *= time_threshold / drone.survived
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
    yaxis: bool = np.random.random() < 0.5
    for i in range(TARGETS):
        # Swap axis every other target
        yaxis: bool = np.random.random() < 0.5 if i%2 == 0 else yaxis
        # Swap side every target
        targets.append(_Target(yaxis, i % 2))

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
        if drone.crash:
            drone.score /= 2
        elif drone.done:
            drone.score *= time_threshold / drone.survived
    # for drone in drones:
    #     avg_vel: float = np.mean(np.abs(np.array(drone.velocities)))
    #     drone.score += drone.completed * avg_vel/10
    #     if drone.crash:
    #         drone.score /= 2
    #     elif drone.done:
    #         drone.score *= time_threshold / drone.survived

    return drones
