import numpy as np
from Drones.droneV1 import AiDrone
from Levels import level1, level2, level3, level4, level5
from geneticFuncs import next_generation
import pygame as pg
import time
import json
import os
import sys

# Initialize Pygame
pg.init()

# GLOBALS SETUP
WIDTH, HEIGHT = 800, 600
WORLDSCALE = 20
SCREEN = pg.display.set_mode((WIDTH, HEIGHT))

# Level settings
# LEVEL 1: FLYING STABILITY
# LEVEL 2: SIGNULAR AXIS MOVEMENT
# LEVEL 3: MULTIPLE AXIS TARGET FOLLOWING
# LEVEL 4: SPEED AND PRECISION
LEVEL = 5
LEVELS = [level1, level2, level3, level4, level5]
RUN_LEVEL = LEVELS[LEVEL-1].run_level_pg
# Shuffle mode
SHUFFLE = True

# Training settings
gen_size: int = 40          # Drones per generation
gen_threshold: int = 5_000  # Total generations to train

# Create drones
drones: list[AiDrone] = [AiDrone([20, 15]) for _ in range(gen_size)]


def load_data():
    # Load/Create drones
    data_path = 'data/level'
    _starting_gen = 0

    # Check if file exist
    try:
        if os.path.exists(f'{data_path}{LEVEL}.json') and not SHUFFLE:  # check if current level exists
            print('Resuming Training:')
            resume = True
            path = f'{data_path}{LEVEL}.json'
        elif os.path.exists(f'{data_path}{LEVEL - 1}.json'):  # check if previous level exists
            print('Beginning Training:')
            resume = False
            path = f'{data_path}{LEVEL - 1}.json'
        else:
            print('\033[91mNo Data Found:\033[0m')
            raise FileNotFoundError  # if none found raise error

        if SHUFFLE:
            path = 'data/shuffle.json'
            resume = True

        # Load data from file
        with open(path, 'r') as f:
            print('\033[33mLoading drones...\033[0m\n')
            _data: dict = json.load(f)
            if resume:
                _starting_gen: int = _data['generation']  # set the starting generation if continuing training

            _all_drones: list = _data['all']
            _drones: list = [AiDrone([20, 15], genome=genome) for genome in _all_drones]

    # Handle file not existing
    except FileNotFoundError:  # if no previous data found
        if LEVEL == 1 or SHUFFLE:  # create new if first level or shuffle mode
            print('\033[33mCreating Drones...\033[0m\n')
            _drones: list[AiDrone] = [AiDrone([20, 15]) for _ in range(gen_size)]
        else:           # else exit program
            print('Previous Level Doesnt Exist')
            sys.exit()

    # Handle data reading error
    except json.decoder.JSONDecodeError as e:
        print('\033[91mData Corrupted:\033[0m')
        if LEVEL == 1:
            print('\033[33mCreating Drones...\033[0m\n')
            _drones: list[AiDrone] = [AiDrone([20, 15]) for _ in range(gen_size)]
        else:
            print(e)
            print(f'\033[91m->\033[0m {path}')
            sys.exit()

    finally:
        _gen = _starting_gen

    return _drones, _gen


if __name__ == '__main__':
    # Load Data
    drones, gen = load_data()
    best_drone = max(drones, key=lambda x: x.score)
    starting_gen = gen
    targs: int = LEVELS[LEVEL-1].TARGETS

    # Check if already Done
    if gen == gen_threshold:
        print('\033[32mTraining Done\033[0m')
        sys.exit()

    # Training loop
    start_time = time.time()
    try:
        while gen < gen_threshold:
            gen += 1

            if SHUFFLE:
                LEVEL = ((gen-100) // 100 % 4) + 2 if gen > 100 else 1
                RUN_LEVEL = LEVELS[LEVEL - 1].run_level_pg
                targs: int = LEVELS[LEVEL - 1].TARGETS

            # simulate drones
            start = time.time()
            drones = RUN_LEVEL(drones, SCREEN)
            end = time.time()
            best_drone = max(drones, key=lambda x: x.score)

            # calculate similarities and diversity:
            for drone in drones:
                drone.calc_similarity(drones)
            diversity = np.std([drone.similarity for drone in drones])

            # Print results
            end_time = time.time()
            if (gen == starting_gen + 1) or (end_time - start_time >= 20):
                print(f"Gen:  {gen:>4} | "
                      f"Score: {best_drone.score:0>6.2f} | "
                      f"{'[CRASH]' if best_drone.crash else '[ALIVE]':<7} {best_drone.survived:0>5.2f}s | "
                      f"Targets [{best_drone.completed:0>2}/{targs:0>2}] | "
                      f"{end_time - start_time:0>5.2f}s | Diversity: {diversity * 100:.2f}")
                start_time = end_time

            # Get the next generation
            drones: list[AiDrone] = next_generation(drones, gen_size)
        print('\033[32mTraining Done\033[0m')

    # Check for Keyboard Interrupt
    except KeyboardInterrupt:
        print('\033[33mSaving Drones...\033[0m')

    # Save Data
    finally:
        # Save the best drone
        if not SHUFFLE:
            path = f'data/level{LEVEL}.json'
        else:
            path = 'data/shuffle.json'
        with open(path, 'w') as f:
            best_genome: dict = best_drone.genome
            for k, v1 in best_genome.items():
                for ix in range(len(v1)):
                    best_genome[k][ix] = v1[ix].tolist()

            all_drones = [drone.genome for drone in drones]
            for drone in all_drones:
                for k, v1 in drone.items():
                    for ix in range(len(v1)):
                        drone[k][ix] = v1[ix].tolist()

            data = {'generation': gen, 'best': best_genome, 'all': all_drones}
            json.dump(data, f)
