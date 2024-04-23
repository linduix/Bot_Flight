import numpy as np

class Drone(object):
    @staticmethod
    def _rotatemat(vec2d: np.ndarray, rads: float) -> np.ndarray:
        cos_rads = np.cos(rads)
        sin_rads = np.sin(rads)
        x, y = vec2d
        final_x = cos_rads * x - sin_rads * y
        final_y = sin_rads * x + cos_rads * y
        return np.array([final_x, final_y])

    def __init__(self):
        self.pos: np.ndarray = np.array([20., 15.])  # x, y
        self.old_pos: np.ndarray = self.pos.copy()   # old postion
        self.mass: float = 1
        self.angle: float = 0  # rads: rotation is Coutner-clockwise from up

        self.output: float = 0  # % output of the thrusters
        self.MaxThrust: float = 20

        self.acc: np.ndarray = np.array([0., 0.])
        self.vel: np.ndarray = np.array([0., 0.])

        self.targets: list[np.ndarray] = [self.pos]  # array of drones targets to hit
        self.target_points: float = 0  # the base points of the target
        self.target_ix: int = 0   # the index of current target
        self.score: float = 0     # total score of the drone: higher is better
        self.done: bool = False   # completion status
        self.crash: bool = False  # crash status

        self.completion_time: float = 0  # total time passed since new target
        self.touch_time: float = 0   # the time spent 'touching' the target
        self.survived: float = 0     # survival time
        self.similarity: float = 0   # similarity score of drone

    def set_targets(self, targs: list[list[float]]):
        self.targets: list[np.ndarray] = [np.array(targ) for targ in targs]
        self.target_points: float = np.linalg.norm(self.targets[self.target_ix] - self.pos)

    def step(self, dt):
        if self.done or self.crash:
            return

        # Force calculations
        force = np.array([0, self.output*self.MaxThrust])
        angled_thrust = self._rotatemat(force, self.angle)

        # Physics updates
        self.acc = angled_thrust / self.mass + np.array([0., -9.81])  # Gravity
        self.vel += self.acc * dt
        self.pos += self.vel * dt

        distance = np.linalg.norm(self.targets[self.target_ix] - self.pos)

        # incremet time variables
        self.touch_time += dt if distance < 1 else 0
        self.completion_time += dt if distance > 1 else 0
        self.survived += dt

        if distance > 50:
            self.crash = True
            self.done = True

        # increment targets if touching > 1s
        if self.touch_time > 1:
            # reset
            self.completion_time = 0
            self.touch_time = 0

            # increment and check if done
            self.target_ix += 1
            self.done = self.target_ix == len(self.targets)

            if not self.done:
                # set the points for next target if not done
                self.target_points = np.linalg.norm(self.targets[self.target_ix] - self.pos)

        self.old_pos = self.pos.copy()

    def rotate(self, inpt: float, dt):
        # 0 is full right turn, 1 is full left turn
        normalized = inpt-0.5
        angle = (normalized * 3*np.pi*dt)
        self.angle = (self.angle + angle) % (2*np.pi)

    def reset(self):
        self.pos: np.ndarray = np.array([20., 12.])
        self.angle: float = 0
        self.output: float = 0
        self.acc: np.ndarray = np.array([0., 0.])
        self.vel: np.ndarray = np.array([0., 0.])
        self.score: float = 0
        self.target_ix: int = 0
        self.touch_time: float = 0
        self.completion_time: float = 0
        self.done: bool = False
        self.crash: bool = False
        self.survived: float = 0
        self.old_pos: np.ndarray = self.pos.copy()
        self.similarity: float = 0
