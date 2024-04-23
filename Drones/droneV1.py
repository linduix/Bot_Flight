import numpy as np
from Drones.NeuralNetwork import neuralNet


class AiDrone(object):
    @staticmethod
    def _rotatemat(vec2d: np.ndarray, rads: float) -> np.ndarray:
        cos_rads = np.cos(rads)
        sin_rads = np.sin(rads)
        x, y = vec2d
        final_x = cos_rads * x - sin_rads * y
        final_y = sin_rads * x + cos_rads * y
        return np.array([final_x, final_y])

    def rotate(self, amt, dt):
        normalized = amt-0.5
        angle = normalized * 3*np.pi*dt
        self.angle = (self.angle + angle) % (2*np.pi)

    def __init__(self, startpos: list[2], genome: dict = None):
        self.startpos = np.array(startpos, dtype=np.float32)
        self.pos: np.ndarray = self.startpos  # x, y
        self.old_pos: np.ndarray = self.pos.copy()   # old postion
        self.mass: float = 1
        self.angle: float = 0  # rads: rotation is Coutner-clockwise from up
        self.brain = neuralNet([8, 10, 10, 2])
        if genome:
            self.genome = genome

        self.old_thrust_output: float = 0
        self.thrust_output: float = 0  # % output of the thrusters
        self.angle_output: float = 0  # rotation target
        self.MaxThrust: float = 20

        self.acc: np.ndarray = np.array([0., 0.])
        self.vel: np.ndarray = np.array([0., 0.])
        self.velocities: list[np.ndarray] = []  # list of recorded velocities

        # self.target: np.ndarray = self.pos  # target position to reach
        self.completed: int = 0  # number of completed targets
        self.score: float = 0  # total score of the drone: higher is better
        self.done: bool = False  # completion status
        self.crash: bool = False  # crash status

        self.completion_time: float = 0  # total time passed since new target
        self.completion_times: list[float] = []
        self.touch_time: float = 0  # the time spent 'touching' the target
        self.survived: float = 0  # survival time

        self.similarity: float = 0  # similarity score of drone

    def process(self, target: list[2]):
        target = np.array(target)
        diffx, diffy = target - self.pos
        velx, vely = self.vel
        accx, accy = self.acc
        angS, angC = np.sin(self.angle), np.cos(self.angle)

        inp = np.array([[diffx, diffy, velx, vely, accx, accy, angS, angC]], dtype=np.float64)
        output = self.brain(inp.T).T

        self.old_thrust_output = self.thrust_output
        self.thrust_output, self.angle_output = output[0]

    def update(self, dt):
        if self.done or self.crash:
            return

        # Rotate drone
        self.rotate(self.angle_output, dt)

        # Force calculations
        force = np.array([0, self.thrust_output*self.MaxThrust])
        angled_thrust = self._rotatemat(force, self.angle)

        # Physics updates
        self.acc = angled_thrust / self.mass + np.array([0., -9.81])  # Gravity
        self.vel += self.acc * dt
        self.pos += self.vel * dt

        self.old_pos = self.pos.copy()

    def reset(self):
        self.pos: np.ndarray = np.array([20., 15.])
        self.angle: float = 0

        self.angle_output: float = 0
        self.thrust_output: float = 0
        self.old_thrust_output: float = 0

        self.acc: np.ndarray = np.array([0., 0.])
        self.vel: np.ndarray = np.array([0., 0.])
        self.velocities: list[np.ndarray] = []

        self.score: float = 0
        self.completed: int = 0

        self.touch_time: float = 0
        self.completion_time: float = 0
        self.completion_times: list[float] = []
        self.survived: float = 0

        self.done: bool = False
        self.crash: bool = False

        self.old_pos: np.ndarray = self.pos.copy()
        self.similarity: float = 0

    @property
    def genome(self):
        return self.brain.state_dict

    @genome.setter
    def genome(self, state_dict):
        self.brain.load(state_dict)

    def calc_similarity(self, drones: list):
        distances = np.zeros(len(drones))
        assert type(distances) is np.ndarray
        for i, drone in enumerate(drones):
            if drone is self:
                continue
            dist = 0
            for k in drone.genome.keys():
                for ix in range(len(drone.genome[k])):
                    tensor1 = self.genome[k][ix]
                    tensor2 = drone.genome[k][ix]
                    diff = np.linalg.norm(tensor1 - tensor2)
                    # Scale factor to give all layers a comparable impact
                    scale_factor = max(np.linalg.norm(tensor1), np.linalg.norm(tensor2))
                    dist += diff / scale_factor

            distances[i] = dist

        # distances: np.ndarray = np.square((distances - distances.mean()) / distances.std())
        self.similarity = 1 / np.mean(distances)


if __name__ == '__main__':
    drone = AiDrone([0, 0])
    print(inp:=np.random.rand(8, 1))
    print(drone.brain(inp))