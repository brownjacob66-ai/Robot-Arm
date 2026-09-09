import importlib.util
import math
import sys
import time
import types
import unittest
from pathlib import Path


REPO_ROOT = Path("/home/runner/work/Robot-Arm/Robot-Arm")
MODULE_PATH = REPO_ROOT / "robot_control_V2.py"


def _install_stub_modules():
    serial = types.ModuleType("serial")
    serial.Serial = object
    sys.modules["serial"] = serial

    numpy = types.ModuleType("numpy")
    numpy.pi = math.pi
    numpy.cos = lambda values: [math.cos(value) for value in values]
    numpy.sin = lambda values: [math.sin(value) for value in values]
    numpy.linspace = lambda start, end, count: [
        start + (end - start) * index / (count - 1) for index in range(count)
    ]
    numpy.full_like = lambda values, fill: [fill for _ in values]
    sys.modules["numpy"] = numpy

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = lambda *args, **kwargs: None
    pyplot.pause = lambda *args, **kwargs: None
    sys.modules["matplotlib"] = types.ModuleType("matplotlib")
    sys.modules["matplotlib.pyplot"] = pyplot
    sys.modules["matplotlib.backends"] = types.ModuleType("matplotlib.backends")
    backend_tkagg = types.ModuleType("matplotlib.backends.backend_tkagg")
    backend_tkagg.FigureCanvasTkAgg = object
    sys.modules["matplotlib.backends.backend_tkagg"] = backend_tkagg

    mplot3d = types.ModuleType("mpl_toolkits.mplot3d")
    mplot3d.Axes3D = object
    sys.modules["mpl_toolkits"] = types.ModuleType("mpl_toolkits")
    sys.modules["mpl_toolkits.mplot3d"] = mplot3d

    class DummyLink:
        def __init__(self, *args, **kwargs):
            pass

    class DummyFKMatrix:
        def __init__(self):
            self._values = {
                (0, 3): 0.0,
                (1, 3): 0.0,
                (2, 3): 0.0,
            }

        def __getitem__(self, key):
            return self._values[key]

    class DummyChain:
        def __init__(self, *args, **kwargs):
            pass

        def forward_kinematics(self, joints):
            return DummyFKMatrix()

        def inverse_kinematics(self, target_position=None, initial_position=None):
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        def plot(self, *args, **kwargs):
            return None

    ikpy = types.ModuleType("ikpy")
    chain_module = types.ModuleType("ikpy.chain")
    chain_module.Chain = DummyChain
    link_module = types.ModuleType("ikpy.link")
    link_module.OriginLink = DummyLink
    link_module.URDFLink = DummyLink
    ikpy.chain = chain_module
    ikpy.link = link_module
    sys.modules["ikpy"] = ikpy
    sys.modules["ikpy.chain"] = chain_module
    sys.modules["ikpy.link"] = link_module

    tkinter = types.ModuleType("tkinter")
    tkinter.Tk = type("Tk", (), {})
    tkinter.LEFT = "left"
    tkinter.RIGHT = "right"
    tkinter.BOTH = "both"
    tkinter.X = "x"
    tkinter.END = "end"
    tkinter.NORMAL = "normal"
    tkinter.DISABLED = "disabled"
    tkinter.StringVar = lambda value=None: types.SimpleNamespace(
        get=lambda: value,
        set=lambda new_value: None,
    )
    sys.modules["tkinter"] = tkinter
    sys.modules["tkinter.ttk"] = types.ModuleType("tkinter.ttk")
    sys.modules["tkinter.scrolledtext"] = types.ModuleType("tkinter.scrolledtext")
    sys.modules["tkinter.messagebox"] = types.ModuleType("tkinter.messagebox")


def load_robot_module():
    _install_stub_modules()
    module_name = "robot_control_V2_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RobotControlSimulationTests(unittest.TestCase):
    def setUp(self):
        self.robot_control = load_robot_module()

    def tearDown(self):
        sys.modules.pop("robot_control_V2_under_test", None)

    def test_pitch_round_trip_preserves_sign(self):
        steps = self.robot_control.convert_angles_to_steps(0.0, 0.0, 0.0, 10.0, 0.0)
        joint_rads = self.robot_control.convert_steps_to_rads(*steps)

        self.assertAlmostEqual(math.degrees(joint_rads[4]), 10.0, places=6)

    def test_simulated_move_completes_with_short_timeout(self):
        robot = self.robot_control.RobotInterface(simulation_mode=True, log_callback=lambda *_: None)
        try:
            self.assertTrue(robot.move_to(100, 200, 300, 400, 500, timeout_sec=0.8))
            self.assertEqual(robot.get_current_position_steps(), [100, 200, 300, 400, 500])
        finally:
            robot.close()

    def test_simulated_move_uses_fixed_start_pose_for_interpolation(self):
        robot = self.robot_control.RobotInterface(simulation_mode=True, log_callback=lambda *_: None)
        try:
            robot.running = False
            robot.read_thread.join(timeout=1.0)

            robot.joint_state = self.robot_control.ArmJointState(base_steps=0)
            robot._sim_move_target = [100, 0, 0, 0, 0]
            robot._sim_move_start_position = [0, 0, 0, 0, 0]
            robot._sim_move_duration_ms = 1000

            robot._sim_move_start_time = time.time() - 0.50
            robot._simulate_move()
            halfway = robot.get_current_position_steps()[0]

            robot._sim_move_start_time = time.time() - 0.75
            robot._simulate_move()
            three_quarters = robot.get_current_position_steps()[0]

            self.assertGreaterEqual(halfway, 49)
            self.assertLessEqual(halfway, 51)
            self.assertGreaterEqual(three_quarters, 74)
            self.assertLessEqual(three_quarters, 76)
        finally:
            robot.close()


if __name__ == "__main__":
    unittest.main()
