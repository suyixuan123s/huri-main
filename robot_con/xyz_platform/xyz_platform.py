import numpy as np
import serial
import robot_con.xyz_platform.xyz_platform_constants as XYZ
from collections import namedtuple
import logging
import time

# logging setup
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MotorInfo = namedtuple("MotorInfo", ["func", "pul_rev", "to_rad", "val_name"])


def convert_rad_to_0_2pi_interval(rad):
    new_rad = np.arctan2(np.sin(rad), np.cos(rad))
    if new_rad < 0:
        new_rad = abs(new_rad) + 2 * (np.pi - abs(new_rad))
    return new_rad


class _XYZPlatformSerial:
    TERMINATOR = '\r'.encode('UTF8')

    def __init__(self, device='COM4', baud=115200, timeout=1):
        self._device = device
        self.time_out = 1
        self.serial = serial.Serial(device, baud, timeout=timeout)

    def receive(self) -> str:
        # print(self.serial.read_all())
        line = self.serial.read_until(self.TERMINATOR)
        return line.decode('UTF8').strip().rstrip("\n").replace(">", "")

    def wait(self, timeout=5):
        time_start = time.time()
        while True:
            data = self.receive()
            if "OK" in data or "O" in data:
                time.sleep(.001)
                print("Motion Finished")
                print(data)
                break
            if time.time() - time_start > timeout:
                print("TIMEOUT and NO DATA Receive")
                break
            print(data)

    def send(self, text: str) -> bool:
        line = '%s\r\f' % text
        self.serial.write(line.encode('UTF8'))
        # the line should be echoed.
        # If it isn't, something is wrong.
        # print(text)
        # print("It is send successfully?", )
        while text.replace(" ", "") != self.receive().replace(" ", ""):
            time.sleep(.001)

    def close(self):
        self.serial.close()


class XYZPlatformController:
    def __init__(self, x_motor_pul_rev=XYZ.X_MOTOR_PUL_REV, y_motor_pul_rev=XYZ.Y_MOTOR_PUL_REV,
                 z_motor_pul_rev=XYZ.Z_MOTOR_PUL_REV, max_freq=XYZ.MAX_FREQ, x_phys_limit=XYZ.X_PHYS_LIMIT,
                 y_phys_limit=XYZ.Y_PHYS_LIMIT, z_phys_limit=XYZ.Z_PHYS_LIMIT, debug=False):
        # communication with PI PICO through serial. Initalized by self.start
        self._serial = None
        # is debug
        self._debug = debug

        # is calibrated
        self._is_calibrated = False

        # setup hardware information
        self._x_motor_pul_rev = x_motor_pul_rev
        self._y_motor_pul_rev = y_motor_pul_rev
        self._z_motor_pul_rev = z_motor_pul_rev
        self._x_phys_limit = x_phys_limit
        self._y_phys_limit = y_phys_limit
        self._z_phys_limit = z_phys_limit
        self._safe_margin = .02

        self._x_lead = .01
        self._y_lead = .005
        self._z_gear_ratio = 3

        # screw lead of the platform
        self.x_plat_screw_lead = None
        self.y_plat_screw_lead = None

        self._max_freq = max_freq

        self._motor_dict = {
            "x": MotorInfo(func="x_motor", pul_rev=self._x_motor_pul_rev,
                           to_rad=lambda v: v / self._x_lead * 2 * np.pi, val_name="x_value"),
            "y": MotorInfo(func="y_motor", pul_rev=self._y_motor_pul_rev,
                           to_rad=lambda v: v / self._y_lead * 2 * np.pi, val_name="y_value"),
            "z": MotorInfo(func="z_motor", pul_rev=self._z_motor_pul_rev,
                           to_rad=lambda v: v * self._z_gear_ratio, val_name="z_value"),
        }

        self._last_sets = {
            'speed_n': None,
            self._motor_dict["x"].val_name: None,
            self._motor_dict["y"].val_name: None,
            self._motor_dict["z"].val_name: None,
        }

        if not debug:
            self.start()

    def start(self):
        self._serial = _XYZPlatformSerial()

    def stop(self):
        if self._serial is not None:
            self._serial.close()

    def calibrate(self):
        # TODO install the 0 position sensors
        if input("Are you sure you are calibrated ?: ").lower() != "y":
            return
        self._is_calibrated = True
        for motor_name in self._motor_dict.keys():
            self._enable_motor(motor_name)
        self._last_sets["x_val"] = 0
        self._last_sets["y_val"] = 0
        self._last_sets["z_val"] = 0

    def set_motor_speed(self, motor_name, speed_n: float = 1):
        """
        :param motor_name: string, name of the motor in ["x", "y", "z"]
        :param speed_n, speed_n = n means speed is n*3.1415 rad/s
        """
        assert motor_name in self._motor_dict
        assert 0 <= speed_n * self._motor_dict[motor_name].pul_rev <= self._max_freq
        req = f"{self._motor_dict[motor_name].func}.set_pwm_freq({round(speed_n * self._motor_dict[motor_name].pul_rev)})"
        logger.debug(req)
        self._last_sets['speed_n'] = speed_n
        if not self._debug:
            self._serial.send(req)

    def _move_motor(self, motor_name, rad, dir="CW", speed_n=1, timeout=10):
        """
        :param motor_name: string, name of the motor in ["x", "y", "z"]
        :param rad: radian to rotate
        :param speed_n, speed_n = n means speed is n*3.1415 rad/s
        """
        assert motor_name in self._motor_dict
        assert dir in ["CW", "ACW"]
        if rad < 1e-7:
            return
            # setup the speed
        self.set_motor_speed(motor_name=motor_name, speed_n=speed_n)
        num_of_rev = rad / (2 * np.pi)
        num_of_pul = round(num_of_rev * self._motor_dict[motor_name].pul_rev)
        req = f"{self._motor_dict[motor_name].func}.revolute(num_of_pul={num_of_pul}, dir='{dir}')"
        logger.debug(req)
        if not self._debug:
            self._serial.send(req)
            self._serial.wait(timeout=timeout)

    def _enable_motor(self, motor_name):
        assert motor_name in self._motor_dict
        req = f"{self._motor_dict[motor_name].func}.enable()"
        logger.debug(req)
        if not self._debug:
            self._serial.send(req)

    def disable_motor(self, motor_name):
        assert motor_name in self._motor_dict or motor_name == "all"
        _l = [motor_name]
        if motor_name == "all":
            _l = list(self._motor_dict.keys())
        for _m_n in _l:
            req = f"{self._motor_dict[_m_n].func}.disable()"
            logger.debug(req)
            if not self._debug:
                self._serial.send(req)

    def _move_motor_sync(self, motor_names, rads, dirs, speed_ns, timeout = 300):
        """
        :param motor_names: List[string], name of the motor in ["x", "y", "z"]
        :param rad: List, radian to rotate for each motor
        :param speed_n, speed_n = n means speed is n*3.1415 rad/s
        """
        motors_string = "["
        num_of_puls_string = "["
        dirs_string = "["
        for ind, motor_n in enumerate(motor_names):
            if motor_n in self._motor_dict:
                motors_string += f"{self._motor_dict[motor_n].func},"
                self.set_motor_speed(motor_name=motor_n, speed_n=speed_ns[ind])
                # num of puls
                num_of_rev = rads[ind] / (2 * np.pi)
                num_of_pul = round(num_of_rev * self._motor_dict[motor_n].pul_rev)
                num_of_puls_string += f"{num_of_pul},"
                # direction
                if dirs[ind] not in ["CW", "ACW"]:
                    raise Exception("Bad direction")
                dirs_string += f"'{dirs[ind]}',"
            else:
                raise Exception("Bad motor name")
        motors_string = motors_string[:-1] + "]"
        num_of_puls_string = num_of_puls_string[:-1] + "]"
        dirs_string = dirs_string[:-1] + "]"

        req = f"move_sync(motors={motors_string},num_of_puls={num_of_puls_string}, dirs={dirs_string})"
        logger.debug(req)
        if not self._debug:
            self._serial.send(req)
            if "x" in motor_names:
                self._serial.wait(timeout)
            if "y" in motor_names:
                self._serial.wait(timeout)
            if "z" in motor_names:
                self._serial.wait(timeout)

    def set_home(self):
        self.set_z(0)

    def move_linear(self, motor_name, val, speed_n=1):
        assert motor_name in self._motor_dict
        if val >= 0:
            _dir = "CW"
        else:
            _dir = "ACW"
        _rad = self._motor_dict[motor_name].to_rad(abs(val))
        self._move_motor(motor_name, rad=_rad, dir=_dir, speed_n=speed_n)
        if self._is_calibrated:
            self._last_sets[self._motor_dict[motor_name].val_name] += val

    def set_x(self, pos, speed_n=1, timeout=10):
        if self._is_calibrated:
            assert self._x_phys_limit[0] <= pos <= self._x_phys_limit[1] - self._safe_margin
            if self._last_sets["x_val"] > pos:
                _dir = "ACW"
            else:
                _dir = "CW"
            _rad = abs(self._last_sets["x_val"] - pos) / self._x_lead * 2 * np.pi
            self._move_motor("x", rad=_rad, dir=_dir, speed_n=speed_n)
            self._last_sets["x_val"] = pos
        else:
            raise Exception("Calibrate the xyz platform first! ")

    def set_y(self, pos, speed_n=.5):
        if self._is_calibrated:
            assert self._y_phys_limit[0] <= pos <= self._y_phys_limit[1] - self._safe_margin
            if self._last_sets["y_val"] > pos:
                _dir = "ACW"
            else:
                _dir = "CW"
            _rad = abs(self._last_sets["y_val"] - pos) / self._y_lead * 2 * np.pi
            self._move_motor("y", rad=_rad, dir=_dir, speed_n=speed_n)
            self._last_sets["y_val"] = pos
        else:
            raise Exception("Calibrate the xyz platform first! ")

    def set_z(self, rad, speed_n=.5):
        if self._is_calibrated:
            _rad = convert_rad_to_0_2pi_interval(rad)
            if self._last_sets["z_val"] > _rad:
                _dir = "CW"
            else:
                _dir = "ACW"
            _rad_diff = abs(self._last_sets["z_val"] - _rad) * self._z_gear_ratio
            self._move_motor("z", rad=_rad_diff, dir=_dir, speed_n=speed_n)
            self._last_sets["z_val"] = _rad
        else:
            raise Exception("Calibrate the xyz platform first! ")

    def set_pos(self, component_name: str, pos, speed_n=1):
        # TODO x y z can set the speed sperately
        if component_name == "all":
            component_name = "xyz"
        if self._is_calibrated:
            assert len(component_name) == len(pos)
            _motion_names = []
            _rads = []
            _dirs = []
            if "x" in component_name:
                _motion_names.append("x")
                _x_ind = component_name.index("x")
                assert self._x_phys_limit[0] <= pos[_x_ind] <= self._x_phys_limit[1] - self._safe_margin
                if self._last_sets["x_val"] > pos[_x_ind]:
                    _dir = "ACW"
                else:
                    _dir = "CW"
                _rad = abs(self._last_sets["x_val"] - pos[_x_ind]) / self._x_lead * 2 * np.pi
                _dirs.append(_dir)
                _rads.append(_rad)
                self._last_sets["x_val"] = pos[_x_ind]
            if "y" in component_name:
                _motion_names.append("y")
                _y_ind = component_name.index("y")
                assert self._y_phys_limit[0] <= pos[_y_ind] <= self._y_phys_limit[1] - self._safe_margin
                if self._last_sets["y_val"] > pos[_y_ind]:
                    _dir = "ACW"
                else:
                    _dir = "CW"
                _rad = abs(self._last_sets["y_val"] - pos[_y_ind]) / self._y_lead * 2 * np.pi
                _dirs.append(_dir)
                _rads.append(_rad)
                self._last_sets["y_val"] = pos[_y_ind]
            if "z" in component_name:
                _motion_names.append("z")
                _z_ind = component_name.index("z")
                _rad = convert_rad_to_0_2pi_interval(pos[_z_ind])
                if self._last_sets["z_val"] > _rad:
                    _dir = "CW"
                else:
                    _dir = "ACW"
                _rad_diff = abs(self._last_sets["z_val"] - _rad) * self._z_gear_ratio
                _dirs.append(_dir)
                _rads.append(_rad_diff)
                self._last_sets["z_val"] = _rad
            if len(_motion_names) < 2:
                raise Exception(f"len(component_name) should no be smaller than 2")
            if "x" in _motion_names and "y" in _motion_names:
                speed_ns = [speed_n] * len(component_name)
                # sync x y speed, (y is slow since lead is small)
                speed_ns[0] = speed_n / (self._x_lead / self._y_lead) * 2
                speed_ns[1] = speed_ns[1] * 2
                if "z" in _motion_names:
                    speed_ns[2] = .5
                self._move_motor_sync(_motion_names, rads=_rads, dirs=_dirs, speed_ns=speed_ns)
            else:
                self._move_motor_sync(_motion_names, rads=_rads, dirs=_dirs, speed_ns=[speed_n] * len(component_name))
        else:
            raise Exception("Calibrate the xyz platform first! ")

    def get_x(self):
        if self._is_calibrated:
            return self._last_sets["X_val"]
        else:
            raise Exception("Calibrate the xyz platform first! ")

    def get_y(self):
        if self._is_calibrated:
            return self._last_sets["y_val"]
        else:
            raise Exception("Calibrate the xyz platform first! ")

    def get_z(self):
        if self._is_calibrated:
            return self._last_sets["z_val"]
        else:
            raise Exception("Calibrate the xyz platform first! ")

    # def __del__(self):
    #     self.set_home()


if __name__ == "__main__":
    test = XYZPlatformController(debug=False)
    test.disable_motor("all")
    # test.calibrate()
    # test._move_motor_sync(["x", "z"],
    #                       [np.radians(3000), np.radians(900)],
    #                       ["CW", "CW", "CW"],
    #                       [1, 1, 1])
    # test.set_pos("xyz", [.01, .01, np.radians(360)], speed_n=1)
    # test.set_x(.02)
    # test.set_pos("xyz", [.02, .02, np.radians(360)], speed_n=1)
    # serial = _XYZPlatformSerial()
    # serial.send("2+2")
