"""
This is the micropython code for the Raspberry PI PICO.
The code is for the NEMA stepper motor with TB6600 motor driver
Author: Chen Hao (chenhao.info)
"""

import machine
from machine import Pin, PWM, Timer
from time import sleep


# basic function for control the motor
class StepperMotor:
    DUTY = 1 / 3
    ENABLE = 0
    DISABLE = 1
    CW = 0
    ACW = 1

    def __init__(self, ena_pin, dir_pin, pul_pin, freq=1600):
        # define the pins
        self.ena_pin = Pin(ena_pin, Pin.OUT)  # Low voltage is enable, High voltage is disable
        self.dir_pin = Pin(dir_pin, Pin.OUT)  # Low voltage is is clockwise, High voltage is anticlockwise
        self.pul_pin = Pin(pul_pin, Pin.OUT)

        # define the pwm pin
        self._pwm_pul_pin = PWM(self.pul_pin)
        self._pwm_pul_pin.freq(int(freq))
        self._freq = int(freq)

        # is calibrated
        self._is_calibrated = False

        # is enable
        self._is_enable = False
        self.enable()

        # is irq free
        self.is_irq_free = True

    def enable(self):
        """
        Enable the motor (default is enabled)
        """
        self.ena_pin.value(self.ENABLE)
        self._is_enable = True

    def disable(self):
        """
        Disable the motor
        """
        self.ena_pin.value(self.DISABLE)
        self._is_enable = False
        print("the motor status running: ", self._is_enable)

    def set_pwm_freq(self, freq):
        """
        set the frequency to send the pwm signal. It is proportional to the speed of the motor
        :param freq: int, the frequency of the pwm
        """
        self._pwm_pul_pin.freq(int(freq))
        self._freq = int(freq)

    def revolute(self, num_of_pul, dir="CW"):
        """
        rotate the stepper motor according to number of pwm signal
        :param num_of_pul : int, number of the PWM signal. It is proportional to the number of the motor's rotation
        :param dir: string[CW, ACW], direction of the motor. Either clockwise (CW) or anticlockwise (ACW)

        """
        assert dir in ["CW", "ACW"]
        if not self._is_enable: return
        if dir == "CW":
            self.dir_pin.value(self.CW)
        else:
            self.dir_pin.value(self.ACW)
        T = 1 / self._freq
        self._pwm_pul_pin.duty_u16(int(65535 * self.DUTY))
        sleep(T * num_of_pul)
        self._pwm_pul_pin.deinit()
        return "OK"

    def revolute_sync(self, num_of_pul, dir="CW"):
        """
        rotate the stepper motor according to number of pwm signal
        :param num_of_pul : int, number of the PWM signal. It is proportional to the number of the motor's rotation
        :param dir: string[CW, ACW], direction of the motor. Either clockwise (CW) or anticlockwise (ACW)

        """
        assert dir in ["CW", "ACW"]
        if not self._is_enable: return
        if dir == "CW":
            self.dir_pin.value(self.CW)
        else:
            self.dir_pin.value(self.ACW)
        tim = Timer()

        def stop_pwm(timer):
            self.stop_pwm()

        T = 1 / self._freq
        self._pwm_pul_pin.duty_u16(int(65535 * self.DUTY))
        tim.init(period=int(T * num_of_pul * 1000), mode=Timer.ONE_SHOT, callback=stop_pwm)

    def stop_pwm(self):
        self._pwm_pul_pin.deinit()

    def zero_position(self, pin):
        if not self.is_irq_free:
            return
        self.is_irq_free = False
        state = machine.disable_irq()
        tim = Timer()

        def fire_irq_func(timer):
            self.is_irq_free = True
            _pin_val = pin.value()
            if _pin_val == 0:
                self.disable()
                self.stop_pwm()
                return "OK"

        tim.init(period=10, mode=Timer.ONE_SHOT, callback=fire_irq_func)
        machine.enable_irq(state)

