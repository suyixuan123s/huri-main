"""
This is the program to run the irq
"""
from machine import Pin


class IRQ:
    PULL_UP = Pin.PULL_UP
    PULL_DOWN = Pin.PULL_DOWN
    IRQ_FALLING = Pin.IRQ_FALLING
    IRQ_RISING = Pin.IRQ_RISING
    HIGH = 1
    LOW = 0

    def __init__(self, irq_pin, resistor="PULL_UP"):
        resistor = resistor.lower()
        assert resistor in ["pull_up", "pull_down", "none"]
        if "pull" in resistor:
            if resistor == "pull_up":
                pull_resistor = self.PULL_UP
            else:
                pull_resistor = self.PULL_DOWN
            self._irq_pin = Pin(irq_pin, Pin.IN, pull_resistor)
        else:
            self._irq_pin = Pin(irq_pin, Pin.IN)
        print("irq is running in Pin: ", irq_pin)

    def register(self, _func, irq_tgr_type="RISING"):
        irq_tgr_type = irq_tgr_type.lower()
        assert irq_tgr_type in ["rising", "falling"]
        if irq_tgr_type == "rising":
            _irq_tgr_type = self.IRQ_RISING
        else:
            _irq_tgr_type = self.IRQ_FALLING
        self._irq_pin.irq(handler=_func, trigger=_irq_tgr_type)

    def get_val(self):
        return self._irq_pin.value()
