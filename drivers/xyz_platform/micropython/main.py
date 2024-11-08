"""
The main program for control the stepper. The code is wirtten in micropython
"""
from stepper import StepperMotor
from irq import IRQ

# Run the main
x_motor = StepperMotor(ena_pin=7, dir_pin=8, pul_pin=9, freq=6400)
y_motor = StepperMotor(ena_pin=3, dir_pin=4, pul_pin=5, freq=3200)
z_motor = StepperMotor(ena_pin=0, dir_pin=1, pul_pin=2, freq=400)


# IRQ for the zero sensor
x_zero_irq = IRQ(irq_pin=10, resistor="pull_down")
x_zero_irq.register(_func=x_motor.zero_position, irq_tgr_type="falling")

print('Initialization Completed')

def move_sync(motors, num_of_puls, dirs):
    for ind, motor in enumerate(motors):
        motor.revolute_sync(num_of_pul=num_of_puls[ind], dir=dirs[ind])

