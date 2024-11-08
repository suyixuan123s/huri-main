# pulse per revolution
X_MOTOR_PUL_REV = 1600
Y_MOTOR_PUL_REV = 6400
Z_MOTOR_PUL_REV = 800

# Max frequency of the TB6600 driver, duty is 1/3. Use 13 kHz for safety
# To change the duty, please revise the setting in the driver
# https://www.mcielectronics.cl/website_MCI/static/documents/TB6600_data_sheet.pdf
MAX_FREQ = 13000

# Physic Constraint of the platform
X_PHYS_LIMIT = (0, .421)
Y_PHYS_LIMIT = (0, .709)
Z_PHYS_LIMIT = (0, 3.1415926535)
