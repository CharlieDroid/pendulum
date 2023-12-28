import pigpio
import time


def button_pressed(gpio, level, tick):
    print(f"{gpio=} {level=} {tick=}")
    global cb
    cb.cancel()


if __name__ == "__main__":
    pi = pigpio.pi()

    button_pin = 26

    pi.set_mode(button_pin, pigpio.INPUT)

    pi.set_pull_up_down(button_pin, pigpio.PUD_UP)

    cb = pi.callback(button_pin, pigpio.RISING_EDGE, button_pressed)
    # add this code to others very good :thumbsup:
    try:
        while True:
            pass
    except KeyboardInterrupt:
        cb.cancel()
