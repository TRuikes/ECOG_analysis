import colormaps as cmaps
import numpy as np


class ProjectColors:
    def __init__(self):
        return

    @staticmethod
    def led_amplitude(led_amp, alpha=1):

        min_led_level = 0
        max_led_level = 3.1

        laser_level = int((led_amp - min_led_level) / (max_led_level - min_led_level) * 100)

        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[laser_level, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def n_turns(n_turns, alpha=1):

        min_n_turns = 25
        max_n_turns = 36

        laser_level = int((n_turns - min_n_turns) / (max_n_turns - min_n_turns) * 100)

        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[laser_level, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def burst_frequency(freq, alpha=1):

        min_led_freq = 0
        max_led_freq = 55

        laser_level = int((freq - min_led_freq) / (max_led_freq - min_led_freq) * 100)

        r, g, b = cmaps.lavender.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[laser_level, :]

        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def burst_duration(bd, alpha=1):
        min_bd = 0
        max_bd = 110

        idx = int((bd - min_bd) / (max_bd - min_bd) * 100)
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[idx, :]
        return f'rgba({r}, {g}, {b}, {alpha})'


    @staticmethod
    def random_color(clr_i):
        r, g, b = cmaps.sinebow_dark.discrete(10).colors[clr_i, :]
        return f'rgba({r}, {g}, {b}, 1)'

    @staticmethod
    def duty_cycle(duty_cycle):
        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.torch.cut(0.2, 'left').cut(0.2, 'right').discrete(100).colors[int(duty_cycle), :]
        return f'rgba({r}, {g}, {b}, 1)'

    @staticmethod
    def repetition_frequency(prf, alpha=1):
        rel_i = int((prf - 1000) / (8000 - 1000) * 100)
        r, g, b = cmaps.torch.cut(0.2, 'left').cut(0.2, 'right').discrete(100).colors[rel_i, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def min_max_map(val, min_val, max_val):
        rel_i = int((val - min_val) / (max_val - min_val) * 100)
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[rel_i, :]
        return f'rgba({r}, {g}, {b}, {1})'


if __name__ == '__main__':
    # p = ProjectColors()
    cmaps.show_cmaps_all()