import pandas as pd
from pathlib import Path

# Parameters describing the date of the experiment
EXPERIMENT_DATE = '17-02-2025'
# Parameter describing the export path for the stimulation files
EXPORT_DIR = Path(rf'C:/axorus/stimfiles/{EXPERIMENT_DATE}')
# Parameter to set the interval between stimulation sets
INTER_STIM_INTERVAL = 5  # s
# Parameter to set the number of times a stimulus should be repeated
REPEATS = 50

def get_binary_stim_code(value, stimulation_frequency, repeats):

    bit_code = format(value, '08b')  # Convert to 8-bit binary

    # Design of bit sequence
    # Times are in ms
    t_pre_high = 1
    t_pre_low = 0.5
    bit_length = 0.5
    bit_high_length = 0.4
    bit_low_length = 0.2

    bit_high_low_length = bit_length - bit_high_length
    bit_low_low_length = bit_length - bit_low_length

    total_duration = 2 * (t_pre_high+t_pre_low) + 8 * bit_length  # ms
    post_seq_duration = 1e6 / stimulation_frequency - total_duration * 1e3

    stim_code = ''

    for i in range(int(repeats)):

        # Start the sequence with 10 ms high, 5 ms low
        stim_code += f'1\t{t_pre_high*1e3:.0f}\t0\t{t_pre_low*1e3:.0f}\t1\n'

        for b in bit_code:

            # 4 ms pulse if bit is 1
            if b == '1':
                stim_code += f'1\t{bit_high_length*1e3:.0f}\t0\t{bit_high_low_length*1e3:.0f}\t1\n'

            # 2 ms pulse if bit is 0
            elif b == '0':
                stim_code += f'1\t{bit_low_length*1e3:.0f}\t0\t{bit_low_low_length*1e3:.0f}\t1\n'

        # Finish the sequence with 10 ms high, 5 ms low
        stim_code += f'1\t{t_pre_high*1e3:.0f}\t0\t{t_pre_low*1e3:.0f}\t1\n'

        # Add inter trial interval
        stim_code += f'0\t{post_seq_duration:.0f}\t0\t0\t1\n'

    return stim_code



def create_file(filename, stim_params: pd.DataFrame):
    # Create output directory
    if not EXPORT_DIR.is_dir():
        EXPORT_DIR.mkdir(parents=True)

    # Randomly shuffle the stimulation parameters
    if 'measure_power' not in stim_params.sequence.values:
        stim_params = stim_params.sample(frac=1).reset_index(drop=True)

    ch2_txt = 'channel: 1\nValue\tTime\tValue\tTime\tRepeat\n'
    ch9_txt = 'channel: 9\nValue\tTime\tValue\tTime\tRepeat\n'
    ch10_txt = 'channel: 10\nValue\tTime\tValue\tTime\tRepeat\n'

    for i, r in stim_params.iterrows():
        value_0 = r.amplitude * 1000  # convert to uV
        time_0 = r.duration * 1000  # convert to us
        value_1 = 0
        frequency = r.frequency
        assert 1e6 / frequency > time_0
        time_1 = 1e6 / frequency - time_0
        repeat = r.repeats

        # Write stim params to each channel
        # Channel 2: stimulation output
        # Channel 9: sync 1
        # Channel 10: sync 2
        ch2_txt += f'{value_0}\t{time_0}\t{value_1}\t{time_1}\t{repeat}\n'
        if r.sequence != 'measure_power':
            ch2_txt += f'{0}\t{INTER_STIM_INTERVAL*1e6:.0f}\t{0}\t{0}\t{1}\n'

        ch9_txt += get_binary_stim_code(i, frequency, repeat)
        if r.sequence != 'measure_power':
            ch9_txt += f'{0}\t{INTER_STIM_INTERVAL*1e6:.0f}\t{0}\t{0}\t{1}\n'

        ch10_txt += f'{1}\t{time_0}\t{0}\t{time_1}\t{repeat}\n'
        if r.sequence != 'measure_power':
            ch10_txt += f'{0}\t{INTER_STIM_INTERVAL*1e6:.0f}\t{0}\t{0}\t{1}\n'

    with open(filename, 'w') as f:

        # Write the header
        f.write('Multi Channel Systems MC_Stimulus II\n')
        f.write('ASCII import Version 1.10\n')

        f.write('channels: 8\n')
        f.write('output mode: voltage\n')
        f.write('format: 2\n\n')

        # Write some data
        f.write(ch2_txt)
        f.write('\n')
        f.write(ch9_txt)
        f.write('\n')
        f.write(ch10_txt)
        f.write('\n')

    stim_params.to_csv(filename.as_posix().split('.')[0] + '.csv')
    print(f'\tsaved: {filename}')


def add_power_sequence(stim_params: pd.DataFrame):
    amplitudes = [0.1, 0.2, 0.5, 1, 1.5, 2, 3]  # V
    duration = 50  # ms
    frequency = 1  # Hz

    current_row = stim_params.shape[0]

    for i, a in enumerate(amplitudes):
        stim_params.at[current_row, 'amplitude'] = a
        stim_params.at[current_row, 'duration'] = duration
        stim_params.at[current_row, 'frequency'] = frequency
        stim_params.at[current_row, 'repeats'] = REPEATS
        stim_params.at[current_row, 'sequence'] = 'power_sequence'

        current_row += 1

    return stim_params

def add_duration_sequence(stim_params: pd.DataFrame, amplitude=None):
    durations = [10, 50, 100, 20]  # ms

    if amplitude is None:
        amplitude = 0.5  # V
    frequency = 1  # Hz
    current_row = stim_params.shape[0]

    for i, d in enumerate(durations):
        stim_params.at[current_row, 'amplitude'] = amplitude
        stim_params.at[current_row, 'duration'] = d
        stim_params.at[current_row, 'frequency'] = frequency
        stim_params.at[current_row, 'repeats'] = REPEATS
        stim_params.at[current_row, 'sequence'] = 'duration_sequence'
        current_row += 1

    return stim_params

def add_frequency_sequence(stim_params: pd.DataFrame, amplitude=None):
    frequencies = [1, 2, 5, 10]
    duration = 20
    if amplitude is None:
        amplitude = 0.5
    current_row = stim_params.shape[0]

    for i, f in enumerate(frequencies):
        stim_params.at[current_row, 'amplitude'] = amplitude
        stim_params.at[current_row, 'duration'] = duration
        stim_params.at[current_row, 'frequency'] = f
        stim_params.at[current_row, 'repeats'] = REPEATS
        stim_params.at[current_row, 'sequence'] = 'duration_sequence'

        current_row += 1

    frequencies = [20, 40, 50]
    duration = 10
    amplitude = 0.5

    for i, f in enumerate(frequencies):
        stim_params.at[current_row, 'amplitude'] = amplitude
        stim_params.at[current_row, 'duration'] = duration
        stim_params.at[current_row, 'frequency'] = f
        stim_params.at[current_row, 'repeats'] = REPEATS
        stim_params.at[current_row, 'sequence'] = 'duration_sequence'
        current_row += 1

    return stim_params

def add_simple_test(stim_params: pd.DataFrame):
    current_row = stim_params.shape[0]
    stim_params.at[current_row, 'amplitude'] = 0.5
    stim_params.at[current_row, 'duration'] = 20
    stim_params.at[current_row, 'frequency'] = 1
    stim_params.at[current_row, 'repeats'] = REPEATS
    stim_params.at[current_row, 'sequence'] = 'duration_sequence'
    return stim_params


def measure_power(stim_params: pd.DataFrame):
    # powers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]
    powers = [5, 4, 3, 2.5, 2.25, 2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.2, 0.1]
    current_row = stim_params.shape[0]

    for p in powers:
        stim_params.at[current_row, 'amplitude'] = p
        stim_params.at[current_row, 'duration'] = 8000
        stim_params.at[current_row, 'frequency'] = 0.1
        stim_params.at[current_row, 'repeats'] = 1
        stim_params.at[current_row, 'sequence'] = 'measure_power'

        current_row += 1

    return stim_params


def main():
    print(f'exporting stimulation files in: {EXPORT_DIR}')

    # Create a file for the power sequence
    stim_params = pd.DataFrame()
    stim_params = add_power_sequence(stim_params)
    create_file(EXPORT_DIR / f'stim_1_power_sequence_{EXPERIMENT_DATE}.dat', stim_params)

    # Create a file for the duration sequence
    stim_params = pd.DataFrame()
    stim_params = add_duration_sequence(stim_params)
    create_file(EXPORT_DIR / f'stim_2_duration_sequence_{EXPERIMENT_DATE}.dat', stim_params)

    # Create a file for the frequency sequence
    stim_params = pd.DataFrame()
    stim_params = add_frequency_sequence(stim_params)
    create_file(EXPORT_DIR / f'stim_3_frequency_sequence_{EXPERIMENT_DATE}.dat', stim_params)

    # Create a file for testing veps
    stim_params = pd.DataFrame()
    stim_params = add_simple_test(stim_params)
    create_file(EXPORT_DIR / f'simple_test{EXPERIMENT_DATE}.dat', stim_params)

    # Create a file that combines all 3 sequences in random order
    stim_params = pd.DataFrame()
    stim_params = add_power_sequence(stim_params)
    stim_params = add_duration_sequence(stim_params)
    stim_params = add_frequency_sequence(stim_params)
    create_file(EXPORT_DIR / f'stim_5_all_sequences_{EXPERIMENT_DATE}.dat', stim_params)

    # Create a stim file for burst duraiton with higher power
    stim_params = pd.DataFrame()
    stim_params = add_duration_sequence(stim_params, 3)
    create_file(EXPORT_DIR / f'stim_6_burst_duration_high_power_{EXPERIMENT_DATE}.dat', stim_params)

    # Create a stim file frequency with higher power
    stim_params = pd.DataFrame()
    stim_params = add_frequency_sequence(stim_params, 3)
    create_file(EXPORT_DIR / f'stim_7_frequency_sequence_{EXPERIMENT_DATE}.dat', stim_params)

    # Create a file that runs the poewr measurement
    stim_params = pd.DataFrame()
    stim_params = measure_power(stim_params)
    create_file(EXPORT_DIR / f'mearure_power.dat', stim_params)


if __name__ == '__main__':
    main()

