import pandas as pd
from pathlib import Path
import allego_file_reader as afr
import utils
import numpy as np
from project_colors import ProjectColors
from scipy.signal import butter, iirnotch, filtfilt, lfilter, sosfiltfilt
import re


def check_recording_datadir(data_dir: Path):
    """"
    Check whether all files are properly organised in the data directory.
    Each recording file should start with 'rec_#_' where # is the recording number.
    Per recording there are 3 files:
    - *json.xdat
    - *_data.xdat
    - *_timestamp.xdat

    Further there should be a 'stimulation per recording' (see example in docs)
    """

    # List of required files in recording dir
    required_files = (
        'stimulation_per_recording.xlsx',  # overview of stimulation per rec file
    )

    # Check if all the filenames are formatted properly
    for f in data_dir.iterdir():

        if ((not bool(re.match(r"^rec_\d+_", f.name))   # file should start with rec_#_
                and f.name not in required_files)                  # or file should be in required files
                and f.name[0] != '~'):                          # skip temporary files
            raise FileExistsError(f, f'see README for which files are allowed')

    # Check if all required files are present
    for f in required_files:
        assert (data_dir / f).exists(), f'{f} does not exist'

    # Check if all files per recording are available
    ro = load_recording_overview(data_dir)

    for i, r in ro.iterrows():
        assert (data_dir / f'{i}.xdat.json').exists(), f'{i}.xdat.json does not exist'
        assert (data_dir / f'{i}_timestamp.xdat').exists(), f'{i}_timestamp.xdat does not exist'
        assert (data_dir / f'{i}_data.xdat').exists(), f'{i}_data.xdat does not exist'

    print(f'Data files ok!')


def load_recording_overview(data_dir: Path):
    ro = pd.read_excel(data_dir / 'stimulation_per_recording.xlsx', engine='openpyxl',
                       index_col=0, header=0)
    ro['rec_nr'] = ro.index.str.split('_').str[1].astype(int)
    return ro


def read_data_files(data_dir, recname):
    # Read datafiles
    filename = (data_dir / f'{recname}').as_posix()
    meta = afr.read_allego_xdat_metadata(filename)
    signals, timestamps, time_samples = afr.read_allego_xdat_all_signals(filename, None, None)

    time_samples -= time_samples[0]      # Set 1st timestamp to 0
    time_samples = time_samples * 1000  # convert to ms

    # Extract metadatanames
    channel_names = meta['sapiens_base']['biointerface_map']['chan_name']
    sys_chan_idx = meta['sapiens_base']['biointerface_map']['sys_chan_idx']
    channel_x = meta['sapiens_base']['biointerface_map']['site_ctr_x']
    channel_y = meta['sapiens_base']['biointerface_map']['site_ctr_y']

    # Create a table with channel names
    channel_df = pd.DataFrame()
    for i, ch_name in enumerate(channel_names):
        channel_df.at[ch_name, 'sys_chan_idx'] = sys_chan_idx[i]
        channel_df.at[ch_name, 'site_ctr_x'] = channel_x[i]
        channel_df.at[ch_name, 'site_ctr_y'] = channel_y[i]

    return signals, time_samples, channel_df


def detect_stim_onsets(signals, time_samples, channel_df):
    # read trigger in signals
    din_1_data = signals[int(channel_df.loc['din_1', 'sys_chan_idx']), :].flatten()
    din_2_data = signals[int(channel_df.loc['din_2', 'sys_chan_idx']), :].flatten()
    aux_1_data = signals[int(channel_df.loc['aux_1', 'sys_chan_idx']), :].flatten()
    aux_2_data = signals[int(channel_df.loc['aux_2', 'sys_chan_idx']), :].flatten()

    # detect pulses on digital 1
    # Detect up slopes and down slopes
    d1_up_idx = np.where(np.diff(din_1_data) == 1)[0]
    d1_down_idx = np.where(np.diff(din_1_data) == -1)[0]

    d1_pulse_df = pd.DataFrame({
        'onset': time_samples[d1_up_idx],
        'offset': time_samples[d1_down_idx],
        'duration': time_samples[d1_down_idx] - time_samples[d1_up_idx],
    })

    # detect pulses on digital 2
    # Detect up slopes and down slopes
    d2_up_idx = np.where(np.diff(din_2_data) > 0)[0]
    d2_down_idx = np.where(np.diff(din_2_data) < 0)[0]

    d2_pulse_df = pd.DataFrame({
        'onset': time_samples[d2_up_idx],
        'offset': time_samples[d2_down_idx],
        'duration': time_samples[d2_down_idx] - time_samples[d2_up_idx],
    })

    # t0 = 11.2
    # t1 = 12
    # idx = np.where((time_samples >= t0 * 1e3) & (time_samples <= t1 * 1e3))[0]
    # x = time_samples[idx]
    # y = din_2_data[idx]
    #
    # onsets = d2_pulse_df.query(f'onset >= {t0 * 1e3} and onset <= {t1 * 1e3}')['onset'].values
    # onsets_y = np.vstack([np.zeros_like(onsets), np.ones_like(onsets), [np.nan for _ in onsets]]).T.flatten()
    # onsets_x = np.vstack([onsets, onsets, [np.nan for _ in onsets]]).T.flatten()
    #
    # offsets = d2_pulse_df.query(f'onset >= {t0 * 1e3} and onset <= {t1 * 1e3}')['offset'].values
    # offsets_y = np.vstack([np.zeros_like(offsets), np.ones_like(offsets), [np.nan for _ in offsets]]).T.flatten()
    # offsets_x = np.vstack([offsets, offsets, [np.nan for _ in offsets]]).T.flatten()
    #
    # fig = utils.simple_fig()
    # fig.add_scatter(x=x, y=y, mode='lines', name='din_1', line=dict(color='black', width=1))
    # fig.add_scatter(x=onsets_x, y=onsets_y, mode='lines', name='onset', line=dict(color='green', width=1))
    # fig.add_scatter(x=offsets_x, y=offsets_y, mode='lines', name='onset', line=dict(color='red', width=1))
    # fig.update_xaxes(tickvals=np.arange(t0*1e3, t1*1e3, 100))
    # utils.save_fig(fig, data_dir.parent / 'test')

    return d1_pulse_df, d2_pulse_df, None, None


def decode_binary_stim_code(pulse_df: pd.DataFrame) -> pd.DataFrame:

    # each pulse starts with 1 ms high + 0.5ms low and ends with 1 ms high + 0.5 ms low
    # each bit has length 0.5 ms
    # a high bit has length 0.4 ms
    # sequence has 8 bits
    # a single pulse has length 2 * 1.5 ms + 8 * 0.5ms = 7 ms

    bin_codes = pd.DataFrame()
    IN_BIT = False  # start reading outside bit, loop over all high points to
        # detect bin onsets and offsets

    bin_nr = 0
    for i, r in pulse_df.iterrows():
        if r.duration > 0.8:  # take a bit of margin, bin on/offsets are marked with 1ms duration
            if not IN_BIT:
                bin_codes.at[bin_nr, 'onset'] = r.onset
                IN_BIT = True
            elif IN_BIT:
                bin_codes.at[bin_nr, 'offset'] = r.offset
                IN_BIT = False
                bin_nr += 1



    return


def detect_stim_onsets_old(time_samples, signals, channel_df, ch_name):
    # Get the digital in data
    din_1_data = signals[int(channel_df.loc[ch_name, 'sys_chan_idx']), :].flatten()

    # Detect up slopes
    up_idx = np.where(np.diff(din_1_data) == 1)[0]
    down_idx = np.where(np.diff(din_1_data) == -1)[0]

    burst_df = pd.DataFrame()

    for b_i, (ui, di) in enumerate(zip(up_idx, down_idx)):
        burst_df.at[b_i, 'burst_onset'] = time_samples[ui]
        burst_df.at[b_i, 'burst_offset'] = time_samples[di]
        burst_df.at[b_i, 'burst_duration_calculated'] = time_samples[di] - time_samples[ui]

        if b_i > 0:
            dt = time_samples[ui] - time_samples[up_idx[b_i-1]]

            if dt < 2e3:
                burst_df.at[b_i, 'burst_frequency_calculated'] = 1e3 / dt
            else:
                burst_df.at[b_i, 'burst_frequency_calculated'] = None
        else:
            burst_df.at[b_i, 'burst_frequency_calculated'] = None

    return burst_df


def align_detected_burst_onset_with_stimulation_files(burst_df, data_dir, stimulation_sequences):
    # Make sure that the nr of sequences in DIN and stimfiles matches

    # Detect how many sequences are measured in the DIN signal
    # a new sequence starts with a time > 10s, as during the experiment,
    # we take a bit more than 30s to setup the next stimulation sequence
    dt = burst_df.burst_onset.diff().unique()
    n_measured_sequences = np.where(dt > 1e4)[0].size + 1

    n_expected_sequences = len(stimulation_sequences)
    assert n_measured_sequences == n_expected_sequences

    # Assign a sequence nr to each row in burst_df
    sequence_nr = 0
    train_nr = 0
    burst_nr = 0

    n_measured_bursts = burst_df.shape[0]

    for i in range(n_measured_bursts):
        if i > 0:
            dt = burst_df.iloc[i].burst_onset - burst_df.iloc[i-1].burst_onset

            if dt > 1500:
                train_nr += 1
                burst_nr = 0

            if dt > 1e4:
                sequence_nr += 1
                train_nr = 0

        burst_df.at[i, 'sequence_nr'] = sequence_nr
        burst_df.at[i, 'train_nr'] = train_nr
        burst_df.at[i, 'burst_nr'] = burst_nr

        burst_nr += 1

    # Verify that the expected nr of pulses matches between the detected and send data
    for sequence_i, sequence_name in enumerate(stimulation_sequences):

        # Read sequence file for this sequence
        dfs = read_stm_file(data_dir / f'{sequence_name}_sequence.stm3')

        # Measure how many trains/burst to expect for this sequence
        n_expected_bursts = 0
        n_expected_trains = 0
        for i, r in dfs.iterrows():
            if r['value_0'] > 0:
                n_expected_bursts += r['row_repeats']
                n_expected_trains += 1

        # Report any mismatches
        n_measured_trains = burst_df.query('sequence_nr == @sequence_i').train_nr.unique().size
        if n_expected_trains != n_measured_trains:
            print(f'WARNING - MISMATCH IN EXPECTED TRAIN COUNT AND MEASURED TRAIN COUNT: {sequence_name}')

        n_measured_bursts = burst_df.query('sequence_nr == @sequence_i').shape[0]
        if n_expected_bursts != n_measured_bursts:
            print(f'WARNING - MISMATCH IN EXPECTED BURST COUNT AND MEASURED BURST COUNT: {sequence_name}')

        # Detect which trains are 'intact'
        # Only align trains which are complete
        train_i = 0
        for i, r in dfs.iterrows():
            if r['value_0'] > 0:  # rows with a zero in the stim files are inter trial intervals
                n_bursts = r['row_repeats']
                df_this_train = burst_df.query('sequence_nr == @sequence_i and train_nr == @train_i')

                if n_bursts == df_this_train.shape[0]:

                    # Assign stimulation parameters to burst dataframe
                    a = r['value_0']
                    d0 = r['duration_0']
                    d1 = r['duration_1']

                    for bdf_idx, bdf_info in burst_df.query('sequence_nr == @sequence_i and train_nr == @train_i').iterrows():

                        assert np.abs(bdf_info.burst_duration_calculated - d0) < 2, f'{bdf_idx} {sequence_name} {d0} {bdf_info.burst_duration_calculated:.03f}'
                        frequency = 1e3 / (d0 + d1)

                        if bdf_info.burst_frequency_calculated is not None:
                            assert np.abs(frequency - bdf_info.burst_frequency_calculated) < 1

                        burst_df.at[int(bdf_idx), 'amplitude'] = a
                        burst_df.at[bdf_idx, 'burst_duration'] = d0
                        burst_df.at[bdf_idx, 'frequency'] = 1e3 / (d0 + d1)
                        burst_df.at[bdf_idx, 'stim_sequence'] = sequence_name
                        burst_df.at[bdf_idx, 'TRIGGER_ALIGNED'] = True

                    train_i += 1

                else:
                    # Assume that from hereon, the data is corrupt
                    print(f'WARNING: DATA IS NOT ALIGNED, STOPPING AFTER TRAIN {train_i} (out of {n_expected_trains})')
                    break

    return burst_df


def get_recording_data(data_dir, recname):

    signals, time_samples, channel_df = read_data_files(data_dir, recname)

    # Detect stimulation onsets in this file
    burst_df = detect_stim_onsets(time_samples, signals, channel_df, 'din_1')
    burst_df['recording_name'] = recording_name
    # plot_din_signal(time_samples, signals, channel_df, burst_df)

    # Align dataframe with sequence
    burst_df = align_detected_burst_onset_with_stimulation_files(burst_df, data_dir, stim_seqs_dict[rec_names[rec_nr]])
    # Only include 'aligned' burst onsets
    burst_df = burst_df.query('TRIGGER_ALIGNED == True')

    return channel_df, burst_df, time_samples, signals


if __name__ == '__main__':
    data_dir = Path(r"E:\250306_PEV\recordings")
    figure_savedir = Path(r"E:\250306_PEV\figures")
    signals, time_samples, channel_df = read_data_files(data_dir, 'rec_1_wl_allseqs')
    d1, d2, a1, a2 = detect_stim_onsets(signals=signals, time_samples=time_samples, channel_df=channel_df)
    print(d2.shape)
    decode_binary_stim_code(d1)
