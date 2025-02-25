import json
import pandas as pd
from pathlib import Path
import allego_file_reader as afr
import utils
import numpy as np
from project_colors import ProjectColors
from scipy.signal import butter, iirnotch, filtfilt, lfilter, sosfiltfilt, medfilt
from PIL import Image
import imageio


def check_duration_unit(dunit):
    if dunit == 0:
        return 1
    elif dunit == 1:
        return 1
    elif dunit == 2:
        return 1e3
    else:
        raise ValueError(dunit)



def read_stm_file(stm_file):
    r = json.loads(open(stm_file.as_posix(), 'r').read())

    chdata = None
    for c in r['ChannelData']:
        if c['ChannelNumber'] == 1:
            chdata = c

    p = chdata['PatternRows']

    df = pd.DataFrame()

    for row_i, rdata in enumerate(p):
        amplitude_units = rdata['AmplitudeUnit']
        for a in amplitude_units:
            assert a == 0 or a == 1

        df.at[row_i, 'value_0'] = rdata['Amplitude'][0]
        df.at[row_i, 'value_1'] = rdata['Amplitude'][1]
        df.at[row_i, 'value_2'] = rdata['Amplitude'][2]

        for i in range(3):
            df.at[row_i, f'duration_{i}'] = rdata['Duration'][i] * check_duration_unit(rdata['DurationUnit'][i])

        df.at[row_i, 'row_repeats'] = rdata['RowRepeat']
    return df



def read_data_files(data_dir, file_nr):
    # Read datafiles
    p = data_dir.expanduser()
    all_xdat_datasource_names = [Path(elem.stem).stem for elem in list(p.glob('**/*xdat.json'))]

    n_files = len(all_xdat_datasource_names)
    print(f'detected {n_files} files')
    for f in all_xdat_datasource_names:
        print(f'\t- {f}')

    filename = all_xdat_datasource_names[file_nr]
    file_path = str(Path(data_dir, filename))
    print(f'reading: {file_path}')
    meta = afr.read_allego_xdat_metadata(file_path)
    signals, timestamps, time_samples = afr.read_allego_xdat_all_signals(file_path, None, None)


    time_samples -= time_samples[0]
    time_samples = time_samples * 1000  # convert to ms

    # Extract metadatanames
    channel_names = meta['sapiens_base']['biointerface_map']['chan_name']
    sys_chan_idx = meta['sapiens_base']['biointerface_map']['sys_chan_idx']
    channel_x = meta['sapiens_base']['biointerface_map']['site_ctr_x']
    channel_y = meta['sapiens_base']['biointerface_map']['site_ctr_y']

    channel_df = pd.DataFrame()
    for i, chname in enumerate(channel_names):
        channel_df.at[chname, 'sys_chan_idx'] = sys_chan_idx[i]
        channel_df.at[chname, 'site_ctr_x'] = channel_x[i]
        channel_df.at[chname, 'site_ctr_y'] = channel_y[i]

    channel_df.head()

    return signals, time_samples, channel_df



def detect_stim_onsets(time_samples, signals, channel_df, ch_name):
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



def get_recording_data(data_dir, rec_nr):
    # Read stimulation sequences
    file_path = data_dir / 'stimulation_sequences.csv'
    stim_seqs = pd.read_csv(file_path, index_col=0, header=0)
    stim_seqs_dict = {}
    for i, r in stim_seqs.iterrows():
        stim_seqs_dict[i] = [s.strip() for s in r['stimulation sequences'].split(',')]
    rec_names = list(stim_seqs_dict.keys())

    recording_name = rec_names[rec_nr]

    signals, time_samples, channel_df = read_data_files(data_dir, rec_nr)
    # signals_filtered = signals
    # plot_probe(channel_df)

    # Detect stimulation onsets in this file
    burst_df = detect_stim_onsets(time_samples, signals, channel_df, 'din_1')
    burst_df['recording_name'] = recording_name
    # plot_din_signal(time_samples, signals, channel_df, burst_df)

    # Align dataframe with sequence
    burst_df = align_detected_burst_onset_with_stimulation_files(burst_df, data_dir, stim_seqs_dict[rec_names[rec_nr]])
    # Only include 'aligned' burst onsets
    burst_df = burst_df.query('TRIGGER_ALIGNED == True')

    return channel_df, burst_df, time_samples, signals


def main():
    data_dir = Path(r'C:\axorus\250120-PEV_test')
    figure_savedir = Path(r'C:\axorus\250120-PEV_test\figures')

    rec_nr = 0
    channel_df, burst_df, time_samples, signals = get_recording_data(data_dir, rec_nr)
    sos = butter(4, (5, 300), btype='bandpass', analog=False, fs=3000,
                 output='sos')

    to_keep = []
    for i, r in channel_df.iterrows():
        if 'pri' in i:
            to_keep.append(i)
    channel_df = channel_df.loc[to_keep]

    signals_filt = np.zeros_like(signals)
    for i in range(signals_filt.shape[0]):
        signals_filt[i, :] = sosfiltfilt(sos, signals[i, :])
        signals_filt[i, :] = medfilt(signals_filt[i, :], 31)


    x_pos = np.arange(-2100, 2100, 600)
    y_pos = np.arange(0, 3900, 600)

    for i, r in channel_df.iterrows():
        channel_df.at[i, 'image_x'] = np.argmin(np.abs(x_pos - r.site_ctr_x))
        channel_df.at[i, 'image_y'] = np.argmin(np.abs(y_pos - r.site_ctr_y))

    bdf_to_plot = burst_df.query('amplitude == 3')
    burst_to_plot = 393
    t_pre = 100
    t_post = 1300

    burst_onset = burst_df.loc[burst_to_plot].burst_onset
    idx = np.where((time_samples >= burst_onset - t_pre) & (time_samples < burst_onset + t_post))[0]

    sig_to_plot = signals_filt[:, idx]
    time_to_plot = time_samples[idx] - burst_onset

    idx = np.where((time_to_plot >= -10) & (time_to_plot < 0))[0]
    for ch_i in range(sig_to_plot.shape[0]):
        sig_to_plot[ch_i, :] = sig_to_plot[ch_i, :] - np.mean(sig_to_plot[ch_i, idx])

    zmin = np.min(sig_to_plot)
    zmax = np.max(sig_to_plot)
    ax_max = np.max([np.abs(zmin), np.abs(zmax)])

    n_x = int(channel_df.image_x.max() + 1)
    n_y = int(channel_df.image_y.max() + 1)

    print(f'n frames: {sig_to_plot.shape[1]}')

    frame_paths = []
    data = np.zeros((n_x, n_y))

    y_plot = sig_to_plot[int(channel_df.loc['pri_6'].sys_chan_idx), :]
    ymin = np.min(y_plot)
    ymax = np.max(y_plot)
    dy = ymax - ymin
    ymin -= 0.1 * dy
    ymax += 0.1 * dy

    for frame_nr in range(sig_to_plot.shape[1]):

        if not frame_nr % 3 == 0:
            continue

        print(frame_nr)

        sname = figure_savedir / f'animation_{rec_nr}_{burst_to_plot}' /  f'fig_{rec_nr}_{frame_nr}'
        frame_paths.append(sname.as_posix())

        data[:] = np.nan

        for i, r in channel_df.iterrows():
            data[int(r.image_x), int(r.image_y)] = sig_to_plot[int(r.sys_chan_idx), frame_nr]

        fig = utils.make_figure(
            width=1, height=1,
            x_domains={1: [[0.1, 0.3], [0.5, 0.9]]},
            y_domains={1: [[0.1, 0.9], [0.1, 0.9]]},
            equal_width_height='y',
            equal_width_height_axes=[[1, 1]],
            subplot_titles={1: [f'{time_to_plot[int(frame_nr)]:.0f}', '']}
        )

        fig.add_heatmap(
            z=data,
            zmin=-ax_max,
            zmax=ax_max,
            colorscale='RdBu_r',
            showscale=False,
            row=1, col=1
        )

        fig.add_scatter(
            x=time_to_plot, y=y_plot,
            row=1, col=2,
            mode='lines', line=dict(color='black', width=1),
            showlegend=False,
        )
        fig.add_scatter(
            x=[time_to_plot[frame_nr], time_to_plot[frame_nr]],
            y=[ymin, ymax],
            mode='lines', line=dict(color='red', width=1),
            row=1, col=2,
            showlegend=False,
        )
        fig.update_xaxes(
            tickvals=np.arange(-800, 800, 100),
            title_text='ms', row=1, col=2,
        )
        fig.update_yaxes(
            range=[ymin, ymax],
            row=1, col=2,
        )

        utils.save_fig(fig, sname, display=False)


    print(frame_paths[0])
    images = [Image.open(fp + '.png') for fp in frame_paths]

    output_filename = (figure_savedir / 'animation.avi').as_posix()
    with imageio.get_writer(output_filename, fps=60) as writer:
        for img in images:

            writer.append_data(imageio.v3.imread(img.filename))

    print(f"Animation saved as {output_filename}")


if __name__ == '__main__':
    main()