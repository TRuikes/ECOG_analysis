import json
import pandas as pd
from pathlib import Path
import allego_file_reader as afr
import utils
import numpy as np
from project_colors import ProjectColors
from scipy.signal import butter, iirnotch, filtfilt, lfilter, sosfiltfilt

data_dir = Path(r'E:\250120-PEV_test')
figure_savedir = Path(r'E:\250120-PEV_test\figures')


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


def plot_din_signal(time_samples, signals, channel_df, burst_df):
    # t0 = 15 * 1e3
    # t1 = 17 * 1e3

    t_pre = 100
    t_post = 1500

    for i, r in burst_df.iterrows():
        i0 = np.where(time_samples >= r.burst_onset - t_pre)[0][0]
        i1 = np.where(time_samples < r.burst_onset + t_post)[0][-1]

        x_plot = time_samples[i0:i1]
        y_plot = signals[int(channel_df.loc['din_1', 'sys_chan_idx']), :].flatten()[i0:i1]

        fig = utils.simple_fig()
        fig.add_scatter(x=x_plot, y=y_plot,
                        mode='lines', line=dict(color='black'),
                        showlegend=False)

        savename = figure_savedir / 'preprocessing' / 'digital_in' / f'burst_{i}'
        utils.save_fig(fig, savename, display=False)


def plot_stimulus_triggered_response(time_samples, signals,
                                     channel_df, burst_df, ch_name):

    t_pre = 100
    t_post = 400

    fig = utils.simple_fig()

    for i, r in burst_df.iterrows():
        i0 = np.where(time_samples >= r.burst_onset - t_pre)[0][0]
        i1 = np.where(time_samples < r.burst_onset + t_post)[0][-1]

        x_plot = time_samples[i0:i1] - r.burst_onset

        y_plot = signals[int(channel_df.loc[ch_name, 'sys_chan_idx']), :].flatten()[i0:i1]
        y_plot = y_plot - np.mean(y_plot[:10])
        yi = np.min(y_plot)
        ya = np.max(y_plot)
        dy = ya - yi
        # ymin = yi - 0.1 * dy
        # ymax = ya + 0.1 * dy

        # y_din = signals[int(channel_df.loc['din_1', 'sys_chan_idx']), :].flatten()[i0:i1]
        # y_din = y_din - np.min(y_din)
        # y_din = y_din / np.max(y_din)
        # y_din = (y_din * (ymax - ymin)) + ymin

        fig.add_scatter(x=x_plot, y=y_plot,
                        mode='lines', line=dict(color='black',
                                                width=0.5),
                        showlegend=False)

    # fig.add_scatter(x=x_plot, y=y_din,
    #                 mode='lines', line=dict(color='red'),
    #                 showlegend=False)

    fig.update_xaxes(
        tickvals=np.arange(-100, 500, 100),
        title_text='time [ms]'
    )

    savename = figure_savedir / 'preprocessing' / 'stimulus_triggered_response' / ch_name / f'average'
    utils.save_fig(fig, savename, display=False)


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


def plot_sequence_all_channels(sequence_name, burst_df, channel_df, time_samples,
                  signals):

    sequence_df = burst_df.query('stim_sequence == @sequence_name')

    assert sequence_df.shape[0] > 0

    all_veps = {}
    t_pre = 100
    t_post = 300
    for trial, btdf in sequence_df.groupby('train_nr'):

        n_samples = (t_pre + t_post) * 3
        n_bursts = btdf.shape[0]
        n_signals = channel_df.shape[0]

        vep = np.zeros((n_bursts, n_signals, n_samples))

        btick = 0
        for burst_i, burst_info in btdf.iterrows():
            burst_onset = burst_info.burst_onset
            i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
            idx = np.arange(i0, i0 + n_samples)

            vep[btick, :, :] = signals[:, idx]
            btick += 1

        if 'burst_duration' in sequence_name:
            name = btdf.iloc[0].burst_duration
        elif 'power' in sequence_name:
            name = btdf.iloc[0].amplitude
        elif 'frequency' in sequence_name:
            name = btdf.iloc[0].frequency
        else:
            raise ValueError('')

        all_veps[name] = vep

    # Setup figure
    n_rows = 7
    n_cols = 7

    x_offset = 0.05
    y_offset = 0.05
    x_spacing = 0.05
    y_spacing = 0.05

    x_width = (1 - 2 * x_offset - (n_cols * x_spacing)) / n_cols
    y_height = (1 - 2 * y_offset - (n_rows * y_spacing)) / n_rows

    x_domains = {}
    y_domains = {}

    for row_i in range(n_rows):
        x_domains[row_i + 1] = []
        y_domains[row_i + 1] = []

        for col_j in range(n_cols):
            x0 = x_offset + col_j * (x_spacing + x_width)
            x_domains[row_i + 1].append([x0, x0 + x_width])

            y0 = 1 - y_offset - row_i * (y_spacing + y_height) - y_height
            y_domains[row_i + 1].append([y0, y0 + y_height])

    # channel_df_cut = channel_df.query('site_ctr_x != 0')
    probe_x_vals = np.sort(channel_df.site_ctr_x.unique())
    probe_y_vals = np.sort(channel_df.site_ctr_y.unique())
    for i, r in channel_df.iterrows():
        row_i = np.where(probe_x_vals == r.site_ctr_x)[0][0] + 1
        col_j = np.where(probe_y_vals == r.site_ctr_y)[0][0] + 1
        channel_df.at[i, 'plot_row'] = int(row_i)
        channel_df.at[i, 'plot_col'] = int(col_j)

    cmap = ProjectColors()

    fig = utils.make_figure(
        width=1.5, height=1.5,
        x_domains=x_domains,
        y_domains=y_domains,
    )

    ymin = None
    ymax = None
    for bd in all_veps.keys():

        data_burst = all_veps[bd]
        mean_data_burst = np.mean(data_burst, axis=0)
        y0 = np.min(mean_data_burst)
        y1 = np.max(mean_data_burst)
        dy = y1 - y0
        ym = np.min(mean_data_burst) - 0.05 * dy
        ymm = np.max(mean_data_burst) + 0.05 * dy

        if ymin is None or ym < ymin:
            ymin = ym
        if ymax is None or ymm > ymax:
            ymax = ymm

        for i, r in channel_df.iterrows():
            pos = dict(row=int(r.plot_row), col=int(r.plot_col))
            # print(r.sys_chan_idx)

            data = all_veps[bd][:, int(r.sys_chan_idx), :]
            x = np.arange(-t_pre, t_post, 1 / 3)
            y = np.mean(data, axis=0)

            if 'frequency' in sequence_name:
                clr = cmap.burst_frequency(bd, 1)
            elif 'burst_duration' in sequence_name:
                clr = cmap.burst_duration(bd, 1)
            elif 'power' in sequence_name:
                clr = cmap.led_amplitude(bd, 1)
            else:
                raise ValueError('')

            fig.add_scatter(x=x, y=y, mode='lines', line=dict(color=clr, width=0.6),
                            showlegend=False, **pos)

            fig.add_scatter(x=[0, 0], y=[ymin, ymax],
                            mode='lines', line=dict(color='red', width=1), showlegend=False,
                            **pos, )

            fig.update_xaxes(
                title_text=f'{i}',
                **pos,
            )

    for row_i in range(n_rows):
        for col_j in range(n_cols):
            fig.update_yaxes(
                range=[ymin, ymax],
                row=row_i+1, col=col_j+1,
            )

    utils.save_fig(fig, figure_savedir / burst_df.iloc[0].recording_name / sequence_name)


def plot_probe(channel_df):
    fig = utils.simple_fig(
        equal_width_height='y',
        width=1, height=1
    )
    fig.add_scatter(
        x=-channel_df.site_ctr_x,
        y=-channel_df.site_ctr_y,
        mode='markers',
        marker=dict(color='white', size=20, line=dict(color='black', width=1)),
    )
    for i, r in channel_df.iterrows():
        fig.add_annotation(
            x=-r.site_ctr_x,
            y=-r.site_ctr_y,
            text=f'{int(i.split("_")[1])+1}',
            showarrow=False,
        )
    utils.save_fig(fig, figure_savedir / 'probe')


def get_recording_data(data_dir, rec_nr):
    # Read stimulation sequences
    file_path = data_dir / 'stimulation_sequences.csv'
    stim_seqs = pd.read_csv(file_path, index_col=0, header=0)
    stim_seqs_dict = {}
    for i, r in stim_seqs.iterrows():
        stim_seqs_dict[i] = [s.strip() for s in r['stimulation sequences'].split(',')]
    rec_names = list(stim_seqs_dict.keys())

    recording_name = rec_names[rec_nr]

    signals, time_samples, channel_df = (read_data_files
                                         (data_dir, rec_nr))
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


def plot_sequence_single_channel(sequence_name, burst_df, time_samples, signals,
                                 channel_info):
    sequence_df = burst_df.query('stim_sequence == @sequence_name')

    assert sequence_df.shape[0] > 0

    all_veps = {}
    t_pre = 100
    t_post = 300
    for trial, btdf in sequence_df.groupby('train_nr'):

        n_samples = (t_pre + t_post) * 3
        n_bursts = btdf.shape[0]

        vep = np.zeros((n_bursts, n_samples))
        btick = 0
        for burst_i, burst_info in btdf.iterrows():
            burst_onset = burst_info.burst_onset
            i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
            idx = np.arange(i0, i0 + n_samples)

            vep[btick, :] = signals[int(channel_info.sys_chan_idx), idx]
            btick += 1

        if 'burst_duration' in sequence_name:
            name = btdf.iloc[0].burst_duration
        elif 'power' in sequence_name:
            name = btdf.iloc[0].amplitude
        elif 'frequency' in sequence_name:
            name = btdf.iloc[0].frequency
        else:
            raise ValueError('')

        all_veps[name] = vep

    cmap = ProjectColors()

    fig = utils.make_figure(
        width=1, height=1,
        x_domains={1: [[0.1, 0.9]]},
        y_domains={1: [[0.1, 0.9]]},
    )

    ymin = None
    ymax = None

    stim_values = list(all_veps.keys())
    stim_values = np.sort(stim_values)

    for bd in stim_values:

        data_burst = all_veps[bd]
        mean_data_burst = np.mean(data_burst, axis=0)
        y0 = np.min(mean_data_burst)
        y1 = np.max(mean_data_burst)
        dy = y1 - y0
        ym = np.min(mean_data_burst) - 0.05 * dy
        ymm = np.max(mean_data_burst) + 0.05 * dy

        if ymin is None or ym < ymin:
            ymin = ym
        if ymax is None or ymm > ymax:
            ymax = ymm

        data = all_veps[bd]
        x = np.arange(-t_pre, t_post, 1 / 3)
        y = np.mean(data, axis=0)
        y_se = np.std(data, axis=0) / np.sqrt(data.shape[0])

        alpha = 0.2
        if 'frequency' in sequence_name:
            clr = cmap.burst_frequency(bd, 1)
            clr_fill = cmap.burst_frequency(bd, alpha)
            name = f'{bd:.0f} Hz'
        elif 'burst_duration' in sequence_name:
            clr = cmap.burst_duration(bd, 1)
            clr_fill = cmap.burst_duration(bd, alpha)
            name = f'{bd:.0f} ms'
        elif 'power' in sequence_name:
            clr = cmap.led_amplitude(bd, 1)
            clr_fill = cmap.led_amplitude(bd, alpha)
            name = f'A: {bd:.0f}'
        else:
            raise ValueError('')

        fig.add_scatter(x=x, y=y-y_se, mode='lines', line=dict(color=clr, width=0),
                        showlegend=False)
        fig.add_scatter(x=x, y=y+y_se, mode='lines', line=dict(color=clr, width=0),
                        fill='tonexty', fillcolor=clr_fill, showlegend=False)

        fig.add_scatter(x=x, y=y, mode='lines', line=dict(color=clr, width=0.6),
                        showlegend=True, name=name)

        fig.add_scatter(x=[0, 0], y=[ymin, ymax],
                        mode='lines', line=dict(color='red', width=1), showlegend=False,
                        )

    fig.update_layout(
        legend=dict(
            xanchor='right',
            x=1,
            yanchor='top',
            y=1,
            entrywidth=0.01,
        )
    )

    fig.update_xaxes(
        tickvals=np.arange(-200, 500, 100),
        title_text='time [ms]',
    )
    fig.update_yaxes(
        range=[ymin, ymax],
        title_text='Amplitude [mV?]'
    )

    utils.save_fig(fig, figure_savedir / burst_df.iloc[0].recording_name /
                   sequence_name / 'per_channel' / channel_info.name,
                   display=False,)


def get_stats(sequence_name, burst_df, channel_df,
              time_samples, signals):

    sequence_df = burst_df.query('stim_sequence == @sequence_name')

    assert sequence_df.shape[0] > 0

    stats = pd.DataFrame()
    t_pre = 100
    t_post = 300
    stats_idx = 0


    for trial, btdf in sequence_df.groupby('train_nr'):
        print(f'\t\t{trial}')

        n_samples = (t_pre + t_post) * 3
        n_channel = signals.shape[0]
        n_bursts = btdf.shape[0]
        amplitudes = np.zeros((n_bursts, n_channel))
        burst_tick = 0

        for burst_i, burst_info in btdf.iterrows():
            burst_onset = burst_info.burst_onset
            i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
            idx = np.arange(i0, i0 + n_samples)
            time = time_samples[idx] - burst_onset
            baseline_idx = np.where((time >= -10) & (time < 0))[0]
            a_base = np.mean(signals[:, idx[baseline_idx]], axis=1)
            stim_idx = np.where(time > 0)[0]
            a_stim = np.max(signals[:, idx[stim_idx]], axis=1)
            amplitudes[burst_tick, :] = a_stim - a_base
            burst_tick += 1

        if 'burst_duration' in sequence_name:
            param_val = btdf.iloc[0].burst_duration
        elif 'power' in sequence_name:
            param_val = btdf.iloc[0].amplitude
        elif 'frequency' in sequence_name:
            param_val = btdf.iloc[0].frequency
        else:
            raise ValueError('')

        for channel_name, channel_info in channel_df.iterrows():
            stats.at[stats_idx, 'recording'] = btdf.iloc[0].recording_name
            stats.at[stats_idx, 'sequence_name'] = sequence_name
            stats.at[stats_idx, 'trial'] = trial
            stats.at[stats_idx, 'channel_name'] = channel_name
            stats.at[stats_idx, 'amplitude'] = np.mean(amplitudes[:, int(channel_info.sys_chan_idx)])
            stats.at[stats_idx, 'amplitude_se'] = np.std(amplitudes[:, int(channel_info.sys_chan_idx)]) / np.sqrt(amplitudes.shape[0])
            stats.at[stats_idx, 'stim_param_val'] = param_val

            stats_idx += 1

    return stats


def plot_stats(stats):
    power_calib_df = read_power_calibration(data_dir)
    rec_names = stats.recording.unique()
    stats['rec_nr'] = stats['recording'].apply(lambda x: np.where(rec_names == x)[0][0])

    for rec_to_plot in range(len(rec_names)):
        for seq_to_plot in stats.sequence_name.unique():

            if 'power' in seq_to_plot:
                x_title = 'Irradiance [mW/cm2]'

            elif 'frequency' in seq_to_plot:
                x_title = 'Frequency [Hz]'

            elif 'duration' in seq_to_plot:
                x_title = 'Duration [ms]'

            df_to_plot = stats.query(f'rec_nr == {rec_to_plot} and sequence_name == "{seq_to_plot}"')

            fig = utils.make_figure(
                width=0.4,
                height=1,
                x_domains={1: [[0.15, 0.95]]},
                y_domains={1: [[0.15, 0.95]]},
            )
            for ch_name, ch_df in df_to_plot.groupby('channel_name'):
                if 'pri' not in ch_name:
                    continue

                x = ch_df.stim_param_val.values

                y = ch_df.amplitude.values
                idx = np.argsort(x)

                if 'frequency' in seq_to_plot or 'duration' in seq_to_plot:
                    xtext = x[idx]
                elif 'power' in seq_to_plot:
                    xtext = [f"{power_calib_df.loc[xx, 'mW/cm2']:.0f}" for xx in x[idx]]

                fig.add_scatter(
                    x=np.arange(y.size),
                    y=y[idx],
                    mode='lines', line=dict(color='black', width=1),
                    showlegend=False,
                )

            fig.update_yaxes(
                tickvals=np.arange(0, 1500, 100),
                title_text='Amplitude'
            )

            fig.update_xaxes(
                title_text=x_title,
                ticktext=xtext,
                tickvals=np.arange(x.size)
            )
            utils.save_fig(fig, figure_savedir / 'stats' / rec_names[rec_to_plot] / seq_to_plot,
                           display=False,)


def read_power_calibration(data_dir):
    file = data_dir / r'power measurement LED 17 Jan 2025.xlsx'
    df = pd.read_excel(file, engine='openpyxl',
                       usecols=['Voltage', 'mW', 'mW/cm2'], index_col='Voltage')
    return df


def plot_individual_trials(sequence_name, burst_df, time_samples, signals,
                                 channel_info):

    sequence_df = burst_df.query('stim_sequence == @sequence_name')

    assert sequence_df.shape[0] > 0

    all_veps = {}
    t_pre = 100
    t_post = 300
    for trial, btdf in sequence_df.groupby('train_nr'):

        n_samples = (t_pre + t_post) * 3
        n_bursts = btdf.shape[0]

        vep = np.zeros((n_bursts, n_samples))
        btick = 0
        for burst_i, burst_info in btdf.iterrows():
            burst_onset = burst_info.burst_onset
            i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
            idx = np.arange(i0, i0 + n_samples)

            vep[btick, :] = signals[int(channel_info.sys_chan_idx), idx]
            btick += 1

        if 'burst_duration' in sequence_name:
            name = btdf.iloc[0].burst_duration
        elif 'power' in sequence_name:
            name = btdf.iloc[0].amplitude
        elif 'frequency' in sequence_name:
            name = btdf.iloc[0].frequency
        else:
            raise ValueError('')

        all_veps[name] = vep

    cmap = ProjectColors()

    ymin = None
    ymax = None

    stim_values = list(all_veps.keys())
    stim_values = np.sort(stim_values)

    for bd in stim_values:
        fig = utils.make_figure(
            width=1, height=1,
            x_domains={1: [[0.1, 0.9]]},
            y_domains={1: [[0.1, 0.9]]},
            subplot_titles={1: [f'{bd}']}
        )

        data_burst = all_veps[bd]
        mean_data_burst = np.mean(data_burst, axis=0)
        y0 = np.min(mean_data_burst)
        y1 = np.max(mean_data_burst)
        dy = y1 - y0
        ym = np.min(mean_data_burst) - 0.05 * dy
        ymm = np.max(mean_data_burst) + 0.05 * dy

        if ymin is None or ym < ymin:
            ymin = ym
        if ymax is None or ymm > ymax:
            ymax = ymm

        data = all_veps[bd]
        x_plot, y_plot = [], []
        xvals = np.arange(-t_pre, t_post, 1/3)
        base_idx = np.where((xvals >= -20) & (xvals < 10))[0]
        for i in range(data.shape[0]):
            x_plot.extend([xvals, None])
            y_plot.extend([data[i, :] - np.mean(data[i, base_idx]), None])

        x_plot = np.hstack(x_plot)
        y_plot = np.hstack(y_plot)
        # x = np.arange(-t_pre, t_post, 1 / 3)
        # y = np.mean(data, axis=0)
        # y_se = np.std(data, axis=0) / np.sqrt(data.shape[0])

        alpha = 0.2
        if 'frequency' in sequence_name:
            clr = cmap.burst_frequency(bd, 1)
            name = f'{bd:.0f} Hz'
        elif 'burst_duration' in sequence_name:
            clr = cmap.burst_duration(bd, 1)
            name = f'{bd:.0f} ms'
        elif 'power' in sequence_name:
            clr = cmap.led_amplitude(bd, 1)
            name = f'A: {bd:.0f}'
        else:
            raise ValueError('')

        fig.add_scatter(x=x_plot, y=y_plot, mode='lines', line=dict(color=clr, width=0.4),
                        showlegend=False, name=name)

        fig.add_scatter(x=[0, 0], y=[ymin, ymax],
                        mode='lines', line=dict(color='red', width=1), showlegend=False,
                        )

        fig.update_xaxes(
            tickvals=np.arange(-200, 500, 100),
            title_text='time [ms]',
        )
        fig.update_yaxes(
            range=[-800, 800],
            tickvals=np.arange(-1000, 1000, 200),
            title_text='Amplitude [mV?]'
        )

        utils.save_fig(fig, figure_savedir / burst_df.iloc[0].recording_name /
                       sequence_name / 'per_channel' / channel_info.name / f'{bd:.2f}',
                       display=False,)


def plot_amplitude_per_burst_single_trial(sequence_name, burst_df, time_samples,
                                          signals, channel_info):

    sequence_df = burst_df.query('stim_sequence == @sequence_name')

    assert sequence_df.shape[0] > 0

    t_pre = 10
    t_post = 100
    for trial, btdf in sequence_df.groupby('train_nr'):
        n_samples = (t_pre + t_post) * 3
        n_bursts = btdf.shape[0]

        burst_nr = np.zeros(n_bursts)
        burst_amplitude = np.zeros(n_bursts)

        btick = 0
        for burst_i, burst_info in btdf.iterrows():
            burst_onset = burst_info.burst_onset
            i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
            idx = np.arange(i0, i0 + n_samples)
            time = time_samples[idx] - burst_onset
            baseline_idx = np.where((time >= -10) & (time < 0))[0]
            a_base = np.mean(signals[int(channel_info.sys_chan_idx), idx[baseline_idx]])
            stim_idx = np.where(time > 0)[0]
            a_stim = np.max(signals[int(channel_info.sys_chan_idx), idx[stim_idx]])

            burst_nr[btick] = btick
            burst_amplitude[btick] = a_stim - a_base
            btick += 1

        burst_amplitude[burst_amplitude > 800] = np.nan

        fig = utils.make_figure(
            width=0.4,
            height=1,
            x_domains={1: [[0.15, 0.95]]},
            y_domains={1: [[0.15, 0.95]]},
        )
        fig.add_scatter(
            x=burst_nr,
            y=burst_amplitude,
            mode='lines', line=dict(color='black', width=1),
            showlegend=False,
        )

        fig.update_yaxes(
            tickvals=np.arange(0, 1500, 100),
            title_text='Amplitude'
        )

        if 'burst_duration' in sequence_name:
            name = btdf.iloc[0].burst_duration
        elif 'power' in sequence_name:
            name = btdf.iloc[0].amplitude
        elif 'frequency' in sequence_name:
            name = btdf.iloc[0].frequency
        else:
            raise ValueError('')

        utils.save_fig(fig, figure_savedir / 'amplitude_per_burst' / burst_df.iloc[0].recording_name /
                       sequence_name / 'per_channel' / channel_info.name / f'{name:.2f}',
                       display=False, )


def main():
    # Create a bandpass filter
    sos = butter(10, (5, 500), btype='bandpass', analog=False, fs=3000,
                 output='sos')

    all_stats = []

    # Load data per recording
    for rec_nr in range(4):

        # Extract recording data
        channel_df, burst_df, time_samples, signals = get_recording_data(data_dir, rec_nr)

        # Filter the ECOG signals
        signals_filt = np.zeros_like(signals)
        signals_filt2 = np.zeros_like(signals)
        for i in range(signals_filt.shape[0]):
            signals_filt[i, :] = sosfiltfilt(sos, signals[i, :])
            signals_filt2[i, :] = sosfiltfilt(sos2, signals_filt[i, :])

        # For each stimulation sequence create a joblist
        for sequence_name in burst_df.stim_sequence.unique():
            print(sequence_name)
            if 'power' not in sequence_name:
                continue
            # plot_sequence_all_channels(sequence_name, burst_df, channel_df, time_samples, signals)

            joblist = []
            joblist2 = []
            for i, r in channel_df.iterrows():
                # plot_sequence_single_channel(sequence_name, burst_df, time_samples, signals, r)
                joblist.append([sequence_name, burst_df, time_samples, signals_filt, r])
                joblist2.append([sequence_name, burst_df, time_samples, signals_filt, r])

            test_job = ['led_power', burst_df, time_samples, signals_filt, channel_df.loc['pri_10']]

            # plot_sequence_single_channel(*test_job)
            plot_individual_trials(*test_job)
            plot_amplitude_per_burst_single_trial(*test_job)

            test_job = ['led_power', burst_df, time_samples, signals_filt2, channel_df.loc['pri_10']]
            plot_sequence_single_channel(*test_job)


            # utils.run_job(plot_sequence_single_channel, 4, joblist)
            # utils.run_job(plot_individual_trials, 4, joblist2)

            # run job that generates amplitude per burst nr figs
            # utils.run_job(plot_amplitude_per_burst_single_trial, 8, joblist)

            # print(rec_nr, sequence_name)
            # stats = get_stats(sequence_name, burst_df, channel_df, time_samples, signals_filt)
            # all_stats.append(stats)

        break

    # all_stats = pd.concat(all_stats)
    # plot_stats(all_stats)


if __name__ == '__main__':
    main()