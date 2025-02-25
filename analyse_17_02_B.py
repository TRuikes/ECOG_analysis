import json
import pandas as pd
from pathlib import Path
import allego_file_reader as afr
import utils
import numpy as np
from project_colors import ProjectColors
from scipy.signal import butter, iirnotch, filtfilt, lfilter, sosfiltfilt, tf2sos

data_dir = Path(r'C:\axorus\250217_PEV_B')
figure_savedir = Path(r'C:\axorus\250217_B\figures')

t_pre = 100
t_post = 600
sample_frequency = 3000


def read_data_files(data_dir, file_nr):
    # Read datafiles
    p = data_dir.expanduser()
    all_xdat_datasource_names = [Path(elem.stem).stem for elem in list(p.glob('**/*xdat.json'))]

    n_files = len(all_xdat_datasource_names)
    # print(f'detected {n_files} files')
    # for f in all_xdat_datasource_names:
    #     print(f'\t- {f}')

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

    return filename, signals, time_samples, channel_df


def detect_stim_onsets(time_samples, signals, channel_df, ch_name):
    # Get the digital in data
    din_1_data = signals[int(channel_df.loc[ch_name, 'sys_chan_idx']), :].flatten()

    # Detect channel high values
    trigger_high = np.where(din_1_data > 2)[0] / 3
    dt = np.diff(trigger_high)
    print(dt.min(), dt.max())

    burst_onsets_idx = np.concatenate([np.array([0]), np.where(dt > 100)[0] + 1])
    burst_offsets_idx = np.concatenate([np.where(dt > 100)[0], np.array([-1])])

    burst_df = pd.DataFrame()

    for b_i, (ui, di) in enumerate(zip(burst_onsets_idx, burst_offsets_idx)):
        burst_df.at[b_i, 'burst_onset'] = trigger_high[ui]
        burst_df.at[b_i, 'burst_offset'] = trigger_high[di]
        burst_df.at[b_i, 'burst_duration_calculated'] = trigger_high[di] - trigger_high[ui]

    burst_df['burst_duration_calculated'] = burst_df['burst_duration_calculated'].round(-1)
    return burst_df


def get_veps(time_samples, signals, burst_df):

    tick = 0
    trial_nr = 0
    for i, r in burst_df.iterrows():
        if tick == 40:
            tick = 0
            trial_nr += 1
        burst_df.at[i, 'tick'] = tick
        burst_df.at[i, 'trial_nr'] = trial_nr
        tick += 1

    veps_per_burst = {}
    for tnr, tnr_df in burst_df.groupby('trial_nr'):

        n_channels = signals.shape[0]
        n_bursts = tnr_df.shape[0]
        n_samples = int((t_pre + t_post) * (sample_frequency/1000))
        veps_burst = np.zeros((n_bursts, n_channels, n_samples))

        tick = 0
        for i, burst_info in tnr_df.iterrows():
            burst_onset = burst_info.burst_onset
            i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
            idx = np.arange(i0, i0 + n_samples)

            if idx[-1] > signals.shape[1] - 1:
                # veps = veps[:i, :, :]
                break

            veps_burst[tick, :, :] = signals[:, idx]
            tick += 1

        veps_per_burst[tnr] = veps_burst

    return veps_per_burst


def plot_single_channel_single_condition(
        recording_df, all_veps, channel_info
):
    sequence_df = recording_df.query('seq_name == "power_sequence" and power == "5"')

    cmap = ProjectColors()

    fig = utils.make_figure(
        width=1, height=1,
        x_domains={1: [[0.1, 0.9]]},
        y_domains={1: [[0.1, 0.9]]},
    )


    # sequence_df.sort_values('power', inplace=True)


    data_trial = all_veps[sequence_df.index.values[0]][:, int(channel_info.sys_chan_idx), :]

    ymin = None
    ymax = None

    for burst_i in range(data_trial.shape[0]):


        x = np.arange(-t_pre, t_post, 1 / 3)
        y = data_trial[burst_i, :]

        idx = np.where((x >= -50) & (x < 0))[0]
        y = y - np.mean(y[idx])

        power = int(5)
        clr = cmap.led_amplitude_discrete(power)
        name = f'A: {power}'

        fig.add_scatter(x=x, y=y, mode='lines', line=dict(color=clr, width=0.2),
                        showlegend=False, name=name)

    fig.add_scatter(x=[0, 0], y=[-3000, 3000],
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
        range=[-1000, 1000],
        tickvals=np.arange(-2000, 2000, 200),
        title_text='Amplitude [mV?]'
    )

    utils.save_fig(fig, figure_savedir / 'power_sequence' /
                   'per_channel_per_condition' / channel_info.name,
                   display=False,)




def plot_sequence_single_channel(rec_nr, recording_df, all_veps,
                                 channel_info):
    sequence_df = recording_df.query(f'rec_nr == "{rec_nr}"')

    cmap = ProjectColors()

    fig = utils.make_figure(
        width=1, height=1,
        x_domains={1: [[0.1, 0.9]]},
        y_domains={1: [[0.1, 0.9]]},
    )

    ymin = None
    ymax = None

    # sequence_df.sort_values('power', inplace=True)

    for bd in all_veps[sequence_df.iloc[0].name].keys():

        data_trial = all_veps[sequence_df.iloc[0].name][bd][:, int(channel_info.sys_chan_idx), :]
        mean_data_trial = np.mean(data_trial, axis=0)
        y0 = np.min(mean_data_trial)
        y1 = np.max(mean_data_trial)
        dy = y1 - y0
        ym = np.min(mean_data_trial) - 0.05 * dy
        ymm = np.max(mean_data_trial) + 0.05 * dy

        if ymin is None or ym < ymin:
            ymin = ym
        if ymax is None or ymm > ymax:
            ymax = ymm

        x = np.arange(-t_pre, t_post, 1 / 3)
        y = np.mean(data_trial, axis=0)
        y_se = np.std(data_trial, axis=0) / np.sqrt(data_trial.shape[0])

        alpha = 0.2
        clr = cmap.increasing_value(bd, min=0, max=11)
        clr_fill =cmap.increasing_value(bd, alpha, min=0, max=11)
        name = f'BD: {bd:.0f}'


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

    utils.save_fig(fig, figure_savedir / 'burst_duration' /
                   f'{sequence_df.iloc[0].rec_nr}_{sequence_df.iloc[0].seq_name}' / 'per_channel' / channel_info.name,
                   display=False,)


def plot_din_signal(time_samples, signals, channel_df, burst_df):
    # t0 = 15 * 1e3
    # t1 = 17 * 1e3

    # t_pre = 100
    # t_post = 1500

    n_samples = (t_pre + t_post) * 3


    for i, r in burst_df.iterrows():

        burst_onset = r.burst_onset
        i0 = np.where(time_samples >= burst_onset - t_pre)[0][0]
        idx = np.arange(i0, i0 + n_samples)

        # i0 = np.where(time_samples >= r.burst_onset - t_pre)[0][0]
        # i1 = np.where(time_samples < r.burst_onset + t_post)[0][-1]

        x_plot = time_samples[idx]
        y_plot = signals[int(channel_df.loc['aux_1', 'sys_chan_idx']), :].flatten()[idx]

        fig = utils.simple_fig()
        fig.add_scatter(x=x_plot, y=y_plot,
                        mode='lines', line=dict(color='black'),
                        showlegend=False)

        savename = figure_savedir / 'preprocessing' / 'digital_in' / f'burst_{i}'
        utils.save_fig(fig, savename, display=False)


def main():
    sos = butter(4, (0.5, 500), btype='bandpass', analog=False, fs=30000,
                 output='sos')

    # b, a = iirnotch(50, 25, 30000)
    sos_notch = butter(4, [48, 52], btype='bandstop', fs=3000, output='sos')
    # sos_notch = tf2sos(b, a)

    p = data_dir.expanduser()
    all_xdat_datasource_names = [Path(elem.stem).stem for elem in list(p.glob('**/*xdat.json'))]

    recording_df = pd.DataFrame()
    for rec_i, rname in enumerate(all_xdat_datasource_names):

        if 'allego' in rname or 'baseline' in rname:
            continue

        _, rec_nr, _, recname, _, _, _  = rname.split('_')

        recording_df.at[rec_i, 'rec_nr'] = rec_nr
        recording_df.at[rec_i, 'rec_name'] = recname

        if 'burst' in recname:
            seq_name = 'burst_duration'
        else:
            seq_name = 'duty_cycle'

        recording_df.at[rec_i, 'seq_name'] = seq_name

    _, _, _, channel_df = read_data_files(data_dir, 0)


    # all_veps = {}
    #
    # for rec_i, r in recording_df.iterrows():
    #     filename, signals, time_samples, channel_df = read_data_files(data_dir, rec_i)
    #     burst_df = detect_stim_onsets(time_samples, signals, channel_df, 'aux_1')
    #
    #
    # #     # plot_din_signal(time_samples, signals, channel_df, burst_df)
    # #
    #     signals_filt = np.zeros_like(signals)
    #     for i in range(signals_filt.shape[0]):
    #         r = sosfiltfilt(sos, signals[i, :])
    #         signals_filt[i, :] = sosfiltfilt(sos_notch, r)
    #
    #     # res = get_veps(time_samples, signals, burst_df)
    #     res = get_veps(time_samples, signals_filt, burst_df)
    #     all_veps[rec_i] = res
    #
    #
    # utils.save_obj(all_veps, figure_savedir / 'all_veps.pkl')

    all_veps = utils.load_obj(figure_savedir / 'all_veps.pkl')

    joblist = []
    joblist2 = []

    for i, r in channel_df.iterrows():
        # plot_single_channel_single_condition(recording_df, all_veps, r)
        # joblist.append([2, recording_df, all_veps, r])
        # joblist.append([3, recording_df, all_veps, r])
        # joblist.append([4, recording_df, all_veps, r])

        # joblist2.append([recording_df, all_veps, r])
        plot_sequence_single_channel(4, recording_df, all_veps, r)
    #
    utils.run_job(plot_sequence_single_channel, 4, joblist)
    # utils.run_job(plot_single_channel_single_condition, 4, joblist2)
    # plot_sequence_single_channel('power_sequence', recording_df, all_veps,
    #                              channel_df.iloc[0])

if __name__ == '__main__':
    main()
