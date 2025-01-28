import numpy as np
from pathlib import Path
import pickle

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.io import write_image
import os
import multiprocessing as mp
import plotly
import sys
# import h5py
# from pptx import Presentation
# from pptx.util import Inches
from tqdm import tqdm

FIG_SCALE = 5

def save_obj(obj, savename: Path):
    """
        Generic function to save an obj with pickle protocol

    Input :
        - obj (python var) : object to be saved in binary format
        - name (str) : path to where the obj shoud be saved

    Possible mistakes :
        - Permissions denied, restart notebook from an admin shell
        - Folders aren't callable, change your folders
    """

    if not savename.parent.is_dir():
        savename.parent.mkdir(parents=True)

    savename = savename.as_posix()
    assert '.pkl' in savename

    with open(savename, 'wb') as f:
        pickle.dump(obj, f, 4)


def load_obj(loadname: Path):
    # assert loadname.is_file(), f'file not found: {loadname}'
    if not loadname.is_file():
        return None
    with open(loadname.as_posix(), 'rb') as f:
        data = pickle.load(f)
    return data


# def store_nested_dict(filename, data):
#     with h5py.File(filename.as_posix(), 'w') as f:
#         _store_nested_dict(f, data)


def _store_nested_dict(hdf5_group, dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            # If the value is a dictionary, create a group and recurse
            subgroup = hdf5_group.create_group(key)
            _store_nested_dict(subgroup, value)
        else:
            # If the value is a NumPy array, create a dataset
            hdf5_group.create_dataset(key, data=value)

# def load_nested_dict(filename):
#     with h5py.File(filename.as_posix(), 'r') as f:
#         return _load_nested_dict(f)
#
# def _load_nested_dict(hdf5_group):
#     result = {}
#     for key, item in hdf5_group.items():
#         if isinstance(item, h5py.Group):
#             # If the item is a group, recurse to load its contents
#             result[key] = _load_nested_dict(item)
#         else:
#             # If the item is a dataset, load it as a NumPy array
#             result[key] = np.array(item)
#     return result



def simple_fig(n_cols=1, n_rows=1, width=1, x_offset=None, height=0.5, equal_width_height=None, equal_width_height_axes='all', subplot_titles=None,
               **kwargs):
    if x_offset is None:
        x_offset = 0.1 if width > 0.5 else 0.2

    x_spacing = 0.05
    y_offset = 0.1
    y_spacing = 0.08
    xwidth = (1 - (2*x_offset) - (n_cols-1) * x_spacing) / n_cols
    yheight = (1 - (2*y_offset) - (n_rows-1) * y_spacing) / n_rows

    x_domains = {}
    y_domains = {}
    st = {}

    for row_i in range(n_rows):
        x_row = []
        st_row = []
        for col_i in range(n_cols):
            x0 = x_offset + (x_spacing+xwidth) * col_i
            x1 = x0+xwidth
            x_row.append([x0, x1])
            st_row.append('')
        x_domains[row_i+1] = x_row
        st[row_i+1] = st_row
        y1 = 1 - y_offset - (y_spacing+yheight) * row_i
        y0 = y1 - yheight
        y_domains[row_i+1] = [[y0, y1] for _ in range(n_cols)]

    if subplot_titles is None:
        subplot_titles = st

    fig = make_figure(
        width=width,
        height=height,
        x_domains=x_domains,
        y_domains=y_domains,
        equal_width_height=equal_width_height,
        subplot_titles=subplot_titles,
        equal_width_height_axes=equal_width_height_axes,
        **kwargs,
    )

    return fig

def make_figure(width, height, x_domains, y_domains, **kwargs) -> go.Figure:
    for i, k in kwargs.items():
        assert i in ['subplot_titles', 'specs', 'equal_width_height', 'equal_width_height_axes', 'bg_color',
                     'xticks', 'yticks'], f'{i}'

    # Check dimensions of x_domains and y_domains
    for row in x_domains.keys():
        assert row in y_domains.keys()
        n_x = len(x_domains[row])
        n_y = len(y_domains[row])
        assert n_x == n_y, f'{n_x} - {n_y}'
        for i in x_domains[row]:
            assert len(i) == 2
        for i in y_domains[row]:
            assert len(i) == 2

    font_famliy = 'calibri'
    figwidth_pxl = 498 * width
    figheight_pxl = 842 * 0.25 * height
    if 'equal_width_height' in kwargs.keys():
        eqw = kwargs['equal_width_height']
        assert eqw in [None, 'x', 'y', 'shortest']
    else:
        eqw = None

    if 'equal_width_height_axes' not in kwargs.keys():
        eqw_axes = 'all'
    else:
        eqw_axes = kwargs['equal_width_height_axes']

    specs_f = dict()
    if 'shared_xaxes' in kwargs.keys():
        specs_f['shared_xaxes'] = kwargs['shared_xaxes']
    # if 'subplot_titles' in kwargs.keys():
    #     specs_f['subplot_titles'] = kwargs['subplot_titles']
    if 'specs' in kwargs.keys():
        specs_f['specs'] = kwargs['specs']

    if 'bg_color' in kwargs.keys():
        plot_bgcolor = kwargs['bg_color']
        plot_papercolor = kwargs['bg_color']
        if plot_bgcolor == 'black':
            font_color = 'white'
        else:
            font_color = 'white'
    else:
        plot_bgcolor = 'white'
        plot_papercolor = 'white'
        font_color = 'black'

    if 'xticks' in kwargs.keys():
        xticks = kwargs['xticks']
    else:
        xticks = False

    if 'yticks' in kwargs.keys():
        yticks = kwargs['yticks']
    else:
        yticks = False

    # Detect the nr of cols
    n_cols = 0
    for row, specs in x_domains.items():
        n_cols = np.max([n_cols, len(specs)])

    n_cols = int(n_cols)
    n_rows = len(x_domains.keys())

    fig = make_subplots(rows=n_rows, cols=n_cols, **specs_f)

    fig.update_layout(
        # Figure dimensions
        autosize=False,
        width=figwidth_pxl,
        height=figheight_pxl,
        margin=dict(l=0, t=0, b=0, r=0),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=plot_papercolor,
        legend=dict(
            borderwidth=0.5,
            bordercolor=font_color,
            font=dict(
                family=font_famliy,
                size=6,
                color=font_color
            )
        ),
    )

    for row_i in range(n_rows):
        n_cols = len(x_domains[row_i+1])

        for col_i in range(n_cols):

            if 'subplot_titles' in kwargs.keys():
                # print(kwargs['subplot_titles'], ax_tick-1)
                fig.add_annotation(
                    x=x_domains[row_i+1][col_i][0]-0.05,
                    y=y_domains[row_i+1][col_i][1]+0.01,
                    text=kwargs['subplot_titles'][row_i+1][col_i],
                    font=dict(size=12, family=font_famliy, color=font_color,),
                    showarrow=False,
                    xanchor='left', yanchor='bottom',
                    xref='paper', yref='paper',
                )

            x0, x1 = x_domains[row_i+1][col_i]
            y0, y1 = y_domains[row_i+1][col_i]
            if eqw_axes == 'all' or [row_i + 1, col_i + 1] in eqw_axes:
                if eqw == 'x':
                    n_px_x = (x1-x0) * figwidth_pxl
                    dy = n_px_x / figheight_pxl
                    y1 = y0 + dy

                elif eqw == 'y':
                    n_px_y = (y1-y0) * figheight_pxl
                    dx = n_px_y / figwidth_pxl
                    x1 = x0 + dx

            fig.update_xaxes(
                row=row_i+1,
                col=col_i+1,
                # automargin=False,
                domain=[x0, x1],

                # Xaxis ticks
                ticks='inside',
                tickmode='array',
                tickvals=[],
                tickwidth=1,
                ticklen=2 if xticks else 0,
                tickangle=0,
                tickcolor='black',
                tickfont=dict(
                    size=8,
                    family=font_famliy,
                    color=font_color,
                ),

                # Xaxis line
                linewidth=0.5,
                linecolor=font_color,
                showline=True,

                # Title properties
                title=dict(
                    standoff=5,
                    font=dict(
                        size=10,
                        family=font_famliy,
                        color=font_color,
                    )
                )

            )
            fig.update_yaxes(
                row=row_i+1,
                col=col_i+1,
                automargin=False,
                domain=[y0, y1],

                # Xaxis ticks
                ticks='inside',
                tickmode='array',
                tickvals=[],
                tickwidth=1,
                ticklen=2 if yticks else 0,
                tickcolor='black',
                tickfont=dict(
                    size=8,
                    family=font_famliy,
                    color=font_color,
                ),

                # Xaxis line
                linewidth=0.5,
                linecolor=font_color,
                showline=True,

                # Title properties
                title=dict(
                    standoff=0,
                    font=dict(
                        size=10,
                        family=font_famliy,
                        color=font_color,
                    )
                )
            )

    return fig


def make_panel(width, height, **kwargs):
    for i, k in kwargs.items():
        assert i in ['tickfont']

    font_famliy = 'calibri'
    figwidth_pxl = 498 * width
    figheight_pxl = 842 * 0.25 * height

    fig = make_subplots(rows=1, cols=1)

    fig.update_layout(
        # Figure dimensions
        autosize=False,
        width=figwidth_pxl,
        height=figheight_pxl,
        margin=dict(l=0, t=0, b=0, r=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            borderwidth=0.5,
            bordercolor='black',
            font=dict(
                family=font_famliy,
                size=6,
                color='black',
            )
        ),
    )

    fig.update_xaxes(
        row=1,
        col=1,
        # automargin=False,
        domain=[0.1, 0.9],

        # Xaxis ticks
        tickmode='array',
        tickvals=[],
        tickwidth=0.5,
        ticklen=0.1,
        tickangle=0,
        # tickcolor='crimson',
        tickfont=dict(
            size=8,
            family=font_famliy,
            color='black',
        ),

        # Xaxis line
        linewidth=0.5,
        linecolor='black',
        showline=True,

        # Title properties
        title=dict(
            standoff=5,
            font=dict(
                size=10,
                family=font_famliy,
                color='black',
            )
        )

    )

    fig.update_yaxes(
        row=1,
        col=1,
        automargin=False,
        domain=[0.1, 0.9],

        # Xaxis ticks
        tickmode='array',
        tickvals=[],
        tickwidth=0.5,
        ticklen=0.1,
        # tickcolor='crimson',
        tickfont=dict(
            size=8,
            family=font_famliy,
            color='black',
        ),

        # Xaxis line
        linewidth=0.5,
        linecolor='black',
        showline=True,

        # Title properties
        title=dict(
            standoff=0,
            font=dict(
                size=10,
                family=font_famliy,
                color='black',
            )
        )
    )
    return fig



def save_fig(fig: go.Figure, savename: Path, formats=None, scale=None, verbose=True,
             display=True):
    if scale is None:
        scale = FIG_SCALE
    if formats is None:
        formats = ['png']

    if not savename.parent.is_dir():
        savename.parent.mkdir(parents=True)

    for f in formats:
        file = savename.parent / (savename.name + f'.{f}')

        if f == 'html':
            fig.write_html(file)
        else:
            write_image(
                fig=fig,
                file=file,
                format=f,
                scale=scale if f == 'png' else 1,  # not used when exporting to svg
                engine='kaleido',
                # width=300,
                # height=180,
            )

        if verbose:
            print(f'saved: {file}')

    if display and 'png' in formats or 'html' in formats:
        print('displaying figure')
        if 'png' in formats:
            file = savename.parent / (savename.name + f'.png')
        elif 'html' in formats:
            file = savename.parent / (savename.name + f'.html')
        os.system(f'start {file}')


def run_job(job_fn, n_proceses, joblist):
    """
    Run a function in parallel

    :param job_fn: function to run the job on
    :param n_proceses: nr of parallel processes
    :param joblist: list of lists with input values for the function
    :return:
    """
    gettrace = getattr(sys, 'gettrace', None)

    if not gettrace():
        pool = mp.Pool(processes=n_proceses)
        for job in joblist:
            pool.apply_async(job_fn, args=job)

        # Call pool.close() and pool.join(), otherwise the main script will not wait for apply_async to
        # finish and kill all workers
        pool.close()
        pool.join()
    else:
        print('DEBUGGING')
        for job in joblist:
            job_fn(*job)


def interp_color(nsteps, step_nr, scalename, alpha, inverted=False):
    cols = getattr(getattr(plotly.colors, scalename[0]), scalename[1])
    if inverted:
        cols = cols[::-1]
    xx = np.linspace(0, 1, len(cols))

    if 'rgb' in cols[0]:
        r = [int(t.split('(')[1].split(',')[0]) for t in cols]
        g = [int(t.split('(')[1].split(',')[1]) for t in cols]
        b = [int(t.split('(')[1].split(',')[2].split(')')[0]) for t in cols]
    else:
        r = [int(tuple(int(t.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[0]) for t in cols]
        g = [int(tuple(int(t.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[1]) for t in cols]
        b = [int(tuple(int(t.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))[2]) for t in cols]

    # fr = interp1d(xx, r)
    # fg = interp1d(xx, g)
    # fb = interp1d(xx, b)

    np.interp(step_nr / nsteps, xx, r)

    # ii = step_nr / nsteps
    # return f'rgba({fr(ii)}, {fg(ii)}, {fb(ii)}, {alpha})'
    return f'rgba({np.interp(step_nr / nsteps, xx, r):.0f}, ' \
           f'{np.interp(step_nr / nsteps, xx, g):.0f}, ' \
           f'{np.interp(step_nr / nsteps, xx, b):.0f}, {alpha})'


def hex_to_rgba(hx, opacity, as_array=False):
    h = hx.lstrip('#')
    res = f'{tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))}'

    if not as_array:
        return f'rgba{res[:-1]}, {opacity})'
    elif as_array:
        res = res.split('(')[1].split(')')[0].split(',')
        res.append(opacity)
        return [int(r) for r in res]


# def create_ppt_with_figures_2_row_1_col(image_paths, output_pptx):
#     """Creates a PowerPoint presentation with Plotly figures in a 2x2 grid per slide."""
#     positions = [
#         (Inches(0.5), Inches(0.5)),  # Top-left
#         # (Inches(5), Inches(0.5)),  # Top-right
#         (Inches(0.5), Inches(4)),  # Bottom-left
#         # (Inches(5), Inches(4))  # Bottom-right
#     ]
#
#     prs = Presentation()
#     # Group images into chunks of 4
#     for slide_i in tqdm(range(0, len(image_paths), 2), 'creating waveforms ppt'):
#
#         slide = prs.slides.add_slide(prs.slide_layouts[5])  # Use blank slide layout
#         for frame_j in range(2):
#             if slide_i + frame_j > len(image_paths) - 1 :
#                 continue
#             slide.shapes.add_picture(
#                 image_paths[slide_i  + frame_j],
#                 positions[frame_j][0], positions[frame_j][1],
#                 width=Inches(8), height=Inches(3))
#
#     prs.save(output_pptx)
#     print(f"PowerPoint saved to {output_pptx}")