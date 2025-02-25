import cv2
import os
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import screeninfo

channel_map = {
    1: (600, 0),
    2: (0, 600),
    3: (0, 1200),
    4: (0, 1800),
    5: (0, 2400),
    6: (1200, 0),
    7: (600, 2400),
    8: (600, 1800),
    9: (600, 600),
    10: (1200, 600),
    11: (600, 1200),
    12: (600, 3000),
    13: (1200, 3000),
    14: (1200, 2400),
    15: (1200, 1800),
    16: (1200, 1200),
    17: (1800, 1200),
    18: (1800, 1800),
    19: (1800, 2400),
    20: (1800, 3000),
    21: (2400, 3000),
    22: (2400, 1200),
    23: (1800, 600),
    24: (2400, 600),
    25: (2400, 1800),
    26: (2400, 2400),
    27: (1800, 0),
    28: (3000, 2400),
    29: (3000, 1800),
    30: (3000, 1200),
    31: (3000, 600),
    32: (2400, 0),
}

# Predefined reference points
reference_points = {
    'line_point_1': dict(name='lambda', x='?', y='?'),
    'line_point_2': dict(name='bregma', x='?', y='?'),
    'grid_top_left': dict(name=32, x=2400, y=0),
    'grid_top_right': dict(name=1, x=600, y=0),
    'grid_bottom_right': dict(name=12, x=600, y=3000),
    'grid_bottom_left': dict(name=21, x=2400, y=3000),
}


# Directory containing images
directory_path = r"C:\axorus\250120-PEV_test"  # Specify the directory containing images


def load_df(csv_path):

    df = pd.read_csv(csv_path, index_col=0)
    df['x_img_pxl'] = df['x']
    df['y_img_pxl'] = df['y']
    for i, r in df.iterrows():
        df.at[i, 'x_ecog_um'] = reference_points[i]['x']
        df.at[i, 'y_ecog_um'] = reference_points[i]['y']
        df.at[i, 'name'] = reference_points[i]['name']

    return df

def get_resolution(grid_df):
    corners = [c for c in grid_df.index.values if 'line' not in c]
    resolution = []
    for ci in corners:
        for cj in corners:
            if ci == cj:
                continue

            n_pixels = np.sqrt((grid_df.loc[ci, 'x_img_pxl'] - grid_df.loc[cj, 'x_img_pxl']) ** 2 +
                               (grid_df.loc[ci, 'y_img_pxl'] - grid_df.loc[cj, 'y_img_pxl']) ** 2)
            dist_ecog = np.sqrt((grid_df.loc[ci, 'x_ecog_um'] - grid_df.loc[cj, 'x_ecog_um']) ** 2 +
                               (grid_df.loc[ci, 'y_ecog_um'] - grid_df.loc[cj, 'y_ecog_um']) ** 2)

            resolution.append((dist_ecog / n_pixels) / 1e3)
    resolution = np.mean(resolution)
    print(f'resolution: {resolution:.3f} mm / pxl')
    return resolution


def compute_transformation(p1, p2):
    # Ensure input points are NumPy arrays
    p1, p2 = np.array(p1), np.array(p2)

    # Compute unit vector along line P (new y-axis)
    y_axis = p2 - p1
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Compute unit vector along new x-axis (orthogonal to y-axis)
    x_axis = np.array([-y_axis[1], y_axis[0]])  # 2D perpendicular vector

    # Construct rotation matrix (column vectors are x_axis and y_axis)
    R = np.column_stack((x_axis, y_axis))

    # Translation vector
    T = p1

    return R.T, -R.T @ T  # Inverting transformation


def transform_point(local_point, R, T):
    # Ensure local_point is a NumPy array
    local_point = np.array(local_point)

    # Apply transformation: global_point = R @ local_point + T
    return R @ local_point + T


def compute_transformation2(P1, P2, Q1, Q2):
    """
    Compute the rotation and translation needed to align line segment P (P1, P2) with Q (Q1, Q2).
    """
    # Compute direction vectors
    P_vec = np.array(P2) - np.array(P1)
    Q_vec = np.array(Q2) - np.array(Q1)

    # Compute angles
    angle_P = np.arctan2(P_vec[1], P_vec[0])
    angle_Q = np.arctan2(Q_vec[1], Q_vec[0])

    # Compute rotation angle
    rotation_angle = angle_Q - angle_P

    # Compute translation
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])
    P1_rotated = R @ np.array(P1)
    translation = np.array(Q1) - P1_rotated

    return rotation_angle, translation


def transform_point2(point, rotation_angle, translation):
    """
    Transform a point from P's coordinate system to Q's coordinate system.
    """
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])
    rotated_point = R @ np.array(point)
    transformed_point = rotated_point + translation
    return transformed_point


def compute_homography(grid_corners, image_corners):
    """
    Computes the homography matrix that maps points from the 6x6 grid to the image.

    Parameters:
        grid_corners (array-like): 4 corner points of the original grid [(x1, y1), ..., (x4, y4)]
        image_corners (array-like): 4 corresponding corner points in the image [(x1', y1'), ..., (x4', y4')]

    Returns:
        H (numpy.ndarray): 3x3 homography matrix
    """
    grid_corners = np.array(grid_corners, dtype=np.float32)
    image_corners = np.array(image_corners, dtype=np.float32)

    # Compute the homography matrix
    H, _ = cv2.findHomography(grid_corners, image_corners)
    return H


def map_grid_point(H, point):
    """
    Maps a point from the grid to the image using the homography matrix.

    Parameters:
        H (numpy.ndarray): 3x3 homography matrix
        point (tuple): (x, y) coordinates of a point in the original 6x6 grid

    Returns:
        (x', y'): Transformed coordinates in the image
    """
    src_point = np.array([[point[0], point[1], 1]], dtype=np.float32).T  # Convert to homogeneous coordinates
    dst_point = H @ src_point  # Apply transformation
    dst_point /= dst_point[2]  # Normalize

    return (dst_point[0][0], dst_point[1][0])


def main():

    # Find the first image in the directory
    images = [f for f in os.listdir(directory_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for image in images:
        first_image_path = os.path.join(directory_path, image)
        csv_path = first_image_path.replace(".jpg", ".csv").replace(".png", ".csv")

        df = load_df(csv_path)

        # Make a dataframe for the measured corners
        grid_df = df.iloc[2:].copy()
        # Make a dataframe for the measured lambda and bregma
        line_df = df.iloc[:2].copy()

        # Compute the resolution using the knowledge of the
        # distance between the corners in the probe
        resolution = get_resolution(grid_df)

        # Add a column to the data with x,y position in MM
        grid_df['x_img_mm'] = grid_df['x_img_pxl'] * resolution
        grid_df['y_img_mm'] = grid_df['y_img_pxl'] * resolution
        line_df['x_img_mm'] = line_df['x_img_pxl'] * resolution
        line_df['y_img_mm'] = line_df['y_img_pxl'] * resolution

        # Find the transformation of the ECOG relative to the lambda-bregma line
        p1 = [line_df.loc['line_point_1', 'x_img_mm'], line_df.loc['line_point_1', 'y_img_mm']]
        p2 = [line_df.loc['line_point_2', 'x_img_mm'], line_df.loc['line_point_2', 'y_img_mm']]

        # Find the translation and rotation of L-B line relative
        # to global reference frame
        R, T = compute_transformation(p1, p2)


        # for i, r in grid_df.iterrows():
        #     pt = transform_point([r['x_img_um'], r['y_img_um']], R, T)
        #     grid_df.at[i, 'ML_um'] = pt[0]
        #     grid_df.at[i, 'AP_um'] = pt[1]

        # Create a probe dataframe using the channel map information
        # Invert the x coordinates since the probe is flipped relative to the
        # channel map (in the catalog its inverted)
        probe_df = pd.DataFrame()
        for ch_nr, (x, y) in channel_map.items():
            probe_df.at[ch_nr, 'x_ecog_mm'] = -x / 1e3
            probe_df.at[ch_nr, 'y_ecog_mm'] = y / 1e3
            probe_df.at[ch_nr, 'x_ecog_pxl'] = -x / resolution
            probe_df.at[ch_nr, 'y_ecog_pxl'] = y / resolution


        # Now measure the tansformation of the ECOG probe in the image
        # because its warped in 3D, there are non-linearities
        # AI suggested to use this DLT approach, which also does a decent job

        # Extract the corners of the measured grid, and the corresponding sites in
        # the ecog probe
        grid_corners, image_corners = [], []
        for i, r in grid_df.iterrows():
            image_corners.append([r.x_img_pxl, r.y_img_pxl])
            probe_info = probe_df.loc[int(r['name'])]
            grid_corners.append([probe_info['x_ecog_pxl'], probe_info['y_ecog_pxl']])

        # Compute the transformation
        H = compute_homography(grid_corners, image_corners)

        # For each site on the probe, find the coordinates in pixels in the image
        # Then transform those into stereotactic coordinates.
        for i, r in probe_df.iterrows():
            # pt = transform_point2([r['x_ecog_pxl'], r['y_ecog_pxl']], R, T)
            pt = [r['x_ecog_pxl'], r['y_ecog_pxl']]
            pt = map_grid_point(H, pt)
            probe_df.at[i, 'x_img_pxl'] = pt[0]
            probe_df.at[i, 'y_img_pxl'] = pt[1]
            probe_df.at[i, 'x_img_mm'] = pt[0] * resolution
            probe_df.at[i, 'y_img_mm'] = pt[0] * resolution


            pt2 = transform_point([pt[0] * resolution, pt[1] * resolution], R, T)
            probe_df.at[i, 'ML_mm'] = pt2[0]
            probe_df.at[i, 'AP_mm'] = pt2[1]


        image = cv2.imread(first_image_path)
        # image[:] = 0
        screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = screen.width, screen.height

        window_name = 'test'
        img_height, img_width = image.shape[:2]
        window_width = screen_width // 2
        aspect_ratio = img_height / img_width
        window_height = int(window_width * aspect_ratio)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        cv2.moveWindow(window_name, (screen_width - window_width) // 2, (screen_height - window_height) // 2)

        # Mark some random channels with their calculated AP and ML coordinates
        to_mark = [28, 2]
        for i, r in probe_df.iterrows():
            x2 = int(r.x_img_pxl)
            y2 = int(r.y_img_pxl)

            if i not in to_mark:
                cv2.circle(image, (x2, y2), 5, (255, 0, 255), -1)

            else:
                txt = f'{i} AP: {r["AP_mm"]:.1f}, ML: {r["ML_mm"]:.1f}'
                cv2.circle(image, (x2, y2), 10, (255, 255, 0), -1)
                cv2.putText(image, txt, (x2 + 10, y2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)


        # Plot the measured corners
        for i, r in grid_df.iterrows():
            x = int(r['x'])
            y = int(r['y'])
            txt = ''
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image, txt, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


