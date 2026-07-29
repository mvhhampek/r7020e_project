import cv2
import numpy as np
import os, re
import yaml
import cv2

# import open3d as o3d


##### TIMESTAMP MATCHING #####
def extract_ts(fn):
    m = re.search(r"(\d+)(?=\.[^.]+$)", fn)
    return int(m.group(1)) if m else None # timestamp


def list_with_ts(folder):
    out = []
    for f in os.listdir(folder):
        ts = extract_ts(f)
        if ts is not None:
            out.append((ts, os.path.join(folder, f)))
    return sorted(out, key=lambda x: x[0])


def nearest(target_ts, lst): 
    # nearest neighbour + binary search
    if not lst:
        return None
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid][0] < target_ts:
            lo = mid + 1
        else:
            hi = mid - 1
    candidates = []
    if 0 <= lo < len(lst):
        candidates.append(lst[lo])
    if 0 <= lo - 1 < len(lst):
        candidates.append(lst[lo - 1])
    return min(candidates, key=lambda x: abs(x[0] - target_ts)) if candidates else None


def get_matched_data(verbose):
    base = "data/raw/test"
    dirs = {
        "depth_points": os.path.join(base, "camera_depth_points"),
        "depth_image": os.path.join(base, "camera_depth_image_raw"),
        "color_image": os.path.join(base, "camera_color_image_raw"),
    }

    depth_points_list = list_with_ts(dirs["depth_points"])
    depth_image_list = list_with_ts(dirs["depth_image"])
    color_image_list = list_with_ts(dirs["color_image"])

    data = {}
    prev_d_ts = None
    for d_ts, d_path in depth_image_list:
        c = nearest(d_ts, color_image_list)
        if not c:
            continue
        c_ts, c_path = c
        p = nearest(d_ts, depth_points_list)
        if not p:
            continue
        p_ts, p_path = p

        color = cv2.imread(c_path, cv2.IMREAD_COLOR)
        depth = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
        pcd = np.asarray(
            []
        )  # o3d.io.read_point_cloud(p_path)


        diff_dc = abs(d_ts - c_ts)
        diff_dp = abs(d_ts - p_ts)
        if verbose:
            print(
                f"d(depth-color)={diff_dc*1e-6:.3f} ms, \td(depth-pcd)={diff_dp*1e-6:.3f} ms"
            )

        # default value 0.05s (only applies to first image)
        duration = 50000000 if prev_d_ts is None else d_ts - prev_d_ts
        data[d_ts] = {
            "color_image": color,
            "depth_image": depth,
            "point_cloud": pcd,
            "duration": duration,
        }
        prev_d_ts = d_ts

    return data


##### CAMERA INFO #####
def read_camera_info():
    with open("data/raw/test/camera_color_camera_info/camera_color_info_1727164479160357265.txt","r",) as f:
        _data = yaml.safe_load(f)

    K = np.array(_data["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(_data["D"], dtype=np.float64).reshape(-1)
    R = np.array(_data["R"], dtype=np.float64).reshape(3, 3)
    P = np.array(_data["P"], dtype=np.float64).reshape(3, 4)

    height = int(_data["height"])
    width = int(_data["width"])
    distortion_model = _data["distortion_model"]
    frame_id = _data["header"]["frame_id"]
    return K, D, R, P, height, width, distortion_model, frame_id


##### 3D LOCALIZATION #####

def sample_depth_mm(depth_img, y, x):
    """
    returns depth of (x,y) in depth_img
    """

    return float(depth_img[int(y), int(x)])


def otsu_peak_pos(img, bins, plus_minus):
    """
    perform otsu threshold on input depth image, select the peak of foreground +/- plus_minus bins,
    create mask of selected depth values, get avg position of pixel with said pixels,
    refine pos by taking the center of the selected pixels row
    """
    # drop noise and zeros (invalid depths have depth = 0, e.x on edges of depth imgs)
    img = np.where(img <= 100, np.nan, img)

    fin = np.isfinite(img)

    vmin, vmax = np.nanmin(img[fin]), np.nanmax(img[fin])
    if vmax <= vmin:
        return np.asarray([0, 0])

    scaled = np.zeros_like(img, dtype=np.uint8)
    scaled[fin] = np.clip(((img[fin] - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    ret, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(vmin + (ret / 255.0) * (vmax - vmin))

    fg = fin & (img < t) # foreground 

    vals = img[fg]
    hist, bin_edges = np.histogram(vals, bins=bins)

    # range of peak +/- plus_minus
    peak_idx = int(np.argmax(hist))
    lo_i = max(0, peak_idx - plus_minus)
    hi_i = min(len(hist) - 1, peak_idx + plus_minus)

    low_val = bin_edges[lo_i]
    high_val = bin_edges[hi_i + 1]

    # pixels with depth within peak +/- plus_minus
    mask = fg & (img >= low_val) & (img < high_val)
    coords = np.argwhere(mask)

    avg_pos = coords.mean(axis=0)

    y, x = avg_pos

    # select center of row with avg pos pixel within otsu mask
    y_int = int(round(y))
    row_xs = np.where(mask[y_int, :])[0]
    if row_xs.size > 0:
        x = row_xs.mean()

    refined_pos = np.array([y, x])

    return refined_pos


def pixel_to_xyz(v, u, depth, P):
    """
    computes and returns the XYZ position  of pixel (u,v) at depth, relative to the camera
    (same distance unit as input depth)
    """
    fx, fy = P[0, 0], P[1, 1]  # focal length
    cx, cy = P[0, 2], P[1, 2]  # image to coordframe offset (roughly half H, W)
    Z = float(depth)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)


def get_detected_object_xyz(depth_img, cropped_depth, P, x1, y1):
    """
    perform otsu to get avg pixel pos of foreground class, sample depth at that point, convert to xyz
    """
    local_pos = otsu_peak_pos(cropped_depth, bins=50, plus_minus=5)
    global_pos = local_pos + [y1, x1]  # local_pos is coords within bb
    v, u = global_pos

    # depth of sample
    depth_mm = sample_depth_mm(depth_img, v, u)

    # XYZ of sample
    XYZ_mm = pixel_to_xyz(v, u, depth_mm, P)

    return np.asarray(XYZ_mm), np.asarray(global_pos)


##### CHANGE BB SIZE #####
def scale_bb(x1, y1, x2, y2, H, W, scale):
    """
    scale bb by input scale, keep within H, W
    """
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * scale
    new_h = h * scale

    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(W - 1, cx + new_w / 2))
    ny2 = int(min(H - 1, cy + new_h / 2))
    return nx1, ny1, nx2, ny2


def min_size_bb(x1, y1, x2, y2, H, W, min_w, min_h):
    """
    set bb size to max of current and input minimum, keep within H, W
    """
    w, h = x2 - x1, y2 - y1

    if w < min_w:
        diff = (min_w - w) // 2
        x1 -= diff
        x2 += diff
    if h < min_h:
        diff = (min_h - h) // 2
        y1 -= diff
        y2 += diff

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    return (x1, y1, x2, y2)
