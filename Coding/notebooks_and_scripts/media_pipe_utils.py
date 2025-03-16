import numpy as np
import cv2
from typing import List, Tuple, Optional
import mediapipe as mp
import realsense_reconstruction as rr
from realsense_reconstruction import vision, solutions
from realsense_reconstruction.camera import CameraCalibration, Intrinsics
import sophus
















def unproj_img_pts(
    pts_im: np.ndarray, depth: np.ndarray, cam: CameraCalibration
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject floating 2.5d pixel coordinates along its ray to the 3d point at the corresponding depth.
    Returns:
    - unprojected 3d points in camera
    - whether a point is valid or not (due to vignette masking, etc.)

    - `pts_im`: shape (N, 3), array of 2.5d points, each has (x, y, relative-depth)
    - `depth`: shape (N,), each entry provides corresponding depth a point should land at
    - `cam`: the camera model used to perform unprojection
    """
    pts_cam = np.zeros([len(pts_im), 3])
    valid_flags = np.zeros(len(pts_im)).astype(bool)
    for i, p in enumerate(pts_im):
        valid = cam.is_visible(p)
        valid_flags[i] = valid
        p_cam = cam.unproject(camera_pixel=p)
        if valid:
            assert p_cam is not None
            # scale by depth
            pts_cam[i] = p_cam * depth[i] / p_cam[2]
        else:
            assert p_cam is None

    # pts_cam[:, 2][valid_flags == True] = depth[valid_flags == True]
    return pts_cam, valid_flags


def get_landmark_img_pts(
    det_res: vision.HandLandmarkerResult,
    img_wh: Tuple[int, int],
    wrist_depth: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 2.5D landmark as np array, where each point follows:
    [pix_x, pix_y, relative_depth_to_wrist]
    Return is a tuple of two hands
    """
    w, h = img_wh
    ret = [None, None]
    for hi, one_hand_lmks in enumerate(det_res.hand_landmarks):
        if len(one_hand_lmks) == 0:
            continue

        hand_cat: mp.tasks.components.containers.Category = det_res.handedness[hi][0]
        hand_index = 0 if hand_cat.category_name == "Left" else 1
        wrist_depth_cur_hand = wrist_depth[hand_index]
        n_lmks = len(one_hand_lmks)
        hand_lmks = np.zeros([n_lmks, 3])

        # Shif all joints by wrist depth
        for i, lmk in enumerate(one_hand_lmks):
            hand_lmks[i] = [lmk.x * w, lmk.y * h, lmk.z]
            if wrist_depth is not None:
                hand_lmks[i][2] += wrist_depth_cur_hand
        ret[hand_index] = hand_lmks
    return tuple(ret)


def get_landmark_cam_pts(
    landmarks_img: np.ndarray, cam: CameraCalibration
) -> Tuple[np.ndarray, np.ndarray]:
    pts_cam, valid_flags = unproj_img_pts(
        pts_im=landmarks_img[:, :2], depth=landmarks_img[:, 2], cam=cam
    )
    return pts_cam, valid_flags


def convert_mp_det_to_cam_landmarks(
    det_res: vision.HandLandmarkerResult,
    wrist_depth: List[float],
    cam: CameraCalibration,
    verbose: bool = False,
) -> List[Optional[np.ndarray]]:
    w, h = cam.get_image_size()
    lmks_per_hand = get_landmark_img_pts(
        det_res=det_res, img_wh=(w, h), wrist_depth=wrist_depth
    )
    pts_per_hand = [None, None]
    for hi, lmks_one_hand in enumerate(lmks_per_hand):
        if lmks_one_hand is not None:
            pts_cam, valid_flags = get_landmark_cam_pts(
                landmarks_img=lmks_one_hand, cam=cam
            )
            if not valid_flags.all():
                if verbose:
                    print(f"Hand {hi} has invalid joints. Skipped.")
                continue
            pts_per_hand[hi] = pts_cam

    return pts_per_hand


def convert_world_to_camera(pts: np.ndarray, cam: CameraCalibration) -> np.ndarray:
    T_device_cam: sophus.SE3 = cam.get_transform_device_camera()
    pts_cam = T_device_cam.inverse() @ pts
    return pts_cam


def compute_hands_with_wrist_location(
    wrist_per_hand_device: np.ndarray,
    det_res: vision.HandLandmarkerResult,
    cam: CameraCalibration,
):
    T_cam_device: sophus.SE3 = cam.get_transform_device_camera().inverse()
    wrist_per_hand_cam = T_cam_device @ wrist_per_hand_device
    wrist_depth = wrist_per_hand_cam.T[:, 2]  # for two hands
    pts_per_hand = convert_mp_det_to_cam_landmarks(det_res, wrist_depth, cam)

    return pts_per_hand


def rr_show_hands_in_cam(
    wrist_per_hand_device: np.ndarray,
    img: np.ndarray,
    det_res: vision.HandLandmarkerResult,
    cam: CameraCalibration,
    cam_label: str,
    hide_hands_w_negative_depth: bool,
    verbose: bool = False,
):
    # Convert hand landmarks into camera space with the given wrist positions
    pts_per_hand = compute_hands_with_wrist_location(
        wrist_per_hand_device=wrist_per_hand_device, det_res=det_res, cam=cam
    )

    pts_per_hand_vis = pts_per_hand
    if hide_hands_w_negative_depth:
        for hi, pts in enumerate(pts_per_hand):
            if pts is None:
                continue
            negative_mask = pts[:, 2] <= 0.0
            if negative_mask.any():
                pts_per_hand_vis[hi] = None
                if verbose:
                    print(f"Hide hand {hi}")

    orig_im = img
    if len(img.shape) == 2:
        # ensure it's RGB
        orig_im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2RGB)
    img_w_hand = draw_landmarks_on_image(orig_im, det_res)

    hands_color = [
        # Left: green
        [0, 128, 0],
        # Right: blue
        [0, 0, 128],
    ]
    for hi, pts in enumerate(pts_per_hand_vis):
        hand_label = f"world/hand_{hi}/joints"
        if pts is not None:
            show_hand_skeleton(
                pts,
                connections=solutions.hands.HAND_CONNECTIONS,
                label=hand_label,
                color_int8=hands_color[hi],
            )
    rr_device_label = f"world/{cam_label}"
    cx, cy = cam.get_principal_point()
    w, h = cam.get_image_size()
    intrinsics = Intrinsics(
        focal=float(cam.get_focal_lengths()[0]), cx=cx, cy=cy, w=w, h=h
    )
    # show camera frustum
    log_cam(sophus.SE3(), intrinsics=intrinsics, label=rr_device_label)
    # show image overlayed with hand
    rr.log(rr_device_label, rr.Image(img_w_hand))