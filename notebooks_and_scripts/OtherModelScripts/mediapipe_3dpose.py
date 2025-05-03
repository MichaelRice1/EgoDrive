import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow logging (1)
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import projectaria_tools.core.mps as mps
import rerun as rr
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from projectaria_tools.core import data_provider, sensor_data, sophus
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

mp_hand_landmarker_task_path =  "C:/Users/athen/Desktop/Github/MastersThesis/models/hand_landmarker.task"
mps_wrist_csv_path = "C:/Users/athen/Desktop/Github/MastersThesis/sampledata/mps_Lab_Test_vrs/hand_tracking/wrist_and_palm_poses.csv"
vrs_path = "C:/Users/athen/Desktop/Github/MastersThesis/sampledata/mps_Lab_Test_vrs/Lab_Test.vrs"



def get_frame_data(
    provider: data_provider.VrsDataProvider, stream_label: str, frame_idx: int
):
    stream_id = provider.get_stream_id_from_label(stream_label)
    image_data = provider.get_image_data_by_index(stream_id, frame_idx)
    return image_data


def get_camera_calibration(
    provider: data_provider.VrsDataProvider, stream_label: str
) -> CameraCalibration:
    device_calib: Optional[DeviceCalibration] = provider.get_device_calibration()
    assert device_calib is not None, "Cannot get device calibration"
    cam_calib: Optional[CameraCalibration] = device_calib.get_camera_calib(stream_label)
    assert cam_calib is not None, f"Cannot get calibration for {stream_label}"

    return cam_calib


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    mediapipe input has to be 3 channel RGB image, so we convert the grayscale
    image to RGB image
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def show_hand_skeleton(
    pts: np.ndarray, connections: List, color_int8: List[int], label: str
):
    """
    Visualize a hand skeleton as 3d line set
    """
    line3d = []
    for edge in connections:
        start_idx, end_idx = edge[0], edge[1]
        landmark_pair = [pts[start_idx], pts[end_idx]]
        line3d.append(landmark_pair)
    colors = [color_int8] * len(line3d)
    rr.log(label, rr.LineStrips3D(line3d, radii=0.002, colors=colors))
    colors = [color_int8] * len(pts)
    rr.log(label, rr.Points3D(pts, radii=0.0025, colors=colors))


@dataclass
class Intrinsics:
    """
    Pinhole camera intrinsics
    """

    focal: float
    cx: float
    cy: float
    w: int
    h: int


def log_cam(T_world_cam: sophus.SE3, intrinsics: Intrinsics, label: str) -> None:
    """
    Logs a pinhole camera with given pose to Rerun
    """
    rr.log(label, ToTransform3D(T_world_cam, False))
    rr.log(
        label,
        rr.Pinhole(
            focal_length=intrinsics.focal,
            width=intrinsics.w,
            height=intrinsics.h,
            principal_point=[intrinsics.cx, intrinsics.cy],
        ),
    )


def log_pose(pose: sophus.SE3, label: str, static=False) -> None:
    rr.log(label, ToTransform3D(pose, False), static=static)



def create_ht_detector():
    base_options = python.BaseOptions(model_asset_path=mp_hand_landmarker_task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    return detector


def run_mediapipe_ht(detector: vision.HandLandmarker, img: np.ndarray):
    # convert grayscale to RGB by replicating channel 3 times
    if img.ndim == 2:
        img = gray_to_rgb(img)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(mp_img)

    return detection_result

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

def run_mp_ht_on_vrs(
    vrs_path: str,
    cam_label: str,
    mps_wrist_csv_path: str,
    start_frame: Optional[int] = None,
    max_frames: Optional[int] = None,
):
    vrs_provider: data_provider.VrsDataProvider = (
        data_provider.create_vrs_data_provider(vrs_path)
    )
    assert vrs_provider is not None, f"Cannot open vrs file {vrs_path}"

    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(
        mps_wrist_csv_path
    )

    stream_id = vrs_provider.get_stream_id_from_label(cam_label)
    timestamps_ns = vrs_provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
    cam = get_camera_calibration(vrs_provider, cam_label)
    detector = create_ht_detector()
    # print(timestamps_ns)

    if start_frame is None or start_frame <= 0:
        start_frame = 0
    if max_frames is None or max_frames <= 0:
        max_frames = len(timestamps_ns)
    end_frame = min(start_frame + max_frames, len(timestamps_ns))
    for t_ns in tqdm(timestamps_ns[start_frame:end_frame]):
        rr.set_time_nanos("synchronization_time", int(t_ns))
        rr.set_time_sequence("timestamp", t_ns)

        img_data: sensor_data.ImageData = vrs_provider.get_image_data_by_time_ns(
            stream_id, t_ns, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
        )[0]
        img = img_data.to_numpy_array()
        det_res: vision.HandLandmarkerResult = run_mediapipe_ht(detector, img=img)

        # get wrist location from MPS HT results
        wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(
            wrist_and_palm_poses, query_timestamp_ns=t_ns
        )
        wrist_per_hand = [
            (
                wrist_and_palm_pose.left_hand.wrist_position_device
                if wrist_and_palm_pose.left_hand.confidence > 0.0
                else None
            ),
            (
                wrist_and_palm_pose.right_hand.wrist_position_device
                if wrist_and_palm_pose.right_hand.confidence > 0.0
                else None
            ),
        ]
        wrist_device_array = np.zeros([3, 2])
        for hi, wrist in enumerate(wrist_per_hand):
            if wrist is None:
                continue
            wrist_device_array[:, hi] = wrist

        # visualize everything for this frame
        for hi in range(2):
            hand_label = f"world/hand_{hi}/joints"
            rr.log(hand_label, rr.Clear(recursive=True))
        rr_show_hands_in_cam(
            wrist_device_array,
            img=img,
            det_res=det_res,
            cam=cam,
            cam_label=cam_label,
            hide_hands_w_negative_depth=True,
        )


rr.init("Hand Tracking on VRS", spawn=True)

run_mp_ht_on_vrs(
    vrs_path=vrs_path,
    cam_label="camera-rgb",
    mps_wrist_csv_path=mps_wrist_csv_path,
    # Set start_frame=0 to start from the beginning of VRS
    start_frame=0,
    # Set max_frames=-1 to use all frames
    max_frames=-1,
)