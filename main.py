from copy import deepcopy
import os
import pickle
import random
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from torchvision.transforms import Compose, ToTensor, Normalize
import json
import pandas as pd
import enlighten
import warnings
from collections import Counter
import argparse
import gc
from easy_ViTPose import VitInference
from huggingface_hub import hf_hub_download
from ast import literal_eval
from compute_recommended_weight import compute_reccomended_weight, get_vm_score
from utils import get_index_body_part, landmark_vitpose, set_all_seeds, draw_keypoints



warnings.filterwarnings("ignore")

set_all_seeds(42)


def compute_overlap_bounding_boxes(hand_rect, box_rect):
    """
    Computes the percentage of the hand's bounding box that overlaps with the package's bounding box.

    hand_rect: (xmin, ymin, xmax, ymax) -> Bounding box of the hand
    box_rect: (xmin, ymin, xmax, ymax) -> Bounding box of the package
    """
    # Get coordinates of intersection area
    xA = max(hand_rect[0], box_rect[0])  # max of xmin
    yA = max(hand_rect[1], box_rect[1])  # max of ymin
    xB = min(hand_rect[2], box_rect[2])  # min of xmax
    yB = min(hand_rect[3], box_rect[3])  # min of ymax

    # Compute width & height of the intersection
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    intersection_area = inter_width * inter_height

    # Compute hand bounding box area
    hand_area = (hand_rect[2] - hand_rect[0]) * (hand_rect[3] - hand_rect[1])

    # Avoid division by zero
    if hand_area == 0:
        return 0.0

    return intersection_area / hand_area


def velocity_estimation():
    lasts = list(frame_information.keys())
    lasts.sort()

    # return the average of the last 5 positions
    if len(lasts) > 5:
        lasts = lasts[-5:]
    elif len(lasts) < 2:
        return 0
    last_positions = [frame_information[i]["box_rect"] for i in lasts]
    test = compute_velocity_polyfit(last_positions, fps)
    timestamps = [i / fps for i in lasts]
    y_values = [pos[3] for pos in last_positions]
    slope, _ = np.polyfit(timestamps, y_values, 1)
    return slope


def compute_velocity_polyfit(bbox_list, frame_rate, degree=2):
    """
    Estimates velocity using polynomial fitting.

    :param bbox_list: List of bounding box coordinates [(x_min, y_min, x_max, y_max), ...]
    :param frame_rate: Frame rate of the video (frames per second)
    :param degree: Degree of polynomial fit (default=2 for quadratic)
    :return: List of estimated velocities
    """
    # Extract center positions
    # centers = [( (x_min + x_max) / 2, (y_min + y_max) / 2 ) for x_min, y_min, x_max, y_max in bbox_list]
    centers = [
        ((x_min + x_max) / 2, y_max) for x_min, y_min, x_max, y_max in bbox_list
    ]  # center of the bottom of the box
    times = np.arange(len(centers)) / frame_rate  # Time for each frame

    velocities = []

    for i in range(1, len(centers)):
        x1, y1 = centers[i - 1]
        x2, y2 = centers[i]

        # Compute displacement
        dx, dy = x2 - x1, y2 - y1
        displacement = np.sqrt(dx**2 + dy**2)

        # Compute velocity (displacement / time per frame)
        velocity = displacement * frame_rate  # pixels/sec

        velocities.append(velocity)

    # Compute average velocity (if no movement, velocity should be 0)
    avg_velocity = np.mean(velocities) if velocities else 0

    return avg_velocity


def get_most_common_side_first_view(datas):
    if len(datas) > 0:
        side_values = [
            info["side_first_view"]
            for info in datas
            if info["side_first_view"] is not None
        ]
        if len(side_values) > 0:
            return Counter(side_values).most_common(1)[0][0]
    return None


def initialize_vitpose(dataset_name="coco_25"):
    MODEL_SIZE = "h"  # @param ['s', 'b', 'l', 'h']
    YOLO_SIZE = "n"  # @param ['s', 'n']
    DATASET = dataset_name  # @param ['coco_25', 'coco', 'wholebody', 'mpii', 'aic', 'ap10k', 'apt36k']
    ext = ".pth"
    ext_yolo = ".pt"

    MODEL_TYPE = "torch"
    YOLO_TYPE = "torch"
    REPO_ID = "JunkyByte/easy_ViTPose"
    FILENAME = (
        os.path.join(MODEL_TYPE, f"{DATASET}/vitpose-" + MODEL_SIZE + f"-{DATASET}")
        + ext
    )
    FILENAME_YOLO = "yolov8/yolov8" + YOLO_SIZE + ext_yolo

    print(f"Downloading model {REPO_ID}/{FILENAME}")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    yolo_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME_YOLO)

    model = VitInference(
        model_path,
        yolo_path,
        MODEL_SIZE,
        det_class="human",
        dataset=DATASET,
        yolo_size=320,
        is_video=True,
        single_pose=True,
    )
    return model


manager = enlighten.get_manager()
DATASET = "wholebody"

# read annotations from json
with open("archives_data/annotations.json") as f:
    data = json.load(f)
df = pd.json_normalize(data)

parser = argparse.ArgumentParser()

parser.add_argument("-if", "--input_folder", nargs="+", help="input_folder", type=str)
parser.add_argument(
    "-c",
    "--camera_calibration",
    help="camera_calibration",
    type=literal_eval,
    choices=[True, False],
    default=False,
)
args = parser.parse_args()

video__folder_path = args.input_folder
camera_calibration = args.camera_calibration

list_video = []
video_folder_mapping = {}  

for folder in video__folder_path:
    for video in os.listdir(folder):
        list_video.append(video)
        video_folder_mapping[video] = folder


pb_video = manager.counter(
    total=len(list_video), desc="Videos", unit="video", color="red", leave=False
)

error_results_dict = {}
rwl_dict = {"ground_truth_values": [], "predicted_values": []}
for video_counter, video in enumerate(list_video[::-1]):  # list_video[::-1]
    print(f"Video {video_counter + 1}/{len(list_video)}. Processing video {video}")
    torch.cuda.empty_cache()

    model = YOLO("yolov8x-world.pt")
    classes = ["person", "box"]
    model.set_classes(classes)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    vitpose = initialize_vitpose(DATASET)

    folder = video_folder_mapping[video]
    video_path = os.path.join(folder, video)

    video_name = video_path.split("/")[-1]

    video_to_process = video_path

    if not os.path.isfile(video_to_process):
        raise FileNotFoundError(f"Video {video_name} not found")
    cap = cv2.VideoCapture(video_to_process)

    annotations = (
        df[df["video"] == video_name].set_index("video").to_dict(orient="index")
    )
    subject_heigth_cm = int(annotations[video_name]["subject_height_cm"])
    print(subject_heigth_cm)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_number_of_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cm_per_pixel = 0

    output_folder = f"output/{video_name}/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_video = f"{output_folder}/video_with_distance.mp4"
    output_log_file = f"{output_folder}/log.txt"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    with open(output_log_file, "a") as f:
        f.write(f"\n\nvideo: {video_name}\n")

    frame_counter = 0
    box_counter_frames = 0
    person_counter_frames = 0    
    pb_frames = manager.counter(
        total=int(video_number_of_frames),
        desc="Frames",
        unit="its",
        color="blue",
        leave=False,
    )
    previous_positions = {}
    frame_information = {}

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        debug_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(frame, verbose=False)  # Detect people and boxes
        max_conf = np.zeros(len(classes))

        h, w, _ = frame.shape

        for r in results:
            box_rect = 0, 0, 0, 0

            hands_dict = {
                "left": {
                    "position": None,
                    "depth": None,
                    "bounding_box": None,
                    "area": 0,
                },
                "right": {
                    "position": None,
                    "depth": None,
                    "bounding_box": None,
                    "area": 0,
                },
            }
            floor_dict = {
                "left": {"position": None, "depth": None},
                "right": {"position": None, "depth": None},
            }

            foot_dict = {
                "left": {"position": None, "depth": None, "area": 0},
                "right": {"position": None, "depth": None, "area": 0},
            }

            ankle_dict = {"left": {"position": None}, "right": {"position": None}}

            subject_perspective = None

            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 1:  # Box class
                    if box.conf > max_conf[cls]:  # avoid to detect multiple boxes
                        box_counter_frames += 1
                        max_conf[cls] = box.conf

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_rect = (x1, y1, x2, y2)
                        cv2.rectangle(
                            frame,
                            (box_rect[0], box_rect[1]),
                            (box_rect[2], box_rect[3]),
                            (0, 255, 255),
                            1,
                        )

                if cls == 0:  # Person class
                    if box.conf > max_conf[cls]:

                        max_conf[cls] = box.conf  # avoid to detect multiple subjects
                        person_counter_frames += 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cropped_frame = frame[y1:y2, x1:x2]

                        width = x2 - x1
                        height = y2 - y1
                        # aspect_ratio = width / height
                        # if aspect_ratio > 1.2:
                        #     subject_perspective = "Side View"
                        # elif aspect_ratio < 0.8:,
                        #     subject_perspective = "Front View"
                        # else:
                        #     subject_perspective = "Intermediate"

                        # drwa the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                        if x2 == w:
                            x2 = -1

                        if y2 == h:
                            y2 = -1

                        left_foot = None
                        right_foot = None
                        right_floor = (int((x1 + x2) / 2), y2)
                        left_floor = (int((x1 + x2) / 2), y2)

                        # assuming the subject is standing upright at the beginning of the video
                        if cm_per_pixel == 0:
                            cm_per_pixel = subject_heigth_cm / abs(y2 - y1)

                        pose_results = vitpose.inference(rgb_frame)
                        # cv2.imwrite("test.jpg", draw_keypoints(pose_results[0], frame, dataset=DATASET))

                        if pose_results:
                            # assert len(pose_results) == 1, "Only one person should be detected"
                            pose_results = pose_results[0]

                            # Get hand positions
                            right_hand_landmarks = landmark_vitpose(
                                pose_results,
                                "right_hands_landmark",
                                dataset_name=DATASET,
                            )
                            left_hand_landmarks = landmark_vitpose(
                                pose_results,
                                "left_hands_landmark",
                                dataset_name=DATASET,
                            )

                            for [hand_landmarks, confs], side in zip(
                                [left_hand_landmarks, right_hand_landmarks],
                                ["left", "right"],
                            ):
                                x_min, y_min = w, h
                                x_max, y_max = 0, 0
                                for hand_point, hand_conf_score in zip(
                                    hand_landmarks, confs
                                ):
                                    if hand_conf_score > 0.5:
                                        x, y = hand_point[0], hand_point[1]
                                        x_min, y_min = min(x, x_min), min(y, y_min)
                                        x_max, y_max = max(x, x_max), max(y, y_max)

                                hands_dict[side]["bounding_box"] = (
                                    x_min,
                                    y_min,
                                    x_max,
                                    y_max,
                                )
                                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                area_bounding_box = (x_max - x_min) * (y_max - y_min)
                                hands_dict[side]["area"] = area_bounding_box

                                keypoint_temp, confidence_temp = landmark_vitpose(
                                    pose_results,
                                    f"{side}_hand_root",
                                    dataset_name=DATASET,
                                )
                                hand_x, hand_y = keypoint_temp[0], keypoint_temp[1]
                                hands_dict[side]["position"] = (hand_x, hand_y)

                            # Get foot positions
                            keypoint_temp, confidence_temp = landmark_vitpose(
                                pose_results, "left_small_toe", dataset_name=DATASET
                            )
                            if confidence_temp > 0.5:
                                left_foot = keypoint_temp

                            keypoint_temp, confidence_temp = landmark_vitpose(
                                pose_results, "right_small_toe", dataset_name=DATASET
                            )
                            if confidence_temp > 0.5:
                                right_foot = keypoint_temp

                            # Get ankle positions
                            keypoint_temp, confidence_temp = landmark_vitpose(
                                pose_results, "left_ankle", dataset_name=DATASET
                            )
                            if confidence_temp > 0.5:
                                left_ankle = keypoint_temp
                            else:
                                left_ankle = None

                            keypoint_temp, confidence_temp = landmark_vitpose(
                                pose_results, "right_ankle", dataset_name=DATASET
                            )
                            if confidence_temp > 0.5:
                                right_ankle = keypoint_temp
                            else:
                                right_ankle = None

                            # Get heel positions
                            keypoint_temp, confidence_temp = landmark_vitpose(
                                pose_results, "left_heel", dataset_name=DATASET
                            )
                            if confidence_temp > 0.5:
                                left_heel = keypoint_temp
                            else:
                                left_heel = None

                            keypoint_temp, confidence_temp = landmark_vitpose(
                                pose_results, "right_heel", dataset_name=DATASET
                            )
                            if confidence_temp > 0.5:
                                right_heel = keypoint_temp
                            else:
                                right_heel = None

                            foot_bounding_box = (0, 0, 0, 0)

                            def herons_formula(ankle, heel, foot):
                                A, B, C = (
                                    np.array(ankle),
                                    np.array(heel),
                                    np.array(foot),
                                )
                                a = np.linalg.norm(B - C)
                                b = np.linalg.norm(A - C)
                                c = np.linalg.norm(A - B)
                                s = (a + b + c) / 2
                                return np.sqrt(s * (s - a) * (s - b) * (s - c))

                            left_area_feet, right_area_feet = 0, 0
                            (
                                left_foot_bounding_triangle,
                                right_foot_bounding_triangle,
                            ) = (None, None)
                            if (
                                left_ankle != None
                                and left_heel != None
                                and left_foot != None
                            ):
                                # area of the triangle using Heron's formula
                                left_area_feet = herons_formula(
                                    left_ankle, left_heel, left_foot
                                )
                                left_foot_bounding_triangle = (
                                    left_ankle,
                                    left_heel,
                                    left_foot,
                                )

                            if (
                                right_ankle != None
                                and right_heel != None
                                and right_foot != None
                            ):
                                right_area_feet = herons_formula(
                                    right_ankle, right_heel, right_foot
                                )
                                right_foot_bounding_triangle = (
                                    right_ankle,
                                    right_heel,
                                    right_foot,
                                )

                            if left_ankle == None and right_ankle == None:
                                left_ankle = left_foot
                                right_ankle = right_foot
                            elif left_ankle == None:
                                left_ankle = right_ankle
                            elif right_ankle == None:
                                right_ankle = left_ankle

                            if left_foot == None:
                                left_foot = (int((x1 + x2) / 2), y2)

                            if right_foot == None:
                                right_foot = (int((x1 + x2) / 2), y2)

                            # floor under left hand
                            if hands_dict["left"]["position"]:
                                left_floor = (
                                    hands_dict["left"]["position"][0],
                                    left_foot[1],
                                )

                            if hands_dict["right"]["position"]:
                                right_floor = (
                                    hands_dict["right"]["position"][0],
                                    right_foot[1],
                                )

                        foot_dict = {
                            "left": {
                                "position": left_foot,
                                "area": left_area_feet,
                                "bounding_triangle": left_foot_bounding_triangle,
                            },
                            "right": {
                                "position": right_foot,
                                "area": right_area_feet,
                                "bounding_triangle": right_foot_bounding_triangle,
                            },
                        }

                        floor_dict = {
                            "left": {
                                "position": left_floor,
                            },
                            "right": {
                                "position": right_floor,
                            },
                        }

                        ankle_dict = {
                            "left": {"position": left_ankle},
                            "right": {"position": right_ankle},
                        }

            hand_distance_dict = {
                "left": {
                    "vertical_hand_2d_foot": None,
                    "vertical_hand_2d_floor": None,
                    "horizontal_hand_mid_ankles": None,
                },
                "right": {
                    "vertical_hand_2d_foot": None,
                    "vertical_hand_2d_floor": None,
                    "horizontal_hand_mid_ankles": None,
                },
            }

            box_distance_dict = {
                "horizontal_box_mid_ankles": None,
                "vertical_box_floor": None,
                "vertical_box_foot": None,
            }

            # Compute distances if hands are detected
            for side in ["left", "right"]:
                side_color = (0, 255, 0) if side == "right" else (255, 0, 0)
                if hands_dict[side]["position"]:
                    side_hand = hands_dict[side]["position"]
                    side_foot = foot_dict[side]["position"]
                    side_floor = floor_dict[side]["position"]

                    # Vertical distances
                    # 2D distance
                    side_distance_2d_floor = abs(side_hand[1] - side_floor[1])  # pixels
                    side_distance_2d_floor = round(
                        side_distance_2d_floor * cm_per_pixel
                    )  # cm
                    side_distance_2d_foot = abs(side_hand[1] - side_foot[1])  # pixels
                    side_distance_2d_foot = round(
                        side_distance_2d_foot * cm_per_pixel
                    )  # cm


                    # Horizontal distance from hand to ankle
                    side_distance_ankle = None
                    if (
                        ankle_dict["right"]["position"] != None
                        and ankle_dict["left"]["position"] != None
                    ):
                        ankle_center = (
                            ankle_dict["left"]["position"][0]
                            + ankle_dict["right"]["position"][0]
                        ) / 2
                        side_distance_ankle = abs(side_hand[0] - ankle_center)  # pixels
                        side_distance_ankle = round(
                            side_distance_ankle * cm_per_pixel
                        )  # cm

                    hand_distance_dict[side] = {
                        "vertical_hand_2d_foot": side_distance_2d_foot,
                        "vertical_hand_2d_floor": side_distance_2d_floor,
                        "horizontal_hand_mid_ankles": side_distance_ankle,
                    }

            list_side_first_view = []
            side_first_view = None
            test_side_variable = None
            if (
                hands_dict["left"]["area"] > 0
                and hands_dict["left"]["area"] > hands_dict["right"]["area"]
            ):
                list_side_first_view.append("left")
                test_side_variable = "left"
            elif (
                hands_dict["right"]["area"] > 0
                and hands_dict["right"]["area"] > hands_dict["left"]["area"]
            ):
                list_side_first_view.append("right")

            if (
                foot_dict["left"]["area"] > 0
                and foot_dict["left"]["area"] > foot_dict["right"]["area"]
            ):
                list_side_first_view.append("left")
                test_side_variable = "left"
            elif (
                foot_dict["right"]["area"] > 0
                and foot_dict["right"]["area"] > foot_dict["left"]["area"]
            ):
                list_side_first_view.append("right")

            if len(list_side_first_view) > 0:
                if len(list_side_first_view) == 1:
                    side_first_view = list_side_first_view[0]
                elif len(list_side_first_view) == 2:
                    if list_side_first_view[0] == list_side_first_view[1]:
                        side_first_view = list_side_first_view[0]
                    else:
                        if frame_counter != 0:
                            side_first_view = get_most_common_side_first_view(
                                frame_information.values()
                            )
                            # side_values = [info["side_first_view"] for info in frame_information.values() if info["side_first_view"] is not None]
                            # if len(side_values) > 0:
                            #     side_first_view = Counter(side_values).most_common(1)[0][0]

            # horizontal distance from box to mid of ankle
            if box_rect != (0, 0, 0, 0) and not None in ankle_dict:
                # box horizontal center
                box_center = (box_rect[0] + box_rect[2]) / 2
                # ankle horizontal center
                ankle_center = (
                    ankle_dict["left"]["position"][0]
                    + ankle_dict["right"]["position"][0]
                ) / 2
                # horizontal distance from box to mid of ankle
                ankle_distance = round(abs(box_center - ankle_center) * cm_per_pixel)

                cv2.putText(
                    frame,
                    f"{ankle_distance}cm horizontal 2D Dist",
                    (
                        int(ankle_center),
                        int(
                            (
                                ankle_dict["left"]["position"][1]
                                + ankle_dict["right"]["position"][1]
                            )
                            / 2
                        )
                        - 10,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

                box_distance_dict["horizontal_box_mid_ankles"] = ankle_distance

            if (
                box_rect != (0, 0, 0, 0)
                and not None in floor_dict
                and side_first_view != None
            ):
                box_distance_dict["vertical_box_floor"] = round(
                    (box_rect[3] - floor_dict[side_first_view]["position"][1])
                    * cm_per_pixel
                )
                cv2.putText(
                    frame,
                    f"{box_distance_dict['vertical_box_floor']}cm vertical box to floor",
                    (box_rect[0], box_rect[3] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            if (
                box_rect != (0, 0, 0, 0)
                and not None in foot_dict
                and side_first_view != None
            ):
                box_distance_dict["vertical_box_foot"] = round(
                    (box_rect[3] - foot_dict[side_first_view]["position"][1])
                    * cm_per_pixel
                )
                cv2.putText(
                    frame,
                    f"{box_distance_dict['vertical_box_foot']}cm vertical box to foot",
                    (box_rect[0], box_rect[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            frame_information[frame_counter] = {
                "box_rect": box_rect,
                "hand_distance": hand_distance_dict,
                "foot": foot_dict,
                "floor": floor_dict,
                "hands": hands_dict,
                "ankle": ankle_dict,
                "box_distance": box_distance_dict,
                "subject_perspective": subject_perspective,
                "side_first_view": side_first_view,
            }

        # write on the video, just based on the first view side
        if side_first_view != None:

            # draw hand bounding box
            temp_draw_side_hand = hands_dict[side_first_view]["bounding_box"]
            x_min, y_min, x_max, y_max = temp_draw_side_hand
            cv2.rectangle(
                frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), thickness=1
            )

            # draw foot bounding box
            temp_draw_side_foot = foot_dict[side_first_view]["bounding_triangle"]
            if temp_draw_side_foot != None:
                cv2.polylines(
                    frame,
                    [np.array(temp_draw_side_foot)],
                    isClosed=True,
                    color=(255, 255, 0),
                    thickness=1,
                )

            # draw line hand to foot
            temp_draw_side_hand = hands_dict[side_first_view]["position"]
            temp_draw_side_foot = foot_dict[side_first_view]["position"]
            temp_draw_side_floor = floor_dict[side_first_view]["position"]
            if temp_draw_side_hand != None and temp_draw_side_foot != None:
                # direct line hand to floor/foot
                cv2.line(
                    frame, temp_draw_side_hand, temp_draw_side_floor, (0, 255, 0), 2
                )

                # draw vertical line from hand to floor/foot
                temp_draw_dist_side = 10 if side_first_view == "left" else -10

                cv2.line(
                    frame,
                    temp_draw_side_hand,
                    (temp_draw_side_hand[0], temp_draw_side_floor[1]),
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    f"{hand_distance_dict[side_first_view]['vertical_hand_2d_floor']}cm V",
                    (
                        temp_draw_side_hand[0] + temp_draw_dist_side,
                        temp_draw_side_hand[1],
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv2.line(
                    frame,
                    temp_draw_side_hand,
                    (temp_draw_side_hand[0], temp_draw_side_foot[1]),
                    (0, 255, 0),
                    1,
                )
                cv2.putText(
                    frame,
                    f"{hand_distance_dict[side_first_view]['vertical_hand_2d_foot']}cm V",
                    (
                        temp_draw_side_hand[0] + temp_draw_dist_side,
                        temp_draw_side_hand[1] + 15,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # draw horizontal line from mid_ankles to hand
                temp_draw_ankle_center = int(
                    (
                        ankle_dict["left"]["position"][0]
                        + ankle_dict["right"]["position"][0]
                    )
                    / 2
                )
                cv2.line(
                    frame,
                    (temp_draw_ankle_center, temp_draw_side_floor[1]),
                    (temp_draw_side_hand[0], temp_draw_side_floor[1]),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{hand_distance_dict[side_first_view]['horizontal_hand_mid_ankles']}cm H",
                    (temp_draw_ankle_center, temp_draw_side_floor[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )
                # cv2.line(frame, temp_draw_side_foot, (temp_draw_side_hand[0], temp_draw_side_foot[1]), (255, 0, 0), 2)

        out.write(frame)
        pb_frames.update()

    pb_frames.close(clear=True)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # print(box_counter_frames) %TODO: remove
    # print(person_counter_frames)

    end_detection = {"right": {}, "left": {}}
    start_detection = {"right": {}, "left": {}}

    frame_counter_list, bbox_list = zip(
        *[
            (k, v["box_rect"])
            for k, v in frame_information.items()
            if v["box_rect"] != (0, 0, 0, 0)
        ]
    )

    frame_rate = fps
    centers = [
        ((x_min + x_max) / 2, (y_min + y_max) / 2)
        for x_min, y_min, x_max, y_max in bbox_list
    ]

    velocities = []
    step_frames = 5
    for i in range(step_frames):
        velocities.append(0)
    for i in range(step_frames, len(centers)):
        x1, y1 = centers[i - step_frames]
        x2, y2 = centers[i]

        # Compute displacement (Euclidean distance)
        dx, dy = x2 - x1, y2 - y1
        displacement = np.sqrt(dx**2 + dy**2)

        # Compute velocity (displacement / time per frame)
        velocity = displacement * (
            frame_counter_list[i] / frame_rate
            - frame_counter_list[i - step_frames] / frame_rate
        )  # pixels/sec

        velocities.append(velocity)  # in pixels/sec

    velocities_per_px = np.array(velocities)
    velocities_per_cm = np.array(velocities) * cm_per_pixel

    # Compute dynamic threshold
    # median_velocity = np.median(velocities)
    # std_velocity = np.std(velocities)
    # threshold = median_velocity + k * std_velocity # TO FIX
    threshold = 0.3  # cm

    # Find movement start and end
    moving_indices = np.where(velocities_per_cm > threshold)[0]

    if len(moving_indices) > 0:
        start_frame = int((frame_counter_list[moving_indices[0]] + frame_counter_list[moving_indices[0] - step_frames]) / 2)
        end_frame = int((frame_counter_list[moving_indices[-1]] + frame_counter_list[moving_indices[-1] - step_frames]) / 2)
        # start_frame = int(frame_counter_list[moving_indices[0] - 1])
        # end_frame = int(frame_counter_list[moving_indices[-1] + 1])
    else:
        start_frame, end_frame = None, None

    def get_specific_frame(video_path, video_time_event_list):
        frames = [None] * len(video_time_event_list)
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            success, image = vidcap.read()
            count += 1
            for index, video_time_event in enumerate(video_time_event_list):
                if count == video_time_event:
                    frames[index] = image
        return frames

    def estimate_hand_distance_value(time_with_none_values, side):
        new_dict = frame_information[time_with_none_values]["hand_distance"][
            side
        ].copy()

        lower_bound = time_with_none_values - 1
        upper_bound = len(frame_information) - time_with_none_values

        time_event = [time_with_none_values]

        for i in range(max(lower_bound, upper_bound)):
            time_event = [None, None]
            try:
                temp_min = frame_information[time_with_none_values - i][
                    "hand_distance"
                ][side]
                time_event[0] = time_with_none_values - i
            except:
                pass
            try:
                temp_plus = frame_information[time_with_none_values + i][
                    "hand_distance"
                ][side]
                time_event[1] = time_with_none_values + i
            except:
                pass

            for keys in new_dict:
                if new_dict[keys] == None:
                    if temp_min[keys] != None and temp_plus[keys] != None:
                        new_dict[keys] = (temp_min[keys] + temp_plus[keys]) / 2
                    elif temp_min[keys] != None:
                        new_dict[keys] = temp_min[keys]
                        time_event[1] = (
                            None  # if the value of t+1 is None, the time_event is None
                        )
                    elif temp_plus[keys] != None:
                        new_dict[keys] = temp_plus[keys]
                        time_event[0] = (
                            None  # if the value of t-1 is None, the time_event is None
                        )
            if None not in new_dict.values():
                break

        time_event = [value for value in time_event if value is not None]
        return new_dict, time_event

    list_video_time_event = [start_frame, end_frame]
    temp_frames = get_specific_frame(video_path, list_video_time_event)

    def get_avg_information(max_index, detection_event):
        frames = frame_information.keys()
        if detection_event == "start":
            frames = [frame for frame in frames if frame <= max_index]
        elif detection_event == "end":
            frames = [frame for frame in frames if frame >= max_index]

        side_first_view = get_most_common_side_first_view(
            [frame_information[frame] for frame in frames]
        )
        # initalize the avg_box_distances as same dict of box_distance, but with values as empty list
        avg_box_distances = frame_information[max_index]["box_distance"].copy()
        for k in avg_box_distances.keys():
            avg_box_distances[k] = []

        avg_hand_distances = frame_information[max_index]["hand_distance"][
            side_first_view
        ].copy()
        for k in avg_hand_distances.keys():
            avg_hand_distances[k] = []

        for pv in frames:
            for k, v in frame_information[pv]["box_distance"].items():
                if v is not None:
                    avg_box_distances[k].append(v)
            for k, v in frame_information[pv]["hand_distance"][side_first_view].items():
                if v is not None:
                    avg_hand_distances[k].append(v)

        for k, v in avg_box_distances.items():
            if v:
                avg_box_distances[k] = round(sum(v) / len(v))
            else:
                avg_box_distances[k] = None

        for k, v in avg_hand_distances.items():
            if v:
                avg_hand_distances[k] = round(sum(v) / len(v))
            else:
                avg_hand_distances[k] = None

        return avg_box_distances, avg_hand_distances

    distances_dict = {}
    side_first_view = get_most_common_side_first_view(frame_information.values())
    for detection_event, video_time_event, temp_frame in zip(
        ["start", "end"], list_video_time_event, temp_frames
    ):
        distances_at_t = {}

        avg_box_distances, avg_hand_distances = get_avg_information(
            video_time_event, detection_event
        )
        distances_at_t["avg_box_distances"] = avg_box_distances
        distances_at_t["avg_hand_distances"] = avg_hand_distances

        # box distances
        box_distance_dict = frame_information[video_time_event]["box_distance"]

        horizontal_distance_box = box_distance_dict["horizontal_box_mid_ankles"]
        distances_at_t["horizontal_box_mid_ankles"] = horizontal_distance_box

        # horizontal_distance_hand = hand_distance_dict[side_first_view][
        #     "horizontal_hand_mid_ankles"
        # ]
        # distances_at_t["horizontal_hand_mid_ankles"] = horizontal_distance_hand

        box_floor_distance = box_distance_dict["vertical_box_floor"]
        distances_at_t["vertical_box_floor"] = box_floor_distance

        box_foot_distance = box_distance_dict["vertical_box_foot"]
        distances_at_t["vertical_box_foot"] = box_foot_distance

        avg_box_horizontal_distance = avg_box_distances["horizontal_box_mid_ankles"]
        avg_box_floor_distance = avg_box_distances["vertical_box_floor"]
        avg_box_foot_distance = avg_box_distances["vertical_box_foot"]

        curr_text_point = 0
        text_space = 15

        with open(output_log_file, "a") as f:

            # box_center = (box_rect[0] + box_rect[2]) / 2
            # ankle_center = (ankle_dict["left"]["position"][0] + ankle_dict["right"]["position"][0]) / 2
            f.write(f"video_time: {video_time_event}, {detection_event} : \n")
            curr_text_point += text_space
            cv2.putText(
                temp_frame,
                f"{horizontal_distance_box}cm H box center-mid ankles",
                (0, curr_text_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            curr_text_point += text_space
            cv2.putText(
                temp_frame,
                f"{avg_box_horizontal_distance}cm H AVG box center-mid ankles",
                (0, curr_text_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            f.write(
                f"          horizontal distance center box - mid ankles: {horizontal_distance_box} | AVG: {avg_box_horizontal_distance}\n"
            )

            curr_text_point += text_space
            cv2.putText(
                temp_frame,
                f"{box_distance_dict['vertical_box_floor']}cm vertical box to floor",
                (0, curr_text_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            curr_text_point += text_space
            cv2.putText(
                temp_frame,
                f"{avg_box_distances['vertical_box_floor']}cm avg vertical box to floor",
                (0, curr_text_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            f.write(
                f"          vertical distance box - floor: {box_floor_distance} | AVG: {avg_box_floor_distance}\n"
            )

            curr_text_point += text_space
            cv2.putText(
                temp_frame,
                f"{box_distance_dict['vertical_box_foot']}cm vertical box to foot",
                (0, curr_text_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            curr_text_point += text_space
            cv2.putText(
                temp_frame,
                f"{avg_box_distances['vertical_box_foot']}cm avg vertical box to foot",
                (0, curr_text_point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            f.write(
                f"          vertical distance box - foot: {box_foot_distance} | AVG: {avg_box_foot_distance}\n"
            )

        # hand distances
        hand_distance_dict = frame_information[video_time_event]["hand_distance"].copy()

        for side in ["right", "left"]:
            if side != side_first_view:
                continue

            hand_horizontal_distance = hand_distance_dict[side]["horizontal_hand_mid_ankles"]
            hand_floor_distance = hand_distance_dict[side]["vertical_hand_2d_floor"]
            hand_foot_distance = hand_distance_dict[side]["vertical_hand_2d_foot"]

            avg_hand_horizontal_distance = avg_hand_distances["horizontal_hand_mid_ankles"]
            avg_hand_floor_distance = avg_hand_distances["vertical_hand_2d_floor"]
            avg_hand_foot_distance = avg_hand_distances["vertical_hand_2d_foot"]

            temp_time_event = [video_time_event]
            if None in hand_distance_dict[side].values():
                temp_distance_dict, temp_time_event = estimate_hand_distance_value(
                    video_time_event, side
                )
                hand_distance_dict[side] = temp_distance_dict.copy()
                # TODO: Check realibility of this
                hand_horizontal_distance = hand_distance_dict[side]["horizontal_hand_mid_ankles"]
                hand_floor_distance = hand_distance_dict[side]["vertical_hand_2d_floor"]
                hand_foot_distance = hand_distance_dict[side]["vertical_hand_2d_foot"]                
                if (
                    None in hand_distance_dict[side].values()
                    and side != frame_information[video_time_event]["side_first_view"]
                ):
                    continue

            with open(output_log_file, "a") as f:
                curr_text_point += text_space
                cv2.putText(
                    temp_frame,
                    f"{hand_horizontal_distance}cm H hand-mid ankles",
                    (0, curr_text_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                curr_text_point += text_space
                cv2.putText(
                    temp_frame,
                    f"{avg_hand_horizontal_distance}cm avg H hand-mid ankles",
                    (0, curr_text_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                f.write(
                    f"          horizontal distance hand - mid ankles: {hand_horizontal_distance} | AVG: {avg_hand_horizontal_distance}\n"
                )

                curr_text_point += text_space
                cv2.putText(
                    temp_frame,
                    f"{hand_floor_distance}cm V hand-floor",
                    (0, curr_text_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                curr_text_point += text_space
                cv2.putText(
                    temp_frame,
                    f"{avg_hand_floor_distance}cm avg V hand-floor",
                    (0, curr_text_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                f.write(
                    f"          vertical distance hand - floor: {hand_floor_distance} | AVG: {avg_hand_floor_distance}\n"
                )

                curr_text_point += text_space
                cv2.putText(
                    temp_frame,
                    f"{hand_foot_distance}cm V hand-foot",
                    (0, curr_text_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                curr_text_point += text_space
                cv2.putText(
                    temp_frame,
                    f"{avg_hand_foot_distance}cm avg V hand-foot",
                    (0, curr_text_point),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                f.write(
                    f"          vertical distance hand - foot: {hand_foot_distance} | AVG: {avg_hand_foot_distance}\n"
                )

            (
                temp_side_hand,
                temp_side_foot,
                temp_side_floor,
                temp_hand_bounding_box,
                temp_ankles_center,
            ) = ([], [], [], [], [])
            for vte in temp_time_event:
                # append if the value is not None
                frame_information[vte]["hands"][side][
                    "position"
                ] is not None and temp_side_hand.append(
                    frame_information[vte]["hands"][side]["position"]
                )
                frame_information[vte]["foot"][side][
                    "position"
                ] is not None and temp_side_foot.append(
                    frame_information[vte]["foot"][side]["position"]
                )
                frame_information[vte]["floor"][side][
                    "position"
                ] is not None and temp_side_floor.append(
                    frame_information[vte]["floor"][side]["position"]
                )
                frame_information[vte]["hands"][side][
                    "bounding_box"
                ] is not None and temp_hand_bounding_box.append(
                    frame_information[vte]["hands"][side]["bounding_box"]
                )

                temp_center = []
                for temp_side in ["left", "right"]:
                    if (
                        frame_information[vte]["ankle"][temp_side]["position"]
                        is not None
                    ):
                        temp_center.append(
                            frame_information[vte]["ankle"][temp_side]["position"]
                        )
                assert len(temp_center) <= 2
                temp_ankles_center.append(
                    sum(pos[0] for pos in temp_center) / len(temp_center)
                    if temp_center
                    else None
                )

            side_hand = (
                tuple(
                    int(sum(values) / len(temp_side_hand))
                    for values in zip(*temp_side_hand)
                )
                if temp_side_hand
                else None
            )
            side_foot = (
                tuple(
                    int(sum(values) / len(temp_side_foot))
                    for values in zip(*temp_side_foot)
                )
                if temp_side_foot
                else None
            )
            side_floor = (
                tuple(
                    int(sum(values) / len(temp_side_floor))
                    for values in zip(*temp_side_floor)
                )
                if temp_side_floor
                else None
            )
            side_hand_bounding_box = (
                tuple(
                    int(sum(values) / len(temp_hand_bounding_box))
                    for values in zip(*temp_hand_bounding_box)
                )
                if temp_hand_bounding_box
                else None
            )
            ankles_center = (
                int(sum(temp_ankles_center) / len(temp_ankles_center))
                if temp_ankles_center
                else None
            )

            # bounding_box hand
            # cv2.rectangle(temp_frame, (side_hand_bounding_box[0], side_hand_bounding_box[1]), (side_hand_bounding_box[2], side_hand_bounding_box[3]), (255, 255, 0), 1)

            # side_color = (255, 0, 0) if side == "left" else (0, 255, 0)
            side_color = (255, 0, 0)
            cv2.line(temp_frame, side_hand, side_foot, side_color, 1)  # Draw line
            cv2.putText(
                temp_frame,
                f"{avg_hand_foot_distance}cm V AVG hand-foot",
                (side_hand[0], side_hand[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            cv2.line(temp_frame, side_hand, side_floor, side_color, 1)  # Draw line
            cv2.putText(
                temp_frame,
                f"{avg_hand_floor_distance}cm V AVG hand-floor",
                (side_hand[0], side_hand[1] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            foot_dict = frame_information[video_time_event]["foot"].copy()
            if foot_dict[side]["position"]:
                cv2.circle(temp_frame, foot_dict[side]["position"], 5, side_color, -1)
            if floor_dict[side]["position"]:
                floor_dict = frame_information[video_time_event]["floor"].copy()
            cv2.circle(temp_frame, floor_dict[side]["position"], 5, side_color, -1)
            cv2.circle(temp_frame, (ankles_center, side_floor[1]), 5, side_color, -1)
            cv2.line(
                temp_frame,
                (ankles_center, side_floor[1]),
                (side_hand[0], side_floor[1]),
                side_color,
                1,
            )
            cv2.putText(
                temp_frame,
                f"{avg_hand_horizontal_distance}cm H AVG hand-mid ankles",
                (ankles_center, side_floor[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                temp_frame,
                f"{avg_hand_horizontal_distance}cm H hand-mid ankles",
                (ankles_center, side_floor[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            if side == side_first_view:
                distances_at_t[f"vertical_hand_2d_foot"] = side_distance_2d_foot
                distances_at_t[f"vertical_hand_2d_floor"] = side_distance_2d_floor
                distances_at_t[f"horizontal_hand_mid_ankles"] = side_distance_ankle

        box_rect = frame_information[video_time_event]["box_rect"]
        cv2.rectangle(
            temp_frame,
            (box_rect[0], box_rect[1]),
            (box_rect[2], box_rect[3]),
            (0, 255, 255),
            2,
        )

        cv2.imwrite(f"{output_folder}/lifting {detection_event}.jpg", temp_frame)
        distances_dict[detection_event] = distances_at_t

    print(f"Video {video_name} processed")

    with open(output_log_file, "a") as f:
        for side in ["right", "left"]:
            for keys in start_detection[side]:
                try:
                    f.write(
                        f"difference in {keys}: {abs(start_detection[side][keys] - end_detection[side][keys])}\n"
                    )
                except:
                    pass

        f.write(f"original annotations: \n")
        for key, value in annotations.items():
            for k, v in value.items():
                f.write(f"  {k}: {v}\n")

    def get_max_distance(values: list):
        max = 0
        for value in values:
            if value != None and not isinstance(value, str):
                value = abs(value)
                if value > max:
                    max = value
        return max

    def get_min_distance(values: list):
        min = 100
        for value in values:
            if value != None and not isinstance(value, str):
                value = abs(value)
                if value < min:
                    min = value
        return min

    def add_distances_to_dict(dict, key, startvalue, endvalue):
        if "vertical" in key:
            try:
                dict[f"difference in {key}"] = abs(endvalue - startvalue)
            except:
                dict[f"difference in {key}"] = get_max_distance([endvalue, startvalue])

            dict[f"min in {key}"] = get_min_distance([startvalue, endvalue])
        elif "horizontal" in key:
            dict[f"max in {key}"] = get_max_distance([endvalue, startvalue])
        else:
            print("HELL???")

    # compute distances and write it to file
    diff_distance_dict = {}
    for key in distances_dict["start"]:
        if isinstance(distances_dict["start"][key], dict):
            for subkey in distances_dict["start"][key]:
                add_distances_to_dict(
                    diff_distance_dict,
                    f"{key}_{subkey}",
                    distances_dict["start"][key][subkey],
                    distances_dict["end"][key][subkey],
                )
        else:
            add_distances_to_dict(
                diff_distance_dict,
                key,
                distances_dict["start"][key],
                distances_dict["end"][key],
            )

    with open(output_log_file, "a") as f:
        for key, value in diff_distance_dict.items():
            f.write(f"{key}: {value}\n")

    # write annotation distances to log file
    diff_annotations_dict = {}
    with open(output_log_file, "a") as f:
        for key, value in annotations.items():
            f.write("box_distances \n")
            annotations_vertical_distance_box = (
                abs(value["action_end.box_height_cm"] - value["action_start.box_height_cm"])
            )
            diff_annotations_dict["difference_vertical_distance_box"] = (
                annotations_vertical_distance_box
            )
            f.write(
                f"annotation vertical lifting distances - box: {annotations_vertical_distance_box}\n"
            )

            annotations_max_horizontal_distance_box = get_max_distance(
                [
                    value["action_end.box_distance_horizontal_from_mid_ankles_cm"],
                    value["action_start.box_distance_horizontal_from_mid_ankles_cm"],
                ]
            )
            diff_annotations_dict["max_horizontal_distance_box"] = (
                annotations_max_horizontal_distance_box
            )
            f.write(
                f"annotation max horizontal distance: {annotations_max_horizontal_distance_box}\n"
            )

            annotations_min_vertical_distance_box = get_min_distance(
                [value["action_end.box_height_cm"], value["action_start.box_height_cm"]]
            )
            diff_annotations_dict["min_vertical_distance_box"] = (
                annotations_min_vertical_distance_box
            )
            f.write(
                f"annotation min vertical distance from floor: {annotations_min_vertical_distance_box}\n"
            )

            f.write("hand_distances \n")

            try: 
                annotations_vertical_distance_hand_floor = abs(
                    value["action_end.hand_vertical_distance_from_floor_cm"]
                    - value["action_start.hand_vertical_distance_from_floor_cm"]
                )
            except TypeError as e:
                if "unsupported operand type(s) for -" in str(e):  #if no annotation == 'unknown' value
                    annotations_vertical_distance_hand_floor = None
                else:
                    raise e

            diff_annotations_dict["difference_vertical_distance_hand"] = (
                annotations_vertical_distance_hand_floor
            )
            f.write(
                f"annotation vertical lifting distances - hand: {annotations_vertical_distance_hand_floor}\n"
            )

            annotations_max_horizontal_distance_hand = None #TODO: no annotations
            diff_annotations_dict["max_horizontal_distance_hand"] = (
                annotations_max_horizontal_distance_hand
            )
            f.write(
                f"annotation max horizontal distance: {annotations_max_horizontal_distance_hand}\n"
            )

            annotations_min_vertical_distance_hand = get_min_distance(
                [
                    value["action_end.hand_vertical_distance_from_floor_cm"],
                    value["action_start.hand_vertical_distance_from_floor_cm"],
                ]
            )
            diff_annotations_dict["min_vertical_distance_hand"] = (
                annotations_min_vertical_distance_hand
            )
            f.write(
                f"annotation min vertical distance from floor: {annotations_min_vertical_distance_hand}\n"
            )

    error_results_dict[video_name] = {}
    error_results_dict[video_name]["V"] = {"box": {}, "hand": {}}
    error_results_dict[video_name]["H"] = {"box": {}, "hand": {}}
    error_results_dict[video_name]["D"] = {"box": {}, "hand": {}}
    for key, value in diff_distance_dict.items():
        for reference_point in ["box", "hand"]:
            if reference_point not in key:
                continue
            if "min" in key:
                error_results_dict[video_name]["V"][reference_point][key] = abs(
                    abs(value) - abs(diff_annotations_dict[f"min_vertical_distance_{reference_point}"])
                )
            elif "max" in key:
                temp_value = diff_annotations_dict[f"max_horizontal_distance_{reference_point}"]
                if temp_value != None: # no annotations
                    error_results_dict[video_name]["H"][reference_point][key] = abs(
                        abs(value) - abs(temp_value)
                    )
            elif "difference" in key:
                temp_value = diff_annotations_dict[f"difference_vertical_distance_{reference_point}"]
                if temp_value != None: # no annotations
                    error_results_dict[video_name]["D"][reference_point][key] = abs(
                        abs(value) - abs(temp_value)
                    )

    print(f"errors estimation in video: {video_name}\n")
    for k, v in error_results_dict[video_name].items():
        print(f"{k}:")
        for sk, sv in v.items():
            print(f"  {sk}:")
            for ssk, ssv in sv.items():
                print(f"    {ssk}: {ssv}")

    # get box sizes
    box_depht = annotations[video_name]["box_depth_cm"]
    temp_min_hand_floor_distance_vertical = abs(
        diff_distance_dict["min in avg_box_distances_vertical_box_foot"]
    )
    temp_vertical_lifting_distance = abs(
        diff_distance_dict["difference in avg_box_distances_vertical_box_foot"]
    )
    temp_max_horizontal_hands_mid_ankle_distance = abs(
        diff_distance_dict["max in avg_box_distances_horizontal_box_mid_ankles"] - (box_depht / 2)
    )

    recommended_weight = 0
    recommended_weight = compute_reccomended_weight(
        gender=annotations[video_name]["subject_gender"],
        age=annotations[video_name]["subject_age"],
        min_hand_floor_distance_vertical=temp_min_hand_floor_distance_vertical,
        vertical_lifting_distance=temp_vertical_lifting_distance,
        max_orizontal_hands_mid_ankle_distance=temp_max_horizontal_hands_mid_ankle_distance,
    )

    with open(output_log_file, "a") as f:
        f.write(f"measurement used to recommended the weight: \n")
        f.write(
            f"  min hand floor distance vertical: {temp_min_hand_floor_distance_vertical}\n"
        )
        f.write(f"  vertical lifting distance: {temp_vertical_lifting_distance}\n")
        f.write(
            f"  max horizontal hands mid ankle distance: {temp_max_horizontal_hands_mid_ankle_distance}\n"
        )
        f.write(f"recommended weight: {recommended_weight}\n")

    annotation_V = diff_annotations_dict["min_vertical_distance_box"]
    annotation_H = diff_annotations_dict["max_horizontal_distance_box"]
    annotation_D = diff_annotations_dict["difference_vertical_distance_box"]
    annotation_recommended_weight = compute_reccomended_weight(
        gender=annotations[video_name]["subject_gender"],
        age=annotations[video_name]["subject_age"],
        min_hand_floor_distance_vertical=abs(annotation_V),
        vertical_lifting_distance=abs(annotation_D),
        max_orizontal_hands_mid_ankle_distance=abs(annotation_H),
    )

    with open(output_log_file, "a") as f:
        f.write(f"recommended weight from original annotations: {annotation_recommended_weight}\n")
        f.write(f"  min hand floor distance vertical: {annotation_V}\n")
        f.write(f"  vertical lifting distance: {annotation_D}\n")
        f.write(f"  max horizontal hands mid ankle distance: {annotation_H}\n")

    rwl_dict["ground_truth_values"].append(annotation_recommended_weight)
    rwl_dict["predicted_values"].append(recommended_weight)

    # recommend the new pose to have

    def compute_vertical_distance(point1, point2):
        return abs(point1[1] - point2[1]) * cm_per_pixel

    def compute_horizontal_distance(point1, point2):
        return (abs(point1[0] - point2[0]) * cm_per_pixel) - 24.6

    def obtain_distances_from_keypoints(pose):
        side_hand, _ = landmark_vitpose(
            pose, f"{side_first_view}_hand_root", dataset_name=DATASET
        )
        side_foot, _ = landmark_vitpose(
            pose, f"{side_first_view}_small_toe", dataset_name=DATASET
        )
        side_floor = (side_hand[0], side_foot[1])
        left_ankle, _ = landmark_vitpose(pose, "left_ankle", dataset_name=DATASET)
        right_ankle, _ = landmark_vitpose(pose, "right_ankle", dataset_name=DATASET)
        # Vertical distances
        vertical_hand_foot = compute_vertical_distance(side_hand, side_foot)

        # horizontal distance from box to mid of ankle
        horizontal_mid_ankle = (left_ankle[0] + right_ankle[0]) / 2
        horizontal_hand_mid_ankle = compute_horizontal_distance(
            side_hand, (horizontal_mid_ankle, 0)
        )

        return vertical_hand_foot, horizontal_hand_mid_ankle
        # return compute_reccomended_weight(
        #     min_hand_floor_distance_vertical=0,
        #     vertical_lifting_distance=vertical_hand_foot,
        #     max_orizontal_hands_mid_ankle_distance=horizontal_hand_mid_ankle,
        # )

    def obtain_recommended_weight_from_keypoints(start_pose, end_pose):
        start_vertical_hand_foot, start_horizontal_hand_mid_ankle = (
            obtain_distances_from_keypoints(start_pose)
        )
        end_vertical_hand_foot, end_horizontal_hand_mid_ankle = (
            obtain_distances_from_keypoints(end_pose)
        )

        lifting_vertical_distance = abs(
            end_vertical_hand_foot - start_vertical_hand_foot
        )

        max_horizontal_distance_frame, horizontal_distance = max(
            enumerate([start_horizontal_hand_mid_ankle, end_horizontal_hand_mid_ankle]),
            key=lambda v: v[1],
        )

        min_vertical_distance_frame = min(
            enumerate(
                [
                    get_vm_score(start_vertical_hand_foot),
                    get_vm_score(end_vertical_hand_foot),
                ]
            ),
            key=lambda v: v[1],
        )[0]
        min_vertical_distance = [start_vertical_hand_foot, end_vertical_hand_foot][
            min_vertical_distance_frame
        ]

        rec_weight = compute_reccomended_weight(
            gender=annotations[video_name]["subject_gender"],
            age=annotations[video_name]["subject_age"],            
            min_hand_floor_distance_vertical=min_vertical_distance,
            vertical_lifting_distance=lifting_vertical_distance,
            max_orizontal_hands_mid_ankle_distance=horizontal_distance,
        )
        if rec_weight == 0:
            rec_weight = 0.0001
        return {
            "rec_weight": rec_weight,
            "max_horizontal_distance_frame": max_horizontal_distance_frame,
            "min_vertical_distance_frame": min_vertical_distance_frame,
            "rec_measures": {
                "lifting_vertical_distance": lifting_vertical_distance,
                "horizontal_distance": horizontal_distance,
                "min_vertical_distance": min_vertical_distance,
            },
        }

    def move_nearby_points(
        old_pose, new_pose, influence_factors=(1.0, 0.8, 0.4, 0.3, 0.1)
    ):
        """
        Moves nearby joints based on the new hand position.

        - `hand`, `wrist`, `elbow`: Current coordinates (numpy arrays or lists).
        - `new_hand_pos`: The corrected hand position.
        - `influence_factors`: How much each joint follows the hand (default: hand=100%, wrist=60%, elbow=30%).
        """
        temp_pose = deepcopy(new_pose)
        old_hand_root, _ = landmark_vitpose(
            old_pose,
            f"{side_first_view}_hand_root",
            dataset_name=DATASET,
        )
        new_hand_root, _ = landmark_vitpose(
            temp_pose,
            f"{side_first_view}_hand_root",
            dataset_name=DATASET,
        )

        hand_movement = (np.array(new_hand_root) - np.array(old_hand_root))[::-1]

        for side in ["right", "left"]:
            hands_indexes = get_index_body_part(f"{side}_hands_landmark", DATASET)
            for i in hands_indexes:
                temp_pose[i][:2] = (
                    old_pose[i][:2] + influence_factors[0] * hand_movement
                ).astype(int)

            wrist_index = get_index_body_part(f"{side}_wrist", DATASET)
            temp_pose[wrist_index][:2] = (
                old_pose[wrist_index][:2] + influence_factors[0] * hand_movement
            ).astype(int)

            elbow_index = get_index_body_part(f"{side}_elbow", DATASET)
            temp_pose[elbow_index][:2] = (
                old_pose[elbow_index][:2] + influence_factors[1] * hand_movement
            ).astype(int)

            hip_index = get_index_body_part(f"{side}_hip", DATASET)
            temp_pose[hip_index][:2] = (
                old_pose[hip_index][:2] + influence_factors[2] * hand_movement
            ).astype(int)

            shoulder_index = get_index_body_part(f"{side}_shoulder", DATASET)
            temp_pose[shoulder_index][:2] = (
                old_pose[shoulder_index][:2] + influence_factors[2] * hand_movement
            ).astype(int)

            ear_index = get_index_body_part(f"{side}_ear", DATASET)
            temp_pose[ear_index][:2] = (
                old_pose[ear_index][:2] + influence_factors[2] * hand_movement
            )

            eye_index = get_index_body_part(f"{side}_eye", DATASET)
            temp_pose[eye_index][:2] = (
                old_pose[eye_index][:2] + influence_factors[2] * hand_movement
            ).astype(int)

            knee_index = get_index_body_part(f"{side}_knee", DATASET)
            temp_pose[knee_index][:2] = (
                old_pose[knee_index][:2] + influence_factors[4] * hand_movement
            ).astype(int)

        faces_indexes = get_index_body_part(f"face_landmark", DATASET)
        for i in faces_indexes:
            temp_pose[i][:2] = (
                old_pose[i][:2] + influence_factors[2] * hand_movement
            ).astype(int)

        nose_index = get_index_body_part(f"nose", DATASET)
        temp_pose[nose_index][:2] = (
            old_pose[nose_index][:2] + influence_factors[2] * hand_movement
        ).astype(int)
        return temp_pose

    def correcting_posture_random(
        pose_start_lifting,
        pose_end_lifting,
        side_first_view,
        input_weight=10,
        index_threshold=0.7,
    ):
        gt_results = obtain_recommended_weight_from_keypoints(
            pose_start_lifting, pose_end_lifting
        )

        count = 0

        temp_results = []
        temp_list_pose_results = []
        body_part_name = f"{side_first_view}_hand_root"
        keypoint_index = get_index_body_part(body_part_name, DATASET)

        # for i in range(1000):
        temp_pose = [deepcopy(pose_start_lifting), deepcopy(pose_end_lifting)]
        while (input_weight/obtain_recommended_weight_from_keypoints(temp_pose[0], temp_pose[1])["rec_weight"] > index_threshold) and count < 100000:
            # while obtain_recommended_weight_and_distances_from_keypoints(temp_side_hand, temp_side_foot, temp_left_ankle, temp_right_ankle) <= gt:
            # if count > 1000:
            #     return None, None
            count += 1
            temp_pose = [deepcopy(pose_start_lifting), deepcopy(pose_end_lifting)]
            percentage = count / 1000
            if (
                gt_results["max_horizontal_distance_frame"]
                == gt_results["min_vertical_distance_frame"]
            ):
                event_index = gt_results["max_horizontal_distance_frame"]
                temp_rand_keypoints = get_new_random_keypoint(
                    temp_pose[event_index],
                    body_part_name,
                    percentage=percentage,
                    vertical=True,
                    horizontal=True,
                )
                temp_pose[event_index][keypoint_index][:2] = temp_rand_keypoints[::-1]
            else:
                temp_rand_keypoints = get_new_random_keypoint(
                    temp_pose[gt_results["max_horizontal_distance_frame"]],
                    body_part_name,
                    percentage=percentage,
                    vertical=False,
                    horizontal=True,
                )
                temp_pose[gt_results["max_horizontal_distance_frame"]][keypoint_index][
                    :2
                ] = temp_rand_keypoints[::-1]

                temp_rand_keypoints = get_new_random_keypoint(
                    temp_pose[gt_results["min_vertical_distance_frame"]],
                    body_part_name,
                    percentage=percentage,
                    vertical=True,
                    horizontal=False,
                )
                temp_pose[gt_results["min_vertical_distance_frame"]][keypoint_index][
                    :2
                ] = temp_rand_keypoints[::-1]

            temp_results.append(
                obtain_recommended_weight_from_keypoints(temp_pose[0], temp_pose[1])[
                    "rec_weight"
                ]
            )
            temp_list_pose_results.append(temp_pose)

        sorted_results = sorted(enumerate(temp_results), key=lambda x: x[1], reverse=True)

        if len(sorted_results) == 0:
            return None, None, None

        temp_pose_results = temp_list_pose_results[sorted_results[0][0]]

        def stacks_images(imgs_list):
            max_height = max(img.shape[0] for img in imgs_list)

            def pad_image(img, target_height):
                h, w, c = img.shape
                pad_top = (target_height - h) // 2
                pad_bottom = target_height - h - pad_top
                return cv2.copyMakeBorder(
                    img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )

            images_padded = [pad_image(img, max_height) for img in imgs_list]
            stacked_image = cv2.hconcat(images_padded)
            return stacked_image

        wrong_points_color_palette = "plasma"
        wrong_skeleton_color_palette = "plasma"
        correct_points_color_palette = "Greens_r"        
        correct_skeleton_color_palette = "Accent"

        drawed_img = []
        for pose in [pose_start_lifting, pose_end_lifting]:
            drawed_img.append(
                draw_keypoints(
                    pose,
                    dataset=DATASET,
                    points_color_palette=wrong_points_color_palette,
                    skeleton_color_palette=wrong_skeleton_color_palette,
                    background="black",
                )
            )
        original_pose_draw = stacks_images(drawed_img)

        drawed_img = []
        for pose in temp_pose_results:
            drawed_img.append(draw_keypoints(pose, dataset=DATASET))
        only_hand_moved = stacks_images(drawed_img)

        # "Accent", "Accent_r", "Blues", "Blues_r", "BrBG", "BrBG_r", "BuGn", "BuGn_r", "BuPu", "BuPu_r", "CMRmap", "CMRmap_r", "Dark2", "Dark2_r", "GnBu", "GnBu_r", "Grays", "Greens", "Greens_r", "Greys", "Greys_r", "OrRd", "OrRd_r", "Oranges", "Oranges_r", "PRGn", "PRGn_r", "Paired", "Paired_r", "Pastel1", "Pastel1_r", "Pastel2", "Pastel2_r", "PiYG", "PiYG_r", "PuBu", "PuBuGn", "PuBuGn_r", "PuBu_r", "PuOr", "PuOr_r", "PuRd", "PuRd_r", "Purples", "Purples_r", "RdBu", "RdBu_r", "RdGy", "RdGy_r", "RdPu", "RdPu_r", "RdYlBu", "RdYlBu_r", "RdYlGn", "RdYlGn_r", "Reds", "Reds_r", "Set1", "Set1_r", "Set2", "Set2_r", "Set3", "Set3_r", "Spectral", "Spectral_r", "Wistia", "Wistia_r", "YlGn", "YlGnBu", "YlGnBu_r", "YlGn_r", "YlOrBr", "YlOrBr_r", "YlOrRd", "YlOrRd_r", "afmhot", "afmhot_r", "autumn", "autumn_r", "binary", "binary_r", "bone", "bone_r", "brg", "brg_r", "bwr", "bwr_r", "cividis", "cividis_r", "cool", "cool_r", "coolwarm", "coolwarm_r", "copper", "copper_r", "cubehelix", "cubehelix_r", "flag", "flag_r", "gist_earth", "gist_earth_r", "gist_gray", "gist_gray_r", "gist_grey", "gist_heat", "gist_heat_r", "gist_ncar", "gist_ncar_r", "gist_rainbow", "gist_rainbow_r", "gist_stern", "gist_stern_r", "gist_yarg", "gist_yarg_r", "gist_yerg", "gnuplot", "gnuplot2", "gnuplot2_r", "gnuplot_r", "gray", "gray_r", "grey", "hot", "hot_r", "hsv", "hsv_r", "inferno", "inferno_r", "jet", "jet_r", "magma", "magma_r", "nipy_spectral", "nipy_spectral_r", "ocean", "ocean_r", "pink", "pink_r", "plasma", "plasma_r", "prism", "prism_r", "rainbow", "rainbow_r", "seismic", "seismic_r", "spring", "spring_r", "summer", "summer_r", "tab10", "tab10_r", "tab20", "tab20_r", "tab20b", "tab20b_r", "tab20c", "tab20c_r", "terrain", "terrain_r", "turbo", "turbo_r", "twilight", "twilight_r", "twilight_shifted", "twilight_shifted_r", "viridis", "viridis_r", "winter", "winter_r"
        drawed_img = []
        overlap_drawed_img = []
        new_poses = []
        for old_pose_results, modified_pose in zip([pose_start_lifting, pose_end_lifting], temp_pose_results):
            pose_recommendation = move_nearby_points(old_pose_results, modified_pose)
            new_poses.append(pose_recommendation)            
            temp = draw_keypoints(
                pose_recommendation,
                dataset=DATASET,
                points_color_palette=correct_points_color_palette,
                skeleton_color_palette=correct_skeleton_color_palette,
                background="black",
            )
            drawed_img.append(temp)
            test = draw_keypoints(old_pose_results, dataset=DATASET, points_color_palette=wrong_points_color_palette, skeleton_color_palette=wrong_skeleton_color_palette, background="black")
            test2 = draw_keypoints(pose_recommendation, test, dataset=DATASET, points_color_palette=correct_points_color_palette, skeleton_color_palette=correct_skeleton_color_palette, background="black")
            overlap_drawed_img.append(test2)            
        recommended_pose = stacks_images(drawed_img)
        overlap_drawed_img = stacks_images(overlap_drawed_img)

        draws = {
            "original_pose": original_pose_draw,
            "only_hand_moved": only_hand_moved,
            "recommended_pose": recommended_pose,
            "overlap_poses": overlap_drawed_img,
        }

        def get_textual_recommendation(old_poses, new_poses, side_first_view, original_pose_li, new_pose_li):
            def get_body_part_distances(old_pose, new_pose, body_part_name):
                old_part = landmark_vitpose(old_pose, body_part_name, dataset_name=DATASET)[0]
                new_part = landmark_vitpose(new_pose, body_part_name, dataset_name=DATASET)[0]
                return [old_part[i] - new_part[i] for i in range(2)]

            def get_knee_angle_difference(old_pose, new_pose):
                def calculate_angle(a, b, c):
                    ba = np.array(a) - np.array(b)
                    bc = np.array(c) - np.array(b)
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    return np.degrees(angle_rad)

                def extract_points(pose):
                    hip = landmark_vitpose(pose, f"{side_first_view}_hip", dataset_name=DATASET)[0]
                    knee = landmark_vitpose(pose, f"{side_first_view}_knee", dataset_name=DATASET)[0]
                    ankle = landmark_vitpose(pose, f"{side_first_view}_ankle", dataset_name=DATASET)[0]
                    return hip, knee, ankle

                old_angle = calculate_angle(*extract_points(old_pose))
                new_angle = calculate_angle(*extract_points(new_pose))
                return new_angle - old_angle  # TODO:check - Positive: more flexed; Negative: straighter
            rec_text = []
            for old_pose, new_pose, event in zip(old_poses, new_poses, ["start", "end"]):
                hand_distance = get_body_part_distances(old_pose, new_pose, f"{side_first_view}_hand_root")
                shoulder_distance = get_body_part_distances(old_pose, new_pose, f"{side_first_view}_shoulder")
                elbow_distance = get_body_part_distances(old_pose, new_pose, f"{side_first_view}_elbow")
                hip_distance = get_body_part_distances(old_pose, new_pose, f"{side_first_view}_hip")
                # knee_distance = get_body_part_distances(old_pose, new_pose, f"{side_first_view}_knee")
                knee_angle_diff = get_knee_angle_difference(old_pose, new_pose)

                text = f"At the {event} of the lifting:"

                if abs(hand_distance[0]) > 0:
                    index = 0 # Horizontal
                    if (hand_distance[0] > 0 and side_first_view == "left") or (
                        hand_distance[0] < 0 and side_first_view == "right"):
                        direction = "farther from your body"
                    else:
                        direction = "closer to your body"
                    text += f"\n• Move your hands at least {round(abs(hand_distance[0])*cm_per_pixel)}cm {direction}."
                elif abs(hand_distance[1]) > 0:
                    index = 1 # Vertical
                    direction = "up" if hand_distance[1] > 0 else "down"
                    text += f"\n• Move your hands at least {round(abs(hand_distance[1])*cm_per_pixel)}cm {direction}."
                else:
                    continue
                shoulder_distance = round(abs(shoulder_distance[index])*cm_per_pixel)
                elbow_distance = round(abs(elbow_distance[index])*cm_per_pixel)
                text += f"\n• This adjustment will likely also move your shoulders ({shoulder_distance}cm) and elbows ({elbow_distance}cm) {direction}."
                hip_distance = round(abs(hip_distance[index])*cm_per_pixel)
                text += f"\n• You should actively shift your hips approximately {hip_distance-2}-{hip_distance+2}cm in the same direction."

                # Knee angle variation output
                if abs(knee_angle_diff) > 3:  # Threshold to avoid noise
                    flexion = "increase" if knee_angle_diff > 0 else "decrease"
                    text += f"\n• This posture change leads to a {flexion} in knee flexion of about {abs(knee_angle_diff):.1f}°."
                rec_text.append(text)
            rec_text.insert(0, f"Lifting Index Original Pose: {original_pose_li:.2f}; Recommended Pose Lifting Index: {new_pose_li:.2f}\n")
            return rec_text

        return temp_pose_results, draws, get_textual_recommendation([pose_start_lifting, pose_end_lifting], new_poses, side_first_view)

    def get_new_random_keypoint(
        pose_results, body_part_name, percentage=0.1, horizontal=True, vertical=True
    ):
        body_part = landmark_vitpose(
            pose_results, body_part_name, dataset_name=DATASET
        )[0]

        x, y = body_part
        # capire l'ordine di grandezza x della caviglia con la x della punta del piede, così da ottenere la direzione in cui può andare la correzione
        side_big_toe, _ = landmark_vitpose(
            pose_results,
            f"{side}_big_toe",
            dataset_name=DATASET,
        )
        side_heel, _ = landmark_vitpose(
            pose_results,
            f"{side}_heel",
            dataset_name=DATASET,
        )

        side_knee, _ = landmark_vitpose(
            pose_results,
            f"{side}_knee",
            dataset_name=DATASET,
        )

        side_small_toe, _ = landmark_vitpose(
            pose_results,
            f"{side}_small_toe",
            dataset_name=DATASET,
        )

        # orientation check
        if side_big_toe[0] > side_heel[0]:
            x_min = side_knee[0]
            x_max = w
            x_min = max(x_min, body_part[0] - body_part[0] * percentage)
            x_max = min(x_max, body_part[0] + body_part[0] * percentage)
            assert side_first_view == "right"
        else:
            x_min = 0
            x_max = side_knee[0]
            x_min = max(x_min, body_part[0] - body_part[0] * percentage)
            x_max = min(x_max, body_part[0] + body_part[0] * percentage)
            assert side_first_view == "left"

        # non può andare sotto i piedi, in quanto nel nostro sistema i piedi sono il pavimento
        y_min = 0
        y_max = side_small_toe[1]

        # y_min, x_min = torch.min(pose_results, axis=0)[0]
        # y_max, x_max = torch.max(pose_results, axis=0)[0]
        y_min = max(y_min, body_part[1] - body_part[1] * percentage)

        y_max = min(y_max, body_part[1] + body_part[1] * percentage)

        y_min, x_min, y_max, x_max = int(y_min), int(x_min), int(y_max), int(x_max)
        if vertical:
            y = random.randint(y_min, y_max)
        if horizontal:
            x = random.randint(x_min, x_max)

        return x, y

    list_video_time_event = [start_frame, end_frame]
    temp_frames = get_specific_frame(video_path, list_video_time_event)
    assert len(temp_frames) == 2
    frame_start_lifting = temp_frames[0].copy()
    frame_end_lifting = temp_frames[1].copy()

    pose_results_start_lifting = vitpose.inference(cv2.cvtColor(frame_start_lifting, cv2.COLOR_BGR2RGB))[0]
    pose_results_end_lifting = vitpose.inference(cv2.cvtColor(frame_end_lifting, cv2.COLOR_BGR2RGB))[0]

    side_first_view = get_most_common_side_first_view(frame_information.values())

    original_pose = [
        deepcopy(pose_results_start_lifting),
        deepcopy(pose_results_end_lifting),
    ]
    corrected_pose_results, draws, text_recommendation = correcting_posture_random(
        pose_results_start_lifting, pose_results_end_lifting, side_first_view
    )
    if corrected_pose_results is not None and draws is not None:
        if not os.path.exists(f"{output_folder}/recommendation/"):
            os.makedirs(f"{output_folder}/recommendation/")
        cv2.imwrite(
            f"{output_folder}/recommendation/original_pose.jpg", draws["original_pose"]
        )
        cv2.imwrite(
            f"{output_folder}/recommendation/only_hand_moved.jpg", draws["only_hand_moved"]
        )
        cv2.imwrite(
            f"{output_folder}/recommendation/recommended_pose.jpg",
            draws["recommended_pose"],
        )
        cv2.imwrite(f"{output_folder}/recommendation/overlap_poses.jpg", draws["overlap_poses"])

        # write text on file
        with open(f"{output_folder}/recommendation/recommendation.txt", "w") as f:
            for text in text_recommendation:
                f.write(f"{text}\n")

        # f = open(f"{output_folder}/test_llama_original_pose", "wb")
        # pickle.dump(original_pose, f)
        # f.close()

        # f = open(f"{output_folder}/test_llama_corrected_pose", "wb")
        # pickle.dump(corrected_pose_results, f)
        # f.close()

    # Garbage collection for VRAM
    del model
    del vitpose
    gc.collect()
    torch.cuda.empty_cache()

    # progress bar update
    pb_video.update()
    break

print("statistics on estimation: errors in difference")
# for different videos, i would to compute the mean of the errors
mean_error_results = {}
for video_name, error_in_video in error_results_dict.items():
    for niosh_attribute, value in error_in_video.items():
        if niosh_attribute not in mean_error_results:
            mean_error_results[niosh_attribute] = {}
        for reference_point, estimation_info in value.items():
            if reference_point not in mean_error_results[niosh_attribute]:
                mean_error_results[niosh_attribute][reference_point] = {}
            for estimation_type, estimation_error in estimation_info.items():
                if estimation_type not in mean_error_results[niosh_attribute][reference_point]:
                    mean_error_results[niosh_attribute][reference_point][estimation_type] = []
                mean_error_results[niosh_attribute][reference_point][estimation_type].append(
                    estimation_error
                )

for niosh_attribute, value in mean_error_results.items():
    for reference_point, estimation_info in value.items():
        for estimation_type, estimations_values in estimation_info.items():
            mean_error_results[niosh_attribute][reference_point][estimation_type] = sum(
                estimations_values
            ) / len(estimations_values)

best_estimation = {}
for niosh_attribute, value in mean_error_results.items():
    best_estimation[niosh_attribute] = {}
    for reference_point, estimation_info in value.items():
        estimation_info = dict(sorted(estimation_info.items(), key=lambda item: item[1]))
        best_estimation[niosh_attribute][reference_point] = {}        
        for enumerate_index, estimation_type in enumerate(estimation_info.keys()):
            print(
                f"{niosh_attribute} - {reference_point} - {estimation_type}: {mean_error_results[niosh_attribute][reference_point][estimation_type]}"
            )        
            if enumerate_index == 0:                    
                best_estimation_types = [
                    k
                    for k, v in estimation_info.items()
                    if v == mean_error_results[niosh_attribute][reference_point][estimation_type]
                ]
                for best_estimation_type in best_estimation_types:
                    best_estimation[niosh_attribute][reference_point][best_estimation_type] = (
                        mean_error_results[niosh_attribute][reference_point][estimation_type]
                    )

print("----------------------------------------")
print("best estimation for each niosh attribute")

for k_niosh_attribute, value in best_estimation.items():
    for reference_point, v_estimation_type in value.items():
        for k_estimation_type, v_mean_error in v_estimation_type.items():
            print(
                f"best estimation for {k_niosh_attribute}-{reference_point} is {k_estimation_type} with error {v_mean_error}"
            )

print("----------------------------------------")
print("recommended weight metrics")

rwl_pred = np.array(rwl_dict["predicted_values"])
rwl_true = np.array(rwl_dict["ground_truth_values"])
# print("predicted values", rwl_pred)
# print("ground truth values", rwl_true)

mae = np.mean(np.abs(rwl_pred - rwl_true))
rmse = np.sqrt(np.mean((rwl_pred - rwl_true) ** 2))


print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
