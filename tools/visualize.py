import os
import sys

import numpy as np
import cv2


def parse_csv_annotations(filepath, num_classes=60):
    """
    Extract annotations from the csv file from one epoch produced in main.py
    :param filepath: path of the csv file
    :param num_classes: number of classes used
    :return: return a list[dict] of record, containing prediction with bbox and score and time in seconds
    """
    records = []

    with open(filepath, "r") as f:
        line = f.readline()
        cur_record = {"predictions": []}
        cur_time = -1
        cur_class = 1
        cur_bbox = None
        cur_scores = {}

        while len(line) > 0:
            line = line.strip().split(",")

            if cur_time == -1:
                cur_time = int(line[1])
            if cur_time != int(line[1]):
                cur_record["video"] = line[0]
                cur_record["time"] = cur_time
                records.append(cur_record)
                cur_record = {"predictions": []}
                cur_class = 1
                cur_scores = {}
                cur_time = int(line[1])

            if cur_class == 1:
                cur_bbox = np.array([float(line[2]), float(line[3]), float(line[4]), float(line[5])])

            cur_scores[int(line[6])] = float(line[7])
            cur_class += 1

            if cur_class > num_classes:
                cur_record["predictions"].append({"bbox": cur_bbox, "scores": cur_scores})
                cur_class = 1
                cur_scores = {}

            line = f.readline()

    return records


def get_bboxes_scores_from_record(record, width, height, idx_2_class, threshold):
    bboxes = np.stack([p["bbox"] for p in record["predictions"]], axis=0)
    bboxes[:, 0] *= width
    bboxes[:, 1] *= height
    bboxes[:, 2] *= width
    bboxes[:, 3] *= height

    bboxes = bboxes.astype(np.int32)

    scores = [list(p["scores"].items()) for p in record["predictions"]]
    final_scores = []
    for score in scores:
        bbox_score = []
        score.sort(key=lambda elt: elt[0])
        posture_idx = np.argmax([s[1] for s in score[:13]])

        bbox_score.append((idx_2_class[score[posture_idx][0]], score[posture_idx][1]))

        for i in range(13, len(score)):
            if score[i][1] > threshold:
                bbox_score.append((idx_2_class[score[i][0]], score[i][1]))

        final_scores.append(bbox_score)

    return bboxes, final_scores


def apply_annotations(root_path, records, idx_2_class, frame_rate=30, threshold=0.5,
                      interpolate=True, image_format="images_%06d.jpg",
                      line_width=2, font_scale=1.5):
    num_record = len(records)
    print(f"num records {num_record}")

    frame_per_clip = 3 * frame_rate + 1
    videos = []
    for idx, record in enumerate(records):
        print(f"applying records - {idx + 1} out of {num_record}", end='\r')
        start_frame = record["time"] * frame_rate + 1 - (frame_per_clip - 1) // 2
        mid_frame_low_bound = record["time"] * frame_rate + 1 - frame_rate // 2
        mid_frame_high_bound = record["time"] * frame_rate + 1 + frame_rate // 2

        is_last = idx == num_record - 1
        is_first = idx == 0

        video_path = os.path.join(root_path, record["video"])
        videos.append(record["video"])

        try:
            height, width = cv2.imread(os.path.join(video_path, image_format % start_frame)).shape[:2]
        except Exception as e:
            print(os.path.join(video_path, image_format % start_frame))
            raise e

        bboxes, scores = get_bboxes_scores_from_record(record, width, height, idx_2_class, threshold)

        frames = list(range(start_frame, start_frame + frame_per_clip))
        for idx_frame, cur_frame in enumerate(frames):
            if not is_first and cur_frame < mid_frame_low_bound:
                continue

            if not is_last and cur_frame >= mid_frame_high_bound:
                continue

            image_path = os.path.join(video_path, image_format % cur_frame)

            img = cv2.imread(image_path)
            for idx_bbox in range(bboxes.shape[0]):
                img = cv2.rectangle(img, tuple(bboxes[idx_bbox, :2]), tuple(bboxes[idx_bbox, 2:]), (0, 0, 255),
                                    thickness=line_width)
                start_origin = bboxes[idx_bbox, :2] + np.array([10, 20], dtype=np.int32)
                for score in scores[idx_bbox]:
                    img = cv2.putText(img, f"{score[0]} - {score[1]}", tuple(start_origin), cv2.FONT_HERSHEY_PLAIN,
                                      fontScale=font_scale, color=(255, 120, 0))
                    start_origin[1] += 20

            cv2.imwrite(image_path[:-4] + "_annotated.jpg", img)

    videos = set(videos)

    print("copying images without annotations")
    for video in videos:
        video_path = os.path.join(root_path, video)
        for entry in os.scandir(video_path):
            if entry.is_file() and entry.name.endswith(".jpg") and not entry.name.endswith("annotated.jpg"):
                annot_path = entry.path[:-4] + "_annotated.jpg"
                if not os.path.isfile(annot_path):
                    ret = os.system(f"cp {entry.path} {annot_path}")
                    assert ret == 0, f"Could not copy {entry.path} to {annot_path}"


if __name__ == "__main__":
    idx_class = [{'name': 'bend/bow (at the waist)', 'id': 1}, {'name': 'crouch/kneel', 'id': 3},
                 {'name': 'dance', 'id': 4}, {'name': 'fall down', 'id': 5}, {'name': 'get up', 'id': 6},
                 {'name': 'jump/leap', 'id': 7}, {'name': 'lie/sleep', 'id': 8}, {'name': 'martial art', 'id': 9},
                 {'name': 'run/jog', 'id': 10}, {'name': 'sit', 'id': 11}, {'name': 'stand', 'id': 12},
                 {'name': 'swim', 'id': 13}, {'name': 'walk', 'id': 14}, {'name': 'answer phone', 'id': 15},
                 {'name': 'carry/hold (an object)', 'id': 17}, {'name': 'climb (e.g., a mountain)', 'id': 20},
                 {'name': 'close (e.g., a door, a box)', 'id': 22}, {'name': 'cut', 'id': 24},
                 {'name': 'dress/put on clothing', 'id': 26},
                 {'name': 'drink', 'id': 27}, {'name': 'drive (e.g., a car, a truck)', 'id': 28},
                 {'name': 'eat', 'id': 29},
                 {'name': 'enter', 'id': 30}, {'name': 'hit (an object)', 'id': 34},
                 {'name': 'lift/pick up', 'id': 36},
                 {'name': 'listen (e.g., to music)', 'id': 37},
                 {'name': 'open (e.g., a window, a car door)', 'id': 38},
                 {'name': 'play musical instrument', 'id': 41}, {'name': 'point to (an object)', 'id': 43},
                 {'name': 'pull (an object)', 'id': 45},
                 {'name': 'push (an object)', 'id': 46}, {'name': 'put down', 'id': 47}, {'name': 'read', 'id': 48},
                 {'name': 'ride (e.g., a bike, a car, a horse)', 'id': 49}, {'name': 'sail boat', 'id': 51},
                 {'name': 'shoot', 'id': 52}, {'name': 'smoke', 'id': 54}, {'name': 'take a photo', 'id': 56},
                 {'name': 'text on/look at a cellphone', 'id': 57}, {'name': 'throw', 'id': 58},
                 {'name': 'touch (an object)', 'id': 59},
                 {'name': 'turn (e.g., a screwdriver)', 'id': 60}, {'name': 'watch (e.g., TV)', 'id': 61},
                 {'name': 'work on a computer', 'id': 62},
                 {'name': 'write', 'id': 63}, {'name': 'fight/hit (a person)', 'id': 64},
                 {'name': 'give/serve (an object) to (a person)', 'id': 65},
                 {'name': 'grab (a person)', 'id': 66}, {'name': 'hand clap', 'id': 67},
                 {'name': 'hand shake', 'id': 68}, {'name': 'hand wave', 'id': 69},
                 {'name': 'hug (a person)', 'id': 70}, {'name': 'kiss (a person)', 'id': 72},
                 {'name': 'lift (a person)', 'id': 73},
                 {'name': 'listen to (a person)', 'id': 74}, {'name': 'push (another person)', 'id': 76},
                 {'name': 'sing to (e.g., self, a person, a group)', 'id': 77},
                 {'name': 'take (an object) from (a person)', 'id': 78},
                 {'name': 'talk to (e.g., self, a person, a group)', 'id': 79},
                 {'name': 'watch (a person)', 'id': 80}]

    idx_2_class = {}
    for c in idx_class:
        idx_2_class[c['id']] = c['name']

    records = parse_csv_annotations(sys.argv[1])

    print("applying annotations")
    apply_annotations("data", records, idx_2_class, frame_rate=int(sys.argv[2]))
