import os
import numpy as np
from MCAF.MCAF import MCAF
from EVAL.evaluation_script import evaluate
import cv2

JSON_GT_FILE = os.path.join('EVAL/kaist_annotations_test20.json' )
result_filename = 'Result/det-test-all.txt'
phase = "Multispectral"

all_detectors_names = list(['rgb', 'ir', 'fusion'])
mcaf = MCAF(all_detectors_names)

det_branch = np.load('EVAL/det_branch.npy', allow_pickle=True)


def save_results(results, result_filename):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    Parameters
    ----------
    results: Dict
        Detection results for each image_id: {image_id: box_xywh + score}
    result_filename: str
        Full path of result file name

    """

    if not result_filename.endswith('.txt'):
        result_filename += '.txt'

    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                f.write(f'{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.8f}\n')

ids = []
image_set = 'EVAL/test-all-20.txt'
for line in open(image_set):
    ids.append(line.strip().replace('/', '_'))

results = dict()
for det in det_branch:
    file_name, bounding_boxes, class_scores, labels, uncertainty = det
    bounding_box, labels, class_score, uncertainty = mcaf.MCAF_result(bounding_boxes, class_scores, labels, uncertainty)
    bounding_box += 1
    
    if bounding_box.shape[0] > 0:
        bounding_box[:, 2] -= bounding_box[:, 0]
        bounding_box[:, 3] -= bounding_box[:, 1]
        image_id = ids.index(file_name)
        results[image_id+1] = np.hstack([bounding_box, class_score[:, np.newaxis]])

save_results(results, result_filename)
evaluate(JSON_GT_FILE, result_filename, phase)


# print('-'*15)
# print('---comparision methods---')
# print('-'*15)
# result_filename = 'comparsion_methods/IAF-det-test-all.txt'
# evaluate(JSON_GT_FILE, result_filename, phase)

# result_filename = 'comparsion_methods/MSDS-det-test-all.txt'
# evaluate(JSON_GT_FILE, result_filename, phase)
