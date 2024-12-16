from tqdm import tqdm
from functools import partial
import argparse
import os
import numpy as np
from PIL import Image
import multiprocessing as mp
import json

import deva.vps_metrics.segmentation_and_tracking_quality as numpy_stq
from evaluation.pq_stat import PQStat

# 常量，用于 STQ
n_classes = 2  # 背景 (0) 和实例 (1)
ignore_label = 255
bit_shift = 16


def read_im(path):
    return np.array(Image.open(path))


def vpq_compute_single_core(categories, nframes, gt_pred_set):
    OFFSET = 256 * 256 * 256
    vpq_stat = PQStat()
    
    for idx in range(0, max(len(gt_pred_set) - nframes + 1, 1)):
        vid_pan_gt, vid_pan_pred = [], []
        
        for _, _, gt_name, pred_name, _ in gt_pred_set[idx:idx + nframes]:
            gt_pan = read_im(gt_name).astype(np.uint32)
            pred_pan = read_im(pred_name).astype(np.uint32)
            
            # 将非0像素作为实例 (类别1)，背景(0)忽略
            pan_gt = np.where(gt_pan != 0, 1, 0).astype(np.uint32)
            pan_pred = np.where(pred_pan != 0, 1, 0).astype(np.uint32)
            
            vid_pan_gt.append(pan_gt)
            vid_pan_pred.append(pan_pred)
        
        # tube-level 数据处理
        vid_pan_gt = np.stack(vid_pan_gt)
        vid_pan_pred = np.stack(vid_pan_pred)
        
        vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
        labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
        
        # 匹配实例，计算 TP、FP 和 FN
        gt_matched, pred_matched = set(), set()
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            
            if gt_id == 0 or pred_id == 0:  # 忽略背景
                continue
            
            union = np.sum(vid_pan_gt == gt_id) + np.sum(vid_pan_pred == pred_id) - intersection
            iou = intersection / union
            
            if iou > 0.5:  # 匹配成功
                vpq_stat[1].tp += 1
                vpq_stat[1].iou += iou
                gt_matched.add(gt_id)
                pred_matched.add(pred_id)
            else:
                vpq_stat[1].fp += 1
                vpq_stat[1].fn += 1
    
    return vpq_stat


def vpq_compute(gt_pred_split, categories, nframes, output_dir, num_processes):
    vpq_stat = PQStat()
    
    with mp.Pool(num_processes) as p:
        for tmp in tqdm(p.imap(partial(vpq_compute_single_core, categories, nframes), gt_pred_split),
                        total=len(gt_pred_split)):
            vpq_stat += tmp
    
    metrics = vpq_stat.pq_average(categories)
    vpq_all = 100 * metrics['pq']
    sq = 100 * metrics['sq']
    rq = 100 * metrics['rq']
    print(f"{nframes}-frame: PQ: {vpq_all:.2f}, SQ: {sq:.2f}, RQ: {rq:.2f}")
    
    save_name = os.path.join(output_dir, f'vpq_{nframes}.txt')
    with open(save_name, 'w') as f:
        f.write(f'VPQ: {vpq_all:.2f}\n')
        f.write(f'SQ: {sq:.2f}\n')
        f.write(f'RQ: {rq:.2f}\n')
    return vpq_all, sq, rq


def eval_vpq(submit_dir, truth_dir, pan_gt_json_file, num_processes):
    output_dir = submit_dir
    
    with open(os.path.join(submit_dir, 'pred.json'), 'r') as f:
        pred_jsons = json.load(f)
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)
    
    categories = {1: {'id': 1, 'name': 'instance', 'isthing': 1}}
    pred_annos = pred_jsons['annotations']
    gt_annos = gt_jsons['annotations']
    
    gt_j = {g_a['video_id']: g_a['annotations'] for g_a in gt_annos}
    pred_j = {p_a['video_id']: p_a['annotations'] for p_a in pred_annos}
    
    gt_pred_split = []
    for video_images in gt_jsons['videos']:
        video_id = video_images['video_id']
        gt_image_jsons = video_images['images']
        gt_js = gt_j[video_id]
        pred_js = pred_j[video_id]
        
        gt_names = [os.path.join(truth_dir, video_id, img_j['file_name']) for img_j in gt_image_jsons]
        pred_names = [os.path.join(submit_dir, 'pan_pred', video_id, img_j['file_name']) for img_j in gt_image_jsons]
        
        gt_pred_split.append(list(zip(gt_js, pred_js, gt_names, pred_names, gt_image_jsons)))
    
    for nframes in [1, 2, 4, 6, 8, 10, 999]:
        vpq_compute(gt_pred_split, categories, nframes, output_dir, num_processes)


def eval_stq(submit_dir, truth_dir, pan_gt_json_file):
    output_dir = submit_dir
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)
    
    categories = gt_jsons['categories']
    
    thing_list_ = []
    for cate_ in categories:
        cat_id = cate_['id']
        isthing = cate_['isthing']
        if isthing:
            thing_list_.append(cat_id)
    
    stq_metric = numpy_stq.STQuality(n_classes, thing_list_, ignore_label, bit_shift, 2**24)
    
    pred_annos = pred_jsons['annotations']
    pred_j = {}
    for p_a in pred_annos:
        pred_j[p_a['video_id']] = p_a['annotations']
    gt_annos = gt_jsons['annotations']
    gt_j = {}
    for g_a in gt_annos:
        gt_j[g_a['video_id']] = g_a['annotations']
    
    pbar = tqdm(gt_jsons['videos'])
    for seq_id, video_images in enumerate(pbar):
        video_id = video_images['video_id']
        pbar.set_description(video_id)
        
        gt_image_jsons = video_images['images']
        gt_js = gt_j[video_id]
        pred_js = pred_j[video_id]
        assert len(gt_js) == len(pred_js)
        
        gt_pans = []
        pred_pans = []
        for imgname_j in gt_image_jsons:
            imgname = imgname_j['file_name']
            image = np.array(Image.open(os.path.join(submit_dir, 'pan_pred', video_id, imgname)))
            pred_pans.append(image)
            image = np.array(Image.open(os.path.join(truth_dir, video_id, imgname)))
            gt_pans.append(image)
        
        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(
                list(zip(gt_js, pred_js, gt_pans, pred_pans, gt_image_jsons))):
            #### Step1. 收集帧级别的 pan_gt, pan_pred 等
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            
            # 现在将图像视为二维，不考虑RGB通道
            pan_gt = gt_pan  # 直接使用二维数组
            pan_pred = pred_pan  # 直接使用二维数组
            
            # 只关注非零且属于类别1的像素（忽略类别0）
            ground_truth_instance = np.zeros_like(pan_gt)
            ground_truth_semantic = np.zeros_like(pan_gt)
            ground_truth_instance[pan_gt != 0] = 1  # 只有一个实例，id 0
            ground_truth_semantic[pan_gt != 0] = 1  # 类别1为唯一实例
            ground_truth = ((ground_truth_semantic << bit_shift) + ground_truth_instance)
            
            prediction_instance = np.zeros_like(pan_pred)
            prediction_semantic = np.zeros_like(pan_pred)
            prediction_instance[pan_pred != 0] = 1  # 只有一个实例，id 0
            prediction_semantic[pan_pred != 0] = 1  # 类别1为唯一实例
            prediction = ((prediction_semantic << bit_shift) + prediction_instance)
            
            # 在调用 update_state 之前确保形状相同
            if ground_truth.shape != prediction.shape:
                raise ValueError(
                    f"Shape mismatch: ground_truth shape {ground_truth.shape}, prediction shape {prediction.shape}")
            
            # 仅更新属于类别1（实例）的像素状态
            stq_metric.update_state(ground_truth.astype(dtype=np.int32),
                                    prediction.astype(dtype=np.int32), seq_id)
    
    result = stq_metric.result()
    print('*' * 100)
    print('STQ : {}'.format(result['STQ']))
    print('AQ :{}'.format(result['AQ']))
    print('IoU:{}'.format(result['IoU']))
    print('STQ_per_seq')
    print(result['STQ_per_seq'])
    print('AQ_per_seq')
    print(result['AQ_per_seq'])
    print('ID_per_seq')
    print(result['ID_per_seq'])
    print('Length_per_seq')
    print(result['Length_per_seq'])
    print('*' * 100)
    
    with open(os.path.join(submit_dir, 'stq.txt'), 'w') as f:
        f.write(f'STQ: {result["STQ"] * 100:.2f}\n')
        f.write(f'AQ : {result["AQ"] * 100:.2f}\n')
        f.write(f'IoU: {result["IoU"] * 100:.2f}\n')
        f.write(f'STQ_per_seq: {result["STQ_per_seq"]}\n')
        f.write(f'AQ_per_seq: {result["AQ_per_seq"]}\n')
        f.write(f'ID_per_seq: {result["ID_per_seq"]}\n')
        f.write(f'Length_per_seq: {result["Length_per_seq"]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VPQ Evaluation')
    parser.add_argument('--submit_dir', '-i', type=str, required=True, help='Test output directory')
    parser.add_argument('--dataset_dir', type=str, help='dataset directory.')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes for computation')
    args = parser.parse_args()
    submit_dir = args.submit_dir
    truth_dir = os.path.join(args.dataset_dir, 'panomasksRGB')
    pan_gt_json_file = os.path.join(args.dataset_dir, 'panoptic_gt_VIPSeg_test.json')
    eval_vpq(args.submit_dir, truth_dir, pan_gt_json_file, args.num_processes)
