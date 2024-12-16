"""
将 VIPSeg 数据集转换为容易处理的格式，保存到文件 panoVIPSeg_categories.json
"""
import os
import json
import numpy as np
import PIL.Image as Image
from panopticapi.utils import IdGenerator, save_json
from multiprocessing import Pool

ROOT_DIR = 'resources/datasets/needle-seg-full'
source_dir = os.path.join(ROOT_DIR, 'panomasks')
target_dir = os.path.join(ROOT_DIR, 'panomasksRGB')
categories_json = os.path.join(ROOT_DIR, 'panoVIPSeg_categories.json')
with open(categories_json, 'r') as f:
    categories = json.load(f)

original_format_folder = source_dir
out_folder = os.path.join(target_dir)
categories_dict = {el['id']: el for el in categories}


def conversion_worker(video):
    videos_dic = {}
    video_id = video
    videos_dic['video_id'] = video_id
    
    images = []
    annotations = []
    id_generator = IdGenerator(categories_dict)
    instid2color = {}
    
    for imgname in sorted(os.listdir(os.path.join(original_format_folder, video))):
        original_format = np.array(Image.open(os.path.join(original_format_folder, video, imgname)))
        image_id = imgname.split('.')[0]
        image_filename = imgname
        images.append({"id": image_id,
                       "width": original_format.shape[1],
                       "height": original_format.shape[0],
                       "file_name": image_filename
                       })
        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
        
        l = np.unique(original_format)
        
        segm_info = {}
        
        for el in l:
            if el == 0:
                continue
            if el < 125:
                semantic_id = el
                is_crowd = 0
            else:
                semantic_id = el // 100
                is_crowd = 0
            semantic_id = semantic_id - 1
            if semantic_id not in categories_dict:
                print(semantic_id, video, l, imgname, list(categories_dict.keys()))
            if semantic_id > 255:
                print(semantic_id, video, l, imgname)
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            mask = (original_format == el)
            
            if el not in instid2color:
                segment_id, color = id_generator.get_id_and_color(semantic_id)
                instid2color[el] = (segment_id, color)
            else:
                segment_id, color = instid2color[el]
            
            pan_format[mask] = color
            segm_info[int(segment_id)] = \
                {"id": int(segment_id),
                 "category_id": int(semantic_id),
                 # "area": int(area),
                 "iscrowd": is_crowd
                 }
        if not os.path.exists(os.path.join(out_folder, video)):
            os.makedirs(os.path.join(out_folder, video))
        
        Image.fromarray(pan_format).save(os.path.join(out_folder, video, image_filename))
        # print('image saved {}'.format(os.path.join(out_folder,video, image_filename)))
        
        gt_pan = np.uint32(pan_format)
        pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
        #         print(np.unique(pan_gt))
        #         exit()
        labels, labels_cnt = np.unique(pan_gt, return_counts=True)
        gt_labels = [_ for _ in segm_info.keys()]
        gt_labels_set = set(gt_labels)
        for label, area in zip(labels, labels_cnt):
            if label == 0:
                continue
            if label not in gt_labels and label > 0:
                print('png label not in json labels.', label, video, gt_labels, l)
            segm_info[label]["area"] = int(area)
            gt_labels_set.remove(label)
        if len(gt_labels_set) != 0:
            raise KeyError('remaining gt_labels json')
        
        segm_info = [v for k, v in segm_info.items()]
        annotations.append({'image_id': image_id,
                            'file_name': image_filename,
                            "segments_info": segm_info
                            })
    v_anno = {'video_id': video_id, 'annotations': annotations}
    videos_dic['images'] = images
    return v_anno, videos_dic
    # return None


if __name__ == "__main__":
    out_file = os.path.join(ROOT_DIR, 'panoptic_gt_VIPSeg.json')
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    
    v_videos = []
    v_annotations = []
    pool = Pool(16)
    
    results = pool.map(conversion_worker, sorted(os.listdir(original_format_folder)), chunksize=8)
    
    for v_anno, videos_dic in results:
        v_videos.append(videos_dic)
        v_annotations.append(v_anno)
    
    d = {'videos': v_videos,
         'annotations': v_annotations,
         'categories': categories,
         }
    
    save_json(d, out_file)
    
    print(f'==> Saved json file at {out_file}')
