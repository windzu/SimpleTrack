'''
Author: windzu windzu1@gmail.com
Date: 2023-04-07 11:25:50
LastEditors: windzu windzu1@gmail.com
LastEditTime: 2023-04-07 19:04:55
Description: 
Copyright (c) 2023 by windzu, All Rights Reserved. 
'''
import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import WaymoLoader

# ros
import rospy
from sensor_msgs.msg import PointCloud2
from tf2_geometry_msgs import PoseStamped

# local msg
from autodriver_msgs import BodyStatus
from autodriver_msgs import DetectedObjectArray


def load_gt_bboxes(gt_folder, data_folder, segment_name, type_token):
    gt_info = np.load(
        os.path.join(gt_folder, '{:}.npz'.format(segment_name)),
        allow_pickle=True)
    ego_info = np.load(
        os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)),
        allow_pickle=True)
    bboxes, ids, inst_types = gt_info['bboxes'], gt_info['ids'], gt_info[
        'types']
    gt_ids, gt_bboxes = utils.inst_filter(
        ids, bboxes, inst_types, type_field=[type_token], id_trans=True)

    ego_keys = sorted(utils.str2int(ego_info.keys()))
    egos = [ego_info[str(key)] for key in ego_keys]
    gt_bboxes = gt_bbox2world(gt_bboxes, egos)
    return gt_bboxes, gt_ids


def gt_bbox2world(bboxes, egos):
    frame_num = len(egos)
    for i in range(frame_num):
        ego = egos[i]
        bbox_num = len(bboxes[i])
        for j in range(bbox_num):
            bboxes[i][j] = BBox.bbox2world(ego, bboxes[i][j])
    return bboxes


def frame_visualization(bboxes,
                        ids,
                        states,
                        gt_bboxes=None,
                        gt_ids=None,
                        pc=None,
                        dets=None,
                        name=''):
    visualizer = visualization.Visualizer2D(name=name, figsize=(12, 12))
    if pc is not None:
        visualizer.handler_pc(pc)
    for _, bbox in enumerate(gt_bboxes):
        visualizer.handler_box(bbox, message='', color='black')
    dets = [d for d in dets if d.s >= 0.1]
    for det in dets:
        visualizer.handler_box(
            det, message='%.2f' % det.s, color='green', linestyle='dashed')
    for _, (bbox, id, state) in enumerate(zip(bboxes, ids, states)):
        if Validity.valid(state):
            visualizer.handler_box(bbox, message=str(id), color='red')
        else:
            visualizer.handler_box(bbox, message=str(id), color='light_blue')
    visualizer.show()
    visualizer.close()


def sequence_mot(configs,
                 data_loader: WaymoLoader,
                 sequence_id,
                 gt_bboxes=None,
                 gt_ids=None,
                 visualize=False):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    IDs, bboxes, states, types = list(), list(), list(), list()
    for frame_index in range(data_loader.cur_frame, frame_num):
        print('TYPE {:} SEQ {:} Frame {:} / {:}'.format(
            data_loader.type_token, sequence_id + 1, frame_index + 1,
            frame_num))

        # input data
        frame_data = next(data_loader)

        # example of frame_data
        # frame_data['time_stamp'] : <class 'float'> 1522688014.970187 此帧时间戳单位：秒
        # frame_data['ego'] ： <class 'numpy.ndarray'> (4, 4) 4x4矩阵 ego to map
        #   [[-1.26406193e-01 -9.91162935e-01 -4.02182839e-02  7.21326592e+03]
        #   [ 9.91978565e-01 -1.26302259e-01 -5.12494962e-03 -1.57112326e+03]
        #   [-0.00000000e+00 -4.05435010e-02  9.99177774e-01  2.11957000e+02]
        #   [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
        # frame_data['pc'] : <class 'numpy.ndarray'> (N, 3)  N为点云数目 3为xyz 单位 cm
        #   [[ 7210.6241365  -1617.7395749    215.87900195]
        #   [ 7201.37000925 -1629.35309694   216.01152451]
        #   [ 7201.53804067 -1627.11640549   215.93148926]
        #   ...
        #   [ 7213.46680204 -1572.30159939   211.9811475 ]
        #   [ 7213.45316792 -1572.2338711    211.97320182]
        #   [ 7213.46086557 -1572.26241484   211.98006699]]
        # frame_data['det_types'] : <class 'list'> (N,) N为检测框数目 具体数值代表检测框类型 1 为车辆 2 为行人 3 为自行车
        # frame_data['dets'] : <class 'list'> (N,) N为检测框数目 list每一个元素为 np.array(8,),
        #   8为[x1, y1, x2, y2, x3, y3, x4, y4]
        #   [
        #       array([ 7.19889407e+03, -1.57989347e+03,  2.12046238e+02, -1.45398743e+00,4.70701504e+00,  2.00523376e+00,  1.51633191e+00,  9.61084366e-01]),
        #       array([ 7.19432788e+03, -1.55129626e+03,  2.12077589e+02, -1.36406784e+00,4.49491453e+00,  1.97391510e+00,  1.51418924e+00,  9.57685351e-01]),
        #       array([ 7.20176441e+03, -1.61714562e+03,  2.12037916e+02, -1.48364371e+00,4.70097828e+00,  2.00820899e+00,  1.52170205e+00,  9.42870736e-01]),
        #   ]
        # frame_data['aux_info'] : <class 'dict'> {'is_key_frame': True, 'velos': None} 关键帧似乎有特殊处理
        for key, value in frame_data.items():
            # print key dan value type
            print(key, type(value))

        frame_data = FrameData(
            dets=frame_data['dets'],
            ego=frame_data['ego'],
            pc=frame_data['pc'],
            det_types=frame_data['det_types'],
            aux_info=frame_data['aux_info'],
            time_stamp=frame_data['time_stamp'])

        # mot
        results = tracker.frame_mot(frame_data)
        result_pred_bboxes = [trk[0] for trk in results]
        result_pred_ids = [trk[1] for trk in results]
        result_pred_states = [trk[2] for trk in results]
        result_types = [trk[3] for trk in results]

        # visualization
        if visualize:
            frame_visualization(
                result_pred_bboxes,
                result_pred_ids,
                result_pred_states,
                gt_bboxes[frame_index],
                gt_ids[frame_index],
                frame_data.pc,
                dets=frame_data.dets,
                name='{:}_{:}_{:}'.format(args.name, sequence_id, frame_index))

        # wrap for output
        IDs.append(result_pred_ids)
        result_pred_bboxes = [
            BBox.bbox2array(bbox) for bbox in result_pred_bboxes
        ]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)
    return IDs, bboxes, states, types


def main(name,
         obj_type,
         config_path,
         data_folder,
         det_data_folder,
         result_folder,
         gt_folder,
         start_frame=0,
         token=0,
         process=1):
    summary_folder = os.path.join(result_folder, 'summary', obj_type)
    # simply knowing about all the segments
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))
    print(file_names[0])

    # load model configs
    configs = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    if obj_type == 'vehicle':
        type_token = 1
    elif obj_type == 'pedestrian':
        type_token = 2
    elif obj_type == 'cyclist':
        type_token = 4
    for file_index, file_name in enumerate(file_names[:]):
        if file_index % process != token:
            continue
        print('START TYPE {:} SEQ {:} / {:}'.format(obj_type, file_index + 1,
                                                    len(file_names)))
        segment_name = file_name.split('.')[0]
        data_loader = WaymoLoader(configs, [type_token], segment_name,
                                  data_folder, det_data_folder, start_frame)
        gt_bboxes, gt_ids = load_gt_bboxes(gt_folder, data_folder,
                                           segment_name, type_token)

        # real mot happens here
        ids, bboxes, states, types = sequence_mot(configs, data_loader,
                                                  file_index, gt_bboxes,
                                                  gt_ids, args.visualize)
        np.savez_compressed(
            os.path.join(summary_folder, '{}.npz'.format(segment_name)),
            ids=ids,
            bboxes=bboxes,
            states=states)


class TestTrack:

    def __init__(self, args):
        pass


global_gnss_pose = PoseStamped()
global_body_status = BodyStatus()


def pc_callback(msg):
    pass


def objects_callback(msg):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_name', type=str, default='simple_track')
    parser.add_argument('--pc_topic', type=str, default='/lidar_points/top')

    args = parser.parse_args()

    # init ros node
    rospy.init_node(args.node_name, anonymous=True)
    pc_sub = rospy.Subscriber(args.pc_topic, PointCloud2, pc_callback)
    objects_sub = rospy.Subscriber('objects', Objects, objects_callback)
    gnss_sub = rospy.Subscriber('gnss', Gnss, gnss_callback)
    body_status_sub = rospy.Subscriber('body_status', BodyStatus,
                                       body_status_callback)
    rospy.spin()

    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        for token in range(args.process):
            result = pool.apply_async(
                main,
                args=(args.name, args.obj_type, args.config_path,
                      args.data_folder, det_data_folder, result_folder,
                      args.gt_folder, 0, token, args.process))
