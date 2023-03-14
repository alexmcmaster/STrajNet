#!/usr/bin/env python3

import sys
import zlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf

from waymo_open_dataset.protos import occupancy_flow_submission_pb2

out_dir = "viz"

def _parse_image_function(example_proto):
    # FIXME: ideally we would import this from train.py, but since that file
    #        is intended as a script, importing it causes much of it to run,
    #        which we don't want.
    feature = {
        'centerlines': tf.io.FixedLenFeature([], tf.string),
        'actors': tf.io.FixedLenFeature([], tf.string),
        #'occl_actors': tf.io.FixedLenFeature([], tf.string),
        'ogm': tf.io.FixedLenFeature([], tf.string),
        'map_image': tf.io.FixedLenFeature([], tf.string),
        'gt_obs_ogm': tf.io.FixedLenFeature([], tf.string),
        #'gt_occ_ogm': tf.io.FixedLenFeature([], tf.string),
        #'gt_flow': tf.io.FixedLenFeature([], tf.string),
        #'origin_flow': tf.io.FixedLenFeature([], tf.string),
        #'vec_flow':tf.io.FixedLenFeature([], tf.string),
        # 'byc_flow':tf.io.FixedLenFeature([], tf.string)
    }
    new_dict = {}
    d =  tf.io.parse_single_example(example_proto, feature)
    new_dict['centerlines'] = tf.cast(tf.reshape(tf.io.decode_raw(
        d['centerlines'],tf.float64),[256,10,7]),tf.float32)
    new_dict['actors'] = tf.cast(tf.reshape(tf.io.decode_raw(
        d['actors'],tf.float64),[48,11,8]),tf.float32)
    new_dict['ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(
        d['ogm'],tf.bool),tf.float32),[512,512,11,2])
    new_dict['gt_obs_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(
        d['gt_obs_ogm'],tf.bool),tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]
    new_dict['map_image'] = tf.cast(tf.reshape(tf.io.decode_raw(
        d['map_image'],tf.int8),[256,256,3]),tf.float32) / 256
    return new_dict

def _make_alpha(im, thresh):
    ret = np.array(im)
    idx = im < thresh
    ret[idx] = 0
    return ret.reshape(im.shape[:2])

print(f"Reading {sys.argv[1]}")
gt_dataset = tf.data.TFRecordDataset(sys.argv[1], compression_type='')
gt_dataset = gt_dataset.map(_parse_image_function)
gt_dataset = list(gt_dataset.batch(1))

print(f"Reading {sys.argv[2]}")
with open(sys.argv[2], "rb") as f:
    raw = f.read()

sub = occupancy_flow_submission_pb2.ChallengeSubmission()
sub.ParseFromString(raw)

SHOW_MAP = True
ALPHA_CONTRAST = 10

for i, scenario in enumerate(sub.scenario_predictions):
    print(f"Processing scenario {scenario.scenario_id}...")
    fig, axs = plt.subplots(2, 8)
    fig.set_size_inches((20, 6))
    fig.suptitle("STrajNet: Predicted Occupancy Map of Observed Vehicles",
                size="x-large")
    axs[0, 0].set_ylabel("Predicted")
    axs[1, 0].set_ylabel("Ground Truth")
    for a in range(8):
        axs[1, a].set_xlabel(f"Waypoint {a+1}")
        fig.colorbar(cm.ScalarMappable(), ax=axs[0, a], location="top", pad=0.05, ticks=[0, 1])
        for b in range(2):
            axs[b, a].set_xticks(list())
            axs[b, a].set_yticks(list())

    for j, wp in enumerate(scenario.waypoints):
        obs_occu_bytes = zlib.decompress(wp.observed_vehicles_occupancy)
        obs_occu_quant = np.frombuffer(obs_occu_bytes, dtype=np.uint8)
        obs_occu = (obs_occu_quant / 255).reshape((256, 256, 1))
        gt_occu = gt_dataset[i]["gt_obs_ogm"][0, j, :, :, 0]
        map_image = gt_dataset[i]["map_image"][0]
        map_image = np.clip(map_image, 0, 1)
        map_image = np.mean(map_image ** 2, axis=2)
        map_image[map_image > 0] = 0.5
        if SHOW_MAP:
            np.save("q", map_image)
            pred_alpha = (obs_occu ** (1/ALPHA_CONTRAST)).reshape(obs_occu.shape[:2])
            axs[0, j].imshow(map_image, cmap="gray", vmin=0, vmax=1)
            axs[0, j].imshow(obs_occu, alpha=pred_alpha, vmin=0, vmax=1)
            gt_alpha = gt_occu
            axs[1, j].imshow(map_image, cmap="gray", vmin=0, vmax=1)
            axs[1, j].imshow(gt_occu, alpha=gt_alpha, vmin=0, vmax=1)
        else:
            axs[0, j].imshow(obs_occu, vmin=0, vmax=1)
            axs[1, j].imshow(gt_occu, vmin=0, vmax=1)

    plt.tight_layout()
    #plt.show()
    #sys.exit(0)
    plt.savefig(f"{out_dir}/{scenario.scenario_id}.png")
