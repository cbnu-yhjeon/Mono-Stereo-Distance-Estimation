#!/usr/bin/env bash
# KITTI Stereo 2015 · Object Detection(training만)을 $KITTI_ROOT(기본 ~/data/kitti)에 받고 푼다.
set -euo pipefail
ROOT="${KITTI_ROOT:-$HOME/data/kitti}"
URL=https://s3.eu-central-1.amazonaws.com/avg-kitti
mkdir -p "$ROOT/zips" "$ROOT/stereo2015" "$ROOT/object"
cd "$ROOT/zips"
for f in data_scene_flow.zip data_object_calib.zip data_object_label_2.zip data_object_image_2.zip data_object_image_3.zip; do
  curl -sSL -C - -o "$f" "$URL/$f" || true
done
unzip -qn data_scene_flow.zip 'training/image_2/*_10.png' 'training/image_3/*_10.png' 'training/disp_occ_0/*' 'training/disp_noc_0/*' -d "$ROOT/stereo2015"
for f in data_object_calib.zip data_object_label_2.zip data_object_image_2.zip data_object_image_3.zip; do
  unzip -qn "$f" 'training/*' -d "$ROOT/object"
done
echo "stereo2015: $(ls "$ROOT/stereo2015/training/disp_occ_0" | wc -l) / object: $(ls "$ROOT/object/training/label_2" | wc -l)"
