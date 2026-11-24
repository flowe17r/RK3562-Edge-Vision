import os

from rknn.api import RKNN

ONNX = 'yolov5s.onnx'
CALIB_DIR = 'calib'
DATASET = 'dataset_calib.txt'


def make_dataset():
    names = sorted(n for n in os.listdir(CALIB_DIR) if n.endswith('.jpg'))
    with open(DATASET, 'w') as f:
        for n in names:
            f.write('%s/%s\n' % (CALIB_DIR, n))
    print('dataset: %d images -> %s' % (len(names), DATASET))


def export(do_quant, out):
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3562', mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
    if rknn.load_onnx(model=ONNX) != 0:
        raise SystemExit('load_onnx failed')
    if do_quant:
        if rknn.build(do_quantization=True, dataset=DATASET) != 0:
            raise SystemExit('build(int8) failed')
    else:
        if rknn.build(do_quantization=False) != 0:
            raise SystemExit('build(fp16) failed')
    if rknn.export_rknn(out) != 0:
        raise SystemExit('export %s failed' % out)
    rknn.release()
    print('exported %s (%.2f MB)' % (out, os.path.getsize(out) / 1048576.0))


if __name__ == '__main__':
    make_dataset()
    export(True, 'yolov5s_int8_calib.rknn')
    export(False, 'yolov5s_fp16.rknn')
    print('done')