import os

from pydicom.data import get_testdata_file

from kaggle_covid import LABELS
from kaggle_covid.utils import convert_boxes_to_coco, generate_coco_annotations, load_image


def test_load_image():
    row = {"boxes": [dict(x=10, y=20, width=30, height=40)]}
    tmp_path = get_testdata_file("CT_small.dcm")
    img, meta = load_image(tmp_path, row, spacing=2.)
    assert img.shape == (64, 64)
    assert meta["interpret"] == 'MONOCHROME2'
    assert meta["boxes"][0] == dict(x=5., y=10., width=15., height=20.)


def test_convert_coco(tmpdir):
    row = {"name": "efgabc151", "class": 0, "boxes": [dict(x=8, y=16, width=32, height=64)]}
    tmp_path = get_testdata_file("CT_small.dcm")
    img, meta = load_image(tmp_path, row)
    assert img.shape == (128, 128)

    bb = convert_boxes_to_coco(meta, img.shape)
    assert bb[0] == {'cls': 0, 'x_center': 0.1875, 'y_center': 0.375, 'width': 0.25, 'height': 0.5}
    meta.update({"bboxes": bb})

    generate_coco_annotations([meta], LABELS, os.path.join(tmpdir, "coco.json"))
