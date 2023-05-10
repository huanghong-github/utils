from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import imgviz
import numpy as np
from matplotlib import pyplot as plt
from path import Path
from PIL import Image, ImageDraw

Coord = namedtuple("Coordinate", ("x", "y"))


@dataclass()
class Polygon:
    Coords: List[Coord]
    name: str


class Label:
    def __init__(self, image_path: Path, label_path: Path, labels: List[str]) -> None:
        self.image_path: Path = image_path
        self.label_path: Path = label_path
        self.stem: str = image_path.stem
        self.root: Path = image_path.parent.parent
        self.img: np.ndarray[int, np.dtype[np.generic]] = cv2.imread(image_path)
        self.height, self.width, self.depth = self.img.shape
        self.polygons: List[Polygon] = None
        self.labels: List[str] = labels
        self.voc_labels = ["background"] + list(labels)

    def unload(self, mode="yolo"):
        func = globals().get(f"to_{mode}")
        func(self)

    def load(self, mode="yolo", **kwargs):
        func = globals().get(f"from_{mode}")
        func(self, **kwargs)

    def convert(self, func):
        self.polygons = func(self)

    def cv_line_show(self):
        import cv2

        im = self.img.copy()
        for polygon in self.polygons:
            coods = np.array(polygon.Coords, dtype=np.int32)
            cv2.polylines(
                img=im,
                pts=coods.reshape((1, -1, 1, 2)),
                isClosed=True,
                color=(255, 0, 0),
            )
            cv2.putText(
                img=im,
                text=polygon.name,
                org=coods.min(0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,  # 字体大小
                color=(255, 0, 0),
                thickness=1,  # 线的粗细
                lineType=cv2.LINE_AA,
            )
        plt.imshow(im)
        plt.show()

    def cv_show(self):
        import cv2

        im = self.img.copy()
        for polygon in self.polygons:
            coods = np.array(polygon.Coords, dtype=np.int32)
            cv2.fillPoly(img=im, pts=coods.reshape((1, -1, 1, 2)), color=(255, 0, 0))
            cv2.putText(
                img=im,
                text=polygon.name,
                org=coods.min(0),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,  # 字体大小
                color=(255, 0, 0),
                thickness=1,  # 线的粗细
                lineType=cv2.LINE_AA,
            )
        plt.imshow(im)
        plt.show()

    def pil_show(self):
        mask = np.zeros([label.height, label.width], dtype=np.uint8)
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        for polygon in label.polygons:
            xy = [Coord(*map(int, coord)) for coord in polygon.Coords]
            draw.polygon(xy=xy, fill=255, outline=255)
        mask = np.array(mask)
        plt.imshow(mask)
        plt.show()

    def viz_cls_show(self):
        cls, _ = shapes_to_label(self)
        clsv = imgviz.label2rgb(
            cls,
            imgviz.rgb2gray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)),
            label_names=self.voc_labels,
            font_size=15,
            loc="rb",
        )
        plt.imshow(clsv)
        plt.show()

    def viz_ins_show(self):
        _, ins = shapes_to_label(self)
        instance_ids = np.unique(ins)
        instance_names = [str(i) for i in range(max(instance_ids) + 1)]
        insv = imgviz.label2rgb(
            ins,
            imgviz.rgb2gray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)),
            label_names=instance_names,
            font_size=15,
            loc="rb",
        )
        plt.imshow(insv)
        plt.show()


# ----------------------------------yolo--------------------------------------


def to_yolo(label: Label):
    target = label.root / "yolo"
    if not target.exists():
        target.mkdir_p()
    target_path = target / f"{label.stem}.txt"
    res = []
    for polygon in label.polygons:
        cls = label.labels.index(polygon.name)
        xy += " ".join(
            f"{x/label.width:<08f} {y/label.height:<08f}" for x, y in polygon.Coords
        )
        res.append(f"{cls:d} {xy}")
    target_path.write_lines(res)


def from_yolo(label: Label):
    polygons = []
    for line in label.label_path.lines():
        items = [float(item) for item in line.strip().split()]
        name = label.labels[int(items[0])]
        Coords = [
            Coord(items[i] * label.width, items[i + 1] * label.height)
            for i in range(1, len(items), 2)
        ]
        polygons.append(Polygon(Coords, name))
    label.polygons = polygons


# -----------------------------------voc--------------------------------------


def shapes_to_label(label: Label):
    import uuid

    cls = np.zeros([label.height, label.width], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for polygon in label.polygons:
        cls_name = polygon.name
        instance = (cls_name, uuid.uuid1())

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label.voc_labels.index(cls_name)

        mask = np.zeros([label.height, label.width], dtype=np.uint8)
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        xy = [Coord(*map(int, coord)) for coord in polygon.Coords]
        draw.polygon(xy=xy, fill=1, outline=1)
        mask = np.array(mask, dtype=bool)

        cls[mask] = cls_id
        ins[mask] = ins_id
    return cls, ins


def lblsave(filename, lbl):
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        lbl_pil.putpalette(colormap.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )


def to_voc(label: Label):
    cls_dir = label.root / "voc/Segmentation"
    if not cls_dir.exists():
        cls_dir.makedirs_p()
    ins_dir = label.root / "voc/SegmentationObject"
    if not ins_dir.exists():
        ins_dir.makedirs_p

    cls, ins = shapes_to_label(label)
    lblsave(cls_dir / f"{label.stem}.png", cls)
    lblsave(ins_dir / f"{label.stem}.png", ins)


def from_voc(
    label: Label,
    convertDict: Dict[Tuple[int, int, int], str],
    min_points: int = 20,
    stride: int = 10,
):
    """
    # convertDict: 转换字典,如{(255, 0, 0): "capacitor"},像素对应标签
    # min_points: polygon的最小点数,小于的全部放弃
    # stride: 步长,轮廓取点很密集,可以用步长降低点数
    """
    from skimage import measure

    mask = cv2.imread(label.label_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    polygons = []
    for color, name in convertDict.items():
        mask_c = mask == color
        mask_c = mask_c[:, :, 0] & mask_c[:, :, 1] & mask_c[:, :, 2]
        contours = measure.find_contours(mask_c, 0.5)
        for contour in contours:
            if len(contour) > min_points:
                polygons.append(Polygon(np.flip(contour)[::stride], name))
    label.polygons = polygons


# ----------------------------------labelme------------------------------------


def from_labelme(label: Label):
    import json

    data = json.loads(label.label_path.read_text())
    label.polygons = [
        Polygon(Coords=shape["points"], name=shape["label"]) for shape in data["shapes"]
    ]


if __name__ == "__main__":
    JPEG = Path(r"D:\dataset\Capacitors\newvoc\train\images\1_1.jpg")
    MASK = Path(r"D:\dataset\Capacitors\newvoc\Segmentation\1_1.png")
    label = Label(JPEG, MASK, ["capacitor"])
    label.load("voc", convertDict={(255, 0, 0): "capacitor"}, min_points=100)
    label.viz_cls_show()
