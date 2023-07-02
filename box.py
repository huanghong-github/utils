from collections import namedtuple
from itertools import chain, product
from typing import List

import cv2
from matplotlib import pyplot as plt
from numpy import dtype, generic, ndarray
from path import Path

Box = namedtuple("Box", ("xmin", "ymin", "xmax", "ymax", "name"))


class Label:
    def __init__(self, image_path: Path, label_path: Path, labels: List[str]) -> None:
        self.image_path: Path = image_path
        self.label_path: Path = label_path
        self.stem: str = image_path.stem
        self.root: Path = image_path.parent.parent
        self.img: ndarray[int, dtype[generic]] = cv2.imread(image_path)
        self.height, self.width, self.depth = self.img.shape
        self.boxes: List[Box] = None
        self.labels: List[str] = labels

    def unload(self, mode="yolo"):
        func = globals().get(f"to_{mode}")
        func(self)

    def load(self, mode="yolo"):
        func = globals().get(f"from_{mode}")
        func(self)

    def convert(self, func):
        self.boxes = func(self)

    def cv_show(self):
        im = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        for box in self.boxes:
            cv2.rectangle(
                img=im,
                pt1=(int(box.xmin), int(box.ymin)),
                pt2=(int(box.xmax), int(box.ymax)),
                color=(255, 0, 0),
                thickness=2,
            )
            cv2.putText(
                img=im,
                text=box.name,
                org=(int(box.xmin), int(box.ymin) - 5),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7,  # 字体大小
                color=(255, 0, 0),
                thickness=1,  # 线的粗细
                lineType=cv2.LINE_AA,
            )
        plt.imshow(im)
        plt.show()

    def plt_show(self):
        im = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        for box in self.boxes:
            rect = plt.Rectangle(
                xy=(box.xmin, box.ymin),
                width=box.xmax - box.xmin,
                height=box.ymax - box.ymin,
                edgecolor="r",
                linewidth=1,
                fill=False,
            )
            plt.text(
                x=box.xmin,
                y=box.ymin,
                s=box.name,
                fontsize=10,
                color="r",
                style="italic",
                weight="light",
            )
            plt.gca().add_patch(rect)
        plt.imshow(im)
        plt.show()

    def viz_show(self):
        import imgviz

        viz = imgviz.instances2rgb(
            image=cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB),
            labels=[self.labels.index(box.name) for box in self.boxes],
            bboxes=[[box.ymin, box.xmin, box.ymax, box.xmax] for box in self.boxes],
            captions=[box.name for box in self.boxes],
            font_size=15,
        )
        plt.imshow(viz)
        plt.show()


class ImageDir:
    def __init__(self, img_dir) -> None:
        self.img_dir = img_dir

    def change_image_format(self, suffix: str = ".jpg"):
        """
        统一当前文件夹下所有图像的格式，如'.jpg'
        :param img_dir: 图片文件夹路径
        :param suffix: 图像文件后缀
        :return:
        """
        externs = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".jfif"]
        externs.remove(suffix)
        files = [self.img_dir.files(match=f"*{extern}") for extern in externs]
        for im in chain(*files):
            cv2.imwrite(im.with_suffix(suffix), cv2.imread(im))
            im.remove()

    def rename(self):
        """
        文件重命名
        """
        for i, im in enumerate(self.img_dir.files()):
            im.rename(self.img_dir / f"{i+1:0>6}{im.ext}")

    def split_dataset(self, test_size=0.3, test=False):
        """
        将文件分为训练集，测试集和验证集
        :param shuffle: 使用shuffle分割数据集
        :param test_size: 分割测试集或验证集的比例
        :param test: 是否使用测试集，默认为False
        :param img_path:当前文件路径
        :return:
        """
        from sklearn.model_selection import train_test_split

        files = [i.stem for i in self.img_dir.files()]
        if test:
            trainval_files, test_files = train_test_split(
                files, test_size=test_size, random_state=55
            )
        else:
            trainval_files, test_files = files, None

        train_files, val_files = train_test_split(
            trainval_files, test_size=test_size, random_state=55
        )

        return train_files, val_files, test_files, files

    def create_yolo(self, test_size=0.1, test=True):
        """批量创建yolo文件夹"""

        def move_files(src_dir: Path, dst_dir: Path, files: List[str]):
            for f in src_dir.files():
                if f.stem in files:
                    f.move(dst_dir)

        root = self.img_dir.parent
        for x, y in product(("train", "valid", "test"), ("images", "labels")):
            (root / x / y).makedirs_p()
        train_files, val_files, test_files, _ = self.split_dataset(test_size, test)
        move_files(self.img_dir, root / "train" / "images", train_files)
        move_files(root / "yolo", root / "train" / "labels", train_files)
        move_files(self.img_dir, root / "valid" / "images", val_files)
        move_files(root / "yolo", root / "valid" / "labels", val_files)
        if test_files:
            move_files(self.img_dir, root / "test" / "images", test_files)
            move_files(root / "yolo", root / "test" / "labels", test_files)

    def create_voc(self, test_size=0.1, test=True):
        root = self.img_dir.parent
        train_files, val_files, test_files, _ = self.split_dataset(test_size, test)
        (root / "train.txt").write_lines(train_files)
        (root / "valid.txt").write_lines(val_files)
        if test_files:
            (root / "test.txt").write_lines(test_files)


# ----------------------------------yolo--------------------------------------


def to_yolo(label: Label):
    target = label.root / "yolo"
    if not target.exists():
        target.mkdir_p()
    target_path = target / f"{label.stem}.txt"
    res = []
    for box in label.boxes:
        center_x = (box.xmin + box.xmax) / 2 / label.width
        center_y = (box.ymin + box.ymax) / 2 / label.height
        w = (box.xmax - box.xmin) / label.width
        h = (box.ymax - box.ymin) / label.height
        cls = label.labels.index(box.name)
        res.append(f"{cls:d} {center_x:<08f} {center_y:<08f} {w:<08f} {h:<08f}")
    target_path.write_lines(res)


def from_yolo(label: Label):
    boxes = []
    for line in label.label_path.lines():
        items = [float(item) for item in line.strip().split()]
        w = items[3] * label.width
        h = items[4] * label.height
        boxes.append(
            Box(
                xmin=items[1] * label.width - w / 2,
                ymin=items[2] * label.height - h / 2,
                xmax=items[1] * label.width + w / 2,
                ymax=items[2] * label.height + h / 2,
                name=label.labels[int(items[0])],
            )
        )
    label.boxes = boxes


# -----------------------------------voc--------------------------------------


def to_voc(label: Label):
    import lxml.builder
    import lxml.etree

    target = label.root / "voc"
    if not target.exists():
        target.mkdir_p()
    target_path = target / f"{label.stem}.xml"
    maker = lxml.builder.ElementMaker()
    xml = maker.annotation(
        maker.folder(),
        maker.filename(label.stem + ".jpg"),
        maker.source(
            maker.database(),  # e.g., The VOC2007 Database
            maker.annotation(),  # e.g., Pascal VOC2007
            maker.image(),  # e.g., flickr
        ),
        maker.size(
            maker.height(str(label.height)),
            maker.width(str(label.width)),
            maker.depth(str(label.depth)),
        ),
        maker.segmented(),
    )

    for box in label.boxes:
        xml.append(
            maker.object(
                maker.name(box.name),
                maker.pose(),
                maker.truncated(),
                maker.difficult(),
                maker.bndbox(
                    maker.xmin(str(box.xmin)),
                    maker.ymin(str(box.ymin)),
                    maker.xmax(str(box.xmax)),
                    maker.ymax(str(box.ymax)),
                ),
            )
        )
    target_path.write_text(lxml.etree.tostring(xml, pretty_print=True))


def from_voc(label: Label):
    import lxml.etree

    xml = lxml.etree.parse(label.label_path)

    def to_box(x):
        return Box(
            xmin=float(x.find("bndbox").find("xmin").text),
            ymin=float(x.find("bndbox").find("ymin").text),
            xmax=float(x.find("bndbox").find("xmax").text),
            ymax=float(x.find("bndbox").find("ymax").text),
            name=x.find("name").text,
        )

    objects = filter(lambda o: o.getchildren(), xml.findall("object"))
    label.boxes = [to_box(o) for o in objects]


# ----------------------------------labelme------------------------------------


def from_labelme(label: Label):
    import json

    data = json.loads(label.label_path.read_text())
    boxes = []
    for shape in data["shapes"]:
        points = shape["points"]
        xmin, ymin = points[0]
        xmax, ymax = points[1]
        boxes.append(Box(xmin, ymin, xmax, ymax, name=shape["label"]))
    label.boxes = boxes


if __name__ == "__main__":
    JPEG = Path(r"D:\Downloads\Capacitors\newvoc\train\images\4_295.jpg")
    MASK = Path(r"D:\Downloads\Capacitors\newvoc\train\labels\4_295.txt")
    label = Label(JPEG, MASK, ["capacitor"])
    label.load("yolo")
    label.plt_show()
