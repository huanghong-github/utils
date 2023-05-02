import cv2
from typing import List
from matplotlib import pyplot as plt
from collections import namedtuple
from path import Path
from numpy import ndarray, dtype, generic
from itertools import chain, product


Anchor = namedtuple(
    "Anchor",
    ('xmin', 'ymin', 'xmax', 'ymax', 'name'))


class Label:
    def __init__(self, image_path: Path, label_path: Path, labels: List[str]) -> None:
        self.image_path: Path = image_path
        self.label_path: Path = label_path
        self.stem: str = image_path.stem
        self.root: Path = image_path.parent.parent
        self.img: ndarray[int, dtype[generic]] = cv2.imread(image_path)
        self.height, self.width, self.depth = self.img.shape
        self.anchors: List[Anchor] = None
        self.labels: List[str] = labels

    def unload(self, mode='yolo'):
        func = globals().get(f'to_{mode}')
        func(self)

    def load(self, mode='yolo'):
        func = globals().get(f'from_{mode}')
        func(self)

    def convert(self, func):
        self.anchors = func(self)

    def cv_show(self):
        im = self.img.copy()
        for box in self.anchors:
            cv2.rectangle(img=im,
                          pt1=(box.xmin, box.ymin),
                          pt2=(box.xmax, box.ymax),
                          color=(255, 0, 0),
                          thickness=2)
            cv2.putText(img=im,
                        text=box.name,
                        org=(box.xmin, box.ymin-5),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.7,  # 字体大小
                        color=(255, 0, 0),
                        thickness=1,  # 线的粗细
                        lineType=cv2.LINE_AA)
        plt.imshow(im)
        plt.show()

    def plt_show(self):
        for box in self.anchors:
            rect = plt.Rectangle(xy=(box.xmin, box.ymin),
                                 width=box.xmax-box.xmin,
                                 height=box.ymax-box.ymin,
                                 edgecolor='r',
                                 linewidth=1,
                                 fill=False)
            plt.text(x=box.xmin,
                     y=box.ymin,
                     s=box.name,
                     fontsize=10,
                     color="r",
                     style="italic",
                     weight="light"
                     )
            plt.gca().add_patch(rect)
        plt.imshow(self.img)
        plt.show()

    def viz_show(self):
        import imgviz
        viz = imgviz.instances2rgb(
            image=self.img,
            labels=[self.labels.index(anchor.name) for anchor in self.anchors],
            bboxes=[[anchor.ymin, anchor.xmin, anchor.ymax, anchor.xmax]
                    for anchor in self.anchors],
            captions=[anchor.name for anchor in self.anchors],
            font_size=15,
        )
        plt.imshow(viz)
        plt.show()


class ImageDir:

    def __init__(self, img_dir) -> None:
        self.img_dir = img_dir

    def change_image_format(self, suffix: str = '.jpg'):
        """
        统一当前文件夹下所有图像的格式，如'.jpg'
        :param img_dir: 图片文件夹路径
        :param suffix: 图像文件后缀
        :return:
        """
        externs = ['.png', '.jpg', '.jpeg', '.bmp']
        externs.remove(suffix)
        files = [self.img_dir.files(match=f'*{extern}') for extern in externs]
        for im in chain(*files):
            cv2.imwrite(im.with_suffix(suffix), cv2.imread(im))
            im.remove()

    def rename(self):
        """
        文件重命名
        """
        for i, im in enumerate(self.img_dir.files()):
            im.rename(self.img_dir/f'{i+1:0>6}{im.ext}')

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
                files, test_size=test_size, random_state=55)
        else:
            trainval_files, test_files = files,  None

        train_files, val_files = train_test_split(
            trainval_files, test_size=test_size, random_state=55)

        return train_files, val_files, test_files, files

    def create_yolo(self):
        """批量创建yolo文件夹"""
        def move_files(src_dir: Path, dst_dir: Path, files: List[str]):
            for f in src_dir.files():
                if f.stem in files:
                    f.move(dst_dir)

        root = self.img_dir.parent
        for x, y in product(('train', 'valid', 'test'), ('images', 'labels')):
            (root/x/y).makedirs_p()
        train_files, val_files, test_files, _ = self.split_dataset()
        move_files(self.img_dir, root/'train'/'images', train_files)
        move_files(root/'yolo', root/'train'/'labels', train_files)
        move_files(self.img_dir, root/'valid'/'images', val_files)
        move_files(root/'yolo', root/'valid'/'labels', val_files)

    def create_voc(self):
        root = self.img_dir.parent
        train_files, val_files, test_files, _ = self.split_dataset()
        (root/'train.txt').write_lines(train_files)
        (root/'valid.txt').write_lines(val_files)
# ----------------------------------yolo--------------------------------------


def to_yolo(label: Label):
    target = label.root/'yolo'
    if not target.exists():
        target.mkdir_p()
    target_path = target/f'{label.stem}.txt'
    res = []
    for anchor in label.anchors:
        center_x = (anchor.xmin + anchor.xmax) / 2 / label.width
        center_y = (anchor.ymin + anchor.ymax) / 2 / label.height
        w = (anchor.xmax - anchor.xmin) / label.width
        h = (anchor.ymax - anchor.ymin) / label.height
        cls = label.labels.index(anchor.name)
        res.append(
            f"{cls:d} {center_x:<08f} {center_y:<08f} {w:<08f} {h:<08f}")
    target_path.write_lines(res)


def from_yolo(label: Label):
    anchors = []
    for line in label.label_path.lines():
        cood = [float(item) for item in line.strip().split()]
        w = cood[3]*label.width
        h = cood[4]*label.height
        anchors.append(Anchor(xmin=cood[1]*label.width-w/2,
                              ymin=cood[2]*label.height-h/2,
                              xmax=cood[1]*label.width+w/2,
                              ymax=cood[2]*label.height+h/2,
                              name=label.labels[int(cood[0])]))
    label.anchors = anchors

# -----------------------------------voc--------------------------------------


def to_voc(label: Label):
    import lxml.builder
    import lxml.etree
    target = label.root/'voc'
    if not target.exists():
        target.mkdir_p()
    target_path = target/f'{label.stem}.xml'
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

    for anchor in label.anchors:
        xml.append(
            maker.object(
                maker.name(anchor.name),
                maker.pose(),
                maker.truncated(),
                maker.difficult(),
                maker.bndbox(
                    maker.xmin(str(anchor.xmin)),
                    maker.ymin(str(anchor.ymin)),
                    maker.xmax(str(anchor.xmax)),
                    maker.ymax(str(anchor.ymax)),
                ),
            )
        )
    target_path.write_text(lxml.etree.tostring(xml, pretty_print=True))


def from_voc(label: Label):
    import lxml.etree
    xml = lxml.etree.parse(label.label_path)

    def to_anchor(x):
        return Anchor(
            xmin=float(x.find('bndbox').find('xmin').text),
            ymin=float(x.find('bndbox').find('ymin').text),
            xmax=float(x.find('bndbox').find('xmax').text),
            ymax=float(x.find('bndbox').find('ymax').text),
            name=x.find("name").text
        )
    objects = filter(lambda o: o.getchildren(), xml.findall('object'))
    label.anchors = [to_anchor(o) for o in objects]
# ----------------------------------labelme------------------------------------


def from_labelme(label: Label):
    import json
    data = json.loads(label.label_path.read_text())
    anchors = []
    for shape in data["shapes"]:
        points = shape['points']
        xmin, ymin = points[0]
        xmax, ymax = points[1]
        anchors.append(Anchor(xmin, ymin, xmax, ymax, name=shape['label']))
    label.anchors = anchors


if __name__ == '__main__':
    JPEG = Path(r'D:\Downloads\Capacitors\archive\Capacitor\Capacitor (1).jpg')
    MASK = Path(r'D:\Downloads\Capacitors\archive\Capacitor\Capacitor (1).json')
    label = Label(JPEG, MASK, ['fire'])
    label.load('labelme')
    label.plt_show()
