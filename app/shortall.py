from xml.etree.ElementTree import parse

def load_lines(xml_path: str):
    tree = parse(xml_path)
    line = tree.getroot()
    tl_x = int(line.find("top").findtext("xleft"))
    tl_y = int(line.find("top").findtext("yleft"))
    tr_x = int(line.find("top").findtext("xright"))
    tr_y = int(line.find("top").findtext("yright"))
    bl_x = int(line.find("bottom").findtext("xleft"))
    bl_y = int(line.find("bottom").findtext("yleft"))
    br_x = int(line.find("bottom").findtext("xright"))
    br_y = int(line.find("bottom").findtext("yright"))
    return (tl_x, tl_y, tr_x, tr_y), (bl_x, bl_y, br_x, br_y)

def get_streight(line):
    lx, ly, rx, ry = line
    a = (ry - ly) / (rx - lx)
    b = ly - a * lx
    return lambda x: a * x + b

def get_streight_inv(line):
    lx, ly, rx, ry = line
    a = (ry - ly) / (rx - lx)
    b = ly - a * lx
    if a == 0:
        a += 1e-6
    return lambda y: (y - b) / a

def iou(box1, box2):
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

class Box:
    def __init__(self, boxes, inference_boxes, detector, seperator):
        self.boxes = boxes
        self.inference_boxes = inference_boxes
        self.detector = detector
        self.seperator = seperator
        self.result = ['empty']*4
    def backfront(self, box):
        lx, ly, rx, ry = box
        if max(self.detector(lx), self.detector(rx)) < max(ly, ry):
            bf = 1
        else:
            bf = 0
        return bf
    
    def short_tall(self, box):
        """
        return -> a list of 0 and 1 (0: short, 1: tall)
        """
        lx, ly, rx, ry = box
        if min(self.seperator(lx), self.seperator(rx)) > min(ly, ry):
            st = '#tall'
        else:
            st = '#short'
        return st
        
    # def make_result(self,):
    #     for box in self.boxes:
    #         iou_list = []
    #         for inference_box in self.inference_boxes:
    #             ttmp = iou(box,inference_box)
    #             log.info(f'ttmp _ {ttmp}')
    #             iou_list.append(ttmp)
    #             log.info(f' iou_list {iou_list}')
    #         max_index = np.argmax(iou_list)
    #         if self.backfront(box):
    #             self.result[max_index] = self.short_tall(box)
    #     return self.result
        
# def box_inheritance(cls):
#     def inherit_base(box, *args, **kwargs):
#         if Box in box.__class__.mro():
#             return type(cls.__name__, (cls, *box.__class__.mro(),), {})(box, *args, **kwargs)
#         else:
#             raise AttributeError("The class should take an instance "
#                                  "derived from 'Box' "
#                                  "as the first argument.")
#     return inherit_base


# @box_inheritance
# class BackFront:
#     def __init__(self, box, detector=None):
#         """
#         src: source image. ndarray shape: (h, w, c)
#         boxes: boxes to detect objects. a list of box: [(lx, ly, rx, ry), ...]
#         detector: the line to detect front boxes. line: (lx, ly, rx, ry)
#         """
#         self.__dict__ = box.__dict__.copy()
#         self.detector = detector

#     @property
#     def back_front(self):
#         """
#         return -> a list of 0 and 1 (0: not front, 1: front)
#         """
#         if self.detector:
#             bf = []
#             for box in self.boxes:
#                 lx, ly, rx, ry = box
#                 if max(self.detector(lx), self.detector(rx)) < max(ly, ry):
#                     bf.append(1)
#                 else:
#                     bf.append(0)
#             return bf


# @box_inheritance
# class ShortTall:
#     def __init__(self, box, seperator=None):
#         """
#         src: source image. ndarray shape: (h, w, c)
#         boxes: boxes to detect objects. a list of box: [(lx, ly, rx, ry), ...]
#         separator: the line to seperate short and tall. line: (lx, ly, rx, ry)
#         """
#         self.__dict__ = box.__dict__.copy()
#         self.seperator = seperator

#     @property
#     def short_tall(self):
#         """
#         return -> a list of 0 and 1 (0: short, 1: tall)
#         """
#         if self.seperator:
#             st = []
#             for box in self.boxes:
#                 lx, ly, rx, ry = box
#                 if min(self.seperator(lx), self.seperator(rx)) > min(ly, ry):
#                     st.append(1)
#                 else:
#                     st.append(0)
#             return st
    
    
#     @property
#     def box(self):
#         return self.boxes
