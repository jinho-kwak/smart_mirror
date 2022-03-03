from iteration_utilities import duplicates, unique_everseen
import copy


class RemoveDuplicatBox():
    '''
    바운딩 박스가 겹쳐졌을 경우, 겹쳐진 박스 중 한 개를 제거하는 로직
    Created by JwMudfish
    '''
    def __init__(self, coor, threshold, limit_area):
        self.coor = coor
        self.threshold = threshold
        self.limit_area = limit_area

    def calc_area(self, corr):
        w = abs(corr[2] - corr[0])
        h = abs(corr[3] - corr[1])
        wh = int(w * h) // 100
        return wh
    
    def area_filter(self, coor):
        del_list = list(filter(lambda x : self.calc_area(x) > self.limit_area, coor))
        #log.info(f'대왕박스 발견 : {del_list}')
        for i in del_list:
            coor.remove(i)   
        return coor

    def iou(self, box1, box2):
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

    def compute_iou(self, coor):
        box_list = copy.deepcopy(coor)
        print('threshold :', self.threshold)
        error_list = []
        for _ in range(len(box_list)):
            box_1 = box_list.pop()
            for box_2 in box_list:
                result = self.iou(box_1, box_2)
                if result > self.threshold :
                    error_list.append([box_1, box_2])
                # print(result)
        return error_list

    def listDupsUnique(self, lists):
        return list(unique_everseen(duplicates(lists)))

    def run(self):

        dup_coor = copy.deepcopy(self.coor)
        dup_coor = self.area_filter(coor = dup_coor)

        error_list = self.compute_iou(self.coor)
        tmp_list = sum(error_list, [])
        print('첫 결과 :', len(dup_coor))
        print('IOU 일정 이상 박스 : ',tmp_list)
        
        del_list = self.listDupsUnique(tmp_list)
        print('del_list : ', del_list)
        
        for coor in del_list:
            dup_coor.remove(coor)
        print('최종 결과 :' , len(dup_coor))
        
        tmp_error_list = self.compute_iou(dup_coor)
        
        for i in tmp_error_list:
            if i[0][1] < i[1][1]:
                i.remove(i[1])
            else:
                i.remove(i[0])
        
        try:
            for i in range(len(tmp_error_list)):
                dup_coor.remove(tmp_error_list[i][0])
        except:
            pass

        return dup_coor
