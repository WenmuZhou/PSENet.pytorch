#encoding:utf-8
from torch.utils.data import Dataset
import torch
from .datautils import *
import logging
import numpy as np
import pathlib
logger = logging.getLogger(__name__)

class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, image, polygons=None):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for polygon in polygons:
                polygon.points[:, 0] = width - polygon.points[:, 0]
        return image, polygons

class ICDAR(Dataset):
    def __init__(self, data_root):
        data_root = pathlib.Path(data_root)

        self.imagesRoot = data_root /'img'
        self.gtRoot =data_root/'gt'
        self.images, self.bboxs, self.transcripts = self.__loadGT()

    def __loadGT(self):
        all_bboxs = []
        all_texts = []
        all_images = []
        for image in self.imagesRoot.glob('*.jpg'):
            all_images.append(image)
            gt = self.gtRoot / 'gt_{}.txt'.format(image.stem)
            with gt.open(mode='r') as f:
                bboxes = []
                texts = []
                for line in f:
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    transcript = text[8]
                    bboxes.append(bbox)
                    texts.append(transcript)
                bboxes = np.array(bboxes)
                all_bboxs.append(bboxes)
                all_texts.append(texts)
        return all_images, all_bboxs, all_texts

    def __getitem__(self, index):
        imageName = self.images[index]
        bboxes = self.bboxs[index] # num_words * 8
        transcripts = self.transcripts[index]

        try:
            return self.__transform((imageName, bboxes, transcripts))
        except:
            return self.__getitem__(np.random.randint(0, len(self)))

    def __len__(self):
        return len(self.images)

    def __transform(self, gt, input_size = 640, random_scale = np.array([0.5,1,2])):
        '''

        :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)

        :return:
        '''

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread(imagePath.as_posix())
        numOfWords = len(wordBBoxes)
        text_polys = wordBBoxes
        transcripts = [word for line in transcripts for word in line.split()]
        text_tags = [True if(tag == '*' or tag == '###') else False for tag in transcripts] # ignore '###'
        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            ## check polys is Valuable
            text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize = None, fx = rd_scale, fy = rd_scale)
            text_polys *= rd_scale
            if np.random.randint(2):
                im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background = False)

            if text_polys.shape[0] == 0:
                raise TypeError('cannot find background')
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype = np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize = (resize_w, resize_h))
            resize_ratio_3_x = resize_w / float(new_w)
            resize_ratio_3_y = resize_h / float(new_h)
            text_polys[:, :, 0] *= resize_ratio_3_x
            text_polys[:, :, 1] *= resize_ratio_3_y
            cv2.imwrite('/data/fxw/PSENet_data/shrink/image_for_train_ori.jpg', im)
            ### flipped
            if np.random.randint(2):
                im = np.fliplr(im)
                _, width, _ = im.shape
                text_polys[:,:,0] = width - text_polys[:,:, 0]
                cv2.imwrite('/data/fxw/PSENet_data/shrink/image_for_train_flipped.jpg', im)
            new_h, new_w, _ = im.shape
            score_map,train_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

            cv2.imwrite('/data/fxw/PSENet_data/shrink/image_for_train_mask.jpg',(train_mask)*255)
            show_images = im[:, :, ::-1].astype(np.float32)

            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            ## normal images
            images = images.astype(np.float32)
            images/= 255.0
            images -= np.array((0.485, 0.456, 0.406))
            images /= np.array((0.229, 0.224, 0.225))

            score_maps =[]
            show_i = 0
            for score_map_item in score_map:
                score_map_item = score_map_item[::1, ::1].astype(np.float32)
                score_maps.append(score_map_item)
                cv2.imwrite('/data/fxw/PSENet_data/shrink/pse_lable_'+str(show_i)+'.jpg',score_map_item*255)
                show_i+=1
            score_maps=np.array(score_maps)
            return images, score_maps,train_mask,show_images
        else:
            raise TypeError('Number of bboxes is inconsist with number of transcripts ')