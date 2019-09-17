import os
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
sys.path.append(os.getcwd())
from data_preprocess.utils import read_class_name, read_images_class_label, classify_images_by_class, classify_images_ids_by_class,\
        check_dir, read_text_vocabulary_json,read_class_ids,  read_text_feature_h5, read_images_filename, read_list_file, read_bbox
import pickle

class Dataset_birds(Dataset):
    stage_classids_pair = {"train":"trainids", "val":"valids", "test":"testids", "trainval":"trainvalids"}
    def __init__(self, CUB_200_2011_path, CUR_200_2011_embedding_path, text_path, branch, stage="train", 
            branch_size_list=[int(64), int(128), int(256), int(512), int(1024)]):
        assert(stage in self.stage_classids_pair.keys())
        self.branch_size_list = branch_size_list
        self.stage = stage
        self.dataset_root = CUB_200_2011_path
        self.text_embedding_root = CUR_200_2011_embedding_path
        self.text_path = text_path
        self.branch = branch
        self.transforms = [transforms.Compose([
                transforms.RandomCrop((branch_size_list[branch_idx], branch_size_list[branch_idx])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]) for branch_idx in range(branch)]
        self.self_class_name = self.__class__.__name__
        #get all filename
        self.class_name_all, self.image_class_label_all, self.images_filename_all, self.bbox = \
            read_class_name(os.path.join(self.dataset_root, "classes.txt")), \
            read_images_class_label(os.path.join(self.dataset_root, "image_class_labels.txt")), \
            read_images_filename(os.path.join(self.dataset_root, "images.txt")), \
            read_bbox(os.path.join(self.dataset_root, "bounding_boxes.txt"))
        # group id --> image id
        self.images_ids_grouped_by_class = classify_images_ids_by_class(
                self.class_name_all,  
                self.image_class_label_all)
            
        #get split ID
        self.selected_class_ids = read_class_ids(os.path.join(self.dataset_root, self.stage_classids_pair[self.stage]+".txt"))
        
        # get all selected image ID
        self.selected_images_ids = [] # image ids
        for idx in self.selected_class_ids:
            self.selected_images_ids += self.images_ids_grouped_by_class[idx]

        #embedding_all
        with open(os.path.join(self.dataset_root, "../train/char-CNN-RNN-embeddings.pickle"), "rb") as f:
            self.embedding_all = pickle.load(f, encoding="bytes")
        #filename_all
        with open(os.path.join(self.dataset_root, "../train/filenames.pickle"), "rb") as f:
            self.filename_train_all = pickle.load(f, encoding="bytes")



    def __len__(self):
        return len(self.selected_images_ids)*10

    def _load_embedding_data(self, embedding_path, embedding_name):
        return read_text_feature_h5(os.path.join(embedding_path, embedding_name))

    def _load_text_data(self, text_path, text_name):
        return read_list_file(os.path.join(text_path, text_name))

    def _load_image_data(self, image_path, image_name):
        image_path_full = os.path.join(image_path, image_name)
        return Image.open(image_path_full).convert("RGB")

    def get_wrong_image_id(self, real_image_class_id):
        wrong_class_id = real_image_class_id
        while wrong_class_id==real_image_class_id:
            wrong_class_idx = np.random.randint(len(self.selected_class_ids))
            wrong_class_id = self.selected_class_ids[wrong_class_idx]
        
        wrong_image_idx = np.random.randint(len(self.images_ids_grouped_by_class[wrong_class_id]))
        wrong_image_id = self.images_ids_grouped_by_class[wrong_class_id][wrong_image_idx]
        return wrong_image_id

    def get_augment_data(self, image, embedding, text, image_wrong, embedding_wrong, image_size=(64,64), image_augment=False):
        '''
        crop, flip for PIL image
        random select a text
        '''
        if image_augment:
            image_random = np.random.randint(10)
            if image_random>=5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image_random -= 5
            img_width, img_height = image.size
            crop_dict = {0:(0, 0, img_width, img_height),
                        1:(0, 0, img_width-2, img_height-2),
                        2:(2, 0, img_width, img_height-2),
                        3:(0, 2, img_width-2, img_height),
                        4:(2, 2, img_width, img_height)}
            #it is a lazy operation, may not copy the memory
            image_crop = image.crop(crop_dict[image_random]).resize(image_size)  
            image_wrong_crop = image_wrong.crop(crop_dict[image_random]).resize(image_size)  
        else:
            image_crop = image.resize(image_size)
            image_wrong_crop = image_wrong.resize(image_size)
            
        text_random = np.random.randint(len(text))
        embedding_select = embedding[text_random]
        text_select = text[text_random]
        
        embedding_wrong_select = embedding_wrong[np.random.randint(len(embedding_wrong))]
        return image_crop, embedding_select, text_select, image_wrong_crop, embedding_wrong_select, text_random
   
    
    def validate_image(self, img):
        img = np.array(img, dtype=np.uint8)
        if len(img.shape) < 3:
            rgb = np.empty(img.size+(3,), dtype=np.float32)
            rgb[:, :, 0] = img
            rgb[:, :, 1] = img
            rgb[:, :, 2] = img
            img = rgb
        return img
    
    @staticmethod
    def custom_crop(image, bbox, single_margin=0.1):
        img_width, img_height = image.size
        width, height = bbox[2]-bbox[0], bbox[3]-bbox[1]
        margin_pixel_width, margin_pixel_height = width*single_margin, height*single_margin
        bbox = [
            bbox[0]-margin_pixel_width if bbox[0]>margin_pixel_width else 0,
            bbox[1]-margin_pixel_height if bbox[1]>margin_pixel_height else 0,
            bbox[2]+margin_pixel_width if bbox[2]<img_width-margin_pixel_width else img_width,
            bbox[3]+margin_pixel_height if bbox[3]<img_height-margin_pixel_height else img_height
        ]
        return image.crop(bbox)

   
    def __getitem__(self, idx):
        # sample{real_image, real_embedding, wrong_embedding, text}
        # (read_text, read_img), (fake_text, real_img)
        img_idx, text_idx = int(idx//10), int(idx%10)
        image_path = os.path.join(self.dataset_root, "images")
        embedding_path = os.path.join(self.text_embedding_root, "text")
        text_path = self.text_path
        image_id = self.selected_images_ids[img_idx]
        class_id = self.image_class_label_all[image_id]
        image_name = self.images_filename_all[image_id]
        text_name = os.path.splitext(image_name)[0]+".txt"
        embedding_name = os.path.splitext(image_name)[0]+".idx.h5"
        real_image, real_embedding, text_data, real_bbox = \
            self._load_image_data(image_path, image_name),\
            self._load_embedding_data(embedding_path, embedding_name), \
            self._load_text_data(text_path, text_name), \
            self.bbox[image_id]
        
        wrong_image_id = self.get_wrong_image_id(class_id)
        wrong_image_name = self.images_filename_all[wrong_image_id]
        wrong_image, wrong_bbox = self._load_image_data(image_path, wrong_image_name), self.bbox[wrong_image_id]

        #wrong_embedding_name = os.path.splitext(wrong_image_name)[0]+".idx.h5"
        #wrong_image = self._load_image_data(image_path, wrong_image_name)
        #wrong_embedding = self._load_embedding_data(embedding_path, wrong_embedding_name)

        # crop bbox
        bbox_width, bbox_height = real_bbox[2], real_bbox[3]
        real_bbox, wrong_bbox = real_bbox[:2]+(real_bbox[0]+real_bbox[2], real_bbox[1]+real_bbox[3]), \
                                wrong_bbox[:2]+(wrong_bbox[0]+wrong_bbox[2], wrong_bbox[1]+wrong_bbox[3])
        real_image, wrong_image = self.custom_crop(real_image, real_bbox, 1.0/8), self.custom_crop(wrong_image, wrong_bbox, 1.0/8)

        real_images, wrong_images = [], []
        for branch_idx in range(self.branch):
            image_size = (self.branch_size_list[branch_idx], self.branch_size_list[branch_idx])
            image_size_temp = (int(image_size[0]*9/8), int(image_size[1]*9/8))
            real_images.append(real_image.resize(image_size_temp))
            wrong_images.append(wrong_image.resize(image_size_temp))

        real_embedding, text_data = real_embedding[text_idx], text_data[text_idx]
        #real_image, real_embedding, text_data, wrong_image, wrong_embedding, text_idx = self.get_augment_data(real_image, real_embedding, 
        #    text_data, wrong_image, wrong_embedding, image_size=image_size_temp, 
        #    image_augment=False)
        real_embedding_anchor = self.embedding_all[img_idx][text_idx, :]
        
        #real_image, real_embedding, text_data, wrong_embedding = real_image.resize((64,64)), real_embedding[text_idx], text_data[text_idx], wrong_embedding[text_idx]
        #real_image, wrong_image = self.validate_image(real_image), self.validate_image(wrong_image)
        
        if self.transforms:
            real_images = [transforms(real_image) for real_image, transforms in zip(real_images, self.transforms)]
            wrong_images = [transforms(wrong_image) for wrong_image, transforms in zip(wrong_images, self.transforms)]

        sample = {"real_image": real_images, 
                "real_embedding": real_embedding,
                "wrong_image":wrong_images,
                "text":text_data}
        return sample

class Dataset_birds_anchor(Dataset):
    def __init__(self, stage="train"):
        super(Dataset_birds_anchor, self).__init__()
        assert(stage in ["train", "test"])
        self.image_folder = "./data/birds/CUB_200_2011/images"
        self.stage = stage
        self.transforms = transforms.Compose([
                transforms.RandomCrop((64,64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        with open("./data/birds/"+self.stage+"/filenames.pickle", "rb") as f:
            self.filenames_all = pickle.load(f, encoding="bytes")
        with open("./data/birds/"+self.stage+"/class_info.pickle", "rb") as f:
            self.class_info_all = pickle.load(f, encoding="bytes")
        with open("./data/birds/"+self.stage+"/char-CNN-RNN-embeddings.pickle", "rb") as f:
            self.embedding_all = pickle.load(f, encoding="bytes")
        with open("./data/birds/"+self.stage+"/76images.pickle","rb") as f:
            self.images_all = pickle.load(f, encoding="bytes")

    def __len__(self):
        return len(self.filenames_all)*10
    

    def __getitem__(self, idx):
        idx_real, idx_text = int(idx//10), int(idx%10)
        while True:
            idx_wrong = np.random.randint(len(self.filenames_all))
            if not self.class_info_all[idx_wrong]==self.class_info_all[idx_real]:
                break

        filename_real, filename_wrong, embedding = \
                self.filenames_all[idx_real], self.filenames_all[idx_wrong], self.embedding_all[idx_real][idx_text]

        image_real, image_wrong = self.images_all[idx_real], self.images_all[idx_wrong]
        text = read_list_file(os.path.join("./data/birds/text", os.path.splitext(filename_real)[0]+".txt"))[idx_text]

        image_real, image_wrong = Image.fromarray(image_real), Image.fromarray(image_wrong)
        image_real, image_wrong = self.transforms(image_real), self.transforms(image_wrong)

        sample = {"real_image": image_real, 
                "real_embedding": embedding,
                "wrong_image":image_wrong,
                "text":text}

        return sample

            
        

class NormalizeRescale(object):
    """normalize a tensor image from [min_in,max_in] to [min_out,max_out]
    """
    def __init__(self,in_range=(0,1), out_range=(-1,1)):
        self.in_range = in_range
        self.out_range = out_range
        self.mid_in = (self.in_range[1]+self.in_range[0])/2
        self.scale_in = (self.in_range[1]-self.in_range[0])/2
        self.mid_out = (self.out_range[1]+self.out_range[0])/2
        self.scale_out = (self.out_range[1]-self.out_range[0])/2
    
    def __call__(self, tensor):
        return tensor.sub(self.mid_in).div(self.scale_in).mul(self.scale_out).add(self.mid_out)


        

if __name__=="__main__":
    CUB_200_2011_path = "./data/birds/CUB_200_2011"
    embedding_path = "./data/birds/CUB_200_2011_embedding"
    text_path = "./data/birds/text"
    dataset = Dataset_birds(CUB_200_2011_path, embedding_path, 
            text_path, 3, stage="train", 
            )
    #dataset = Dataset_birds_anchor()
    param = {}  #{"num_workers":2, "pin_memory":True}
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, **param)
    for idx, sample in enumerate(dataloader):
        print("[test for dataloader]:{}, image length:{}".format(list(sample.keys()), len(sample["real_image"])))


    