import os
import time
import json
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from functools import partial
import torch
torch.cuda.is_available()
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG
from tokenizer import Tokenizer
from utils import (
    seed_everything,
    load_checkpoint,
    test_generate,
    postprocess,
    permutations_to_polygons,
)
from models.model import (
    Encoder,
    Decoder,
    EncoderDecoder
)

from torch.utils.data import DataLoader
from datasets.dataset_whu_buildings_coco import WHUBuildingsCocoDataset_val
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset to use for evaluation.")
parser.add_argument("-e", "--experiment_path", help="path to experiment folder to evaluate.")
parser.add_argument("-c", "--checkpoint_name", help="Choice of checkpoint to evaluate in experiment.")
parser.add_argument("-o", "--output_dir", help="Name of output subdirectory to store part predictions.")
args = parser.parse_args()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DATASET = f"{args.dataset}"
# VAL_DATASET_DIR = f"./data/{DATASET}/test"
# PART_DESC = f"{args.output_dir}"

# EXPERIMENT_NAME = os.path.basename(os.path.realpath(args.experiment_path))
# CHECKPOINT_PATH = f"runs/{EXPERIMENT_NAME}/logs/checkpoints/{args.checkpoint_name}.pth"
# BATCH_SIZE = 24

# Debug
DATASET = 'whu_buildings_224'
VAL_DATASET_DIR = r"C:\Users\HDSL36\Desktop\Work\Projects\Change_detection\thirdparty/Pix2Poly/data/whu_buildings_224/val"
PART_DESC = r"C:\Users\HDSL36\Desktop\Work\Projects\Change_detection\thirdparty/Pix2Poly/data/results/whu_buildings_224"

EXPERIMENT_NAME = 'whu_buildings_224'
CHECKPOINT_PATH = r"C:\Users\HDSL36\Desktop\Work\Projects\Change_detection\data/weights/runs_share/Pix2Poly_whu_building_224_coco/logs/checkpoints/epoch_499.pth"
TEST_DATASET_DIR = r"C:\Users\HDSL36\Desktop\Work\Projects\Change_detection\data\pdok\set_1\Images"
#OUTPUT_PATH = 
BATCH_SIZE = 24


def bounding_box_from_points(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0]/2) * 2
    odd_locations = even_locations + 1
    X = np.take(points, even_locations.tolist())
    Y = np.take(points, odd_locations.tolist())
    bbox = [X.min(), Y.min(), X.max()-X.min(), Y.max()-Y.min()]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(image_id, poly):
    _result = {}
    _result["image_id"] = int(image_id)
    _result["category_id"] = 100 
    _result["score"] = 1
    _result["segmentation"] = poly
    _result["bbox"] = bounding_box_from_points(_result["segmentation"])
    return _result


def collate_fn(batch, max_len, pad_idx):
    """
    if max_len:
        the sequences will all be padded to that length.
    """

    image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, idx_batch = [], [], [], [], [], []
    for image, mask, c_mask, seq, perm_mat, idx in batch:
        image_batch.append(image)
        mask_batch.append(mask)
        coords_mask_batch.append(c_mask)
        coords_seq_batch.append(seq)
        perm_matrix_batch.append(perm_mat)
        idx_batch.append(idx)

    coords_seq_batch = pad_sequence(
        coords_seq_batch,
        padding_value=pad_idx,
        batch_first=True
    )

    if max_len:
        pad = torch.ones(coords_seq_batch.size(0), max_len - coords_seq_batch.size(1)).fill_(pad_idx).long()
        coords_seq_batch = torch.cat([coords_seq_batch, pad], dim=1)

    image_batch = torch.stack(image_batch)
    mask_batch = torch.stack(mask_batch)
    coords_mask_batch = torch.stack(coords_mask_batch)
    perm_matrix_batch = torch.stack(perm_matrix_batch)
    idx_batch = torch.stack(idx_batch)
    return image_batch, mask_batch, coords_mask_batch, coords_seq_batch, perm_matrix_batch, idx_batch


def main():
    seed_everything(42)

    valid_transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    val_ds = WHUBuildingsCocoDataset_val(
        cfg=CFG,
        dataset_dir=VAL_DATASET_DIR,
        transform=valid_transforms,
        tokenizer=tokenizer,
        shuffle_tokens=CFG.SHUFFLE_TOKENS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=partial(collate_fn, max_len=CFG.MAX_LEN, pad_idx=CFG.PAD_IDX),
        num_workers=2
    )

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)
    model.eval()

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epochs_run']

    print(f"Model loaded from epoch: {epoch}")
    ckpt_desc = f"epoch_{epoch}"
    if "best_valid_loss" in os.path.basename(CHECKPOINT_PATH):
        ckpt_desc = f"epoch_{epoch}_bestValLoss"
    elif "best_valid_metric" in os.path.basename(CHECKPOINT_PATH):
        ckpt_desc = f"epoch_{epoch}_bestValMetric"
    else:
        pass

    mean_iou_metric = BinaryJaccardIndex()
    mean_acc_metric = BinaryAccuracy()


    with torch.no_grad():
        cumulative_miou = []
        cumulative_macc = []
        speed = []
        predictions = []
        for i_batch, (x, y_mask, y_corner_mask, y, y_perm, idx) in enumerate(tqdm(val_loader)):
            all_coords = []
            all_confs = []
            t0 = time.time()
            batch_preds, batch_confs, perm_preds = test_generate(model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
            speed.append(time.time() - t0)
            vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

            all_coords.extend(vertex_coords)
            all_confs.extend(confs)

            coords = []
            for i in range(len(all_coords)):
                if all_coords[i] is not None:
                    coord = torch.from_numpy(all_coords[i])
                else:
                    coord = torch.tensor([])

                padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                coord = torch.cat([coord, padd], dim=0)
                coords.append(coord)
            batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]
            # pred_polygons = permutations_to_polygons(perm_preds, coords, out='coco')  # [0, 224]

            for ip, pp in enumerate(batch_polygons):
                for p in pp:
                    p = torch.fliplr(p)
                    p = p[p[:, 0] != CFG.PAD_IDX]
                    p = p * (CFG.IMG_SIZE / CFG.INPUT_WIDTH)
                    p = p.view(-1).tolist()
                    if len(p) > 0:
                        predictions.append(single_annotation(idx[ip], [p]))

            B, C, H, W = x.shape

            polygons_mask = np.zeros((B, 1, H, W))
            for b in range(len(batch_polygons)):
                for c in range(len(batch_polygons[b])):
                    poly = batch_polygons[b][c]
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    cnt = np.flip(np.int32(poly.cpu()), 1)
                    if len(cnt) > 0:
                        cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.)
            polygons_mask = torch.from_numpy(polygons_mask)

            batch_miou = mean_iou_metric(polygons_mask, y_mask)
            batch_macc = mean_acc_metric(polygons_mask, y_mask)

            cumulative_miou.append(batch_miou)
            cumulative_macc.append(batch_macc)

            pred_grid = make_grid(polygons_mask).permute(1, 2, 0)
            gt_grid = make_grid(y_mask).permute(1, 2, 0)
            plt.subplot(211), plt.imshow(pred_grid) ,plt.title("Predicted Polygons") ,plt.axis('off')
            plt.subplot(212), plt.imshow(gt_grid) ,plt.title("Ground Truth") ,plt.axis('off')

            if not os.path.exists(os.path.join(f"runs/{EXPERIMENT_NAME}", 'test_preds', DATASET, PART_DESC, ckpt_desc)):
                os.makedirs(os.path.join(f"runs/{EXPERIMENT_NAME}", 'test_preds', DATASET, PART_DESC, ckpt_desc))
            plt.savefig(f"{PART_DESC}/{ckpt_desc}/batch_{i_batch}.png")
            plt.close()

        print("Average model speed: ", np.mean(speed) / BATCH_SIZE, " [s / image]")

        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}")
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}")

    with open(f"runs/{EXPERIMENT_NAME}/predictions_{DATASET}_{PART_DESC}_{ckpt_desc}.json", "w") as fp:
        fp.write(json.dumps(predictions))

    with open(f"runs/{EXPERIMENT_NAME}/test_metrics_{DATASET}_{PART_DESC}_{ckpt_desc}.txt", 'w') as ff:
        print(f"Average Mean IOU: {torch.tensor(cumulative_miou).nanmean()}", file=ff)
        print(f"Average Mean Acc: {torch.tensor(cumulative_macc).nanmean()}", file=ff)
import os
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class WHUBuildingsTestDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, resize=None):
        """
        Args:
            dataset_dir (str): path to dataset folder containing 'images'
            transform: albumentations or torchvision transforms
            resize (tuple or None): (width, height) to resize with PIL before transform
        """
        self.image_dir = dataset_dir
        self.transform = transform
        self.resize = resize
        self.images = [file for file in os.listdir(self.image_dir) 
                       if osp.isfile(osp.join(self.image_dir, file)) and file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = osp.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # optional resize with PIL
        if self.resize is not None:
            image = image.resize(self.resize, Image.BILINEAR)

        image = np.array(image)

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations['image']

        return image, img_name


def test():
    seed_everything(42)

    valid_transforms = A.Compose(
        [
            A.Resize(height=CFG.INPUT_HEIGHT, width=CFG.INPUT_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format='yx', remove_invisible=False)
    )

    tokenizer = Tokenizer(
        num_classes=1,
        num_bins=CFG.NUM_BINS,
        width=CFG.INPUT_WIDTH,
        height=CFG.INPUT_HEIGHT,
        max_len=CFG.MAX_LEN
    )
    CFG.PAD_IDX = tokenizer.PAD_code

    test_ds = WHUBuildingsTestDataset(
        dataset_dir=TEST_DATASET_DIR,
        transform=valid_transforms      # Albumentations pipeline
        #resize=(512, 512)                 # (width, height) before transforms
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    encoder = Encoder(model_name=CFG.MODEL_NAME, pretrained=True, out_dim=256)
    decoder = Decoder(
        cfg=CFG,
        vocab_size=tokenizer.vocab_size,
        encoder_len=CFG.NUM_PATCHES,
        dim=256,
        num_heads=8,
        num_layers=6
    )
    model = EncoderDecoder(cfg=CFG, encoder=encoder, decoder=decoder)
    model.to(CFG.DEVICE)
    model.eval()

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epochs_run']

    print(f"Model loaded from epoch: {epoch}")
    ckpt_desc = f"epoch_{epoch}"
    if "best_valid_loss" in os.path.basename(CHECKPOINT_PATH):
        ckpt_desc = f"epoch_{epoch}_bestValLoss"
    elif "best_valid_metric" in os.path.basename(CHECKPOINT_PATH):
        ckpt_desc = f"epoch_{epoch}_bestValMetric"
    else:
        pass
    with torch.no_grad():
        cumulative_miou = []
        cumulative_macc = []
        speed = []
        predictions = []
        for i_batch, (x, names) in enumerate(tqdm(test_loader)):
            all_coords = []
            all_confs = []
            t0 = time.time()
            batch_preds, batch_confs, perm_preds = test_generate(model, x, tokenizer, max_len=CFG.generation_steps, top_k=0, top_p=1)
            speed.append(time.time() - t0)
            vertex_coords, confs = postprocess(batch_preds, batch_confs, tokenizer)

            all_coords.extend(vertex_coords)
            all_confs.extend(confs)

            coords = []
            for i in range(len(all_coords)):
                if all_coords[i] is not None:
                    coord = torch.from_numpy(all_coords[i])
                else:
                    coord = torch.tensor([])

                padd = torch.ones((CFG.N_VERTICES - len(coord), 2)).fill_(CFG.PAD_IDX)
                coord = torch.cat([coord, padd], dim=0)
                coords.append(coord)
            batch_polygons = permutations_to_polygons(perm_preds, coords, out='torch')  # [0, 224]
            # pred_polygons = permutations_to_polygons(perm_preds, coords, out='coco')  # [0, 224]
            idx = 0
            for ip, pp in enumerate(batch_polygons):
                img_id = names[ip].split('_')[0].replace('_', '')
                for p in pp:
                    p = torch.fliplr(p)
                    p = p[p[:, 0] != CFG.PAD_IDX]
                    p = p * (CFG.IMG_SIZE / CFG.INPUT_WIDTH)
                    p = p.view(-1).tolist()
                    if len(p) > 0:
                        predictions.append(single_annotation(img_id, [p]))

            B, C, H, W = x.shape
            save_dir = f"{PART_DESC}/{ckpt_desc}"   
            polygons_mask = np.zeros((B, 1, H, W))
            for b in range(len(batch_polygons)):
                name = names[b].split('.')[0] + '.jpg'
                for c in range(len(batch_polygons[b])):
                    poly = batch_polygons[b][c]
                    poly = poly[poly[:, 0] != CFG.PAD_IDX]
                    cnt = np.flip(np.int32(poly.cpu()), 1)
                    if len(cnt) > 0:
                        cv2.fillPoly(polygons_mask[b, 0], pts=[cnt], color=1.)
                save_path = os.path.join(save_dir, name)
                cv2.imwrite(save_path, polygons_mask[b, 0] * 255)
            # polygons_mask = torch.from_numpy(polygons_mask)
            # pred_grid = make_grid(polygons_mask).permute(1, 2, 0)
            # plt.imshow(pred_grid) ,plt.title("Predicted Polygons") ,plt.axis('off')
            # plt.savefig(f"{PART_DESC}/{ckpt_desc}/batch_{i_batch}.png")
            # plt.close()
            
    
if __name__ == "__main__":
    test()
    #main()

