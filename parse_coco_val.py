import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def main(clip_model_type: str):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_val.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open('./data/coco/annotations/val_caption_coco_format.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    
    all_embeddings = []
    embedding_id2image_id = []

    for i in tqdm(range(len(data['images']))):
        d = data["images"][i]
        img_id = d["id"]
        filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            raise IndexError
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        all_embeddings.append(prefix)
        embedding_id2image_id.append(img_id)

    all_captions = [{"image_id":embedding_id2image_id[i], "clip_embedding":i, "caption":[]} for i in range(len(embedding_id2image_id))]

    for i in tqdm(range(len(data['annotations']))):
        d = data['annotations'][i]
        img_id = d["image_id"]
        all_captions[embedding_id2image_id.index(img_id)]["caption"].append((data['annotations'][i]['caption']))

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
