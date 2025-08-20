import torch
import os
import json
import argparse

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoProcessor
from PIL import Image
from V2P.constants import chat_template
from V2P.modeling import Qwen2VLForConditionalGenerationWithPointer
from V2P.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from V2P.inference import inference, ForceFollowTokensLogitsProcessor
from V2P.utils import do_boxes_overlap
from V2P.constants import DEFAULT_POINTER_PAD_TOKEN, DEFAULT_POINTER_END_TOKEN

IMAGE_PATCH_SIZE =14

def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    # if bbox_x1y1x2y2 is not normalized to [0, 1], normalize it
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    else:
        x1 = x1 / img_width
        y1 = y1 / img_height
        x2 = x2 / img_width
        y2 = y2 / img_height
        return x1, y1, x2, y2

def evaluate(model_name_or_path, model_type, data_fn, image_dir, use_placeholder, topk, resize_to_pixels=None):
    # initialize model
    data_processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = data_processor.tokenizer
    for k, v in tokenizer.added_tokens_encoder.items():
        print(v, k)

    if model_type == "qwen2vl":
        print(f"Loading model with Qwen2-VL backbone from {model_name_or_path}")
        model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()
        grounding_system_message = "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task."
    elif model_type == "qwen25vl":
        print(f"Loading model with Qwen2.5-VL backbone from {model_name_or_path}")
        model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2"
        ).eval()
        grounding_system_message = "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    print(f"Loaded model from {model_name_or_path}")

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[
            tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        ]
    )

    # load data
    with open(data_fn, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_fn}")

    results = []
    for i, example in tqdm(enumerate(data), total=len(data)):

        ele = {
            "file_name": example["image_path"],
            "ui_type": example["GUI_types"][0] if example["GUI_types"] else "unknown",  # 取第一个 GUI 类型            "group": example["group"]
            "id": example["id"],
            "instruction": example["instruction"],
            "img_size": example["image_size"],
            "boxes_type": example["box_type"],
            "box_coordinates": example["box_coordinates"],
            "hit_top1": 0,
            "overlap_top1": 0,
            "hit_topk": 0,
            "overlap_topk": 0,
        }
        
        image_path = os.path.join(image_dir, example["image_path"])
        image = Image.open(image_path)
        # resize the image if needed
        image_width, image_height = example["image_size"]
        if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
            resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
            image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
            image = image.resize((image_width_resized, image_height_resized))
            ele["img_size_resized"] = [image_width_resized, image_height_resized]
        else:
            ele["img_size_resized"] = None
        
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": grounding_system_message,
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image, # PIL.Image.Image or str to path
                        # "image_url": "https://xxxxx.png" or "https://xxxxx.jpg" or "file://xxxxx.png" or "data:image/png;base64,xxxxxxxx", will be split by "base64,"
                    },
                    {
                        "type": "text",
                        "text": example["instruction"]
                    },
                ],
            },
        ]

        pred = inference(conversation, model, tokenizer, data_processor, logits_processor=logits_processor_pointer, use_placeholder=use_placeholder, topk=3)
        topk_points = pred["topk_points"]
        # gt_bbox = ele["bbox_x1y1x2y2"]
        boxes_type = ele["boxes_type"]

        # compute 

        def _is_point_in_polygon(point, polygon):
            x, y = point
            n = len(polygon) // 2
            inside = False

            j = n - 1
            for i in range(n):
                xi, yi = polygon[i * 2], polygon[i * 2 + 1]
                xj, yj = polygon[j * 2], polygon[j * 2 + 1]

                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                    inside = not inside
                j = i

            return inside
        
        if boxes_type == "bbox":
          bbox_coordinates = ele['box_coordinates']
          gt_bbox = normalize_bbox(
            [bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[0]+bbox_coordinates[2], bbox_coordinates[1]+ bbox_coordinates[3]],
            ele['img_size'][0],
            ele['img_size'][1]
          )
          px, py = topk_points[0]
          x1, y1, x2, y2 = gt_bbox

          if (x1 <= px <= x2) and (y1 <= py <= y2):
              ele["hit_top1"] = 1
              ele["hit_topk"] = 1

          # pred_bbox = [px - IMAGE_PATCH_SIZE, py - IMAGE_PATCH_SIZE, px + IMAGE_PATCH_SIZE, py + IMAGE_PATCH_SIZE]
          # if do_boxes_overlap(pred_bbox, gt_bbox):
          #     ele["overlap_top1"] = 1
          #     ele["overlap_topk"] = 1

          for px, py in topk_points[1:]:
              if (x1 <= px <= x2) and (y1 <= py <= y2):
                  ele["hit_topk"] = 1
              # pred_bbox = [px - IMAGE_PATCH_SIZE, py - IMAGE_PATCH_SIZE, px + IMAGE_PATCH_SIZE, py + IMAGE_PATCH_SIZE]
              # if do_boxes_overlap(pred_bbox, gt_bbox):
              #     ele["overlap_topk"] = 1
        elif boxes_type == "polygon":
          center_point = topk_points[0][0]*ele["img_size"][0] , topk_points[0][1]*ele["img_size"][1]
          bbox_coordinates = ele["box_coordinates"]
          print(ele["img_size"][0], ele["img_size"][1])
          print(center_point)
          print(bbox_coordinates)
          print("-"*20)
          if _is_point_in_polygon(center_point, bbox_coordinates):
            ele["hit_top1"] = 1
            ele["hit_topk"] = 1

          for px, py in topk_points[1:]:
            center_point = px*ele["img_size"][0] , py*ele["img_size"][1]
            if _is_point_in_polygon(center_point, bbox_coordinates):
              ele["hit_topk"] = 1
        elif  boxes_type == "refusal": # 暂时不支持
          pass

        results.append(ele)
    
    return results


def get_metric(list_of_examples):
    classification_result = {}
    accuracy_dict_group = {}
    correct = 0
    total = len(list_of_examples)

    with open(
            "/ossfs/workspace/OSWorld-G/benchmark/classification_result.json"
      ) as f:
      classification_result = json.load(f)

      for item in list_of_examples:
        instance_group_list = []
        for cls_type, classification_items in classification_result[
                "classified"
            ].items():
                for classification_item in classification_items:
                    if classification_item["id"] == item["id"]:
                        instance_group_list.append(cls_type)
                        break
        item["instance_group_list"] = instance_group_list
        if len(instance_group_list) == 0:
          instance_group_list.append("unclassified")
          if "unclassified" not in accuracy_dict_group:
              accuracy_dict_group["unclassified"] = {
                  "total": 0,
                  "correct": 0,
                  "accuracy": 0,
              }
          accuracy_dict_group["unclassified"]["total"] += 1
        else:
          for instance_group in instance_group_list:
              if instance_group not in accuracy_dict_group:
                  accuracy_dict_group[instance_group] = {
                      "total": 0,
                      "correct": 0,
                      "accuracy": 0,
                  }
              accuracy_dict_group[instance_group]["total"] += 1
        if item["hit_top1"] == 1:
          correct += 1
          for instance_group in item["instance_group_list"]:
              accuracy_dict_group[instance_group]["correct"] += 1
      accuracy = correct / total
      for group in accuracy_dict_group:
        accuracy_dict_group[group][
                "accuracy"
            ] = f"{(accuracy_dict_group[group]['correct'] / accuracy_dict_group[group]['total'])*100:.2f}%"
      return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_dict_group": accuracy_dict_group,
        }
      




"""
# cd to project root directory
python eval/screenSpot_pro.py --save_path <path_to_save_results> --data_path <path_to_data>
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="qwen25vl", choices=["qwen2vl", "qwen25vl"])
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/GUI-Actor-7B-Qwen2.5-VL")
    parser.add_argument("--save_path", type=str, default="./test_code_result_v2/")
    parser.add_argument("--data_path", type=str, default="/ossfs/workspace/OSWorld-G/benchmark")
    parser.add_argument("--resize_to_pixels", type=int, default=3200*1800, help="If set to <0, will not resize the image.")
    parser.add_argument('--no-placeholder', dest='use_placeholder', action='store_false', help='Disable the placeholder')
    parser.add_argument('--topk', type=int, default=3, help='Topk')
    parser.set_defaults(use_placeholder=True)

    args = parser.parse_args()

    resize_to_pixels = args.resize_to_pixels if args.resize_to_pixels > 0 else None
    image_dir = os.path.join(args.data_path, "images")
    data_fn = os.path.join(args.data_path, "OSWorld-G_refined.json")
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    pred_path = f"{save_path}/OSWorld-G_all_preds_StandardResize.json"
    metric_path = f"{save_path}/OSWorld-G_all_preds_StandardResize.txt"

    if os.path.exists(metric_path):
        exit()

    if os.path.exists(pred_path):
        print(f"Loading predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        print(f"Evaluating {args.model_name_or_path}...")
        results = evaluate(args.model_name_or_path, args.model_type, data_fn, image_dir, args.use_placeholder, args.topk, resize_to_pixels)
        with open(pred_path, "w") as f:
            json.dump(results, f)
        print(f"Saved {len(results)} predictions to {pred_path}")

    if not os.path.exists(metric_path):
        results = get_metric(results)
        print(f"Evaluation Results:")
        print(f"Total samples: {results['total']}")
        print(f"Correct predictions: {results['correct']}")
        print(f"Accuracy: {results['accuracy']*100:.2f}%")
        print(f"Accuracy by Group:")
        print(results["accuracy_dict_group"])
        for group, stats in results["accuracy_dict_group"].items():
            print(
                f"  {group}: {stats['accuracy']*100:.2}% ({stats['correct']}/{stats['total']})"
            )