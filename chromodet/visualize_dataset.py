# Copyright (c) OpenMMLab. All rights reserved.
import json
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


def print_coco_structure(json_file: str):

    def print_json_tree(node, dash=0, is_first_in_list=False):
        """
        递归打印JSON的树状结构
        :param node: 当前处理的JSON节点（dict/list/基本类型）
        :param dash: 缩进空格数（控制层级）
        :param is_first_in_list: 是否为列表中的第一个元素（避免重复打印相同结构）
        """
        if isinstance(node, dict):
            for key in node:
                print(' ' * dash + f'├─ {key}:')
                # 递归处理子节点，缩进+2
                print_json_tree(node[key], dash + 2)

        elif isinstance(node, list):
            if len(node) > 0:
                print(' ' * dash + f'├─ List: 共{len(node)}个')
                print_json_tree(node[0], dash + 2, is_first_in_list=True)
        else:
            value_str = str(node)
            if len(value_str) > 50:  # 过长内容截断
                value_str = value_str[:50] + '...'
            print(' ' * dash + f'└─ 值: {value_str}')

    try:
        # 加载JSON文件
        with open(json_file, encoding='utf-8') as f:
            coco_data = json.load(f)

        print('COCO JSON注释结构 (树状视图):')
        print('=' * 50)
        # 开始递归打印树状结构
        print_json_tree(coco_data)

    except FileNotFoundError:
        print(f'错误: 未找到文件 {json_file}')
    except json.JSONDecodeError:
        print(f'错误: {json_file} 不是有效的JSON文件')
    except Exception as e:
        print(f'处理出错: {e!s}')


def visualize_bbox_sizes(json_file: str, save_path: str = None):
    """
    Visualize the distribution of bounding box sizes in a COCO dataset

    :param json_file: Path to the COCO annotation JSON file
    :param save_path: Path to save the figure. If None, tries to display it
    """
    try:
        # Load JSON file
        with open(json_file, encoding='utf-8') as f:
            coco_data = json.load(f)

        # Extract bounding boxes
        bboxes = []
        areas = []

        for annotation in coco_data['annotations']:
            bbox = annotation['bbox']  # [x, y, width, height]
            bboxes.append(bbox)
            areas.append(bbox[2] * bbox[3])  # width * height

        if not bboxes:
            print('No bounding boxes found in the dataset')
            return

        bboxes = np.array(bboxes)
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        areas = np.array(areas)

        # Calculate aspect ratios (width/height)
        aspect_ratios = widths / heights

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('COCO Bounding Box Size Distribution')

        # Plot 1: Width vs Height scatter plot
        axes[0, 0].scatter(widths, heights, alpha=0.5)
        axes[0, 0].set_xlabel('Width')
        axes[0, 0].set_ylabel('Height')
        axes[0, 0].set_title('Bounding Box Width vs Height')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Histogram of areas
        axes[0, 1].hist(areas, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Area (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Bounding Box Areas')
        axes[0, 1].set_yscale('log')

        # Plot 3: Histogram of widths
        axes[0, 2].hist(widths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Width (pixels)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Bounding Box Widths')
        axes[0, 2].set_yscale('log')

        # Plot 4: Histogram of heights
        axes[1, 0].hist(heights, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Height (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Bounding Box Heights')
        axes[1, 0].set_yscale('log')

        # Plot 5: Histogram of aspect ratios
        axes[1, 1].hist(aspect_ratios, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Aspect Ratio (width/height)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Bounding Box Aspect Ratios')
        axes[1, 1].set_yscale('log')

        # Plot 6: Aspect ratio vs Area scatter plot
        axes[1, 2].scatter(areas, aspect_ratios, alpha=0.5)
        axes[1, 2].set_xlabel('Area (pixels)')
        axes[1, 2].set_ylabel('Aspect Ratio (width/height)')
        axes[1, 2].set_title('Aspect Ratio vs Area')
        axes[1, 2].set_xscale('log')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # If save_path is provided, save the figure, otherwise try to display it
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Figure saved to {save_path}')
        else:
            # Try to display the plot
            try:
                plt.show()
            except:
                # If display fails, save to a default location
                default_save_path = 'bbox_size_distribution.png'
                plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
                print(f'Display failed. Figure saved to {default_save_path}')

        plt.close()  # Close the figure to free memory

        # Print statistics
        print(f'Total bounding boxes: {len(bboxes)}')
        print(f'Average width: {np.mean(widths):.2f} pixels')
        print(f'Average height: {np.mean(heights):.2f} pixels')
        print(f'Average area: {np.mean(areas):.2f} pixels²')
        print(f'Min area: {np.min(areas):.2f} pixels²')
        print(f'Max area: {np.max(areas):.2f} pixels²')
        print(f'Average aspect ratio: {np.mean(aspect_ratios):.2f}')
        print(f'Min aspect ratio: {np.min(aspect_ratios):.2f}')
        print(f'Max aspect ratio: {np.max(aspect_ratios):.2f}')

    except FileNotFoundError:
        print(f'Error: File not found {json_file}')
    except json.JSONDecodeError:
        print(f'Error: {json_file} is not a valid JSON file')
    except Exception as e:
        print(f'Error processing file: {e!s}')


def visualize_center_distances(json_file: str, save_path: str = None):
    """
    Visualize the distribution of distances between chromosome centers in a COCO dataset,
    grouped by chromosome categories.

    :param json_file: Path to the COCO annotation JSON file
    :param save_path: Path to save the figure. If None, tries to display it
    """
    try:
        # Load JSON file
        with open(json_file, encoding='utf-8') as f:
            coco_data = json.load(f)

        # Create category id to name mapping
        category_map = {
            cat['id']: cat['name'] for cat in coco_data['categories']
        }

        # Group annotations by image
        image_annotations = defaultdict(list)
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']  # [x, y, width, height]
            # Calculate center point (x + width/2, y + height/2)
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            category_id = annotation['category_id']
            category_name = category_map[category_id]

            image_annotations[image_id].append(
                {
                    'center': (center_x, center_y),
                    'category_id': category_id,
                    'category_name': category_name,
                }
            )

        if not image_annotations:
            print('No annotations found in the dataset')
            return

        # Calculate distances between centers
        all_distances = []
        same_category_distances = defaultdict(list)
        diff_category_distances = defaultdict(list)

        for image_id, annotations in image_annotations.items():
            # Calculate distances between all pairs in the same image
            for ann1, ann2 in combinations(annotations, 2):
                # Calculate Euclidean distance
                dist = np.sqrt(
                    (ann1['center'][0] - ann2['center'][0]) ** 2
                    + (ann1['center'][1] - ann2['center'][1]) ** 2
                )

                all_distances.append(dist)

                # Group by category relationship
                if ann1['category_id'] == ann2['category_id']:
                    # Same category
                    same_category_distances[ann1['category_name']].append(dist)
                else:
                    # Different categories
                    diff_category_distances['between_categories'].append(dist)

        if not all_distances:
            print('No pairs of chromosomes found in the dataset')
            return

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chromosome Center Distance Distribution')

        # Plot 1: Overall distance distribution
        axes[0, 0].hist(all_distances, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Distance (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Distance Distribution')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Same category distances (sample up to 10 categories)
        categories_sampled = list(same_category_distances.keys())[:10]
        same_category_data = [
            same_category_distances[cat] for cat in categories_sampled
        ]
        if same_category_data:
            axes[0, 1].hist(
                same_category_data,
                bins=30,
                alpha=0.7,
                label=categories_sampled,
            )
            axes[0, 1].set_xlabel('Distance (pixels)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Same Category Distance Distribution')
            axes[0, 1].set_yscale('log')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Between categories vs within categories
        data_for_comparison = []
        labels = []
        if diff_category_distances['between_categories']:
            data_for_comparison.append(
                diff_category_distances['between_categories']
            )
            labels.append('Between Categories')
        if same_category_distances:
            # Combine all same category distances
            all_same = []
            for distances in same_category_distances.values():
                all_same.extend(distances)
            data_for_comparison.append(all_same)
            labels.append('Within Categories')

        if data_for_comparison:
            axes[1, 0].hist(
                data_for_comparison, bins=50, alpha=0.7, label=labels
            )
            axes[1, 0].set_xlabel('Distance (pixels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Within vs Between Categories Distance')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Box plot of same category distances
        if same_category_data:
            axes[1, 1].boxplot(
                same_category_data,
                labels=[cat[:10] for cat in categories_sampled],
            )  # Truncate labels
            axes[1, 1].set_title('Distance Distribution by Category')
            axes[1, 1].set_ylabel('Distance (pixels)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # If save_path is provided, save the figure, otherwise try to display it
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Figure saved to {save_path}')
        else:
            # Try to display the plot
            try:
                plt.show()
            except:
                # If display fails, save to a default location
                default_save_path = 'center_distances_distribution.png'
                plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
                print(f'Display failed. Figure saved to {default_save_path}')

        plt.close()  # Close the figure to free memory

        # Print statistics
        print(f'Total chromosome pairs analyzed: {len(all_distances)}')
        print(f'Average distance: {np.mean(all_distances):.2f} pixels')
        print(f'Min distance: {np.min(all_distances):.2f} pixels')
        print(f'Max distance: {np.max(all_distances):.2f} pixels')
        print(f'Median distance: {np.median(all_distances):.2f} pixels')

    except FileNotFoundError:
        print(f'Error: File not found {json_file}')
    except json.JSONDecodeError:
        print(f'Error: {json_file} is not a valid JSON file')
    except Exception as e:
        print(f'Error processing file: {e!s}')


def main():
    dataset_dir = '/home/linkst/workplace/datasets/Chromosome20240904_NoAug_NoResize_coco/'
    anno_file = dataset_dir + 'train/' + '_annotations.coco.json'
    # print_coco_structure(anno_file)
    visualize_bbox_sizes(anno_file, 'bbox_visualization.png')
    visualize_center_distances(anno_file, 'center_distances_visualization.png')


if __name__ == '__main__':
    main()
