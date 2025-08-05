import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


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
                print(" " * dash + f"├─ {key}:")
                # 递归处理子节点，缩进+2
                print_json_tree(node[key], dash + 2)
        
        elif isinstance(node, list):
            if len(node) > 0:
                print(" " * dash + f"├─ List: 共{len(node)}个")
                print_json_tree(node[0], dash + 2, is_first_in_list=True)
        else:
            value_str = str(node)
            if len(value_str) > 50:  # 过长内容截断
                value_str = value_str[:50] + "..."
            print(" " * dash + f"└─ 值: {value_str}")
    
    try:
        # 加载JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        print("COCO JSON注释结构 (树状视图):")
        print("=" * 50)
        # 开始递归打印树状结构
        print_json_tree(coco_data)
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 {json_file}")
    except json.JSONDecodeError:
        print(f"错误: {json_file} 不是有效的JSON文件")
    except Exception as e:
        print(f"处理出错: {str(e)}")


def visualize_bbox_sizes(json_file: str, save_path: str = None):
    """
    Visualize the distribution of bounding box sizes in a COCO dataset
    
    :param json_file: Path to the COCO annotation JSON file
    :param save_path: Path to save the figure. If None, tries to display it
    """
    try:
        # Load JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Extract bounding boxes
        bboxes = []
        areas = []
        
        for annotation in coco_data['annotations']:
            bbox = annotation['bbox']  # [x, y, width, height]
            bboxes.append(bbox)
            areas.append(bbox[2] * bbox[3])  # width * height
        
        if not bboxes:
            print("No bounding boxes found in the dataset")
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
            print(f"Figure saved to {save_path}")
        else:
            # Try to display the plot
            try:
                plt.show()
            except:
                # If display fails, save to a default location
                default_save_path = "bbox_size_distribution.png"
                plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
                print(f"Display failed. Figure saved to {default_save_path}")
        
        plt.close()  # Close the figure to free memory
        
        # Print statistics
        print(f"Total bounding boxes: {len(bboxes)}")
        print(f"Average width: {np.mean(widths):.2f} pixels")
        print(f"Average height: {np.mean(heights):.2f} pixels")
        print(f"Average area: {np.mean(areas):.2f} pixels²")
        print(f"Min area: {np.min(areas):.2f} pixels²")
        print(f"Max area: {np.max(areas):.2f} pixels²")
        print(f"Average aspect ratio: {np.mean(aspect_ratios):.2f}")
        print(f"Min aspect ratio: {np.min(aspect_ratios):.2f}")
        print(f"Max aspect ratio: {np.max(aspect_ratios):.2f}")
        
    except FileNotFoundError:
        print(f"Error: File not found {json_file}")
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not a valid JSON file")
    except Exception as e:
        print(f"Error processing file: {str(e)}")


def main():
    dataset_dir = "/home/linkst/workplace/datasets/Chromosome20240904_NoAug_NoResize_coco/"
    anno_file = dataset_dir + "train/" + "_annotations.coco.json"
    # print_coco_structure(anno_file)
    visualize_bbox_sizes(anno_file, "bbox_visualization.png")
    

if __name__ == "__main__":
    main()