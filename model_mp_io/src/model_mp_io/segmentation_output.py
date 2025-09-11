import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class SegmentationOutput:

    def __init__(self, colors=None, alpha=0.5, class_names=None):
        """
        Initialize the segmentation output visualization engine.
        
        Sets up color palettes, transparency settings, and class name mappings
        for professional segmentation result visualization. Provides defaults
        for COCO dataset classes with 80 standard object categories.
        
        Args:
            colors (List[Tuple[int, int, int]], optional): Custom color palette for classes.
                                                          If None, generates 80 random colors.
                                                          Each color is a BGR tuple for OpenCV.
            alpha (float): Mask transparency factor [0.0, 1.0]. Default: 0.5
                          0.0 = fully transparent, 1.0 = fully opaque
            class_names (Dict[int, str], optional): Custom class ID to name mapping.
                                                   If None, uses COCO dataset class names.
                                                   
        Example:
            # Default configuration
            seg_viz = SegmentationOutput()
            
            # Custom transparency
            seg_viz = SegmentationOutput(alpha=0.3)
            
            # Custom classes
            custom_classes = {0: 'background', 1: 'foreground'}
            seg_viz = SegmentationOutput(class_names=custom_classes)
        """
        self.colors = colors or self._generate_colors(80)  # Default: 80 distinct colors
        self.alpha = alpha
        # Default COCO class names mapping - can be customized for different models
        self.class_names = class_names or {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
            19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
            24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
            28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
            74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
            79: 'toothbrush'
        }
    
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """
        Generate a diverse color palette for class visualization.
        
        Creates visually distinct colors using random generation with a fixed
        seed to ensure consistent color assignment across different runs.
        
        Args:
            num_colors (int): Number of distinct colors to generate
            
        Returns:
            List[Tuple[int, int, int]]: List of BGR color tuples for OpenCV
            
        Note:
            Uses a fixed random seed (42) to ensure reproducible color assignment
            for consistent visualization across different runs.
        """
        np.random.seed(42)  # Fixed seed for consistent color generation
        colors = []
        for _ in range(num_colors):
            colors.append(tuple(np.random.randint(0, 255, 3).tolist()))
        return colors
    
    def draw_segmentation(self, image: np.ndarray, segmentation_results: Dict[str, Any], 
                         draw_boxes: bool = True, draw_masks: bool = True,
                         thickness: int = 2, font_scale: float = 0.6, display_time: float = 0) -> np.ndarray:
        """
        Render segmentation results onto an image with comprehensive visualization.
        
        Combines mask overlay and bounding box visualization to provide complete
        segmentation result display. Supports both instance and semantic segmentation
        formats with configurable styling options.
        
        Args:
            image (np.ndarray): Input image in BGR format with shape (H, W, C)
            segmentation_results (Dict[str, Any]): Segmentation results containing:
                                                   - 'boxes': List of bounding boxes [x1, y1, x2, y2]
                                                   - 'scores': List of confidence scores [0, 1]
                                                   - 'class_ids': List of class identifiers
                                                   - 'masks': List of binary masks as numpy arrays
            draw_boxes (bool): Whether to draw bounding boxes around segments (default: True)
            draw_masks (bool): Whether to draw segmentation masks as colored overlays (default: True)
            thickness (int): Bounding box line thickness in pixels (default: 2)
            font_scale (float): Text scaling factor for class labels (default: 0.6)
            display_time (float): Time to display image in seconds, 0 = no display (default: 0)
            
        Returns:
            np.ndarray: Annotated image with segmentation visualizations in BGR format
            
        Note:
            Masks are blended with the original image using the alpha transparency
            setting defined during initialization. Each class receives a unique color
            for easy visual distinction.
            
        Example:
            seg_viz = SegmentationOutput(alpha=0.4)
            result_img = seg_viz.draw_segmentation(
                image, seg_results, draw_boxes=True, thickness=3
            )
        """
        result_image = image.copy()
        
        boxes = segmentation_results.get('boxes', [])
        scores = segmentation_results.get('scores', [])
        class_ids = segmentation_results.get('class_ids', [])
        masks = segmentation_results.get('masks', [])
        
        # Create mask overlay layer for transparent blending
        if draw_masks and masks:
            mask_overlay = np.zeros_like(image, dtype=np.uint8)
            
            for i, mask in enumerate(masks):
                if i < len(class_ids):
                    class_id = class_ids[i]
                    color = self.colors[class_id % len(self.colors)]
                    
                    # Apply mask to overlay layer with class-specific color
                    mask_colored = np.zeros_like(image, dtype=np.uint8)
                    mask_colored[mask > 0] = color
                    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_colored, 1.0, 0)
            
            # Blend mask overlay with original image using alpha transparency
            result_image = cv2.addWeighted(result_image, 1.0, mask_overlay, self.alpha, 0)
        
        # Draw bounding boxes and labels for each detected segment
        if draw_boxes and boxes:
            for i, box in enumerate(boxes):
                if i < len(scores) and i < len(class_ids):
                    x1, y1, x2, y2 = map(int, box)
                    score = scores[i]
                    class_id = class_ids[i]
                    
                    # Get color for this class
                    color = self.colors[class_id % len(self.colors)]
                    
                    # Get human-readable class name
                    class_name = self.get_class_name(class_id)
                    
                    # Draw bounding box rectangle
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
                    
                    # Prepare label text with class name, ID, and confidence
                    label = f"{class_name} ({class_id}): {score:.2f}"
                    
                    # Calculate text dimensions for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    
                    # Draw text background rectangle for better visibility
                    cv2.rectangle(
                        result_image,
                        (x1, y1 - text_height - baseline - 5),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw text label on background for visibility
                    cv2.putText(
                        result_image,
                        label,
                        (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),  # White text for contrast
                        thickness
                    )
        
        # Handle image display based on display_time parameter
        if display_time > 0:
            # Create descriptive window title
            window_name = f"Segmentation Results ({len(boxes)} objects)"
            cv2.imshow(window_name, result_image)
            
            # Calculate wait time in milliseconds
            wait_time_ms = int(display_time * 1000)
            print(f"Displaying segmentation results for {wait_time_ms} milliseconds...")
            
            # Display image and wait for specified duration
            cv2.waitKey(wait_time_ms)
            
            # Clean up window
            cv2.destroyWindow(window_name)
        
        return result_image
    
    def get_class_name(self, class_id: int) -> str:
        """
        Retrieve human-readable class name from class ID.
        
        Converts numerical class identifiers to descriptive names using
        the configured class name mapping. Supports both dictionary and
        list-based class name configurations.
        
        Args:
            class_id (int): Numerical class identifier
            
        Returns:
            str: Human-readable class name or default fallback format
            
        Example:
            seg_viz = SegmentationOutput()
            name = seg_viz.get_class_name(0)  # Returns: 'person'
            name = seg_viz.get_class_name(999)  # Returns: 'Class 999'
        """
        if isinstance(self.class_names, dict):
            return self.class_names.get(class_id, f"Class {class_id}")
        elif isinstance(self.class_names, list) and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        else:
            return f"Class {class_id}"
    
    def draw_masks_only(self, image: np.ndarray, masks: List[np.ndarray], 
                       class_ids: Optional[List[int]] = None) -> np.ndarray:
        """
        Render only segmentation masks without bounding boxes or labels.
        
        Provides a clean mask-only visualization by applying colored overlays
        for each segmentation mask with alpha blending.
        
        Args:
            image (np.ndarray): Input image in BGR format
            masks (List[np.ndarray]): List of binary segmentation masks
            class_ids (Optional[List[int]]): Optional class IDs for color selection.
                                           If None, uses sequential numbering.
            
        Returns:
            np.ndarray: Image with colored mask overlays applied
            
        Example:
            masks_only = seg_viz.draw_masks_only(image, mask_list, class_ids)
        """
        result_image = image.copy()
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            if class_ids and i < len(class_ids):
                class_id = class_ids[i]
            else:
                class_id = i
            
            color = self.colors[class_id % len(self.colors)]
            
            # Apply mask to overlay layer with class-specific color
            mask_colored = np.zeros_like(image, dtype=np.uint8)
            mask_colored[mask > 0] = color
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_colored, 1.0, 0)
        
        # Blend mask overlay with original image
        result_image = cv2.addWeighted(result_image, 1.0, mask_overlay, self.alpha, 0)
        return result_image
    
    def save_segmentation(self, image: np.ndarray, segmentation_results: Dict[str, Any], 
                         output_path: str, **kwargs) -> None:
        """
        Save image with rendered segmentation results to file.
        
        Convenience method that combines segmentation visualization and file saving.
        
        Args:
            image (np.ndarray): Input image in BGR format
            segmentation_results (Dict[str, Any]): Segmentation results to visualize
            output_path (str): File path for saving the annotated image
            **kwargs: Additional arguments passed to draw_segmentation method
        """
        result_image = self.draw_segmentation(image, segmentation_results, **kwargs)
        cv2.imwrite(output_path, result_image)
    
    def save_masks_separately(self, masks: List[np.ndarray], output_dir: str, 
                             prefix: str = "mask") -> None:
        """
        Save individual segmentation masks as separate image files.
        
        Exports each mask as a grayscale image file for individual analysis
        or further processing. Automatically creates output directory if needed.
        
        Args:
            masks (List[np.ndarray]): List of binary segmentation masks
            output_dir (str): Output directory path for saving mask files
            prefix (str): Filename prefix for generated mask files (default: "mask")
            
        Example:
            seg_viz.save_masks_separately(masks, "output/masks/", "object_mask")
            # Creates: object_mask_000.png, object_mask_001.png, etc.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, mask in enumerate(masks):
            mask_path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            # Convert mask to 0-255 grayscale range for image saving
            mask_image = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_image)
    
    def print_segmentation_summary(self, segmentation_results: Dict[str, Any]) -> None:
        """
        Print comprehensive summary of segmentation results.
        
        Displays statistical information about detected objects, class distribution,
        and average confidence scores in a readable format.
        
        Args:
            segmentation_results (Dict[str, Any]): Segmentation results dictionary
            
        Example:
            seg_viz.print_segmentation_summary(results)
            # Output:
            # Segmentation Results Summary:
            #   Detected Objects: 5
            #   Masks Generated: 5
            #   Class Distribution: {0: 2, 1: 3}
            #   Average Confidence: 0.847
        """
        boxes = segmentation_results.get('boxes', [])
        scores = segmentation_results.get('scores', [])
        class_ids = segmentation_results.get('class_ids', [])
        masks = segmentation_results.get('masks', [])
        
        print(f"Segmentation Results Summary:")
        print(f"  Detected Objects: {len(boxes)}")
        print(f"  Masks Generated: {len(masks)}")
        
        if class_ids:
            class_counts = {}
            for class_id in class_ids:
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            print(f"  Class Distribution: {class_counts}")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  Average Confidence: {avg_score:.3f}")
    
    def get_segmentation_stats(self, segmentation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics from segmentation results.
        
        Analyzes segmentation data to extract key metrics including object counts,
        class distribution, confidence scores, and total mask coverage area.
        
        Args:
            segmentation_results (Dict[str, Any]): Segmentation results dictionary
            
        Returns:
            Dict[str, Any]: Statistics dictionary containing:
                          - 'num_objects': Total number of detected objects
                          - 'num_masks': Total number of generated masks
                          - 'class_counts': Distribution of objects by class ID
                          - 'avg_score': Average confidence score across all detections
                          - 'total_mask_area': Total pixel area covered by all masks
                          
        Example:
            stats = seg_viz.get_segmentation_stats(results)
            print(f"Total coverage: {stats['total_mask_area']} pixels")
        """
        boxes = segmentation_results.get('boxes', [])
        scores = segmentation_results.get('scores', [])
        class_ids = segmentation_results.get('class_ids', [])
        masks = segmentation_results.get('masks', [])
        
        stats = {
            'num_objects': len(boxes),
            'num_masks': len(masks),
            'class_counts': {},
            'avg_score': 0.0,
            'total_mask_area': 0
        }
        
        if class_ids:
            for class_id in class_ids:
                stats['class_counts'][class_id] = stats['class_counts'].get(class_id, 0) + 1
        
        if scores:
            stats['avg_score'] = sum(scores) / len(scores)
        
        if masks:
            total_area = 0
            for mask in masks:
                total_area += np.sum(mask > 0)
            stats['total_mask_area'] = total_area
        
        return stats
