#!/usr/bin/env python3
"""
Main Point Cloud Completion Script
Python implementation equivalent to main.m

This script processes building point clouds through:
1. Downsampling
2. Noise removal
3. Orientation adjustment
4. Surface completion in 6 directions
5. Result refinement and saving
"""

import numpy as np
import os
import sys
from pointcloud_completion import PointCloudProcessor
from buliaomoni import tianbu_guocheng


def main():
    """
    Main processing function - equivalent to main.m
    """
    print("Starting Point Cloud Completion Process...")
    
    # Initialize processor
    processor = PointCloudProcessor()
    
    # Set input file
    input_file = 'B5.txt'  # Replace with your filename
    
    # Read point cloud
    print(f"Reading point cloud from {input_file}...")
    data_all = processor.read_point_cloud(input_file)
    if data_all is None:
        print("Failed to read input file")
        return
    
    datas = data_all[:, 0:3]  # X Y Z coordinates
    
    # Keep colors if available
    if processor.colors is not None:
        colors = processor.colors  # R G B colors
        print(f"Color information available: {colors.shape}")
    
    print(f"Original point cloud: {datas.shape[0]} points")
    
    # Step 1: Downsample
    print("Step 1: Downsampling...")
    datas_downsample = processor.pointcloud_downsample(datas, 0.2)
    print(f"After downsampling: {datas_downsample.shape[0]} points")
    
    # Step 2: Remove noise
    print("Step 2: Noise removal...")
    datas_qz = processor.datas_quzao(datas_downsample)
    print(f"After noise removal: {datas_qz.shape[0]} points")
    
    # Step 3: Adjust building orientation using BBVM method
    print("Step 3: Orientation adjustment...")
    data_new, R2, RR2 = processor.tiaozheng_zitai3(datas_qz)
    print(f"After orientation adjustment: {data_new.shape[0]} points")
    
    # Save intermediate result
    processor.save_data(data_new[:, 0:3], '/Users/duli/CS/Project/little Project/20250902', 'data_new.txt')
    
    data_new = data_new[:, 0:3]
    
    # Step 4: Surface completion process
    print("Step 4: Surface completion...")
    
    # Complete surfaces in 6 directions
    print("  Completing surface in +X direction...")
    data_new1, buliao_out_x1 = tianbu_guocheng(data_new, np.array([1, 0, 0]), '1', 5, 0)
    print(f"    Added {len(buliao_out_x1)} points, total: {data_new1.shape[0]}")
    
    print("  Completing surface in -X direction...")
    data_new2, buliao_out_x2 = tianbu_guocheng(data_new1, np.array([-1, 0, 0]), '2', 5, 0)
    print(f"    Added {len(buliao_out_x2)} points, total: {data_new2.shape[0]}")
    
    print("  Completing surface in +Y direction...")
    data_new3, buliao_out_y1 = tianbu_guocheng(data_new2, np.array([0, 1, 0]), '3', 5, 0)
    print(f"    Added {len(buliao_out_y1)} points, total: {data_new3.shape[0]}")
    
    print("  Completing surface in -Y direction...")
    data_new4, buliao_out_y2 = tianbu_guocheng(data_new3, np.array([0, -1, 0]), '4', 5, 0)
    print(f"    Added {len(buliao_out_y2)} points, total: {data_new4.shape[0]}")
    
    print("  Completing surface in +Z direction...")
    data_new5, buliao_out_z1 = tianbu_guocheng(data_new4, np.array([0, 0, 1]), '5', 5, 0)
    print(f"    Added {len(buliao_out_z1)} points, total: {data_new5.shape[0]}")
    
    print("  Completing surface in -Z direction...")
    data_new6, buliao_out_z2 = tianbu_guocheng(data_new5, np.array([0, 0, -1]), '6', 5, 0)
    print(f"    Added {len(buliao_out_z2)} points, total: {data_new6.shape[0]}")
    
    print(f"After surface completion: {data_new6.shape[0]} points")
    
    # Step 5: Iterative multi-layer completion process
    print("Step 5: Iterative completion...")
    directionx2 = np.array([-1, 0, 0])
    x2 = 'x2'
    data_new_in = data_new
    k = 1
    
    for i in range(7):
        print(f"  Iteration {i+1}/7...")
        data_new_out, buliao_out_x1 = tianbu_guocheng(data_new_in, directionx2, x2, k, 1)
        k += 3
        data_new_in = data_new_out
    
    data_new1 = data_new_in
    data_new2 = data_new4
    
    data_f = data_new2
    print(f"After iterative completion: {data_f.shape[0]} points")
    
    # Step 6: Transform back to original coordinate system
    print("Step 6: Coordinate transformation...")
    # TEMPORARILY skip coordinate transformation to test if completion points exist
    data_final = data_f[:, 0:3]
    print(f"TEMPORARILY skipping coordinate transformation: {data_final.shape[0]} points")
    
    # Step 7: Final filtering and refinement
    print("Step 7: Final filtering...")
    data_out = processor.xiufu_datas_qz(data_final, datas)
    print(f"After final filtering: {data_out.shape[0]} points")
    
    # Step 8: Save final result
    print("Step 8: Saving results...")
    success = processor.save_data(data_out[:, 0:3], '/Users/duli/CS/Project/little Project/20250902', 'data_final.txt')
    
    if success:
        print("Point cloud completion completed successfully!")
        print("\nSummary:")
        print(f"  Original points: {datas.shape[0]}")
        print(f"  After downsampling: {datas_downsample.shape[0]}")
        print(f"  After noise removal: {datas_qz.shape[0]}")
        print(f"  After orientation adjustment: {data_new.shape[0]}")
        print(f"  After surface completion: {data_f.shape[0]}")
        print(f"  Final output points: {data_out.shape[0]}")
        print(f"  Completion ratio: {data_out.shape[0] / datas.shape[0] * 100:.1f}%")
    else:
        print("Error saving final results")


def test_with_sample_data():
    """
    Test function with sample data
    """
    print("Testing with sample data...")
    
    # Create sample point cloud data
    np.random.seed(42)
    n_points = 1000
    
    # Create a simple building-like structure
    x = np.random.uniform(-5, 5, n_points)
    y = np.random.uniform(-5, 5, n_points)
    z = np.random.uniform(0, 10, n_points)
    
    # Add some structure (walls)
    wall_mask = (np.abs(x) > 4.5) | (np.abs(y) > 4.5)
    z[wall_mask] = np.random.uniform(8, 10, np.sum(wall_mask))
    
    sample_data = np.column_stack([x, y, z])
    
    # Save sample data
    sample_file = 'sample_building.txt'
    np.savetxt(sample_file, sample_data, fmt='%.6f', delimiter='\t')
    print(f"Sample data saved to {sample_file}")
    
    # Process sample data
    processor = PointCloudProcessor()
    data_all = processor.read_point_cloud(sample_file)
    
    if data_all is not None:
        print(f"Sample data loaded: {data_all.shape[0]} points")
        
        # Test downsampling
        downsampled = processor.pointcloud_downsample(data_all, 0.5)
        print(f"Downsampled to: {downsampled.shape[0]} points")
        
        # Test noise removal
        filtered = processor.datas_quzao(downsampled)
        print(f"After noise removal: {filtered.shape[0]} points")
        
        # Test orientation adjustment
        oriented, R2, RR2 = processor.tiaozheng_zitai3(filtered)
        print(f"After orientation adjustment: {oriented.shape[0]} points")
        
        print("Sample data processing completed successfully!")
    
    # Clean up
    if os.path.exists(sample_file):
        os.remove(sample_file)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_with_sample_data()
    else:
        main()
