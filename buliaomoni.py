#!/usr/bin/env python3
"""
Main Surface Completion Function
Implements the buliaomoni function from MATLAB
"""

import numpy as np
from scipy.spatial import cKDTree
from surface_completion import SurfaceCompletion


class BuliaoMoni:
    """Main surface completion class"""
    
    def __init__(self):
        self.surface_completion = SurfaceCompletion()
    
    def buliaomoni(self, datas, kk, normal_pts, juli, zhengping):
        """
        Main surface completion function
        Equivalent to: buliaomoni(datas, kk, normal_pts, juli, zhengping)
        
        Parameters:
        - datas: input point cloud data
        - kk: processing mode (1 for direct, 2 for with rotation)
        - normal_pts: normal vector for rotation
        - juli: distance parameter
        - zhengping: refinement flag (0 or 1)
        """
        
        # Rotate to Z-axis normal coordinate system
        if kk != 1:
            normal2 = normal_pts.flatten()
            if np.linalg.norm(normal2) == 0:
                normal2 = np.array([0, 0, 1])
            else:
                normal2 = normal2 / np.linalg.norm(normal2)
            
            if not np.allclose(normal2, [0, 0, 1], atol=1e-12):
                R = self.xuanzhuanjuzhen2(normal2, np.array([0, 0, 1]))
                R2 = self.xuanzhuanjuzhen2(np.array([0, 0, 1]), normal2)
            else:
                R = np.eye(3)
                R2 = np.eye(3)
            
            center = np.mean(datas[:, 0:3], axis=0)
            data = datas[:, 0:3] - center
            data = data @ R
        else:
            data = datas[:, 0:3]
            center = np.array([0, 0, 0])
            R2 = np.eye(3)
        
        # Determine boundaries and create plane
        data_zhi = data[:, 0:3]
        data_out, xmin, xmax, ymin, ymax, zmax = self.surface_completion.quedingbianjie(data_zhi, 20)
        
        # Calculate step size
        tree = cKDTree(data[:, 0:3])
        distances, _ = tree.query(data[:, 0:3], k=3)
        d = np.mean(distances[:, 1])  # Average distance to second nearest neighbor
        
        # Create initial plane grid
        k = 0
        xi = 1
        plane = []
        
        for i in np.arange(xmin, xmax + d, d):
            yi = 1
            for j in np.arange(ymin, ymax + d, d):
                plane.append([i, j, xi, yi])
                yi += 1
                k += 1
            xi += 1
        
        plane = np.array(plane)
        
        # Initialize buliao array
        buliao = np.column_stack([
            plane[:, 0:2],  # x, y
            np.full(len(plane), zmax),  # z (zmax)
            np.zeros(len(plane)),  # occupancy flag
            np.zeros(len(plane)),  # value (0 = unprocessed, 1 = processed)
            plane[:, 2:4]  # grid indices
        ])
        
        # Downward completion process
        kkk = 0
        high = []
        
        for zz in np.arange(zmax - juli, zmax + 0.1, 0.1):
            buliao = self.surface_completion.down_buliao(data, buliao, d * 2, 1, 0.1)
            high.append(zmax - kkk * 0.1)
            kkk += 1
        
        high.append(zmax - kkk * 0.1)
        high = np.array(high)
        
        # Remove noise
        buliao = self.surface_completion.quzao(buliao)
        buliao_out = buliao.copy()
        
        # Reorder columns following MATLAB logic exactly:
        # buliao_out = [buliao_out(:,1:3), buliao_out(:,4), buliao_out(:,6:7), buliao_out(:,5)]
        buliao_out = np.column_stack([
            buliao_out[:, 0:3],  # x, y, z (columns 1-3)
            buliao_out[:, 3],    # occupancy flag (column 4)
            buliao_out[:, 5:7],  # grid indices (columns 6-7)
            buliao_out[:, 4]     # value (column 5)
        ])
        
        # Surface interpolation
        data_out = self.surface_completion.fantan(buliao_out, high)
        print(f"    Debug buliaomoni: fantan returned {len(data_out)} points")
        
        if len(data_out) == 0:
            # Return empty result if no interpolation possible
            return np.array([]).reshape(0, 3), data_out, high, np.array([]).reshape(0, 3)
        
        minhigh = np.min(high)
        data_zhi_3D = data_zhi[data_zhi[:, 2] > minhigh, :]
        
        # Create polygon for filtering
        polyin = self.surface_completion.wanggehua(data_zhi_3D)
        print(f"    Debug buliaomoni: polygon created: {polyin is not None}")
        
        # Follow MATLAB logic exactly: data_out2 = data_out(:,1:3)
        data_out2 = data_out[:, 0:3]
        data_out2_2D = data_out2[:, 0:2]
        
        print(f"    Debug buliaomoni: completion points range:")
        print(f"      X: {np.min(data_out2_2D[:, 0]):.3f} to {np.max(data_out2_2D[:, 0]):.3f}")
        print(f"      Y: {np.min(data_out2_2D[:, 1]):.3f} to {np.max(data_out2_2D[:, 1]):.3f}")
        print(f"    Debug buliaomoni: data_zhi_3D range:")
        print(f"      X: {np.min(data_zhi_3D[:, 0]):.3f} to {np.max(data_zhi_3D[:, 0]):.3f}")
        print(f"      Y: {np.min(data_zhi_3D[:, 1]):.3f} to {np.max(data_zhi_3D[:, 1]):.3f}")
        print(f"      Z: {np.min(data_zhi_3D[:, 2]):.3f} to {np.max(data_zhi_3D[:, 2]):.3f}")
        
        # Use relaxed polygon filtering - keep points that are close to polygon boundary
        # Filter points inside polygon: TFin = isinterior(polyin, data_out2_2D(:,1), data_out2_2D(:,2))
        if polyin is not None:
            try:
                from shapely.geometry import Point
                # Check if points are inside polygon
                inside_polygon = np.array([polyin.contains(Point(x, y)) for x, y in data_out2_2D])
                # Also check if points are within a small distance of polygon boundary
                boundary_distance = np.array([polyin.exterior.distance(Point(x, y)) for x, y in data_out2_2D])
                # Keep points that are either inside or very close to boundary
                TFin = inside_polygon | (boundary_distance < 1.0)  # 1.0 unit tolerance
                print(f"    Debug buliaomoni: points inside polygon: {np.sum(inside_polygon)}/{len(inside_polygon)}")
                print(f"    Debug buliaomoni: points near boundary: {np.sum(boundary_distance < 1.0)}/{len(boundary_distance)}")
                print(f"    Debug buliaomoni: total kept: {np.sum(TFin)}/{len(TFin)}")
            except:
                TFin = np.ones(len(data_out2_2D), dtype=bool)
                print(f"    Debug buliaomoni: polygon filtering failed, keeping all points")
        else:
            TFin = np.ones(len(data_out2_2D), dtype=bool)
            print(f"    Debug buliaomoni: no polygon created, keeping all points")
        
        # Refinement: data_out2 = data_out2(TFin,:)
        if zhengping == 1 and len(data_out) > 50:
            data_out2 = self.surface_completion.refine_buliao3(data_out[TFin, :])
        else:
            data_out2 = data_out2[TFin, :]
        
        print(f"    Debug buliaomoni: after polygon filtering: {len(data_out2)} points")
        
        if data_out2.shape[1] != 3:
            print(f"Warning: data_out2 shape is {data_out2.shape}, expected (N, 3)")
            if len(data_out2) > 0:
                data_out2 = data_out2[:, 0:3]
            else:
                data_out2 = np.array([]).reshape(0, 3)
        
        # Rotate back to original coordinate system
        if kk != 1:
            if R2.shape != (3, 3):
                print(f"Warning: R2 shape is {R2.shape}, expected (3, 3)")
                R2 = np.eye(3)
            
            if len(center) != 3:
                print(f"Warning: center length is {len(center)}, expected 3")
                center = np.array([0, 0, 0])
            
            center = center.reshape(1, 3)
            rotated = data_out2 @ R2
            data_out3 = rotated + np.tile(center, (len(rotated), 1))
            buliao_out = np.column_stack([
                buliao_out[:, 0:3] @ R2,
                buliao_out[:, 3],
                buliao_out[:, 4:6],
                buliao_out[:, 6]
            ])
        else:
            data_out3 = data_out2
            buliao_out = np.column_stack([
                buliao_out[:, 0:3],
                buliao_out[:, 3],
                buliao_out[:, 4:6],
                buliao_out[:, 6]
            ])
        
        # Return the newly completed points (not the original transformed points)
        # data_out3 contains the newly generated points
        buliao_out = data_out3
        
        return buliao_out, data_out, high, data_out3
    
    def xuanzhuanjuzhen2(self, normal1, normal2):
        """
        Create rotation matrix from normal1 to normal2 using quaternions
        """
        v1 = normal1 / np.linalg.norm(normal1)
        v2 = normal2 / np.linalg.norm(normal2)
        
        if np.linalg.norm(v1 + v2) == 0:
            q = np.array([0, 0, 0, 0])
        else:
            u = np.cross(v1, v2)
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-10:
                # Vectors are parallel, no rotation needed
                return np.eye(3)
            u = u / u_norm
            
            theta = np.arccos(np.clip(np.dot(v1, v2), -1, 1)) / 2
            q = np.array([np.cos(theta), np.sin(theta) * u[0], 
                         np.sin(theta) * u[1], np.sin(theta) * u[2]])
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [2*q[0]**2-1+2*q[1]**2, 2*(q[1]*q[2]+q[0]*q[3]), 2*(q[1]*q[3]-q[0]*q[2])],
            [2*(q[1]*q[2]-q[0]*q[3]), 2*q[0]**2-1+2*q[2]**2, 2*(q[2]*q[3]+q[0]*q[1])],
            [2*(q[1]*q[3]+q[0]*q[2]), 2*(q[2]*q[3]-q[0]*q[1]), 2*q[0]**2-1+2*q[3]**2]
        ])
        
        return R


def tianbu_guocheng(data_new, normal, num, juli, zhengping):
    """
    Surface completion process wrapper
    Equivalent to: tianbu_guocheng(data_new, normal, num, juli, zhengping)
    """
    buliao_processor = BuliaoMoni()
    
    buliao_out_x1, data_out, high, data_out3 = buliao_processor.buliaomoni(
        data_new, 2, normal, juli, zhengping
    )
    
    print(f"    Debug tianbu_guocheng: buliao_out_x1 shape: {buliao_out_x1.shape}")
    print(f"    Debug tianbu_guocheng: data_new shape: {data_new.shape}")
    
    if len(buliao_out_x1) > 0:
        data_new1 = np.vstack([data_new, buliao_out_x1[:, 0:3]])
        print(f"    Debug tianbu_guocheng: data_new1 shape: {data_new1.shape}")
    else:
        data_new1 = data_new
        print(f"    Debug tianbu_guocheng: no completion points, data_new1 = data_new")
    
    return data_new1, buliao_out_x1
