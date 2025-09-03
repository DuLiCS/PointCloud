#!/usr/bin/env python3
"""
Surface Completion Module
Implements the complex surface completion algorithms from the MATLAB codebase
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Point, Polygon as ShapelyPolygon
import warnings
warnings.filterwarnings('ignore')


class SurfaceCompletion:
    """Class for surface completion operations"""
    
    def __init__(self):
        pass
    
    def quedingbianjie(self, datas, dd):
        """
        Determine boundaries and filter data
        Equivalent to: quedingbianjie(datas, dd)
        """
        data = datas.copy()
        
        # Project to 2D plane
        data_plane = np.column_stack([data[:, 0:2], np.zeros(data.shape[0])])
        xy = data_plane[:, 0:2]
        
        xmm, xmn = np.max(xy[:, 0]), np.min(xy[:, 0])
        ymm, ymn = np.max(xy[:, 1]), np.min(xy[:, 1])
        
        # Create grid indices
        ge = np.column_stack([
            np.ceil(np.abs(xy[:, 0] - xmn) / 0.5),
            np.ceil(np.abs(xy[:, 1] - ymn) / 0.5)
        ]).astype(int)
        
        # Group points by grid
        data1 = {}
        for i in range(1, np.max(ge[:, 0]) + 1):
            for j in range(1, np.max(ge[:, 1]) + 1):
                mask = (ge[:, 0] == i) & (ge[:, 1] == j)
                if np.sum(mask) > 0:
                    data1[(i, j)] = data[mask, 0:3]
        
        # Collect valid grid cells
        data_out = np.array([[0], [0], [0]])
        for (i, j), cell_data in data1.items():
            if cell_data.shape[0] > dd:
                data_out = np.column_stack([data_out, cell_data.T])
        
        if data_out.shape[1] > 1:
            data_out = data_out[:, 1:].T
        else:
            data_out = data
        
        xmin, xmax = np.min(data_out[:, 0]), np.max(data_out[:, 0])
        ymin, ymax = np.min(data_out[:, 1]), np.max(data_out[:, 1])
        
        # Z-direction processing
        data_z = data[:, 2]
        mmz, mnz = np.max(data_z), np.min(data_z)
        
        gz = np.ceil(np.abs(data_z - mnz) / 0.2).astype(int)
        
        dataz = {}
        for i in range(1, np.max(gz) + 1):
            mask = (gz == i)
            if np.sum(mask) > 0:
                dataz[i] = data[mask, 0:3]
        
        data_outz = np.array([[0], [0], [0]])
        for i, cell_data in dataz.items():
            if cell_data.shape[0] > 100:
                data_outz = np.column_stack([data_outz, cell_data.T])
        
        if data_outz.shape[1] > 1:
            data_outz = data_outz[:, 1:].T
        else:
            data_outz = data
        
        zmax = np.max(data_outz[:, 2])
        
        return data_out, xmin, xmax, ymin, ymax, zmax
    
    def down_buliao(self, data, buliao, d, full, disn):
        """
        Downward surface completion
        Equivalent to: down_buliao(data, buliao, d, full, disn)
        """
        # Find unoccupied points
        idx = (buliao[:, 3] == 0)
        buliao_down = buliao[idx, :]
        
        # Find nearest neighbors
        tree = cKDTree(data[:, 0:3])
        distances, _ = tree.query(buliao[:, 0:3], k=1)
        
        idx_dis = distances > d
        idx2 = np.where(idx_dis, 0, np.arange(len(distances)))
        
        buliao_new = buliao.copy()
        # Following MATLAB logic: buliao_new(idx2~=0,4)=1 and buliao(idx2~=0,4)=1
        # Column 4 in MATLAB is occupancy flag (index 3 in Python)
        buliao_new[idx2 != 0, 3] = 1  # occupancy flag
        buliao[idx2 != 0, 3] = 1      # occupancy flag
        
        xx = np.unique(buliao_new[:, 0])
        yy = np.unique(buliao_new[:, 1])
        
        if full == 2:
            # Fill gaps in X direction (following MATLAB logic exactly)
            for i in range(len(xx)):
                idxx = (buliao[:, 0] == xx[i])
                nx = np.sum(buliao[idxx, 3])  # occupancy flag
                if nx >= 2:
                    datax = buliao[idxx, :]
                    idxx2 = (buliao[idxx, 3] == 1)  # occupancy flag == 1
                    xymax = np.max(datax[idxx2, 1])
                    xymin = np.min(datax[idxx2, 1])
                    
                    for j in range(buliao.shape[0]):
                        if (buliao[j, 0] == xx[i] and buliao[j, 3] == 0 and 
                            xymin < buliao[j, 1] < xymax):
                            buliao_new[j, 3] = 1  # occupancy flag
            
            # Fill gaps in Y direction
            for i in range(len(yy)):
                idxy = (buliao[:, 1] == yy[i])
                ny = np.sum(buliao[idxy, 3])
                if ny >= 2:
                    datay = buliao[idxy, :]
                    idxy2 = (buliao[idxy, 3] == 1)
                    yxmax = np.max(datay[idxy2, 0])
                    yxmin = np.min(datay[idxy2, 0])
                    
                    for j in range(buliao.shape[0]):
                        if (buliao[j, 1] == yy[i] and buliao[j, 3] == 0 and 
                            yxmin < buliao[j, 0] < yxmax):
                            buliao_new[j, 3] = 1
        
        # Update unoccupied points (following MATLAB logic exactly)
        idx3 = (buliao_new[:, 3] == 0)  # occupancy flag == 0
        buliao_new[idx3, 2] -= disn     # z coordinate -= disn
        buliao_new[idx3, 4] += 1        # value field += 1
        
        return buliao_new
    
    def quzao(self, datas):
        """
        Remove noise from surface data
        Equivalent to: quzao(datas)
        """
        data = datas.copy()
        
        # Find k-nearest neighbors
        tree = cKDTree(data[:, 0:3])
        distances, _ = tree.query(data[:, 0:3], k=5)
        
        d = np.mean(distances[:, 1])
        
        # Mark points as noise if 5th nearest neighbor distance >= 4*d
        # Following MATLAB logic: data(idx,4)=0 (set value field to 0)
        # In MATLAB: column 4 is the value field (0-indexed in Python: column 3)
        idx = (distances[:, 4] >= 4 * d)
        data[idx, 3] = 0  # Set value field to 0
        
        return data
    
    def find_pp(self, vector, vector2, p, maxz):
        """
        Find projection points
        Equivalent to: find_pp(vector, vector2, p, maxz)
        """
        n = len(vector)
        
        if p == 1 or p == n:
            p1, p2 = 0, 0
            Z1, Z2 = maxz, maxz
        else:
            # Find left neighbor
            p1, Z1 = 0, maxz
            for i in range(p-1, -1, -1):
                if vector[i] == 1:
                    p1 = i
                    Z1 = vector2[i]
                    break
            
            # Find right neighbor
            p2, Z2 = 0, maxz
            for i in range(p, n):
                if vector[i] == 1:
                    p2 = i
                    Z2 = vector2[i]
                    break
        
        return p1, p2, Z1, Z2
    
    def fantan(self, datas, high):
        """
        Surface interpolation using reflection method
        Equivalent to: fantan(datas, high)
        """
        points = datas[:, 0:3]
        value = datas[:, 4]
        XYZ = np.column_stack([datas[:, 5], datas[:, 5:7]])
        
        maxz = np.max(XYZ[:, 2])
        zmax = np.max(points[:, 2])
        
        x = XYZ[:, 0].astype(int)
        y = XYZ[:, 1].astype(int)
        z = XYZ[:, 2]
        v = value
        idx_p = np.arange(len(x))
        
        m, n = int(np.max(XYZ[:, 0])), int(np.max(XYZ[:, 1]))
        
        # Create sparse matrices
        graph = np.zeros((m, n))
        graph_v = np.zeros((m, n))
        graph_idx = np.zeros((m, n), dtype=int)
        
        for i in range(len(x)):
            if x[i] < m and y[i] < n:
                graph[x[i]-1, y[i]-1] = z[i]
                # graph_v should be 1 where there are points, 0 elsewhere
                graph_v[x[i]-1, y[i]-1] = 1 if v[i] != 0 else 0
                graph_idx[x[i]-1, y[i]-1] = idx_p[i]
        
        pp = np.zeros((len(x), 9))
        
        for i in range(m):
            for j in range(n):
                if graph_v[i, j] == 1:
                    pp[graph_idx[i, j], :] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
                else:
                    vector_x = graph_v[i, :]
                    vector_zx = graph[i, :]
                    p_x = j + 1
                    vector_y = graph_v[:, j]
                    vector_zy = graph[:, j]
                    p_y = i + 1
                    
                    px1, px2, zx1, zx2 = self.find_pp(vector_x, vector_zx, p_x, maxz)
                    py1, py2, zy1, zy2 = self.find_pp(vector_y, vector_zy, p_y, maxz)
                    

                    # Simplified condition - accept more points for interpolation
                    # Check if we have at least some neighbors
                    if ((px1 != 0 and py1 != 0) or (px2 != 0 and py2 != 0) or 
                        (px1 != 0 and py2 != 0) or (px2 != 0 and py1 != 0)):
                        p5 = 1
                    else:
                        p5 = 0
                    
                    pp[graph_idx[i, j], :] = [px1, px2, py1, py2, p5, zx1, zx2, zy1, zy2]
        
        out = np.column_stack([points, value, XYZ, pp])
        # Follow MATLAB logic exactly: idx=(out(:,4)==0) and idx2=(out(idx,12)==1)
        # After reordering, value field is in column 7 (index 6)
        idx = (out[:, 6] == 0)  # value field == 0 (column 7 after reordering)
        idx2 = (out[idx, 11] == 1)  # p5 field == 1 (column 12 in MATLAB is index 11 in Python)
        out2 = out[idx, :]
        out3 = out2[idx2, :]
        
        if len(out3) == 0:
            return np.array([]).reshape(0, 7)
        
        p1 = out3[:, 12]
        p2 = out3[:, 13]
        p3 = out3[:, 14]
        p4 = out3[:, 15]
        
        fin_z = np.zeros(len(out3))
        for i in range(len(out3)):
            pp1 = np.sort([p1[i], p2[i]])
            pp2 = np.sort([p3[i], p4[i]])
            ppp1 = np.sort([pp1[0], pp2[0]])
            ppp2 = np.sort([pp1[1], pp2[1]])
            
            if ppp1[1] > ppp2[0]:
                pp = [ppp2[0], ppp1[1], ppp1[1], ppp2[1], ppp1[0]]
            else:
                pp = [ppp1[1], ppp1[1], ppp2[0], ppp2[1], ppp1[0]]
            
            fin_z[i] = pp[0]
        
        # Follow MATLAB format exactly: [out3(:,1:2) high(fin_z')' out3(:,4:6) fin_z']
        data_out = np.column_stack([
            out3[:, 0:2],                    # x, y coordinates
            high[fin_z.astype(int)-1],       # z coordinates from high array
            out3[:, 3:6],                    # other fields (value, grid indices)
            fin_z                            # fin_z field
        ])
        
        return data_out
    
    def shunshizhen_order(self, jiaodian):
        """
        Order points in clockwise direction
        Equivalent to: shunshizhen_order(jiaodian)
        """
        points2 = jiaodian.copy()
        xy = points2[:, 0:2]
        
        mmx, mnx = np.max(xy[:, 0]), np.min(xy[:, 0])
        mmy, mny = np.max(xy[:, 1]), np.min(xy[:, 1])
        xy2 = np.array([[mmx, mmy], [mnx, mmy], [mmx, mny], [mnx, mny]])
        center = np.mean(xy2, axis=0)
        
        b = np.array([0, 1])
        angle = np.zeros(len(points2))
        
        for i in range(len(points2)):
            a = np.array([points2[i, 0] - center[0], points2[i, 1] - center[1]])
            
            # Calculate angle using dot product
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle1 = np.arccos(cos_angle) * 180 / np.pi
            
            if a[0] >= 0 and a[1] >= 0:
                angle[i] = angle1
            elif a[0] > 0 and a[1] < 0:
                angle[i] = 180 - angle1
            elif a[0] < 0 and a[1] < 0:
                angle[i] = angle1 + 180
            elif a[0] < 0 and a[1] > 0:
                angle[i] = 360 - angle1
        
        order = np.argsort(angle)
        return order
    
    def wanggehua(self, datas):
        """
        Create polygon from point cloud
        Equivalent to: wanggehua(datas)
        """
        minz = np.min(datas[:, 2])
        data_2D = datas[(datas[:, 2] > (minz + 0.2)) & (datas[:, 2] < (minz + 0.4)), 0:3]
        
        if len(data_2D) == 0:
            return None
        
        k3 = self.shunshizhen_order(data_2D)
        data2D = data_2D[:, 0:2]
        
        sit = 1  # If cross-section is complete, no need to reverse
        if sit == 1:
            try:
                # Create polygon from ordered points
                polygon_points = data_2D[k3, 0:2]
                pogen2 = ShapelyPolygon(polygon_points)
                return pogen2
            except:
                # If polygon creation fails, return None
                return None
        else:
            # Alternative method using RANSAC and line fitting
            # This is a simplified version
            return None
    
    def RANSAC_para_once(self, data_in):
        """
        Single RANSAC plane fitting
        Equivalent to: RANSAC_para_once(data_in)
        """
        data = data_in[:, 0:3]
        iter_count = 5000
        number = len(data)
        data = data.T
        
        pretotal = 0
        sigma = 0.02
        bestplane = np.array([0, 0, -1, 0])
        
        for i in range(iter_count):
            # Randomly select three points
            idx = np.random.choice(number, 3, replace=False)
            sample = data[:, idx]
            
            # Fit plane equation z = ax + by + c
            x = sample[0, :]
            y = sample[1, :]
            z = sample[2, :]
            
            try:
                # Calculate plane parameters
                denom = (x[0] - x[1]) * (y[0] - y[2]) - (x[0] - x[2]) * (y[0] - y[1])
                if abs(denom) < 1e-10:
                    continue
                
                a = ((z[0] - z[1]) * (y[0] - y[2]) - (z[0] - z[2]) * (y[0] - y[1])) / denom
                b = ((z[0] - z[2]) - a * (x[0] - x[2])) / (y[0] - y[2])
                c = z[0] - a * x[0] - b * y[0]
                
                plane = np.array([a, b, -1, c])
                
                # Calculate distances to plane
                distances = np.abs(plane @ np.vstack([data, np.ones((1, data.shape[1]))])) / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
                
                total = np.sum(distances < sigma)
                
                if total > pretotal:
                    pretotal = total
                    bestplane = plane
                    
            except:
                continue
        
        return bestplane
    
    def VerticalFootCoordinates(self, PlanePara, PointInput):
        """
        Calculate vertical foot coordinates on plane
        Equivalent to: VerticalFootCoordinates(PlanePara, PointInput)
        """
        a, b, c, d = PlanePara[0], PlanePara[1], PlanePara[2], PlanePara[3]
        
        # Avoid division by zero
        if a == 0:
            a = 0.001
        if b == 0:
            b = 0.001
        if c == 0:
            c = 0.001
        
        VerticalFoot = np.zeros((len(PointInput), 3))
        
        for i in range(len(PointInput)):
            x, y, z = PointInput[i, 0], PointInput[i, 1], PointInput[i, 2]
            
            A = np.array([
                [a, b, c],
                [b, -a, 0],
                [c, 0, -a]
            ])
            
            Y = -np.array([
                d,
                -b * x + a * y,
                -c * x + a * z
            ])
            
            try:
                VerticalFoot[i, :] = np.linalg.solve(A.T @ A, A.T @ Y)
            except:
                VerticalFoot[i, :] = PointInput[i, :]
        
        return VerticalFoot
    
    def refine_buliao3(self, datas):
        """
        Refine surface completion results
        Equivalent to: refine_buliao3(datas)
        """
        if len(datas) == 0 or datas.shape[1] < 3:
            return datas[:, 0:3] if len(datas) > 0 else np.array([]).reshape(0, 3)
        
        data = datas.copy()
        
        # Group filtering if group information exists
        if data.shape[1] >= 7:
            grp = np.unique(data[:, 6])
            keepMask = np.ones(len(data), dtype=bool)
            
            for g in grp:
                idx = (data[:, 6] == g)
                if np.sum(idx) < 10:
                    keepMask[idx] = False
            
            if np.sum(keepMask) >= 6:
                data = data[keepMask, :]
        
        if len(data) < 6:
            return data[:, 0:3]
        
        # Point cloud segmentation
        bbox = np.max(data[:, 0:3], axis=0) - np.min(data[:, 0:3], axis=0)
        scale = np.linalg.norm(bbox)
        
        if scale == 0:
            return data[:, 0:3]
        
        minDistance = 0.5
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=minDistance, min_samples=5).fit(data[:, 0:3])
        labels = clustering.labels_
        numClusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Collect valid clusters
        cluster_data = []
        for j in range(numClusters):
            idx = (labels == j)
            if np.sum(idx) > 5:
                cluster_data.append(data[idx, :])
        
        if len(cluster_data) == 0:
            cluster_data = [data]
        
        # Fit planes to each cluster
        p = []
        for cluster in cluster_data:
            points = cluster
            n1 = len(np.unique(points[:, 0]))
            n2 = len(np.unique(points[:, 1]))
            n3 = len(np.unique(points[:, 2]))
            
            if n1 > 1 and n2 > 1 and n3 >= 1:
                try:
                    bestplane = self.RANSAC_para_once(points)
                    projected = self.VerticalFootCoordinates(bestplane, points)
                    p.append(projected)
                except:
                    continue
        
        if len(p) == 0:
            return data[:, 0:3]
        
        data_out = np.vstack(p)
        
        # Add small noise
        z = np.random.uniform(-0.05, 0.05, (len(data_out), 1))
        data_out = data_out + z
        
        return data_out
