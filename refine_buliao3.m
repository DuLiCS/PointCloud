function data_out = refine_buliao3(datas)
% 输入: datas (N×>=3)，至少包含 xyz；可有第7列做组标签
% 输出: data_out (N×3) 经过平面RANSAC投影并加微噪声的点

    % ---------- 1) 基本检查 ----------
    if isempty(datas) || size(datas,2) < 3
        warning('refine_buliao3: 输入数据不足(需要>=3列xyz)，已原样返回。');
        data_out = datas(:,1:min(3,end));
        return;
    end

    data = datas;

    % ---------- 2) 组内点数过滤（仅当第7列存在时启用） ----------
    if size(data,2) >= 7
        grp = unique(data(:,7));
        keepMask = true(size(data,1),1);
        for i = 1:numel(grp)
            idx = (data(:,7) == grp(i));
            if sum(idx) < 10
                keepMask(idx) = false;
            end
        end
        % 若过滤过严导致点太少，则撤销过滤
        if nnz(keepMask) >= 6
            data = data(keepMask,:);
        end
    end

    % 若仍然点很少，直接返回 xyz（不做投影）
    if size(data,1) < 6
        data_out = data(:,1:3);
        return;
    end

    % ---------- 3) 基于距离的点云分割（pcsegdist），阈值自适应 ----------
    bbox  = max(data(:,1:3),[],1) - min(data(:,1:3),[],1);
    scale = norm(bbox);
    if scale == 0
        data_out = data(:,1:3);
        return;
    end
    minDistance = 0.5;   % 包围盒对角线的2%作为阈值

    ptsObj = pointCloud(data(:,1:3));
    [labels, numClusters] = pcsegdist(ptsObj, minDistance);

    % 收集有效簇（每簇>5点）
    cluster_data = {};
    k = 1;
    for j = 1:numClusters
        idx = (labels == j);
        if nnz(idx) > 5
            cluster_data{k} = data(idx,:); %#ok<AGROW>
            k = k + 1;
        end
    end
    if isempty(cluster_data)
        cluster_data = {data};  % 退化为单簇
    end

    % ---------- 4) 对每个簇做平面拟合 & 投影 ----------
    p = {};        
    for i = 1:numel(cluster_data)
        points = cluster_data{i};
        n1 = numel(unique(points(:,1)));
        n2 = numel(unique(points(:,2)));
        n3 = numel(unique(points(:,3)));
        if n1>1 && n2>1 && n3>=1
            try
                % 项目内接口：返回 [a b c d]
                bestplane = RANSAC_para_once(points);
                % 垂足坐标（项目内接口）
                p{end+1} = VerticalFootCoordinates(bestplane, points); %#ok<AGROW>
            catch
                % 拟合/投影失败则跳过该簇
            end
        end
    end

    % ---------- 5) 汇总并兜底 ----------
    if isempty(p)
        % 没有任何有效投影，直接返回 xyz
        data_out = data(:,1:3);
        return;
    end

    data_out = vertcat(p{:});

    % ---------- 6) 加微噪声（与原逻辑一致） ----------
    z = unifrnd(-0.05, 0.05, size(data_out,1), 1);
    data_out = [data_out(:,1)+z, data_out(:,2)+z, data_out(:,3)+z];
end
