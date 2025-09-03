function [buliao_out,data_out,high,data_out3] = buliaomoni(datas,kk,normal_pts,juli,zhengping)
% 依赖:
%   quedingbianjie, down_buliao, quzao, fantan, wanggehua,
%   xuanzhuanjuzhen2, refine_buliao3
%
% 本版为"严格掩膜"：
% - 始终按 TFin 掩膜，不再在点太少时放弃掩膜
% - 不再在掩膜后为空时做回退（与样例行为一致）
% - R/R2 用 eye(3) 作单位旋转，避免原来用 1 的维度问题

    % ---------- 旋转到 Z 轴法向的坐标系 ----------
    if kk ~= 1
        normal2 = normal_pts(:).';
        if norm(normal2) == 0
            normal2 = [0 0 1];
        else
            normal2 = normal2 ./ norm(normal2);
        end

        if ~all(abs(normal2 - [0 0 1]) < 1e-12)
            R  = xuanzhuanjuzhen2(normal2, [0 0 1]);
            R2 = xuanzhuanjuzhen2([0 0 1], normal2);
        else
            R  = eye(3);
            R2 = eye(3);
        end

        center = mean(datas(:,1:3), 1);
        center = reshape(center,1,3);
        data   = datas(:,1:3) - center;
        data   = data * R;
    else
        data   = datas(:,1:3);
        center = [0 0 0];
        R2     = eye(3);
    end

    % ---------- 确定边界 & 网格平铺 ----------
    data_zhi = data(:,1:3);
    [data_out, xmin, xmax, ymin, ymax, zmax] = quedingbianjie(data_zhi, 20);

    % 邻域尺度用于网格步长
    [~, dis] = knnsearch(data(:,1:3), data(:,1:3), ...
                         'Distance','euclidean', 'NSMethod','kdtree', 'K', 3); %#ok<ASGLU>
    d = mean(dis(:,2));   % 与样例一致：保持系数 1.0

    k  = 1;  xi = 1;
    for i = xmin:d:xmax
        yi = 1;
        for j = ymin:d:ymax
            plane(k,:) = [i j xi yi]; %#ok<AGROW>
            yi  = yi + 1;
            k   = k + 1;
        end
        xi = xi + 1;
    end

    buliao = [ ...
        plane(:,1:2), repmat(zmax,size(plane,1),1), ...   % x y z(zmax)
        repmat(0, size(plane,1),1), ...                   % 占位列
        repmat(1, size(plane,1),1), ...                   % 标记列
        plane(:,3:4) ...                                  % 网格索引
    ];

    % ---------- 向下填补 ----------
    kkk = 1;
    for zz = zmax - juli : 0.1 : zmax %#ok<NASGU>
        buliao     = down_buliao(data, buliao, d*2, 1, 0.1);
        high(kkk)  = zmax - (kkk-1)*0.1; %#ok<AGROW>
        kkk        = kkk + 1;
    end
    high(kkk) = zmax - (kkk-1)*0.1; %#ok<AGROW>

    buliao     = quzao(buliao);
    buliao_out = buliao;
    buliao_out = [buliao_out(:,1:3), buliao_out(:,4), buliao_out(:,6:7), buliao_out(:,5)];

    % ---------- 反摊 & 多边形严格筛选 ----------
    data_out      = fantan(buliao_out, high);
    minhigh       = min(high);
    data_zhi_3D   = data_zhi(data_zhi(:,3) > minhigh, :);

    polyin        = wanggehua(data_zhi_3D);
    % plot(polyin); % 需要可打开

    data_out2     = data_out(:,1:3);
    data_out2_2D  = data_out2(:,1:2);
    TFin          = isinterior(polyin, data_out2_2D(:,1), data_out2_2D(:,2));

    % —— 严格掩膜（与样例一致）：无兜底
    if zhengping == 1 && size(data_out,1) > 50
        data_out2 = refine_buliao3(data_out(TFin,:));
    else
        data_out2 = data_out2(TFin,:);
    end

    % —— 不做"为空回退"的兜底（严格模式）
    if size(data_out2,2) ~= 3
        error('buliaomoni: data_out2 非 N×3，当前为 %dx%d', size(data_out2,1), size(data_out2,2));
    end

    % ---------- 旋转回原坐标并平移 ----------
    if kk ~= 1
        if ~isequal(size(R2), [3 3])
            error('buliaomoni: R2 应为 3×3，当前为 %dx%d', size(R2,1), size(R2,2));
        end
        if numel(center) ~= 3
            error('buliaomoni: center 应含 3 个元素，当前为 %d', numel(center));
        end
        center   = reshape(center,1,3);
        rotated  = data_out2 * R2;                         % N×3（可为空）
        data_out3 = rotated + repmat(center, size(rotated,1), 1);
        buliao_out = [buliao_out(:,1:3)*R2, buliao_out(:,4), buliao_out(:,6:7), buliao_out(:,5)];
    else
        data_out3  = data_out2;                            % 直通
        buliao_out = [buliao_out(:,1:3),   buliao_out(:,4), buliao_out(:,6:7), buliao_out(:,5)];
    end

    % 最终只输出 data_out3（与原逻辑一致）
    buliao_out = [data_out3];
end
