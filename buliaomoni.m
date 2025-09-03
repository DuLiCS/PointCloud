function [buliao_out,data_out,high,data_out3] = buliaomoni(datas,kk,normal_pts,juli,zhengping)
% ����:
%   quedingbianjie, down_buliao, quzao, fantan, wanggehua,
%   xuanzhuanjuzhen2, refine_buliao3
%
% ����Ϊ"�ϸ���Ĥ"��
% - ʼ�հ� TFin ��Ĥ�������ڵ�̫��ʱ������Ĥ
% - ��������Ĥ��Ϊ��ʱ�����ˣ���������Ϊһ�£�
% - R/R2 �� eye(3) ����λ��ת������ԭ���� 1 ��ά������

    % ---------- ��ת�� Z �ᷨ�������ϵ ----------
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

    % ---------- ȷ���߽� & ����ƽ�� ----------
    data_zhi = data(:,1:3);
    [data_out, xmin, xmax, ymin, ymax, zmax] = quedingbianjie(data_zhi, 20);

    % ����߶��������񲽳�
    [~, dis] = knnsearch(data(:,1:3), data(:,1:3), ...
                         'Distance','euclidean', 'NSMethod','kdtree', 'K', 3); %#ok<ASGLU>
    d = mean(dis(:,2));   % ������һ�£�����ϵ�� 1.0

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
        repmat(0, size(plane,1),1), ...                   % ռλ��
        repmat(1, size(plane,1),1), ...                   % �����
        plane(:,3:4) ...                                  % ��������
    ];

    % ---------- ����� ----------
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

    % ---------- ��̯ & ������ϸ�ɸѡ ----------
    data_out      = fantan(buliao_out, high);
    minhigh       = min(high);
    data_zhi_3D   = data_zhi(data_zhi(:,3) > minhigh, :);

    polyin        = wanggehua(data_zhi_3D);
    % plot(polyin); % ��Ҫ�ɴ�

    data_out2     = data_out(:,1:3);
    data_out2_2D  = data_out2(:,1:2);
    TFin          = isinterior(polyin, data_out2_2D(:,1), data_out2_2D(:,2));

    % ���� �ϸ���Ĥ��������һ�£����޶���
    if zhengping == 1 && size(data_out,1) > 50
        data_out2 = refine_buliao3(data_out(TFin,:));
    else
        data_out2 = data_out2(TFin,:);
    end

    % ���� ����"Ϊ�ջ���"�Ķ��ף��ϸ�ģʽ��
    if size(data_out2,2) ~= 3
        error('buliaomoni: data_out2 �� N��3����ǰΪ %dx%d', size(data_out2,1), size(data_out2,2));
    end

    % ---------- ��ת��ԭ���겢ƽ�� ----------
    if kk ~= 1
        if ~isequal(size(R2), [3 3])
            error('buliaomoni: R2 ӦΪ 3��3����ǰΪ %dx%d', size(R2,1), size(R2,2));
        end
        if numel(center) ~= 3
            error('buliaomoni: center Ӧ�� 3 ��Ԫ�أ���ǰΪ %d', numel(center));
        end
        center   = reshape(center,1,3);
        rotated  = data_out2 * R2;                         % N��3����Ϊ�գ�
        data_out3 = rotated + repmat(center, size(rotated,1), 1);
        buliao_out = [buliao_out(:,1:3)*R2, buliao_out(:,4), buliao_out(:,6:7), buliao_out(:,5)];
    else
        data_out3  = data_out2;                            % ֱͨ
        buliao_out = [buliao_out(:,1:3),   buliao_out(:,4), buliao_out(:,6:7), buliao_out(:,5)];
    end

    % ����ֻ��� data_out3����ԭ�߼�һ�£�
    buliao_out = [data_out3];
end
