# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from chromodet.model.dit.blocks import DiTBlock

# # 导入原有的基础组件
from projects.DiffusionDet.diffusiondet.head import *


# -------------------- 新增：大核DWConv上下文模块 --------------------
class LargeKernelContext(nn.Module):
    """
    多分支大核深度可分离卷积，上下文增强：
    - 分支1：大核 DWConv (k x k)
    - 分支2：3x3 DWConv, dilation=d1
    - 分支3：5x5 DWConv, dilation=d2
    # - 融合：sum -> 1x1 Conv -> 残差 （取消）
     ---> New 融合：SK Attention (Selective Kernel) 动态加权
     ---> 1x1 Conv 残差
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 31,
        dilations: Tuple[int, int] = (3, 5),
        act_cfg: dict = dict(type='ReLU', inplace=True),
        norm_groups: int = 32,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size 必须为奇数，保证same padding'
        d1, d2 = dilations
        pad_big = kernel_size // 2
        self.dw_big = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=pad_big,
            groups=channels,
            bias=True,
        )
        self.dw_d1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=d1,
            dilation=d1,
            groups=channels,
            bias=True,
        )
        self.dw_d2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=d2 * 2,
            dilation=d2,
            groups=channels,
            bias=True,
        )
        # SK Attention 组件
        reduction = 16
        hidden_dim = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fc_select = nn.Conv2d(hidden_dim, channels * 3, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # 归一化（GroupNorm）以稳定跨数据/层的分布
        self.gn_big = nn.GroupNorm(norm_groups, channels)
        self.gn_d1 = nn.GroupNorm(norm_groups, channels)
        self.gn_d2 = nn.GroupNorm(norm_groups, channels)
        self.gn_pw = nn.GroupNorm(norm_groups, channels)
        self.act = build_activation_layer(act_cfg)
        # 温和残差
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_big = self.act(self.gn_big(self.dw_big(x)))
        y_d1 = self.act(self.gn_d1(self.dw_d1(x)))
        y_d2 = self.act(self.gn_d2(self.dw_d2(x)))
        # y = y_big + y_d1 + y_d2

        # 2. SK Attention 融合
        batch_size, C, H, W = x.shape
        feats = torch.stack([y_big, y_d1, y_d2], dim=1)  # [B, 3, C, H, W]

        # 全局信息聚合
        U = torch.sum(feats, dim=1)  # [B, C, H, W]
        S = self.avg_pool(U)  # [B, C, 1, 1]

        # 生成注意力权重
        Z = self.fc_reduce(S)
        attn = self.fc_select(Z)  # [B, 3*C, 1, 1]
        attn = attn.view(batch_size, 3, C, 1, 1)
        attn = self.softmax(attn)  # [B, 3, C, 1, 1]

        # 动态加权
        V = torch.sum(feats * attn, dim=1)  # [B, C, H, W]

        # 后处理与残差连接
        y = self.gn_pw(self.pw(V))
        y = self.act(y)
        return x + self.alpha * y


# -------------------- 新增结束 --------------------


@MODELS.register_module()
class ChromoDetDynamicHead(DynamicDiffusionDetHead):
    """
    染色体检测专用的动态扩散检测头

    主要改进：
    1. 形态感知的噪声调度
    2. 长度先验引导的扩散采样
    3. 拓扑约束损失
    4. 重叠敏感的NMS
    """

    def __init__(
        self,
        num_classes=24,  # 24种染色体类型
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        prior_prob=0.01,
        snr_scale=2.0,
        timesteps=1000,
        sampling_timesteps=1,
        self_condition=False,
        box_renewal=True,
        use_ensemble=True,
        deep_supervision=True,
        ddim_sampling_eta=1.0,
        # 染色体特化参数
        use_morphology_aware: bool = False,  # 形态感知
        aspect_ratio_gamma=10.0,
        use_length_prior: bool = False,  # 长度感知
        length_prior_weight=0.1,
        use_topology_pairing: bool = False,  # 拓扑匹配
        topology_loss_weight=0.05,
        use_length_ordering: bool = False,
        criterion=dict(
            type='ChromoDetCriterion',
            num_classes=24,
            assigner=dict(
                type='ChromoDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost', alpha=2.0, gamma=0.25, weight=2.0
                    ),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0),
                ],
                center_radius=2.5,
                candidate_topk=5,
            ),
        ),
        single_head=dict(
            type='ChromoDetSingleHead',
            num_cls_convs=1,
            num_reg_convs=3,
            dim_feedforward=2048,
            num_heads=8,
            dropout=0.0,
            act_cfg=dict(type='ReLU'),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
        ),
        roi_extractor=None,
        train_cfg=None,
        test_cfg=None,
    ) -> None:
        """染色体特化参数"""
        self.use_morphology_aware = use_morphology_aware  # 形态感知
        self.aspect_ratio_gamma = aspect_ratio_gamma
        self.use_length_prior = use_length_prior  # 长度先验
        self.length_prior_weight = length_prior_weight
        self.use_topology_pairing = use_topology_pairing  # 拓扑匹配
        self.topology_loss_weight = topology_loss_weight
        assert topology_loss_weight > 0, 'topology_loss_weight should be > 0'
        self.use_length_ordering = use_length_ordering  # 长度排序

        super().__init__(
            num_classes=num_classes,
            feat_channels=feat_channels,
            num_proposals=num_proposals,
            num_heads=num_heads,
            prior_prob=prior_prob,
            snr_scale=snr_scale,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            self_condition=self_condition,
            box_renewal=box_renewal,
            use_ensemble=use_ensemble,
            deep_supervision=deep_supervision,
            ddim_sampling_eta=ddim_sampling_eta,
            criterion=criterion,
            single_head=single_head,
            roi_extractor=roi_extractor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )

        # 形态特征编码器
        if use_morphology_aware:
            self._build_morphology_aware_noise()

        # 拓扑关系建模
        if use_topology_pairing:
            self.topology_encoder = nn.MultiheadAttention(
                feat_channels, num_heads=4, dropout=0.1
            )

    def _build_morphology_aware_noise(self):
        """构建形态感知的扩散调度"""
        # 基础扩散调度
        betas = cosine_beta_schedule(self.timesteps)
        # 针对细长目标的调整
        # 早期时间步使用更小的噪声（保持形态）
        early_steps = self.timesteps // 4
        betas[:early_steps] *= 0.5
        # 中期时间步正常噪声
        # 后期时间步稍微增大噪声（增强随机性）
        late_steps = 3 * self.timesteps // 4
        betas[late_steps:] *= 1.2
        betas = betas.clamp(min=0.0, max=1 - 1e-6)  # 防止超过[0,1]范围

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 其他扩散参数计算（与原版相同）
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            'sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    def forward(self, features, init_bboxes, init_t, init_features=None):
        """前向传播，加入拓扑关系建模"""
        time = self.time_mlp(init_t)

        inter_class_logits = []
        inter_pred_bboxes = []
        inter_pred_lengths = []

        bs = len(features[0])
        bboxes = init_bboxes

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None

        for head_idx, single_head in enumerate(self.head_series):
            # 拓扑特征层在最后一层前， 按旧版逻辑则会多加一层 single_head，不能控制变量
            if (
                head_idx == len(self.head_series) - 1
                and self.use_topology_pairing
            ):
                # 自注意力建模染色体间关系
                topo_features, _ = self.topology_encoder(
                    proposal_features, proposal_features, proposal_features
                )
                proposal_features = proposal_features + topo_features

            if self.use_length_prior or self.use_length_ordering:
                class_logits, pred_bboxes, proposal_features, pred_lengths = (
                    single_head(
                        features,
                        bboxes,
                        proposal_features,
                        self.roi_extractor,
                        time,
                    )
                )
            else:
                class_logits, pred_bboxes, proposal_features = single_head(
                    features,
                    bboxes,
                    proposal_features,
                    self.roi_extractor,
                    time,
                )
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
                if self.use_length_prior or self.use_length_ordering:
                    inter_pred_lengths.append(pred_lengths)
            bboxes = pred_bboxes.detach()

        if self.deep_supervision:
            res = [
                torch.stack(inter_class_logits),
                torch.stack(inter_pred_bboxes),
            ]
            if self.use_length_prior or self.use_length_ordering:
                res.append(torch.stack(inter_pred_lengths))
            return tuple(res)
        else:
            res = [
                class_logits[None, ...],
                pred_bboxes[None, ...],
            ]
            if self.use_length_prior or self.use_length_ordering:
                res.append(pred_lengths[None, ...])
            return tuple(res)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """损失计算，加入染色体特化损失"""

        prepare_outputs = self.prepare_training_targets(batch_data_samples)
        batch_gt_instances, batch_pred_instances, _, batch_img_metas = (
            prepare_outputs
        )

        batch_diff_bboxes = torch.stack(
            [
                pred_instances.diff_bboxes_abs
                for pred_instances in batch_pred_instances
            ]
        )
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances]
        )

        # 前向传播
        pred_results = self(x, batch_diff_bboxes, batch_time)
        if self.use_length_prior or self.use_length_ordering:
            pred_logits, pred_bboxes, pred_lengths = pred_results
        else:
            pred_logits, pred_bboxes = pred_results
        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1],
        }
        if self.use_length_prior or self.use_length_ordering:
            output['pred_lengths'] = pred_lengths[-1]

        if self.deep_supervision:
            if self.use_length_prior or self.use_length_ordering:
                output['aux_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b, 'pred_lengths': c}
                    for a, b, c in zip(
                        pred_logits[:-1], pred_bboxes[:-1], pred_lengths[:-1]
                    )
                ]
            else:
                output['aux_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b}
                    for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])
                ]

        losses = self.criterion(output, batch_gt_instances, batch_img_metas)

        return losses

    def predict_by_feat(
        self,
        x,
        time_pairs,
        batch_noise_bboxes,
        batch_noise_bboxes_raw,
        batch_image_size,
        device,
        batch_img_metas=None,
        cfg=None,
        rescale=True,
    ):
        """
        根据特征进行预测

        Args:
            x: 特征金字塔
            time_pairs: 时间对列表,用于反向扩散
            batch_noise_bboxes: 批次噪声边界框（xyxy格式）
            batch_noise_bboxes_raw: 批次原始噪声边界框（未处理）
            batch_image_size: 批次图像尺寸
            device: 设备
            batch_img_metas: 批次图像元信息
            cfg: 配置
            rescale: 是否缩放

        Returns:
            预测结果列表
        """
        batch_size = len(batch_img_metas)  # 批次大小

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        ensemble_score, ensemble_label, ensemble_coord = (
            [],
            [],
            [],
        )  # 集成预测结果
        # 遍历时间对（反向扩散过程）
        for time, time_next in time_pairs:
            # 构造时间张量
            batch_time = torch.full(
                (batch_size,), time, device=device, dtype=torch.long
            )  # shape: [batch_size]
            # 前向传播
            pred_results = self(x, batch_noise_bboxes, batch_time)
            # 兼容所有输出
            pred_logits, pred_bboxes = pred_results[:2]

            x_start = pred_bboxes[-1]  # 预测的去噪结果

            # 转换为cx,cy,w,h格式并归一化
            x_start = (
                x_start / batch_image_size[:, None, :]
            )  # shape: [batch_size, num_proposals, 4]
            x_start = bbox_xyxy_to_cxcywh(x_start)  # 转换为cx,cy,w,h格式
            x_start = (
                x_start * 2 - 1.0
            ) * self.snr_scale  # 缩放到[-snr_scale, snr_scale]
            x_start = torch.clamp(
                x_start, min=-1 * self.snr_scale, max=self.snr_scale
            )  # 限制范围
            # 从去噪结果预测噪声
            pred_noise = self.predict_noise_from_start(
                batch_noise_bboxes_raw, batch_time, x_start
            )

            pred_noise_list, x_start_list = (
                [],
                [],
            )  # 每个图像的预测噪声和去噪结果
            noise_bboxes_list, num_remain_list = [], []  # 噪声框和保留数量
            if self.box_renewal:  # 如果使用框更新
                score_thr = cfg.get('score_thr', 0)  # 置信度阈值
                # 对批次中每个图像处理
                for img_id in range(batch_size):
                    score_per_image = pred_logits[-1][
                        img_id
                    ]  # 当前图像的分类得分

                    score_per_image = torch.sigmoid(
                        score_per_image
                    )  # sigmoid激活
                    value, _ = torch.max(
                        score_per_image, -1, keepdim=False
                    )  # 每个框的最大类别得分
                    keep_idx = value > score_thr  # 保留高置信度框

                    num_remain_list.append(torch.sum(keep_idx))  # 保留框数量
                    pred_noise_list.append(
                        pred_noise[img_id, keep_idx, :]
                    )  # 保留框对应的预测噪声
                    x_start_list.append(
                        x_start[img_id, keep_idx, :]
                    )  # 保留框对应的去噪结果
                    noise_bboxes_list.append(
                        batch_noise_bboxes[img_id, keep_idx, :]
                    )  # 保留的噪声框

            # 如果是最后一步
            if time_next < 0:
                # 不同于原始DiffusionDet
                if self.use_ensemble and self.sampling_timesteps > 1:
                    # 使用集成预测
                    box_pred_per_image, scores_per_image, labels_per_image = (
                        self.inference(
                            box_cls=pred_logits[-1],
                            box_pred=pred_bboxes[-1],
                            cfg=cfg,
                            device=device,
                        )
                    )
                    ensemble_score.append(scores_per_image)
                    ensemble_label.append(labels_per_image)
                    ensemble_coord.append(box_pred_per_image)
                continue

            # DDIM采样参数计算
            alpha = self.alphas_cumprod[time]  # 当前时间步alpha累积值
            alpha_next = self.alphas_cumprod[
                time_next
            ]  # 下一时间步alpha累积值

            sigma = (
                self.ddim_sampling_eta
                * (
                    (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
                ).sqrt()
            )  # sigma参数
            c = (1 - alpha_next - sigma**2).sqrt()  # c参数

            batch_noise_bboxes_list = []  # 新的噪声框列表
            batch_noise_bboxes_raw_list = []  # 新的原始噪声框列表
            # 对批次中每个图像处理
            for idx in range(batch_size):
                pred_noise = pred_noise_list[idx]  # 预测噪声
                x_start = x_start_list[idx]  # 去噪结果
                noise_bboxes = noise_bboxes_list[idx]  # 当前噪声框
                num_remain = num_remain_list[idx]  # 保留框数量
                noise = torch.randn_like(noise_bboxes)  # 新的随机噪声
                """ DDIM采样步骤:
                    x_{t-1} =
                    sqrt(alpha_{t-1}) * x_0 +
                    sqrt(1 - alpha_{t-1} - sigma^2) * pred_noise +
                    sigma * noise
                """
                noise_bboxes = (
                    x_start * alpha_next.sqrt()
                    + c * pred_noise
                    + sigma * noise
                )

                if self.box_renewal:  # 如果使用框更新
                    # 用随机框补充
                    if num_remain < self.num_proposals:
                        # 如果保留框少于建议框数量,用随机框填充
                        noise_bboxes = torch.cat(
                            (
                                noise_bboxes,
                                torch.randn(
                                    self.num_proposals - num_remain,
                                    4,
                                    device=device,
                                ),
                            ),
                            dim=0,
                        )  # shape: [num_proposals, 4]
                    else:
                        # 如果保留框多于建议框数量,随机选择
                        # select_mask = [True] * self.num_proposals + \
                        #               [False] * (num_remain -
                        #                          self.num_proposals)
                        # random.shuffle(select_mask)
                        select_mask = torch.randperm(num_remain)[
                            : self.num_proposals
                        ]  # 使用torch自带随机序号方法
                        noise_bboxes = noise_bboxes[select_mask]

                    # 保存原始噪声框
                    batch_noise_bboxes_raw_list.append(noise_bboxes)
                    # 处理噪声框: 转换为xyxy格式并缩放到图像尺寸
                    noise_bboxes = torch.clamp(
                        noise_bboxes,
                        min=-1 * self.snr_scale,
                        max=self.snr_scale,
                    )  # 限制范围
                    noise_bboxes = (
                        (noise_bboxes / self.snr_scale) + 1
                    ) / 2  # 转换到[0,1]
                    noise_bboxes = bbox_cxcywh_to_xyxy(
                        noise_bboxes
                    )  # 转换为xyxy格式
                    noise_bboxes = (
                        noise_bboxes * batch_image_size[idx]
                    )  # 缩放到绝对坐标

                batch_noise_bboxes_list.append(noise_bboxes)
            # 更新噪声框
            batch_noise_bboxes = torch.stack(batch_noise_bboxes_list)
            batch_noise_bboxes_raw = torch.stack(batch_noise_bboxes_raw_list)

            # 如果使用集成预测
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = (
                    self.inference(
                        box_cls=pred_logits[-1],
                        box_pred=pred_bboxes[-1],
                        cfg=cfg,
                        device=device,
                    )
                )
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        # 如果使用集成预测
        if self.use_ensemble and self.sampling_timesteps > 1:
            steps = len(ensemble_score)  # 集成步数
            results_list = []
            # 对批次中每个图像处理
            for idx in range(batch_size):
                # 收集所有步的预测结果
                ensemble_score_per_img = [
                    ensemble_score[i][idx] for i in range(steps)
                ]
                ensemble_label_per_img = [
                    ensemble_label[i][idx] for i in range(steps)
                ]
                ensemble_coord_per_img = [
                    ensemble_coord[i][idx] for i in range(steps)
                ]

                # 拼接所有步的结果
                scores_per_image = torch.cat(ensemble_score_per_img, dim=0)
                labels_per_image = torch.cat(ensemble_label_per_img, dim=0)
                box_pred_per_image = torch.cat(ensemble_coord_per_img, dim=0)

                # 如果使用NMS
                use_nms = bool(getattr(cfg, 'use_nms', False))
                if use_nms:
                    nms_cfg = getattr(cfg, 'nms', None)
                    class_agnostic = False
                    if isinstance(nms_cfg, dict):
                        class_agnostic = bool(
                            nms_cfg.pop('class_agnostic', False)
                        )
                    idxs_for_nms = (
                        torch.zeros_like(labels_per_image)
                        if class_agnostic
                        else labels_per_image
                    )
                    det_bboxes, keep_idxs = batched_nms(
                        box_pred_per_image,
                        scores_per_image,
                        labels_per_image,
                        cfg.nms,
                    )
                    box_pred_per_image = box_pred_per_image[keep_idxs]
                    labels_per_image = labels_per_image[keep_idxs]
                    scores_per_image = det_bboxes[:, -1]  # NMS可能重新加权得分
                # 创建结果对象
                results = InstanceData()
                results.bboxes = box_pred_per_image
                results.scores = scores_per_image
                results.labels = labels_per_image
                results_list.append(results)
        else:
            # 不使用集成预测,直接使用最后一步的结果
            box_cls = pred_logits[-1]
            box_pred = pred_bboxes[-1]
            results_list = self.inference(box_cls, box_pred, cfg, device)

        # 如果需要缩放结果
        if rescale:
            results_list = self.do_results_post_process(
                results_list, cfg, batch_img_metas=batch_img_metas
            )
        return results_list


@MODELS.register_module()
class ChromoDetSingleHead(SingleDiffusionDetHead):
    """染色体专用的单头检测器"""

    def __init__(
        self,
        num_classes,
        use_length_prior: bool = False,
        feat_channels=256,
        dim_feedforward=2048,
        num_cls_convs=1,
        num_reg_convs=3,
        num_heads=8,
        dropout=0.0,
        pooler_resolution=7,
        bbox_weights=(2.0, 2.0, 1.0, 1.0),
        use_focal_loss=True,
        use_fed_loss=False,
        act_cfg=dict(type='ReLU', inplace=True),
        dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
        # 大核参数
        use_large_kernel: bool = False,
        lk_kernel_size: int = 31,
        lk_dilations: Tuple[int, int] = (3, 5),
        use_large_kernel_levels: Optional[list] = None,  # 仅在指定FPN层使用
        lk_alpha_init: float = 0.1,
        lk_norm_groups: int = 32,
        # -------------------- DiT ------------------------
        use_dit: bool = True,
        dit_heads: int = 8,
        # -------------------------------------------------
    ):
        super().__init__(
            num_classes=num_classes,
            feat_channels=feat_channels,
            dim_feedforward=dim_feedforward,
            num_cls_convs=num_cls_convs,
            num_reg_convs=num_reg_convs,
            num_heads=num_heads,
            dropout=dropout,
            pooler_resolution=pooler_resolution,
            use_focal_loss=use_focal_loss,
            use_fed_loss=use_fed_loss,
            act_cfg=act_cfg,
            dynamic_conv=dynamic_conv,
        )

        self.use_large_kernel = use_large_kernel
        self.use_large_kernel_levels = use_large_kernel_levels
        if self.use_large_kernel:
            self.lk_block = LargeKernelContext(
                channels=feat_channels,
                kernel_size=lk_kernel_size,
                dilations=lk_dilations,
                act_cfg=act_cfg,
                norm_groups=lk_norm_groups,
                alpha_init=lk_alpha_init,
            )
        # ---------------------- 新增: DiT --------------------------
        self.use_dit = use_dit
        if self.use_dit:
            # kv_group 取 dit_heads // 4，至少为1
            kv_group = max(1, dit_heads // 4)
            self.dit_block = DiTBlock(
                hidden_size=self.feat_channels,
                num_heads=dit_heads,
                kv_group=kv_group,
            )

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        前向传播

        Args:
            features: 特征金字塔,列表形式
            bboxes: 边界框,shape: (N, num_boxes, 4)
            pro_features: 提案特征,shape: (N, num_boxes, feat_channels)
            pooler: ROI池化器
            time_emb: 时间嵌入,shape: (N, time_dim)

        Returns:
            class_logits: 分类logits,shape: (N, num_boxes, num_classes)
            pred_bboxes: 预测边界框,shape: (N, num_boxes, 4)
            obj_features: 对象特征,shape: (1, N*num_boxes, feat_channels)
        """
        if hasattr(self, 'lk_block'):
            enhanced_feats = []
            for x in features:
                # 对每个FPN层做大核DWConv增强（共享权重），保持尺寸不变
                enhanced_feats.append(self.lk_block(x))
            features = tuple(enhanced_feats)
        # ------------------------------------------------
        N, num_boxes = bboxes.shape[:2]  # 获取批次大小和框数量

        # ROI特征提取
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b])  # 收集每张图像的边界框
        rois = bbox2roi(proposal_boxes)  # 转换为ROI格式

        roi_features = pooler(features, rois)  # ROI池化提取特征

        # 处理提案特征
        if pro_features is None:
            # 如果没有提供提案特征,则从ROI特征计算均值
            pro_features = roi_features.view(
                N, num_boxes, self.feat_channels, -1
            ).mean(-1)  # shape: (N, num_boxes, feat_channels)

        # 调整ROI特征形状以适应注意力机制
        roi_features = roi_features.view(
            N * num_boxes, self.feat_channels, -1
        ).permute(2, 0, 1)  # shape: (49, N*num_boxes, feat_channels)

        # 自注意力
        pro_features = pro_features.view(
            N, num_boxes, self.feat_channels
        ).permute(1, 0, 2)  # shape: (num_boxes, N, feat_channels)
        # 执行自注意力: Query=Key=Value=pro_features
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features
        )[0]  # shape: (num_boxes, N, feat_channels)
        # 残差连接和归一化
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # 实例交互
        # 调整形状以进行实例交互
        pro_features = (
            pro_features.view(num_boxes, N, self.feat_channels)
            .permute(1, 0, 2)
            .reshape(1, N * num_boxes, self.feat_channels)
        )  # shape: (1, N*num_boxes, feat_channels)
        # 动态卷积交互
        pro_features2 = self.inst_interact(pro_features, roi_features)
        # 残差连接和归一化
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(
            pro_features
        )  # shape: (1, N*num_boxes, feat_channels)

        # 对象特征处理（前馈网络）
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features)))
        )  # 前馈网络
        obj_features = obj_features + self.dropout3(obj_features2)  # 残差连接
        obj_features = self.norm3(obj_features)  # 归一化
        # 新增：DiT 跨 proposal 注意力（按 [N, num_boxes, C] 排列）
        if self.use_dit:
            N, num_boxes = bboxes.shape[:2]
            obj_seq = obj_features.view(N, num_boxes, self.feat_channels)
            obj_seq = self.dit_block(obj_seq)
            obj_features = obj_seq.view(1, N * num_boxes, self.feat_channels)
        # 时间嵌入条件化
        fc_feature = obj_features.transpose(0, 1).reshape(
            N * num_boxes, -1
        )  # shape: (N*num_boxes, feat_channels)

        scale_shift = self.block_time_mlp(time_emb)  # 时间嵌入MLP
        scale_shift = torch.repeat_interleave(
            scale_shift, num_boxes, dim=0
        )  # 扩展到每个框
        scale, shift = scale_shift.chunk(2, dim=1)  # 分割为缩放和偏移
        # 时间条件化: feature = feature * (scale + 1) + shift
        fc_feature = fc_feature * (scale + 1) + shift

        # 分类和回归分支
        cls_feature = fc_feature.clone()  # 分类特征
        reg_feature = fc_feature.clone()  # 回归特征
        # 通过分类模块
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        # 通过回归模块
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        # 预测分类logits和边界框偏移
        class_logits = self.class_logits(
            cls_feature
        )  # shape: (N*num_boxes, num_classes)
        bboxes_deltas = self.bboxes_delta(
            reg_feature
        )  # shape: (N*num_boxes, 4)
        pred_bboxes = self.apply_deltas(
            bboxes_deltas, bboxes.view(-1, 4)
        )  # 应用偏移到边界框

        # 调整输出形状
        return (
            class_logits.view(N, num_boxes, -1),
            pred_bboxes.view(N, num_boxes, -1),
            obj_features,
        )
