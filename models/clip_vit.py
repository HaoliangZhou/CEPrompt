from collections import OrderedDict
import torch
import torch.nn as nn
from models.cat import CATPromptLearner, TextEncoder as TextEncoder_cat


class CLIPVIT(nn.Module):
    def __init__(self, args, clip_model, embed_dim=768,):
        super().__init__()

        self.final_dim = 512
        # self.final_dim = 768  # for vit-l/14
        self.global_only = False  # global+local
        self.local_only = False  # global+local

        # visual
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.use_clip_proj = True  # 默认用CLIP的proj

        if not self.use_clip_proj:  # 如果不用CLIP的proj, 则用自己的proj
            self.projection = nn.Sequential(OrderedDict([  # No
                ('fc1', nn.Linear(embed_dim, self.final_dim)),
                ('act', nn.Tanh()),
                ('fc2', nn.Linear(self.final_dim, self.final_dim)),
            ], ))

        self.projection_dist = clip_model.visual.proj  # 全局头用1层fc
        self.alpha = args.alpha
        self.topk = args.topk

        # text
        self.clip_text_encoder = clip_model.encode_text
        self.stage2_name = args.stage2_name  # cat
        # stage2, text, cat
        if self.stage2_name == "cat":
            self.compound_prompt_nctx = args.n_ctx
            self.num_layers = len(self.transformer.resblocks)  # 12
            self.dtype = clip_model.dtype
            self.text_encoder = TextEncoder_cat(args, clip_model)
            self.prompt_learner = MultiModalPromptLearner(args, clip_model)
            self.depth = args.prompts_depth

    # ========================= Stage1 =========================
    # 相当于走了一遍CLIP的visual部分(VisualTransformer), 截止到proj.
    def forward_features(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid], (bs,768,14,14)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2], (bs,768,196)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width], (bs,196,768)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width], (bs,197,768)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND, (197,bs,768)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD  (bs,197,768)

        x = self.ln_post(x)
        return x

    def forward_text(self, label_token):
        with torch.no_grad():
            x = self.clip_text_encoder(label_token)

        return x

    def forward(self, x, label_token, norm_pred=True):
        # text forward
        label_embed = self.forward_text(label_token)
        label_embed = label_embed / label_embed.norm(dim=-1, keepdim=True)

        # image forward
        x = self.forward_features(x)  # (bs,197,768), 其中x[:, 1:]为local features(图中o_patch);  x[:, 0]为global feature(图中o_cls & o_dist)
        dist_feat = x[:, 0] @ self.projection_dist  # (bs,512), x的第一列特征(class token) 做线性变换(图中e_cls)
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)
        # dist_feat = self.dist_norm(dist_feat)  # 二选一即可,都用效果不好88-89

        # For Global Head Only Ablation
        if self.global_only:
            # print("Global Only Ablation")
            score = dist_feat @ label_embed.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat
        # For Local Head Only Ablation
        elif self.local_only:
            # print("Local Only Ablation")
            pred_feat = x[:, 1:] @ self.projection_dist
            pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
            score = torch.topk(pred_feat @ label_embed.t(), k=self.topk, dim=1)[0].mean(dim=1)
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, x[:, 1:], dist_feat

        # Default, Global and Local
        else:
            # print("Default: Global and Local")
            if not self.use_clip_proj:
                pred_feat = self.projection(x[:, 1:])
            else:
                pred_feat = x[:, 1:] @ self.projection_dist  # (bs,196,512), local features(图中o_patch) 经过线性投影层(图中e_1~e_N)
            pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
            # pred_feat = self.pred_norm(pred_feat)

            score1 = torch.topk(pred_feat @ label_embed.t(), k=self.topk, dim=1)[0].mean(dim=1)  # (bs,7), 即Top-K Mean pooling (而且如图特征空间: 有没有可能FER是要多个e_x的组合才能预测一个label?)
            score2 = dist_feat @ label_embed.t()  # (bs,7), 图中S_global
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)

            score = self.alpha * score1 + (1-self.alpha) * score2
            # score = (score1 + score2) / 2

            return score, pred_feat, dist_feat  # 输出了这张图片的预测分数, 及经过投影后的local和global特征

    # 仅image encoder, test时用
    def encode_img(self, x):
        x = self.forward_features(x)
        pred_feat = x[:, 1:] @ self.projection_dist
        dist_feat = x[:, 0] @ self.projection_dist
        pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)
        return pred_feat, dist_feat

    # ========================= Stage2 =========================
    # 二阶段cat的过程1
    def forward_vitblocks(self, inputs):
        hidden_states = None
        x = inputs[0]  # (197,bs,768)
        compound_prompts_deeper = inputs[1]
        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](x)
            elif i < self.depth:
                # 沿袭vpt
                prefix = hidden_states[:1, :, :]  # (1,bs,768)
                suffix = hidden_states[1 + self.compound_prompt_nctx:, :, :]  # (194,bs,768)

                # cat自己的visual prompt，改1，差不多结果
                # prefix = hidden_states[0:x.shape[0] - self.compound_prompt_nctx, :, :]  # (197-2,bs,768)

                # Create/configure learnable tokens of this layer
                textual_context = compound_prompts_deeper[i - 1]  # (2,768)
                textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()  # (2,bs,768)
                # Add the learnable tokens of this layer with the input, replaced by previous

                # layer learnable tokens
                hidden_states = torch.cat([prefix, textual_context, suffix], dim=0)  # 沿袭vpt  # (197,bs,768)
                # hidden_states = torch.cat([prefix, textual_context], dim=0)  # cat自己的visual prompt，改2
                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = self.transformer.resblocks[i](hidden_states)
        return hidden_states

    # 二阶段cat的过程2
    def get_features(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND (197,bs,768)
        inputs = [x, compound_deeper_prompts, 0]  # Again combine the inputs, so nn.sequential can work
        x = self.forward_vitblocks(inputs)  # third argument is counter
        # x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD (bs,197,768)

        x = self.ln_post(x)

        return x

    # 二阶段cat的总过程
    def forward_cat(self, x, norm_pred=True, label=None):
        # text forward
        # tokenized_prompts = self.prompt_learner.tokenized_prompts if label is None else self.prompt_learner.tokenized_prompts[label]
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # image forward
        img_features = self.get_features(x.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        dist_feat = img_features[:, 0] @ self.projection_dist  # (bs,512), x的第一列特征(class token) 做线性变换(图中e_cls)
        dist_feat = dist_feat / dist_feat.norm(dim=-1, keepdim=True)

        # For Global Head Only Ablation
        if self.global_only:
            score = dist_feat @ text_features.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return score, img_features[:, 1:], dist_feat

        # Default, Global and Local
        else:
            if not self.use_clip_proj:
                pred_feat = self.projection(img_features[:, 1:])
            else:
                pred_feat = img_features[:, 1:] @ self.projection_dist  # (bs,196,512), local features(图中o_patch) 经过线性投影层(图中e_1~e_N)
            pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
            # pred_feat = self.pred_norm(pred_feat)

            score1 = torch.topk(pred_feat @ text_features.t(), k=self.topk, dim=1)[0].mean(dim=1)  # (bs,7), 即Top-K Mean pooling (而且如图特征空间: 有没有可能FER是要多个e_x的组合才能预测一个label?)
            score2 = dist_feat @ text_features.t()  # (bs,7), 图中S_global
            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)

            score = self.alpha * score1 + (1 - self.alpha) * score2  # (bs,7), 图中S
            # score = (score1 + score2) / 2

            return score, pred_feat, dist_feat

