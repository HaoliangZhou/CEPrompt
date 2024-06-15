import copy
import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(args):
    model_path = args.clip_path
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.num_layers = len(self.transformer.resblocks)
        self.depth = args.prompts_depth
        self.compound_prompt_nctx = args.n_ctx

    def forward_resblocks(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        hidden_states = None
        x = inputs[0]
        compound_prompts_deeper = inputs[1]

        for i in range(self.num_layers):
            # print("i = ", i)
            if i == 0:
                hidden_states = self.transformer.resblocks[i](x)
            elif i < self.depth:
                prefix = hidden_states[:1, :, :]
                suffix = hidden_states[1 + self.compound_prompt_nctx:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = compound_prompts_deeper[i - 1]
                textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # print("textual_context.shape = ", textual_context.shape)
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                hidden_states = torch.cat([prefix, textual_context, suffix], dim=0)
                # print("hidden_states.shape = ", hidden_states.shape)
                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                # For the last layer, we do not need to add any duplicate, as it is already added
                # as the shallow version
                hidden_states = self.transformer.resblocks[i](hidden_states)

        return hidden_states

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        x = self.forward_resblocks(combined)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CATPromptLearner(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        classnames = args.label_nms
        n_cls = len(classnames)
        n_ctx = args.n_ctx
        # ctx_init = None   # 不用a photo初始化
        # ctx_init = "a facial expression of"
        ctx_init = args.ctxinit
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.input_size
        CSC = False # True for CSC, False for UC
        self.use_class_invariant = args.use_class_invariant
        # self.use_class_invariant = True  # 如果默认使用class-invariant, 就解除注释, 令其为True
        print("is use_class_invariant:", self.use_class_invariant)
        # Default is 1, which is compound shallow prompting
        # assert cfg.TRAINER.cat.PROMPT_DEPTH >= 1, "For cat, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.prompts_depth  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            print("ctx_init:",ctx_init)
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt.to(args.device)).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('CAT design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of CAT context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        if self.use_class_invariant:
            n_ctx_1 = 1
            print("Initializing class-invariant contexts")
            ctx_vectors_invariant = torch.empty(n_ctx_1, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_vectors_invariant, std=0.02)
            prompt_prefix = " ".join(["X"] * (n_ctx_1 + n_ctx))

            print(f'with class-invariant, Initial context: "{prompt_prefix}"')
            print(f"with class-invariant, Number of context words (tokens): {n_ctx + n_ctx_1}")

            self.ctx_invariant = nn.Parameter(ctx_vectors_invariant)

        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512)) for _ in range(self.compound_prompts_depth - 1)])

        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)\

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.to(args.device)).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS

        if self.use_class_invariant:
            self.register_buffer("token_suffix", embedding[:, 1 + (n_ctx + n_ctx_1):, :])
            self.n_ctx_1 = n_ctx_1
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        use_class_invariant = self.use_class_invariant
        if use_class_invariant:
            ctx_invariant = self.ctx_invariant
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx_invariant.repeat(self.n_cls, 1, 1),
                    ctx,
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            prompts = torch.cat(
                [
                    prefix,  # (dim0, 1, dim)
                    ctx,  # (dim0, n_ctx, dim)
                    suffix,  # (dim0, *, dim)
                ],
                dim=1,
            )

        return prompts

    def forward(self, label=None):
        if label is not None:
            ctx = self.ctx[label]
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        # prompts = self.construct_prompts(ctx, prefix, suffix, label) # if use con_loss, (B, n_ctx, ctx_dim)
        prompts = self.construct_prompts(ctx, prefix, suffix)  # if use ce_loss (cls, n_ctx, ctx_dim)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts  # pass here original, as for visual 768 is required


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
