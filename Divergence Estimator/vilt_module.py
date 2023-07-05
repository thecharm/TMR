import torch
import torch.nn as nn
from . import vision_transformer as vit

from transformers.pytorch_transformers.modeling_bert import BertConfig, BertEmbeddings
from . import heads
from .vilt_utils import SupervisedContrastiveLoss
from ..datasets.oscar_tsv import _transform


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class ViLTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        bert_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.hidden_size * config.mlp_ratio,
            max_position_embeddings=config.max_text_len,
            hidden_dropout_prob=config.drop_rate,
            attention_probs_dropout_prob=config.drop_rate,
        )
        self.config = config
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(init_weights)

        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings.apply(init_weights)
        if self.config.load_path == "":
            self.transformer = getattr(vit, self.config.vit)(
                pretrained=True, config=self.config
            )
        else:
            self.transformer = getattr(vit, self.config.vit)(
                pretrained=False, config=self.config
            )

        self.pooler = heads.Pooler(config.hidden_size)
        self.pooler.apply(init_weights)
        hs = config.hidden_size
        vs = self.config.interaction_label_size
        self.interaction_classifier = nn.Sequential(
            nn.Linear(hs, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, vs),
        )
        self.losses = SupervisedContrastiveLoss()
        self.interaction_classifier.apply(init_weights)

        if self.config.load_path != "":
            ckpt = torch.load(self.config.load_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, text_ids,
                input_masks,
                images,
                mask_image=False,
                image_token_type_idx=1):
        text_embeds = self.text_embeddings(text_ids)
        text_masks = input_masks[:, :len(text_ids[0])]
        (image_embeds, image_masks, patch_index, image_labels) = self.transformer.visual_embed(
            images,
            max_image_len=self.config.max_image_len,
            mask_it=mask_image,
        )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        cls_feats = self.pooler(x)
        selection_logits = self.softmax(self.interaction_classifier(cls_feats))

        strong_selection_weight, weak_selection_weight = selection_logits[:, 2], selection_logits[:, 1]
        summation_weight = strong_selection_weight + weak_selection_weight
        return strong_selection_weight / summation_weight, weak_selection_weight / summation_weight
