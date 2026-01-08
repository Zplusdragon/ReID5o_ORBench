from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights, Transformer, LayerNorm ,QuickGELU
import torch
import copy
import itertools
import torch.nn as nn
from collections import OrderedDict
from .mmencoder_withlora import MMTransformer_withlora

######################ReID5o Model########################
class VisionTokenizer(nn.Module):
    def __init__(self, conv1, class_embedding, positional_embedding,ln_pre):
        super(VisionTokenizer, self).__init__()
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x


class MultimodalVisionEncoder(nn.Module):
    def __init__(self, transformer, ln_post, proj):
        super(MultimodalVisionEncoder, self).__init__()
        self.transformer = transformer
        self.ln_post = ln_post
        self.proj = proj

    def forward(self, x,modality='RGB'):
        x = x.permute(1, 0, 2)
        x = self.transformer(x,modality)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, token_embedding, positional_embedding,transformer,ln_final,text_projection):
        super(CLIPTextEncoder, self).__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection

    def forward(self, text, dtype):
        x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection
        eot_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        embeds = x
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return eot_embed, embeds


class ReID5oModel(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,args.stride_size)
        self.embed_dim = base_cfg['embed_dim']  # 512
        self.width = base_cfg['vision_width']  # 768
        self.image_resolution = base_cfg['image_resolution']
        self.encoder_layers = base_cfg['vision_layers']
        
        # self.heads = self.embed_dim //64 fixed this bug
        self.heads = self.width//64

        self.mmt_depth = args.mmt_depth

        self.clip_text_encoder = self.build_clip_pretrained_text_encoder(base_model)
        self.rgb_tokenizer = self.build_vision_tokenizer(base_model)
        self.nir_tokenizer = self.build_vision_tokenizer(base_model)
        self.cp_tokenizer = self.build_vision_tokenizer(base_model)
        self.sk_tokenizer = self.build_vision_tokenizer(base_model)
        self.vision_encoder = self.build_vision_encoder(base_model,args)

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'mm' in args.loss_names:
            self.create_mm_fusion_module()

        if 'id' in args.loss_names:
            self.create_id_classifier()

    def create_id_classifier(self):
        print('num_classes:{}'.format(self.num_classes))
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)

    def create_mm_fusion_module(self):
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=self.mmt_depth,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5

        self.ln_pre = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # init cross attn
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                         ('gelu', QuickGELU()),
                         ('ln', LayerNorm(self.embed_dim)),
                         ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
        # init mlm head
        nn.init.normal_(self.mm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mm_head.fc.weight, std=proj_std)

    def mm_fusion(self, q, k, v):
        x = self.cross_attn(self.ln_pre(q),self.ln_pre(k),self.ln_pre(v),need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = self.mm_head(x)
        x =  torch.mean(x,dim=1).float()
        return x

    @property
    def dtype(self):
        return self.rgb_tokenizer.conv1.weight.dtype

    def build_vision_tokenizer(self,base_model):
        conv1 = copy.deepcopy(base_model.visual.conv1)
        cls = copy.deepcopy(base_model.visual.class_embedding)
        pe = copy.deepcopy(base_model.visual.positional_embedding)
        ln_pre = copy.deepcopy(base_model.visual.ln_pre)
        return VisionTokenizer(conv1, cls, pe, ln_pre)

    def build_vision_encoder(self,base_model,args):
        if args.add_lora:
            transformer = MMTransformer_withlora(width=self.width,layers=self.encoder_layers,heads=self.heads,lora_r=args.lora_r, num_loras=args.num_loras,lora_layers=args.lora_layers,lora_mode=args.lora_mode)
            stat = copy.deepcopy(base_model.visual.transformer).state_dict()
            transformer.load_state_dict(stat,strict=False)
            print('Pretrained Multimodal Encoder with LoRAs Loaded, with LoRA_r={}, LoRA_layers={}'.format(args.lora_r,args.lora_layers))
        else:
            transformer = copy.deepcopy(base_model.visual.transformer)
        ln_post = copy.deepcopy(base_model.visual.ln_post)
        proj = copy.deepcopy(base_model.visual.proj)
        encoder = MultimodalVisionEncoder(transformer,ln_post,proj)
        return encoder

    def build_clip_pretrained_text_encoder(self,base_model):
        transformer = copy.deepcopy(base_model.transformer)
        token_embedding = copy.deepcopy(base_model.token_embedding)
        positional_embedding = copy.deepcopy(base_model.positional_embedding)
        ln_final = copy.deepcopy(base_model.ln_final)
        text_projection = copy.deepcopy(base_model.text_projection)
        return CLIPTextEncoder(token_embedding, positional_embedding,transformer,ln_final,text_projection)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_rgb_cls(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x[:, 0, :].float()

    def encode_nir_cls(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x[:, 0, :].float()

    def encode_cp_cls(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x[:, 0, :].float()

    def encode_sk_cls(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x[:, 0, :].float()

    def encode_text_cls(self,x):
        x,_ = self.clip_text_encoder(x,self.dtype)
        x = x.float()
        return x

    def encode_rgb_embeds(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x

    def encode_nir_embeds(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x

    def encode_cp_embeds(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x

    def encode_sk_embeds(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x

    def encode_text_embeds(self,x):
        eot,x = self.clip_text_encoder(x,self.dtype)
        return eot,x

    def router_multimodal_embeds(self,rgb,nir,cp,sk,text):
        rgb_embeds = self.encode_rgb_embeds(rgb)
        nir_embeds = self.encode_nir_embeds(nir)
        cp_embeds = self.encode_cp_embeds(cp)
        sk_embeds = self.encode_sk_embeds(sk)
        text_eot,text_embeds = self.encode_text_embeds(text)
        mm_embeds = [nir_embeds,cp_embeds,sk_embeds,text_embeds]
        combined_embeds_for_one = [rgb_embeds[:,0,:].float(),nir_embeds[:,0,:].float(),cp_embeds[:,0,:].float(),sk_embeds[:,0,:].float(),text_eot.float()]

        combinations_for_two = itertools.combinations(mm_embeds, 2)
        combined_embeds_for_two = []
        for combo in combinations_for_two:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_two.append(combined)

        combinations_for_three = itertools.combinations(mm_embeds, 3)
        combined_embeds_for_three = []
        for combo in combinations_for_three:
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_three.append(combined)

        combined_embeds_for_four = [torch.cat(mm_embeds,dim=1)]

        return combined_embeds_for_one,combined_embeds_for_two,combined_embeds_for_three,combined_embeds_for_four

    def forward(self, batch):
        ret = dict()
        rgbs = batch['rgbs']
        nirs = batch['nirs']
        cps = batch['cps']
        sks = batch['sks']
        texts = batch['caption_ids']
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'mm_sdm' in self.current_task:
            cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_sdm(rgb_feat, cone_feat, batch['pids'], logit_scale))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_sdm(rgb_feat, ctwo_feat, batch['pids'], logit_scale))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_sdm(rgb_feat, cthree_feat, batch['pids'], logit_scale))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_sdm(rgb_feat, cfour_feat, batch['pids'], logit_scale))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmsdm_loss': cone_aver_loss})
            ret.update({'ctwo_mmsdm_loss': ctwo_aver_loss})
            ret.update({'cthree_mmsdm_loss': cthree_aver_loss})
            ret.update({'cfour_mmsdm_loss': cfour_aver_loss})

            if 'id' in self.current_task:
                all_feats = [rgb_feat] + cone_feats + ctwo_feats + cthree_feats + cfour_feats
                assert len(all_feats) == 16
                logits_list = []
                for feat in all_feats:
                    logits = self.classifier(feat.half()).float()
                    logits_list.append(logits)
                ret.update({'id_loss': objectives.compute_id(logits_list,batch['pids']) * self.args.id_loss_weight})
                return ret

        if 'mm_itc' in self.current_task:
            cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_itc(rgb_feat, cone_feat, logit_scale))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_itc(rgb_feat, ctwo_feat, logit_scale))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_itc(rgb_feat, cthree_feat, logit_scale))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_itc(rgb_feat, cfour_feat, logit_scale))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmitc_loss': cone_aver_loss})
            ret.update({'ctwo_mmitc_loss': ctwo_aver_loss})
            ret.update({'cthree_mmitc_loss': cthree_aver_loss})
            ret.update({'cfour_mmitc_loss': cfour_aver_loss})
            return ret

        if 'mm_supitc' in self.current_task:
            cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_supitc(rgb_feat, cone_feat, batch['pids']))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_supitc(rgb_feat, ctwo_feat, batch['pids']))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_supitc(rgb_feat, cthree_feat, batch['pids']))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_supitc(rgb_feat, cfour_feat, batch['pids']))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmsupitc_loss': cone_aver_loss})
            ret.update({'ctwo_mmsupitc_loss': ctwo_aver_loss})
            ret.update({'cthree_mmsupitc_loss': cthree_aver_loss})
            ret.update({'cfour_mmsupitc_loss': cfour_aver_loss})
            return ret

        if 'mm_cmpm' in self.current_task:
            cone_embeds, ctwo_embeds, cthree_embeds, cfour_embeds = self.router_multimodal_embeds(rgbs,nirs,cps,sks,texts)

            cone_feats = cone_embeds
            ctwo_feats,cthree_feats,cfour_feats = [],[],[]

            for i in range(len(ctwo_embeds)):
                ctwo_embed = ctwo_embeds[i]
                ctwo_feat = self.mm_fusion(ctwo_embed,ctwo_embed,ctwo_embed)
                ctwo_feats.append(ctwo_feat)

            for i in range(len(cthree_embeds)):
                cthree_embed = cthree_embeds[i]
                cthree_feat = self.mm_fusion(cthree_embed, cthree_embed, cthree_embed)
                cthree_feats.append(cthree_feat)

            for i in range(len(cfour_embeds)):
                cfour_embed = cfour_embeds[i]
                cfour_feat = self.mm_fusion(cfour_embed, cfour_embed, cfour_embed)
                cfour_feats.append(cfour_feat)

            rgb_feat = cone_feats[0]
            cone_feats = cone_feats[1:]

            cone_losses,ctwo_losses,cthree_losses,cfour_losses = [],[],[],[]

            for cone_feat in cone_feats:
                cone_losses.append(objectives.compute_cmpm(rgb_feat, cone_feat, batch['pids']))

            for ctwo_feat in ctwo_feats:
                ctwo_losses.append(objectives.compute_cmpm(rgb_feat, ctwo_feat, batch['pids']))

            for cthree_feat in cthree_feats:
                cthree_losses.append(objectives.compute_cmpm(rgb_feat, cthree_feat, batch['pids']))

            for cfour_feat in cfour_feats:
                cfour_losses.append(objectives.compute_cmpm(rgb_feat, cfour_feat, batch['pids']))

            cone_aver_loss = torch.mean(torch.stack(cone_losses))
            ctwo_aver_loss = torch.mean(torch.stack(ctwo_losses))
            cthree_aver_loss = torch.mean(torch.stack(cthree_losses))
            cfour_aver_loss = torch.mean(torch.stack(cfour_losses))

            ret.update({'cone_mmcmpm_loss': cone_aver_loss})
            ret.update({'ctwo_mmcmpm_loss': ctwo_aver_loss})
            ret.update({'cthree_mmcmpm_loss': cthree_aver_loss})
            ret.update({'cfour_mmcmpm_loss': cfour_aver_loss})
            return ret

        rgb_feats = self.encode_rgb_cls(rgbs)
        nir_feats = self.encode_nir_cls(nirs)
        cp_feats = self.encode_cp_cls(cps)
        sk_feats = self.encode_sk_cls(sks)
        text_feats = self.encode_text_cls(texts)

        if 'itc' in self.current_task:
            ret.update({'nir_itc_loss': objectives.compute_itc(rgb_feats, nir_feats, logit_scale)})
            ret.update({'cp_itc_loss': objectives.compute_itc(rgb_feats, cp_feats, logit_scale)})
            ret.update({'sk_itc_loss': objectives.compute_itc(rgb_feats, sk_feats, logit_scale)})
            ret.update({'txt_itc_loss': objectives.compute_itc(rgb_feats, text_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'nir_sdm_loss': objectives.compute_sdm(rgb_feats, nir_feats, batch['pids'], logit_scale)})
            ret.update({'cp_sdm_loss': objectives.compute_sdm(rgb_feats, cp_feats, batch['pids'], logit_scale)})
            ret.update({'sk_sdm_loss': objectives.compute_sdm(rgb_feats, sk_feats, batch['pids'], logit_scale)})
            ret.update({'txt_sdm_loss': objectives.compute_sdm(rgb_feats, text_feats, batch['pids'], logit_scale)})
        return ret

def build_model(args, num_classes=11003):
    model = ReID5oModel(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
