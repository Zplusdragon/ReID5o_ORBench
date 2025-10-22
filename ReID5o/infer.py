import torch
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.iotools import load_train_configs, read_image
from utils.simple_tokenizer import SimpleTokenizer
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import textwrap

class ImageRetriever:
    def __init__(self, config_path: str, device: str = "cuda"):
        self.device = device
        self.args = self._load_config(config_path)
        self.model = self._build_model()
        self.transform = self._build_transform()
        self.tokenizer = SimpleTokenizer()

    def _load_config(self, config_path: str):
        args = load_train_configs(config_path)
        args.batch_size = 1024
        args.training = False
        return args

    def _build_model(self):
        model = build_model(self.args, num_classes=1000)
        checkpointer = Checkpointer(model)
        checkpointer.load(f=op.join(self.args.output_dir, 'best.pth'))
        return model.to(self.device).eval()

    def _build_transform(self):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        return T.Compose([
            T.Resize((384, 128)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def tokenize(self, caption: str, text_length: int = 77) -> torch.LongTensor:
        if not caption:
            return torch.zeros(text_length, dtype=torch.long).to(self.device)

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self.tokenizer.encode(caption) + [eot_token]

        result = torch.zeros(text_length, dtype=torch.long)
        if len(tokens) > text_length:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token

        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def extract_gallery_features(self, gallery_loader) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        gids, gfeats, gpaths = [], [], []
        for batch in gallery_loader:
            pid, img = batch
            img_paths = gallery_loader.dataset.img_paths[len(gpaths):len(gpaths) + img.size(0)]
            with torch.no_grad():
                img_feat = self.model.encode_rgb_cls(img.to(self.device))
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
            gpaths.extend(img_paths)

        gids = torch.cat(gids, 0)
        gfeats = F.normalize(torch.cat(gfeats, 0), p=2, dim=1)
        return gids, gfeats, gpaths

    def extract_query_features(self,
                               nir_path: Optional[str] = None,
                               cp_path: Optional[str] = None,
                               sk_path: Optional[str] = None,
                               text: Optional[str] = None) -> torch.Tensor:
        modality_embeddings = []
        modality_clses = []

        if nir_path and op.exists(nir_path):
            nir_tensor = self.transform(read_image(nir_path)).to(self.device).unsqueeze(0)
            with torch.no_grad():
                nir_embeds = self.model.encode_nir_embeds(nir_tensor)
                nir_cls = self.model.encode_nir_cls(nir_tensor)
            modality_embeddings.append(nir_embeds)
            modality_clses.append(nir_cls)

        if cp_path and op.exists(cp_path):
            cp_tensor = self.transform(read_image(cp_path)).to(self.device).unsqueeze(0)
            with torch.no_grad():
                cp_embeds = self.model.encode_cp_embeds(cp_tensor)
                cp_cls = self.model.encode_cp_cls(cp_tensor)
            modality_embeddings.append(cp_embeds)
            modality_clses.append(cp_cls)

        if sk_path and op.exists(sk_path):
            sk_tensor = self.transform(read_image(sk_path)).to(self.device).unsqueeze(0)
            with torch.no_grad():
                sk_embeds = self.model.encode_sk_embeds(sk_tensor)
                sk_cls = self.model.encode_sk_cls(sk_tensor)
            modality_embeddings.append(sk_embeds)
            modality_clses.append(sk_cls)

        if text and text.strip():
            txt_tensor = self.tokenize(text).to(self.device).unsqueeze(0)
            with torch.no_grad():
                _, text_embeds = self.model.encode_text_embeds(txt_tensor)
                text_cls = self.model.encode_text_cls(txt_tensor)
            modality_embeddings.append(text_embeds)
            modality_clses.append(text_cls)

        if not modality_embeddings:
            raise ValueError("At least one modality input!")

        with torch.no_grad():
            if len(modality_embeddings) == 1:
                fusion_feats = modality_clses[0]
            else:
                combined = torch.cat(modality_embeddings, dim=1)
                fusion_feats = self.model.mm_fusion(combined, combined, combined)

        return F.normalize(fusion_feats, p=2, dim=1)

    def retrieve(self, query_features: torch.Tensor, gallery_features: torch.Tensor, topk: int = 10):
        similarity = query_features @ gallery_features.t()
        _, indices = torch.topk(similarity, k=topk, dim=1, largest=True, sorted=True)
        return indices.cpu()


def retrieve_image_paths_and_ids(indices: torch.Tensor, gallery_paths: List[str], gallery_true_ids: List[int]) -> Tuple[
    List[str], List[int]]:
    retrieved_paths = [gallery_paths[j] for j in indices[0]]
    retrieved_ids = [gallery_true_ids[j] for j in indices[0]]
    return retrieved_paths, retrieved_ids


def plot_retrieval_results(retrieved_paths: List[str],
                           retrieved_ids: List[int],
                           query_id: int,
                           nir_path: Optional[str] = None,
                           cp_path: Optional[str] = None,
                           sk_path: Optional[str] = None,
                           text: Optional[str] = None,
                           save_path: str = None):
    query_modalities = []
    modality_titles = []

    if nir_path and op.exists(nir_path):
        query_modalities.append(nir_path)
        modality_titles.append("NIR")

    if cp_path and op.exists(cp_path):
        query_modalities.append(cp_path)
        modality_titles.append("CP")

    if sk_path and op.exists(sk_path):
        query_modalities.append(sk_path)
        modality_titles.append("SK")

    if text and text.strip():
        query_modalities.append(text)
        modality_titles.append("Text")

    n_queries = len(query_modalities)
    n_results = len(retrieved_paths)
    total_cols = n_queries + n_results

    if total_cols == 0:
        print("No results to be displayed.")
        return

    fig, axes = plt.subplots(1, total_cols, figsize=(3.5 * total_cols, 6.5))

    if total_cols == 1:
        axes = [axes]

    for i, (modality, title) in enumerate(zip(query_modalities, modality_titles)):
        if title == "Text":
            axes[i].axis('off')
            wrapped_text = textwrap.fill(modality, width=30)
            axes[i].text(0.5, 0.5, f'Text Query:\n{wrapped_text}',
                         transform=axes[i].transAxes,
                         fontsize=20,
                         ha='center',
                         va='center',
                         wrap=True)
        else:
            img = Image.open(modality).resize((128, 256))
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=18)
            axes[i].axis('off')

    for i, (img_path, retrieved_id) in enumerate(zip(retrieved_paths, retrieved_ids)):
        img = Image.open(img_path).resize((128, 256))
        axes[i + n_queries].imshow(img)

        if retrieved_id == query_id:
            title_color = 'green'
            match_status = "✓"
        else:
            title_color = 'red'
            match_status = "✗"

        axes[i + n_queries].set_title(f"Top {i + 1} (ID:{retrieved_id}) {match_status}",
                                      fontsize=18, color=title_color)
        axes[i + n_queries].axis('off')

    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.2)
    plt.show()


def main():
    # Initialize retriever
    retriever = ImageRetriever('logs/reid5o_ckpt/configs.yaml')

    # Build gallery data loaders
    data_loaders = build_dataloader(retriever.args)
    test_gallery_loader = data_loaders[0]

    # Extract gallery features and IDs
    gallery_ids, gallery_features, gallery_paths = retriever.extract_gallery_features(test_gallery_loader)

    # Example 1
    query_inputs_full = {
        'id': 888,
        'nir_path': "/data/970ep/jlongzuo/ORBench/nir/0706/0706_sysu_0228_cam3_0013_nir.jpg",
        'cp_path': "/data/970ep/jlongzuo/ORBench/cp/0706/0706_sysu_0228_back_1_colorpencil.jpg",
        'sk_path': "/data/970ep/jlongzuo/ORBench/sk/0706/0706_sysu_0228_front_4_sketch.jpg",
        'text': "A young man, of medium to slim build, with short black hair and wearing a pair of black-rimmed glasses. He is dressed in a yellow short-sleeved T-shirt, with the collar and cuffs designed in black. Below, he pairs this with a set of blue trousers that hit right at the ankle. On his feet are a pair of sport shoes with gray uppers and white soles. He carries a blue backpack, its zipper parts accentuated with fluorescent green."
    }

    # Example 2
    query_inputs_nir_sk_text = {
        'id': 888,
        'nir_path': "/data/970ep/jlongzuo/ORBench/nir/0706/0706_sysu_0228_cam3_0013_nir.jpg",
        'sk_path': "/data/970ep/jlongzuo/ORBench/sk/0706/0706_sysu_0228_front_4_sketch.jpg",
        'text': "A young man, of medium to slim build, with short black hair and wearing a pair of black-rimmed glasses. He is dressed in a yellow short-sleeved T-shirt, with the collar and cuffs designed in black. Below, he pairs this with a set of blue trousers that hit right at the ankle. On his feet are a pair of sport shoes with gray uppers and white soles. He carries a blue backpack, its zipper parts accentuated with fluorescent green."
    }

    # Example 3
    query_inputs_sk_text = {
        'id': 888,
        'sk_path': "/data/970ep/jlongzuo/ORBench/sk/0706/0706_sysu_0228_front_4_sketch.jpg",
        'text': "A young man, of medium to slim build, with short black hair and wearing a pair of black-rimmed glasses. He is dressed in a yellow short-sleeved T-shirt, with the collar and cuffs designed in black. Below, he pairs this with a set of blue trousers that hit right at the ankle. On his feet are a pair of sport shoes with gray uppers and white soles. He carries a blue backpack, its zipper parts accentuated with fluorescent green."
    }

    # Example 4
    query_inputs_nir_text = {
        'id': 888,
        'nir_path': "/data/970ep/jlongzuo/ORBench/nir/0706/0706_sysu_0228_cam3_0013_nir.jpg",
        'text': "A young man, of medium to slim build, with short black hair and wearing a pair of black-rimmed glasses. He is dressed in a yellow short-sleeved T-shirt, with the collar and cuffs designed in black. Below, he pairs this with a set of blue trousers that hit right at the ankle. On his feet are a pair of sport shoes with gray uppers and white soles. He carries a blue backpack, its zipper parts accentuated with fluorescent green."
    }

    # Example 5
    query_inputs_text = {
        'id': 888,
        'text': "A young man, of medium to slim build, with short black hair and wearing a pair of black-rimmed glasses. He is dressed in a yellow short-sleeved T-shirt, with the collar and cuffs designed in black. Below, he pairs this with a set of blue trousers that hit right at the ankle. On his feet are a pair of sport shoes with gray uppers and white soles. He carries a blue backpack, its zipper parts accentuated with fluorescent green."
    }

    # Example 6
    query_inputs_sk = {
        'id': 888,
        'sk_path': "/data/970ep/jlongzuo/ORBench/sk/0706/0706_sysu_0228_front_4_sketch.jpg",
    }

    # Example 7
    query_inputs_nir = {
        'id': 888,
        'nir_path': "/data/970ep/jlongzuo/ORBench/nir/0706/0706_sysu_0228_cam3_0013_nir.jpg",
    }

    # Example 8
    query_inputs_cp = {
        'id': 888,
        'cp_path': "/data/970ep/jlongzuo/ORBench/cp/0706/0706_sysu_0228_back_1_colorpencil.jpg",
    }


    for idx, query_inputs in enumerate([query_inputs_full,query_inputs_nir_sk_text,query_inputs_sk_text,query_inputs_nir_text,query_inputs_text,query_inputs_sk,query_inputs_nir,query_inputs_cp]):
        # Extract query features and retrieve
        query_features = retriever.extract_query_features(
            nir_path=query_inputs.get('nir_path'),
            cp_path=query_inputs.get('cp_path'),
            sk_path=query_inputs.get('sk_path'),
            text=query_inputs.get('text')
        )
        indices = retriever.retrieve(query_features, gallery_features, topk=10)

        retrieved_paths, retrieved_ids = retrieve_image_paths_and_ids(indices, gallery_paths, gallery_ids)

        plot_retrieval_results(
            retrieved_paths=retrieved_paths,
            retrieved_ids=retrieved_ids,
            query_id=query_inputs['id'],
            nir_path=query_inputs.get('nir_path'),
            cp_path=query_inputs.get('cp_path'),
            sk_path=query_inputs.get('sk_path'),
            text=query_inputs.get('text'),
            save_path=f'retrieval_result_{idx}.png'
        )

        matched_count = sum(1 for rid in retrieved_ids if rid == query_inputs['id'])
        print(f"Retrieval Results {idx}:")
        print(f"Query ID: {query_inputs['id']}")
        print(f"Matched Number: {matched_count}/{len(retrieved_ids)}")
        print(f"Matched Ratio: {matched_count / len(retrieved_ids) * 100:.1f}%")


if __name__ == "__main__":
    main()











