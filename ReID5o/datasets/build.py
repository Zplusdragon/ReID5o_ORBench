import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .bases import (
    ImageTextDataset, GalleryDataset, SingleQueryTextDataset, SingleQueryVisionDataset,
    TwoQueryTextVisionDataset, TwoQueryVisionTextDataset, TwoQueryAllVisionDataset,
    ThreeQueryTextVisionVisionDataset, ThreeQueryVisionVisionTextDataset, ThreeQueryAllVisionDataset,
    FourQueryTextVisionVisionVisionDataset, FourQueryVisionVisionVisionTextDataset
)
from .orbench import ORBench

__factory = {'ORBench': ORBench}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform


def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
            batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def _create_query_datasets(test_queries, val_transforms):
    """创建所有查询类型的Dataset"""
    query_datasets = {}

    # 单模态查询
    single_modalities = ['NIR', 'CP', 'SK', 'TEXT']
    for modality in single_modalities:
        if modality == 'TEXT':
            query_datasets[modality] = SingleQueryTextDataset(test_queries[modality])
        else:
            query_datasets[modality] = SingleQueryVisionDataset(test_queries[modality], val_transforms)

    # 双模态查询
    two_modality_queries = {
        'NIR+CP': TwoQueryAllVisionDataset,
        'CP+NIR': TwoQueryAllVisionDataset,
        'NIR+SK': TwoQueryAllVisionDataset,
        'SK+NIR': TwoQueryAllVisionDataset,
        'NIR+TEXT': TwoQueryVisionTextDataset,
        'TEXT+NIR': TwoQueryTextVisionDataset,
        'CP+SK': TwoQueryAllVisionDataset,
        'SK+CP': TwoQueryAllVisionDataset,
        'CP+TEXT': TwoQueryVisionTextDataset,
        'TEXT+CP': TwoQueryTextVisionDataset,
        'SK+TEXT': TwoQueryVisionTextDataset,
        'TEXT+SK': TwoQueryTextVisionDataset,
    }

    for query_key, dataset_class in two_modality_queries.items():
        query_datasets[query_key] = dataset_class(test_queries[query_key], val_transforms)

    # 三模态查询
    three_modality_queries = {
        'NIR+CP+SK': ThreeQueryAllVisionDataset,
        'CP+NIR+SK': ThreeQueryAllVisionDataset,
        'SK+NIR+CP': ThreeQueryAllVisionDataset,
        'NIR+CP+TEXT': ThreeQueryVisionVisionTextDataset,
        'CP+NIR+TEXT': ThreeQueryVisionVisionTextDataset,
        'TEXT+NIR+CP': ThreeQueryTextVisionVisionDataset,
        'NIR+SK+TEXT': ThreeQueryVisionVisionTextDataset,
        'SK+NIR+TEXT': ThreeQueryVisionVisionTextDataset,
        'TEXT+NIR+SK': ThreeQueryTextVisionVisionDataset,
        'CP+SK+TEXT': ThreeQueryVisionVisionTextDataset,
        'SK+CP+TEXT': ThreeQueryVisionVisionTextDataset,
        'TEXT+CP+SK': ThreeQueryTextVisionVisionDataset,
    }

    for query_key, dataset_class in three_modality_queries.items():
        query_datasets[query_key] = dataset_class(test_queries[query_key], val_transforms)

    # 四模态查询
    four_modality_queries = {
        'NIR+CP+SK+TEXT': FourQueryVisionVisionVisionTextDataset,
        'CP+NIR+SK+TEXT': FourQueryVisionVisionVisionTextDataset,
        'SK+NIR+CP+TEXT': FourQueryVisionVisionVisionTextDataset,
        'TEXT+NIR+CP+SK': FourQueryTextVisionVisionVisionDataset,
    }

    for query_key, dataset_class in four_modality_queries.items():
        query_datasets[query_key] = dataset_class(test_queries[query_key], val_transforms)

    return query_datasets


def _create_dataloaders(query_datasets, test_batch_size, num_workers):
    """为所有查询Dataset创建DataLoader"""
    dataloaders = {}

    for query_key, dataset in query_datasets.items():
        dataloaders[query_key] = DataLoader(
            dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )

    return dataloaders


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("ORBench.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    if tranforms:
        val_transforms = tranforms
    else:
        val_transforms = build_transforms(img_size=args.img_size, is_train=False)

    if args.training:
        train_transforms = build_transforms(
            img_size=args.img_size, aug=args.img_aug, is_train=True
        )

        train_set = ImageTextDataset(
            dataset.train, train_transforms, text_length=args.text_length
        )

        logger.info('using random sampler')
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate
        )

        # use test set as validate set
        ds = dataset.test
        test_gallery_set = GalleryDataset(
            ds['gallery_pids'], ds['gallery_paths'], val_transforms
        )
        test_gallery_loader = DataLoader(
            test_gallery_set, batch_size=args.test_batch_size,
            shuffle=False, num_workers=num_workers
        )

        # 创建所有查询的Dataset和DataLoader
        query_datasets = _create_query_datasets(ds['queries'], val_transforms)
        query_dataloaders = _create_dataloaders(
            query_datasets, args.test_batch_size, num_workers
        )

        # 按照原始顺序返回所有loader
        return_order = [
            'train_loader', 'test_gallery_loader',
            # 单模态
            'NIR', 'CP', 'SK', 'TEXT',
            # 双模态
            'NIR+CP', 'CP+NIR', 'NIR+SK', 'SK+NIR', 'NIR+TEXT', 'TEXT+NIR',
            'CP+SK', 'SK+CP', 'CP+TEXT', 'TEXT+CP', 'SK+TEXT', 'TEXT+SK',
            # 三模态
            'NIR+CP+SK', 'CP+NIR+SK', 'SK+NIR+CP', 'NIR+CP+TEXT', 'CP+NIR+TEXT',
            'TEXT+NIR+CP', 'NIR+SK+TEXT', 'SK+NIR+TEXT', 'TEXT+NIR+SK',
            'CP+SK+TEXT', 'SK+CP+TEXT', 'TEXT+CP+SK',
            # 四模态
            'NIR+CP+SK+TEXT', 'CP+NIR+SK+TEXT', 'SK+NIR+CP+TEXT', 'TEXT+NIR+CP+SK',
            'num_classes'
        ]

        result = [train_loader, test_gallery_loader]
        for key in return_order[2:-1]:  # 跳过train_loader, test_gallery_loader和num_classes
            result.append(query_dataloaders[key])
        result.append(num_classes)

        return tuple(result)

    else:
        # build dataloader for testing
        ds = dataset.test

        test_gallery_set = GalleryDataset(
            ds['gallery_pids'], ds['gallery_paths'], val_transforms
        )
        test_gallery_loader = DataLoader(
            test_gallery_set, batch_size=args.test_batch_size,
            shuffle=False, num_workers=num_workers
        )

        # 创建所有查询的Dataset和DataLoader
        query_datasets = _create_query_datasets(ds['queries'], val_transforms)
        query_dataloaders = _create_dataloaders(
            query_datasets, args.test_batch_size, num_workers
        )

        # 按照原始顺序返回所有loader
        return_order = [
            'test_gallery_loader',
            # 单模态
            'NIR', 'CP', 'SK', 'TEXT',
            # 双模态
            'NIR+CP', 'CP+NIR', 'NIR+SK', 'SK+NIR', 'NIR+TEXT', 'TEXT+NIR',
            'CP+SK', 'SK+CP', 'CP+TEXT', 'TEXT+CP', 'SK+TEXT', 'TEXT+SK',
            # 三模态
            'NIR+CP+SK', 'CP+NIR+SK', 'SK+NIR+CP', 'NIR+CP+TEXT', 'CP+NIR+TEXT',
            'TEXT+NIR+CP', 'NIR+SK+TEXT', 'SK+NIR+TEXT', 'TEXT+NIR+SK',
            'CP+SK+TEXT', 'SK+CP+TEXT', 'TEXT+CP+SK',
            # 四模态
            'NIR+CP+SK+TEXT', 'CP+NIR+SK+TEXT', 'SK+NIR+CP+TEXT', 'TEXT+NIR+CP+SK',
            'num_classes'
        ]

        result = [test_gallery_loader]
        for key in return_order[1:-1]:  # 跳过test_gallery_loader和num_classes
            result.append(query_dataloaders[key])
        result.append(num_classes)

        return tuple(result)