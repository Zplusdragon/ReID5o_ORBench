import os.path as op
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs
import logging

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID5o Evaluation")
    parser.add_argument("--config_file", default='logs/reid5o_ckpt/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('ORBench', save_dir=args.output_dir, if_train=args.training)
    device = "cuda"
    logger.info(args)

    test_gallery_loader, \
    nir_query_loader, \
    cp_query_loader, \
    sk_query_loader, \
    text_query_loader, \
    nir_cp_query_loader, \
    cp_nir_query_loader, \
    nir_sk_query_loader, \
    sk_nir_query_loader, \
    nir_text_query_loader, \
    text_nir_query_loader, \
    cp_sk_query_loader, \
    sk_cp_query_loader, \
    cp_text_query_loader, \
    text_cp_query_loader, \
    sk_text_query_loader, \
    text_sk_query_loader, \
    nir_cp_sk_query_loader, \
    cp_nir_sk_query_loader, \
    sk_nir_cp_query_loader, \
    nir_cp_text_query_loader, \
    cp_nir_text_query_loader, \
    text_nir_cp_query_loader, \
    nir_sk_text_query_loader, \
    sk_nir_text_query_loader, \
    text_nir_sk_query_loader, \
    cp_sk_text_query_loader, \
    sk_cp_text_query_loader, \
    text_cp_sk_query_loader, \
    nir_cp_sk_text_query_loader, \
    cp_nir_sk_text_query_loader, \
    sk_nir_cp_text_query_loader, \
    text_nir_cp_sk_query_loader, num_classes = build_dataloader(args)

    model = build_model(args, num_classes=1000)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    # 初始化 Evaluator
    evaluator = Evaluator(
        gallery_loader=test_gallery_loader,
        get_mAP=False,
        # 单模态
        nir_query_loader=nir_query_loader,
        cp_query_loader=cp_query_loader,
        sk_query_loader=sk_query_loader,
        text_query_loader=text_query_loader,
        # 双模态
        nir_cp_query_loader=nir_cp_query_loader,
        cp_nir_query_loader=cp_nir_query_loader,
        nir_sk_query_loader=nir_sk_query_loader,
        sk_nir_query_loader=sk_nir_query_loader,
        nir_text_query_loader=nir_text_query_loader,
        text_nir_query_loader=text_nir_query_loader,
        cp_sk_query_loader=cp_sk_query_loader,
        sk_cp_query_loader=sk_cp_query_loader,
        cp_text_query_loader=cp_text_query_loader,
        text_cp_query_loader=text_cp_query_loader,
        sk_text_query_loader=sk_text_query_loader,
        text_sk_query_loader=text_sk_query_loader,
        # 三模态
        nir_cp_sk_query_loader=nir_cp_sk_query_loader,
        cp_nir_sk_query_loader=cp_nir_sk_query_loader,
        sk_nir_cp_query_loader=sk_nir_cp_query_loader,
        nir_cp_text_query_loader=nir_cp_text_query_loader,
        cp_nir_text_query_loader=cp_nir_text_query_loader,
        text_nir_cp_query_loader=text_nir_cp_query_loader,
        nir_sk_text_query_loader=nir_sk_text_query_loader,
        sk_nir_text_query_loader=sk_nir_text_query_loader,
        text_nir_sk_query_loader=text_nir_sk_query_loader,
        cp_sk_text_query_loader=cp_sk_text_query_loader,
        sk_cp_text_query_loader=sk_cp_text_query_loader,
        text_cp_sk_query_loader=text_cp_sk_query_loader,
        # 四模态
        nir_cp_sk_text_query_loader=nir_cp_sk_text_query_loader,
        cp_nir_sk_text_query_loader=cp_nir_sk_text_query_loader,
        sk_nir_cp_text_query_loader=sk_nir_cp_text_query_loader,
        text_nir_cp_sk_query_loader=text_nir_cp_sk_query_loader
    )


    logger = logging.getLogger("ORBench.test")
    logger.info("Enter evaluating")
    top1 = evaluator.eval(model.eval())

