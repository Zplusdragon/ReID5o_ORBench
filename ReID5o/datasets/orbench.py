import os.path as op
import random
from utils.iotools import read_json
from .bases import BaseDataset
import os

class ORBench(BaseDataset):
    dataset_dir = 'ORBench'

    def __init__(self, root='', verbose=True):
        super(ORBench, self).__init__()
        self.dataset_root = op.join(root,self.dataset_dir)
        self.train_anno_path = op.join(self.dataset_root, 'train_annos.json')
        self.test_anno_path = op.join(self.dataset_root, 'test_gallery_and_queries.json')

        self.train_annos = read_json(self.train_anno_path)
        self.test_annos = read_json(self.test_anno_path)
        self.nir_paths, self.cp_paths, self.sk_paths = self.get_paths()
        self.random_sampling()

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)

        if verbose:
            self.logger.info("=> ORBench Images and Captions are loaded")
            self.show_dataset_info()

    def random_sampling(self):
        print("Random Sampling Processing...")
        for anno in self.train_annos:
            real_identity = anno['file_path'].split('/')[1]
            anno['nir_path'] = random.sample(self.nir_paths[real_identity],1)[0]
            anno['cp_path'] = random.sample(self.cp_paths[real_identity], 1)[0]
            anno['sk_path'] = random.sample(self.sk_paths[real_identity], 1)[0]
        print("Random Sampling Completed!")

    def get_paths(self):
        nir_paths = {}
        for nir_identity in os.listdir(os.path.join(self.dataset_root,'nir')):
            if nir_identity not in nir_paths.keys():
                nir_paths[nir_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root,'nir',nir_identity))
            for path in identity_paths:
                nir_paths[nir_identity].append(os.path.join('nir',nir_identity,path))

        cp_paths = {}
        for cp_identity in os.listdir(os.path.join(self.dataset_root,'cp')):
            if cp_identity not in cp_paths.keys():
                cp_paths[cp_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root,'cp', cp_identity))
            for path in identity_paths:
                cp_paths[cp_identity].append(os.path.join('cp', cp_identity, path))

        sk_paths = {}
        for sk_identity in os.listdir(os.path.join(self.dataset_root, 'sk')):
            if sk_identity not in sk_paths.keys():
                sk_paths[sk_identity] = []
            identity_paths = os.listdir(os.path.join(self.dataset_root, 'sk', sk_identity))
            for path in identity_paths:
                sk_paths[sk_identity].append(os.path.join('sk', sk_identity, path))

        return nir_paths, cp_paths, sk_paths

    def _process_anno(self, annos, training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1  # make pid begin from 0
                pid_container.add(pid)
                rgb_path = op.join(self.dataset_root, anno['file_path'])
                nir_path = op.join(self.dataset_root, anno['nir_path'])
                cp_path = op.join(self.dataset_root, anno['cp_path'])
                sk_path = op.join(self.dataset_root, anno['sk_path'])

                caption = anno['caption']  # caption list
                dataset.append((pid, image_id, rgb_path, nir_path, cp_path, sk_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container

        else:
            gallery_paths = []
            gallery_pids = []

            for anno in annos['RGB_GALLERY']:
                pid = int(anno[0])
                pid_container.add(pid)
                img_path = op.join(self.dataset_root, anno[1])
                gallery_paths.append(img_path)
                gallery_pids.append(pid)

            queries = annos.copy()
            queries.pop('RGB_GALLERY',None)

            for key in queries.keys():
                for i in range(len(queries[key])):
                    for j in range(len(queries[key][i])):
                        if type(queries[key][i][j]) == int:
                            continue
                        if '.jpg' in queries[key][i][j]:
                            queries[key][i][j] = op.join(self.dataset_root,queries[key][i][j])

            dataset = {
                "gallery_pids": gallery_pids,
                "gallery_paths": gallery_paths,
                "queries": queries
            }
            return dataset, pid_container

