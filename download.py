#!/usr/bin/env python

""" Downloads FaceForensics++ and Deep Fake Detection public data release """

import argparse
import os
import urllib
import urllib.request
import tempfile
import time
import sys
import json

from tqdm import tqdm
from os.path import join


FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'

DEEPFAKES_MODEL_NAMES = [
    'decoder_A.h5',
    'decoder_B.h5',
    'encoder.h5'
]


DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}


ALL_DATASETS = [
    'original',
    'DeepFakeDetection_original',
    'Deepfakes',
    'DeepFakeDetection',
    'Face2Face',
    'FaceShifter',
    'FaceSwap',
    'NeuralTextures'
]


COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():

    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.'
    )

    parser.add_argument(
        'output_path',
        type=str,
        help='Output directory.'
    )

    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='all',
        choices=list(DATASETS.keys()) + ['all']
    )

    parser.add_argument(
        '-c',
        '--compression',
        type=str,
        default='raw',
        choices=COMPRESSION
    )

    parser.add_argument(
        '-t',
        '--type',
        type=str,
        default='videos',
        choices=TYPE
    )

    parser.add_argument(
        '-n',
        '--num_videos',
        type=int,
        default=None
    )

    parser.add_argument(
        '--server',
        type=str,
        default='EU',
        choices=SERVERS
    )


    args = parser.parse_args()


    if args.server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'

    elif args.server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'

    elif args.server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'

    else:
        raise Exception("Wrong server")


    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'

    args.deepfakes_model_url = (
        server_url +
        'v3/manipulated_sequences/Deepfakes/models/'
    )


    return args



def download_files(
        filenames,
        base_url,
        output_path,
        report_progress=True
):

    os.makedirs(output_path, exist_ok=True)

    if report_progress:
        filenames = tqdm(filenames)


    for filename in filenames:
        download_file(
            base_url + filename,
            join(output_path, filename)
        )



def download_file(
        url,
        out_file,
        report_progress=False
):

    out_dir = os.path.dirname(out_file)

    if not os.path.isfile(out_file):

        fh, out_file_tmp = tempfile.mkstemp(
            dir=out_dir
        )

        f = os.fdopen(fh, 'w')
        f.close()


        if report_progress:
            urllib.request.urlretrieve(
                url,
                out_file_tmp
            )

        else:
            urllib.request.urlretrieve(
                url,
                out_file_tmp
            )


        os.rename(
            out_file_tmp,
            out_file
        )

    else:
        print(
            "Skipping existing file:",
            out_file
        )



def main(args):

    print(
        "By pressing any key you confirm agreement to FaceForensics terms."
    )

    print(args.tos_url)

    input("Press Enter to continue...")


    if args.dataset == 'all':
        datasets = ALL_DATASETS

    else:
        datasets = [args.dataset]


    for dataset in datasets:

        dataset_path = DATASETS[dataset]


        print(
            "\nDownloading:",
            dataset_path
        )


        if (
            'DeepFakeDetection' in dataset_path
            or 'actors' in dataset_path
        ):

            filepaths = json.loads(
                urllib.request.urlopen(
                    args.base_url + DEEPFEAKES_DETECTION_URL
                )
                .read()
                .decode("utf-8")
            )


            if 'actors' in dataset_path:
                filelist = filepaths['actors']

            else:
                filelist = filepaths['DeepFakesDetection']


        elif 'original' in dataset_path:

            file_pairs = json.loads(
                urllib.request.urlopen(
                    args.base_url + FILELIST_URL
                )
                .read()
                .decode("utf-8")
            )


            filelist = []

            for pair in file_pairs:
                filelist += pair


        else:

            file_pairs = json.loads(
                urllib.request.urlopen(
                    args.base_url + FILELIST_URL
                )
                .read()
                .decode("utf-8")
            )


            filelist = []

            for pair in file_pairs:

                filelist.append(
                    '_'.join(pair)
                )

                if args.type != 'models':

                    filelist.append(
                        '_'.join(pair[::-1])
                    )



        if args.num_videos:

            print(
                f"Downloading first {args.num_videos} videos"
            )

            filelist = filelist[:args.num_videos]



        dataset_url = (
            args.base_url +
            f"{dataset_path}/{args.compression}/{args.type}/"
        )


        output_dir = join(
            args.output_path,
            dataset_path,
            args.compression,
            args.type
        )


        if args.type == 'videos':

            filelist = [
                file + ".mp4"
                for file in filelist
            ]

            download_files(
                filelist,
                dataset_url,
                output_dir
            )



if __name__ == "__main__":

    args = parse_args()

    main(args)