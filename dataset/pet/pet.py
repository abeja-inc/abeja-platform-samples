from collections import namedtuple
from pathlib import Path
from utils.data_utils import get_file
import xmltodict

PetData = namedtuple('PetData', ('image_path', 'class_id', 'species', 'breed_id', 'size', 'bbox', 'annotated'))


def load_data(cache_dir: str = './cache') -> (list, list):
    """The Oxford-IIIT Pet Dataset (http://www.robots.ox.ac.uk/~vgg/data/pets/)


    ID: 1:37 Class ids
    SPECIES: 1:Cat 2:Dog
    BREED ID: 1-25:Cat 1:12:Dog

    Returns:
        dataset_trainval (list): list of PetData(namedtuple) with below format
            PetData(
                    image_path,
                    class_id,
                    species,
                    breed_id,
                    size,
                    bbox
                )
        dataset_test (list): same as `dataset_trainval`
    """

    IMAGE_URL = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    ANNOTATION_URL = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'

    cache_dir = Path(cache_dir)
    image_root_dir = cache_dir / 'datasets' / 'images'
    xml_root_dir = cache_dir / 'datasets' / 'annotations' / 'xmls'

    _download(ANNOTATION_URL, IMAGE_URL, cache_dir)

    trainval_txt_filepath = cache_dir / 'datasets' / 'annotations' / 'trainval.txt'
    dataset_trainval = _parse_txt(trainval_txt_filepath, image_root_dir, xml_root_dir)

    test_txt_filepath = cache_dir / 'datasets' / 'annotations' / 'test.txt'
    dataset_test = _parse_txt(test_txt_filepath, image_root_dir, xml_root_dir)

    return dataset_trainval, dataset_test


def _download(annotation_url, image_url, cache_dir):
    if not cache_dir.exists():
        cache_dir.mkdir()
    get_file('annotations', annotation_url, cache_dir=cache_dir, untar=True)
    get_file('images', image_url, cache_dir=cache_dir, untar=True)


def _parse_txt(filepath, image_root_dir, xml_root_dir):
    parsed_lines = []
    with open(filepath, 'r') as lines:

        def _f(line):
            filename_base, class_id, species, breed_id = line.rstrip().split(' ')
            return filename_base, class_id, species, breed_id

        for line in lines:
            if line[0] == '#':
                continue
            parsed = _f(line)
            parsed_lines.append(parsed)

    # parse XML files and store information
    parsed_data = []
    for filename_base, class_id, species, breed_id in parsed_lines:
        xml_filepath = xml_root_dir / f'{filename_base}.xml'
        annotated = True
        try:
            size, bbox = _parse_xml(xml_filepath)
        except FileNotFoundError as e:
            size = dict(width=-1, height=-1)
            bbox = dict(xmin=-1, ymin=-1, xmax=-1, ymax=-1)
            annotated = False
        except TypeError as e:
            print('parse error at:', xml_filepath)
            print(e)
            continue

        size_int = dict(width=int(size['width']), height=int(size['height']))
        bbox_int = dict(xmin=int(bbox['xmin']),
                        xmax=int(bbox['xmax']),
                        ymin=int(bbox['ymin']),
                        ymax=int(bbox['ymax']))

        datum = PetData(
            image_path=f'{image_root_dir}/{filename_base}.jpg',
            class_id=int(class_id),
            species=int(species),
            breed_id=int(breed_id),
            size=size_int,
            bbox=bbox_int,
            annotated=int(annotated)
        )
        parsed_data.append(datum)

    return parsed_data


def _parse_xml(xml_filepath):
    with open(xml_filepath, 'r') as f:
        txt = f.read()
    xml_dict = xmltodict.parse(txt)
    size_dict = xml_dict['annotation']['size']
    
    bbdict = xml_dict['annotation']['object']['bndbox']
    return size_dict, bbdict


category_to_int = {
    'abyssinian': 1,
    'american_bulldog': 2,
    'american_pit_bull_terrier': 3,
    'basset_hound': 4,
    'beagle': 5,
    'bengal': 6,
    'birman': 7,
    'bombay': 8,
    'boxer': 9,
    'british_shorthair': 10,
    'chihuahua': 11,
    'egyptian_mau': 12,
    'english_cocker_spaniel': 13,
    'english_setter': 14,
    'german_shorthaired': 15,
    'great_pyrenees': 16,
    'havanese': 17,
    'japanese_chin': 18,
    'keeshond': 19,
    'leonberger': 20,
    'maine_coon': 21,
    'miniature_pinscher': 22,
    'newfoundland': 23,
    'persian': 24,
    'pomeranian': 25,
    'pug': 26,
    'ragdoll': 27,
    'russian_blue': 28,
    'saint_bernard': 29,
    'samoyed': 30,
    'scottish_terrier': 31,
    'shiba_inu': 32,
    'siamese': 33,
    'sphynx': 34,
    'staffordshire_bull_terrier': 35,
    'wheaten_terrier': 36,
    'yorkshire_terrier': 37
}

int_to_category = {
    1: 'abyssinian',
    2: 'american_bulldog',
    3: 'american_pit_bull_terrier',
    4: 'basset_hound',
    5: 'beagle',
    6: 'bengal',
    7: 'birman',
    8: 'bombay',
    9: 'boxer',
    10: 'british_shorthair',
    11: 'chihuahua',
    12: 'egyptian_mau',
    13: 'english_cocker_spaniel',
    14: 'english_setter',
    15: 'german_shorthaired',
    16: 'great_pyrenees',
    17: 'havanese',
    18: 'japanese_chin',
    19: 'keeshond',
    20: 'leonberger',
    21: 'maine_coon',
    22: 'miniature_pinscher',
    23: 'newfoundland',
    24: 'persian',
    25: 'pomeranian',
    26: 'pug',
    27: 'ragdoll',
    28: 'russian_blue',
    29: 'saint_bernard',
    30: 'samoyed',
    31: 'scottish_terrier',
    32: 'shiba_inu',
    33: 'siamese',
    34: 'sphynx',
    35: 'staffordshire_bull_terrier',
    36: 'wheaten_terrier',
    37: 'yorkshire_terrier'
}

dog_breeds = {
    1: 'american_bulldog',
    2: 'american_pit_bull_terrier',
    3: 'basset_hound',
    4: 'beagle',
    5: 'boxer',
    6: 'chihuahua',
    7: 'english_cocker_spaniel',
    8: 'english_setter',
    9: 'german_shorthaired',
    10: 'great_pyrenees',
    11: 'havanese',
    12: 'japanese_chin',
    13: 'keeshond',
    14: 'leonberger',
    15: 'miniature_pinscher',
    16: 'newfoundland',
    17: 'pomeranian',
    18: 'pug',
    19: 'saint_bernard',
    20: 'samoyed',
    21: 'scottish_terrier',
    22: 'shiba_inu',
    23: 'staffordshire_bull_terrier',
    24: 'wheaten_terrier',
    25: 'yorkshire_terrier'
}

cat_breeds = {
    1: 'abyssinian',
    2: 'bengal',
    3: 'birman',
    4: 'bombay',
    5: 'british_shorthair',
    6: 'egyptian_mau',
    7: 'maine_coon',
    8: 'persian',
    9: 'ragdoll',
    10: 'russian_blue',
    11: 'siamese',
    12: 'sphynx'
}

if __name__ == "__main__":
    dataset_trainval, dataset_test = load_data()
    trainval_has_size = list(filter(lambda x: x.annotated, dataset_trainval))
    print(f'trainval: {len(dataset_trainval)} with annotation {len(trainval_has_size)}')
    test_has_size = list(filter(lambda x: x.annotated, dataset_test))
    print(f'test: {len(dataset_test)} with annotation {len(test_has_size)}')
