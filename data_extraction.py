import tarfile
import gzip
import glob
import os
import sys
from tqdm import tqdm

RAW_DATA_PATH = 'raw-data/'
SPLIT_DATA_PATH = 'split-data/'
SENTENCES_TO_READ = 125000


def read_file_by_chuncks(gz_src_file, gz_trg_file, language_pair):
    src_sentences = []
    trg_sentences = []

    print('Reading ' + language_pair + ' source file')
    with gzip.open(gz_src_file, 'rt', encoding='utf-8') as gz1:
        for _ in tqdm(range(SENTENCES_TO_READ)):
            src_chunck = gz1.readline().strip()
            src_sentences.append(src_chunck)
    
    print('Reading ' + language_pair + ' target file')
    with gzip.open(gz_trg_file, 'rt', encoding='utf-8') as gz2:
        for i in tqdm(range(SENTENCES_TO_READ)):
            trg_chunck = gz2.readline().strip()
            trg_sentences.append(trg_chunck)
    
    return src_sentences, trg_sentences
  

def write_file(sentences, file_name):
    print('Writing ' + file_name)
    with open(file_name, 'w', encoding='utf-8') as f:
        # Write each sentence in the list to the file, followed by a newline character
        for sentence in tqdm(sentences):
            f.write(sentence + '\n')

def read_file():
    raw_languages = glob.glob(RAW_DATA_PATH + '*.tar')
    for raw_language in raw_languages:
        with tarfile.open(raw_language, 'r') as tar:
            files = tar.getnames()
            # print(files)
            
            gz_src_file = [file for file in files if file.endswith('train.src.gz')]
            gz_trg_file = [file for file in files if file.endswith('train.trg.gz')]
            # print(gz_src_file[0])
            # print(gz_trg_file[0])
            
            # Create a name for the file
            language_pair = gz_src_file[0].split('/')[-2].split('.')[0]
            src_file_name = 'source_' + language_pair + '.txt'
            trg_file_name = 'target_' + language_pair + '.txt'
            # print(src_file_name)
            # print(trg_file_name)
            
            src_sentences, trg_sentences = read_file_by_chuncks(tar.extractfile(gz_src_file[0]), tar.extractfile(gz_trg_file[0]), language_pair)
            print()
            
            write_file(src_sentences, SPLIT_DATA_PATH + src_file_name)
            write_file(trg_sentences, SPLIT_DATA_PATH + trg_file_name)
            print()    
            
            # TODO: If the tar has not 1000000 sentences, clean the emtpy lines generated in the txt file
          
            

if __name__ == '__main__':
    if not os.path.exists(RAW_DATA_PATH) or len(os.listdir(RAW_DATA_PATH)) == 0:
        sys.exit('ERROR! Please download the raw data from https://github.com/Helsinki-NLP/Tatoeba-Challenge and place it in the raw-data folder')
    
    if not os.path.exists(SPLIT_DATA_PATH):
        os.makedirs(SPLIT_DATA_PATH)
    
    read_file()