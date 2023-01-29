import tarfile
import gzip
import glob

BYTES_TO_READ = 1024 * 1024 * 50 # (*1024 -> 1 GB)
RAW_DATA_PATH = 'raw-data/'
SPLIT_DATA_PATH = 'split-data/'

def read_file_by_chuncks(gz_file):
    print('Reading file...')
    bytes_read = 0
    sentences = []
    
    with gzip.open(gz_file, 'rb') as gz:
        while True:
            chunk = gz.read(1024*1024).decode('utf-8', errors='ignore')
            bytes_read += len(chunk)
            if bytes_read >= BYTES_TO_READ or len(chunk) == 0: # len(chunk) == 0 is for the case when the file is smaller than BYTES_TO_READ
                break
            # Add all the lines to lines
            sentences.extend(chunk.split('\n'))
    return sentences   

def write_file(sentences, file_name):
    print('Writing to file...')
    with open(file_name, 'w', encoding='utf-8') as f:
        # Write each sentence in the list to the file, followed by a newline character
        for sentence in sentences:
            f.write(sentence + '\n')

def read_file():
    raw_languages = glob.glob(RAW_DATA_PATH + '*.tar')
    for raw_language in raw_languages:
        with tarfile.open(raw_language, 'r') as tar:
            files = tar.getnames()
            # print(files)
            
            gz_src_file = [file for file in files if file.endswith('train.src.gz')]
            gz_trg_file = [file for file in files if file.endswith('train.trg.gz')]
            # print(gz_src_file)
            # print(gz_trg_file)
            
            src_sentences = read_file_by_chuncks(tar.extractfile(gz_src_file[0]))
            trg_sentences = read_file_by_chuncks(tar.extractfile(gz_trg_file[0]))
            
            # Create a name for the file
            language_pair = gz_src_file[0].split('/')[-2].split('.')[0]
            src_file_name = 'source_' + language_pair + '.txt'
            trg_file_name = 'target_' + language_pair + '.txt'
            # print(src_file_name)
            # print(trg_file_name)
            
            write_file(src_sentences, SPLIT_DATA_PATH + src_file_name)
            write_file(trg_sentences, SPLIT_DATA_PATH + trg_file_name)                    
            
            
    
            
if __name__ == '__main__':
    read_file()