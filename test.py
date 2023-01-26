import tarfile
import gzip

bytes_to_read = 1024 * 1024 * 256 # 1 GB
bytes_read = 0

trg_sentences = []
src_sentences = []
    
# Open the tar file
with tarfile.open('eng-spa.tar', 'r') as tar:
    print('Reading target sentences...')
    # Get a file-like object for the 'other.trg.gz' file
    gz_file_trg = tar.extractfile('data/release/v2021-08-07/eng-spa/train.trg.gz')
    # Open the 'other.trg.gz' file using gzip
    with gzip.open(gz_file_trg, 'rb') as gz:
         while True:
            chunk = gz.read(1024*1024).decode('utf-8', errors='ignore')
            bytes_read += len(chunk)
            if bytes_read >= bytes_to_read:
                break
            # Add all the lines to lines
            trg_sentences.extend(chunk.split('\n'))
            
    bytes_read = 0
    print('Reading source sentences...')
    gz_file_src = tar.extractfile('data/release/v2021-08-07/eng-spa/train.src.gz')
    # Open the 'other.trg.gz' file using gzip
    with gzip.open(gz_file_src, 'rb') as gz:
         while True:
            chunk = gz.read(1024*1024).decode('utf-8', errors='ignore')
            bytes_read += len(chunk)
            if bytes_read >= bytes_to_read:
                break
            # Add all the lines to lines
            src_sentences.extend(chunk.split('\n'))

print('Writing target to file...')
# Open a file for writing
with open('target.txt', 'w', encoding='utf-8') as f:
    # Write each sentence in the list to the file, followed by a newline character
    for sentence in trg_sentences:
        f.write(sentence + '\n')
        
print('Writing source to file...')
with open('source.txt', 'w', encoding='utf-8') as f:
    for sentence in src_sentences:
        f.write(sentence + '\n')
    


    

