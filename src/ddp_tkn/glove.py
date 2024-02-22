from typing import Tuple
import zipfile
import torch
import urllib.request
import os
import re
from ddp_tkn.abstract_tokenizer import t_token
from .abstract_tokenizer import Tokenizer
from tqdm import tqdm


def resumable_download(url, filename):
    existing_file_size = 0
    
    if os.path.exists(filename):
        existing_file_size = os.path.getsize(filename)
        

    req = urllib.request.Request(url)
    req.get_method = lambda: 'HEAD'
    with urllib.request.urlopen(req) as response:
        file_size = int(response.info().get('Content-Length'))
    
    if existing_file_size == file_size:
        print(f"'{filename}' is already downloaded and complete.")
        return
    
    # Adjust the start range based on the existing file size
    headers = {}
    if existing_file_size:
        headers['Range'] = f'bytes={existing_file_size}-'
        
    with tqdm(total=file_size, initial=existing_file_size, unit='B', unit_scale=True, desc=filename) as pbar:
        # Define a custom reporthook function to update the progress bar
        def reporthook(block_num, block_size, total_size):
            if pbar.total != total_size:
                pbar.reset(total=total_size)
            downloaded = block_num * block_size + existing_file_size
            pbar.update(downloaded - pbar.n)  # update progress
            
        # Open file in append-binary mode and start downloading from where it left
        with open(filename, 'ab') as file:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                file.write(response.read())
                # Update the progress bar manually for the last chunk
                pbar.update(file_size - existing_file_size)

def resumable_download(url, filename): # todo(anguelos) make the above function work instaed of this
    cmd = f"wget -c {url} -O {filename}"
    os.system(cmd)


class TokenizerGlove(Tokenizer):
    def __init__(self, zip_storage="./glove.6B.zip", zip_url="https://nlp.stanford.edu/data/glove.6B.zip", txt_storage="", str_storage="", fd_storage=None, dims=50) -> None:
        assert dims in [50, 100, 200, 300], "Glove only supports 50, 100, 200, 300 dimensions"
        # no super the parent has no constructor
        self.zip_storage = zip_storage
        self.zip_url = zip_url
        if fd_storage is not None:
            lines = fd_storage.read().decode('utf-8').strip().split('\n')
        elif str_storage != '':
            lines = str_storage.strip().split('\n')
        elif txt_storage != '' and os.path.exists(txt_storage):
            with open(txt_storage, 'r') as file:
                lines = file.read().strip().split('\n')
        elif os.path.exists(zip_storage):
            try:
                zip_ref = zipfile.ZipFile(self.zip_storage, 'r')
                with zip_ref.open(f'glove.6B.{dims}d.txt') as file_inside_zip:
                    lines = [l.split() for l in file_inside_zip.read().decode('utf-8').strip().split('\n')]
            except (FileNotFoundError, zipfile.BadZipFile) as e:
                resumable_download(self.zip_url, self.zip_storage)
                with zipfile.ZipFile(self.zip_storage, 'r') as zip_ref:
                    with zip_ref.open(f'glove.6B.{dims}d.txt') as file_inside_zip:
                        lines = [l.split() for l in file_inside_zip.read().decode('utf-8').strip().split('\n')]
        else:
            raise FileNotFoundError(f"Could not load embeddings from {zip_storage} or {txt_storage} or {str_storage} or {fd_storage} or download from {zip_url}")
            
        self.vocabulary = ['<unk>'] + [l[0] for l in lines]
        embeddings = torch.tensor([list(map(float, l[1:])) for l in lines])
        unknown_embedding = embeddings.mean(dim=0, keepdim=True)
        self.embeddings = torch.cat([unknown_embedding, embeddings])
        self.token_idx = {w: n for n, w in enumerate(self.vocabulary)}
        self.inv_token_idx = {n: w for n, w in enumerate(self.vocabulary)}
    
    @classmethod
    def ID(cls):
        return "Glove"
    
    def create_dictionary_tokens(self, text: str) -> Tuple[Tuple[int, int, int], ...]:
        reg = re.compile(r"([^\sA-Za-z]+|\.|[A-Za-z]+)")
        id_from_to = [(self.str2num(m.group()), m.start(), m.end()) for m in reg.finditer(text)]
        return tuple(id_from_to)
    
    def num2str(self, num: int) -> str:
        return self.inv_token_idx.get(num)
    
    def str2num(self, string: str) -> int:
        return self.token_idx.get(string.lower(), 0)
    