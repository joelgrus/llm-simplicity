import multiprocessing as mp

import tqdm

from eda import chats
from chatgpt import do_analysis


if __name__ == "__main__":
    pool = mp.Pool(processes=2)
    for _ in tqdm.tqdm(pool.imap_unordered(do_analysis, chats), total=len(chats)):
        pass