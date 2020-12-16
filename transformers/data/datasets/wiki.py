from __future__ import absolute_import, division, print_function

import os
import os.path as osp
from typing import List

from transformers.utils.file_io import PathManager


def load_wiki(root: str) -> List[List[str]]:
    """
    Loads the documents from wiki directory.
    """
    documents = []
    # file path looks like: root/wiki_0, root/wiki_1

    documents = []
    # file path looks like: root/wiki_0, root/wiki_1
    for file_name in os.listdir(root):
        file_path = osp.join(root, file_name)
        with open(PathManager.get_local_path(file_path), "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        article_lines = []
        article_open = False
        for line in original_lines:
            line = line.strip()
            if "<doc id=" in line:
                article_open = True
            elif "</doc>" in line:
                article_open = False
                # ignore the first line since it is the title
                cur_doc = [
                    line for line in article_lines[1:] if len(line) > 0 and not line.isspace()
                ]
                documents.append(cur_doc)
                article_lines = []
            else:
                if article_open:
                    article_lines.append(line)

    return documents
