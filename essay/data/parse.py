import xml.etree.ElementTree as ET
import os
import tqdm

BAWE_PATH = "/home/nickatomlin/vivek/plag/aesop/download/CORPUS_ASCII/"


def extract_body_content(xml_content):
    root = ET.fromstring(xml_content)
    body = root.find('text/body')

    content = ''
    for div in body.findall('div1'):
        for p in div.findall('p'):
            for s in p.findall('s'):
                content += s.text.strip() + ' '
            content = content.strip() + '\n'

    return content.strip()


n = 0
for file in tqdm.tqdm(sorted(os.listdir(BAWE_PATH))):
    if file == "tei_bawe.dtd":
        continue
    with open(f"{BAWE_PATH}/{file}") as f:
        xml = f.read().strip()
        content = extract_body_content(xml)

    if len(content.split(" ")) < 50:
        continue

    with open(f"human/{n}.txt", "w") as f:
        f.write(content)

    n += 1
