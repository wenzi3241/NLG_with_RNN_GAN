"""
author: Wen Cui
Nov 21, 2017
Reference: https://github.com/skashyap7/TBBTCorpus/blob/master/preprocessing/util.py
"""

import json
import requests
from lxml import html
import os

allTranscripts = {}
"""
aim to crawl data from link specified in episode_links.json
"""

def getEpisodeText(season, index, episodeInfo):
    ep, title, link = episodeInfo[season][index]
    try:
        page = requests.get(link)
        tree = html.fromstring(page.content)
        p_count = tree.find_class('entrytext')[0]
        result = p_count.text_content()
    except:
        raise Exception("Failed for {ep}, {season}".format(ep=ep, season=season))
    # Remove advertisements
    result, _, _ = result.partition('Advertisements')
    return result, ep


def getLinksList(filename):
    try:
        with open(filename, "r") as fhandle:
            episodeInfo = json.load(fhandle)
            fhandle.close()
        for season in episodeInfo:
            # Season is the key
            print(season)
            for idx in range(0, len(episodeInfo[season])):
                transcript, episode = getEpisodeText(season, idx, episodeInfo)
                ep_id = season + "_" + str(episode) + ".txt"
                path = os.path.join("../raw_corpus/", ep_id)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(transcript)
                    fh.close()
                print("Downloaded the transcripts into raw_corpus directory")
                allTranscripts[ep_id] = transcript
        with open("corpus.json", "w") as corpus_file:
            json.dump(allTranscripts, corpus_file, indent=4)
            corpus_file.close()

    except FileNotFoundError:
        print("Could not file {fname}".format(fname=filename))


# Main Function
def main():
    getLinksList("./episode_links.json")


# Entry point
if __name__ == "__main__":
    main()