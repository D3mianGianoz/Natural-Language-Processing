import re
import sys
import xml.etree.ElementTree as eT

from lxml import etree as exml
from tqdm import tqdm

from Wsd.utilities import get_sense_index, lesk, pos_validity


def parse_xml(path):
    """ It parses the SemCor corpus, which has been annotated by hand on WordNet synsets by Rada Mihalcea and her team.

    In order:
    1) Load XML file
    2) Took all the tags "s"
    3) Extract the sentence
    4) Select the words to disambiguate (select only the needed ones) with total number of senses >= 2
    5) Extract Golden annotated sense from WSN

    Args:
        path: the path to the XML file (Brown Corpus)
    Returns:
        [(sentence, [(word, gold)])]
    """

    with open(path, 'r') as fileXML:
        data = fileXML.read()

        # fixing XML's bad formatting
        data = data.replace('\n', '')
        replacer = re.compile("=([\w|:|\-|$|(|)|']*)")
        data = replacer.sub(r'="\1"', data)

        result = []
        try:
            root = exml.XML(data)
            paragraphs = root.findall("./context/p")
            sentences = []
            for p in paragraphs:
                sentences.extend(p.findall("./s"))
            for sentence in sentences:
                words = sentence.findall('wf')
                sent = ""
                tuple_list = []
                for word in words:
                    w = word.text
                    pos = word.attrib['pos']
                    sent = sent + w + ' '
                    if pos_validity(pos=pos, text=w, word=word):
                        sense = word.attrib['wnsn']
                        t = (w, sense)
                        tuple_list.append(t)
                result.append((sent, tuple_list))
        except Exception as e:
            raise NameError('xml: ' + str(e))
    return result


def word_sense_disambiguation(options):
    """ Word Sense Disambiguation: Extracts sentences from the SemCor corpus
    (corpus annotated with WN synset) and disambiguate at least one noun per
    sentence. It also calculates the accuracy based on the senses noted in
    SemCor. Writes the output into a xml file.

    Args:
        options: a dictionary that contains the input and output paths.
    Format:
        { "input": "...", "output": "..." }
    """

    list_xml = parse_xml(options["input"])

    result = []
    count_word = 0
    count_exact = 0

    # showing progress bar
    progress_bar = tqdm(desc="Percentage", total=50, file=sys.stdout)

    i = 0
    while i in range(len(list_xml)) and len(result) < 50:
        dict_list = []
        sentence = list_xml[i][0]
        words = list_xml[i][1]
        for t in words:
            sense = lesk(t[0], sentence)  # running lesk algorithm
            value = str(get_sense_index(t[0], sense))
            golden = t[1]
            count_word += 1
            if golden == value:
                count_exact += 1
            dict_list.append({'word': t[0], 'gold': golden, 'value': value})

        if len(dict_list) > 0:
            result.append((sentence, dict_list))
            progress_bar.update(1)

        i += 1

    accuracy = count_exact / count_word

    with open(options["output"] / 'task1.2_output.xml', 'wb') as out:
        out.write('<results accuracy="{0:.2f}">'.format(accuracy).encode())
        out.write("\n".encode())
        for j in range(len(result)):
            xml_s = eT.Element('sentence_wrapper')
            xml_s.set('sentence_number', str(j + 1))
            xml_sentence = eT.SubElement(xml_s, 'sentence')
            xml_sentence.text = result[j][0]
            for tword in result[j][1]:
                xml_word = eT.SubElement(xml_sentence, 'word')
                xml_word.text = tword['word']
                xml_word.set('golden', tword['gold'])
                xml_word.set('sense', tword['value'])

            tree = eT.ElementTree(xml_s)
            tree.write(out)
            out.write("\n".encode())
        out.write(b'</results>')
