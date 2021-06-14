
# bpemb

Complete pipeline to train byte-pair-encoded embeddings (BPEmb) from scratch. Fully compatible with [bpemb](https://github.com/bheinzerling/bpemb). Recreating the following paper: _BPEmb. Tokenization-free Pre-trained Subword Embeddings in 275 Languages_

## run

- ```pip install sentencepiece```
- ```git clone https://github.com/stanfordnlp/glove && cd glove && make```
- set parameter in: bpemb.py, glove/demo.sh
- ```./bpemb.py```

## about the paper

Benjamin Heinzerling and Michael Strube. 2018. _BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages._ In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

> We apply BPE1 to all Wikipedias2 of sufficient size with various o and pre-train embeddings for the resulting BPE symbol using GloVe (Pennington et al., 2014), resulting in byte-pair embeddings for 275 languages. To allow study- ing the effect the number of BPE merge operations and of the embedding dimensionality, we provide embeddings for 1000, 3000, 5000, 10000, 25000, 50000, 100000 and 200000 merge operations, with dimensions 25, 50, 100, 200, and 300.

## good to know

- Vocabulary size: Depending on your target domain, you will have to experiment with the vocabulary size. A vocabulary of 200k is good for a web corpus, otherwise the model gets too sparse. Non-Internet-size data works well with 10-50k items. See the (mostly German) implementation notes below for more.

- Branch ```dataload``` integrates with a complete MLOps infrastructure: a data loader for various text classification data sets (```dataload```), optionally backed by GridFS (```datalake```), and feeding into an automated training and evaluation pipeline (```autofit```).

## implementation notes

1. subwords lernen
    - https://github.com/google/sentencepiece
    - python wrapper: https://github.com/google/sentencepiece/blob/master/python/README.md
    - training parameters: https://github.com/google/sentencepiece/blob/master/doc/options.md
        - input_sentence_size und andere formen von sampling -> keine gute idee, performance beim training wesentlich besser mit unlimited data size
    - input: path to single txt file, sentence iterator
    - bzgl. große vocabulary size
        - zu wenig daten für 200k, pieces kommen tlw. nur 1x vor
        - var 1: bpe mit kleinerem vocab.
            - en-custom-10k: ['▁🤣', '🤣', '🤣', '🤣', '🤣']
            - en-custom-20k: ['▁🤣', '🤣🤣', '🤣🤣']
            - en-custom-50k: ['▁🤣🤣🤣', '🤣🤣']
        - var 2: mehr daten
            - wikipedia corpora
                - aber: definieren von user_defined_symbols (z.b. emoji) garantiert nicht, dass diese im vokabular enthalten sind -> "If this symbol is included in the input text, this symbol is always extracted as one piece.", vgl. https://github.com/google/sentencepiece/blob/master/doc/special_symbols.md + dazu kommt: pieces aus den twitter daten würden aufgrund der viel geringeren datenmenge nicht im vokabular landen
            - fazit: für bessere emoji-tokenisierung hilft nur mehr twitter scrapen

2. embedding lernen
    - texte mit bpe model encoden
    - train glove
        - https://github.com/stanfordnlp/GloVe
        - modify demo.sh: https://stackoverflow.com/a/54567091
    - embedding dimensions: 300
    - input: path to single txt file
    - output: write_header=1
        - grund: bpemb lädt glove embedding via gensim.models.KeyedVectors.load_word2vec_format()

3. in bpemb laden
    - BPEmb(model_file='en-custom-10k.model', emb_file='en-custom-10k-vectors.txt', dim=300)
    - model_file: ``Path'', optional (default = None)
        Path to a custom SentencePiece model file.
    - emb_file: ``Path'', optional (default = None)
        Path to a custom embedding file. Supported formats are Word2Vec
        plain text and GenSim binary.

- bzgl. alignment von bpe model + glove embedding
    - indexerror: bpe model erzeugt id, die nicht im embedding enthalten ist
        - die vocabularies von bpe model + embedding müssen identisch sein
    - grund 1
        - warum sind sie nicht identisch? evtl. weil bpe beim training pieces erzeugt, die nachher nicht zum einsatz kommen, z.b. weil ein längeres/passenderes piece in frage kommt
    - lösung 1
        - alle pieces im bpe model triggern
        - encode() parameter: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.h
            ```
            for _ in range(10):
            ...     sp.encode('This is a test', out_type=str, enable_sampling=True, alpha=0.1)
            ... 
            ['▁', 'This', '▁', 'is', '▁a', '▁', 't', 'e', 'st']
            ['▁T', 'h', 'i', 's', '▁is', '▁a', '▁', 'te', 's', 't']
            ['▁T', 'h', 'is', '▁', 'is', '▁', 'a', '▁', 't', 'est']
            ['▁', 'This', '▁is', '▁', 'a', '▁', 't', 'e', 'st']
            ['▁', 'This', '▁', 'is', '▁', 'a', '▁', 't', 'e', 's', 't']
            ['▁This', '▁is', '▁a', '▁', 'te', 's', 't']
            ['▁This', '▁is', '▁', 'a', '▁', 't', 'e', 'st']
            ['▁', 'T', 'h', 'is', '▁', 'is', '▁', 'a', '▁', 'te', 'st']
            ['▁', 'This', '▁', 'i', 's', '▁a', '▁', 't', 'e', 'st']
            ['▁This', '▁', 'is', '▁a', '▁', 't', 'est']
            ```
    - grund 2
        - spezielle unicode character entfallen beim glove vokabular bzw. spätestens wenn die embedding file von bpemb() geladen wird, kommt es wieder zum mismatch der vokabulare -> indexerror
    - lösung 2
        - check unicode characters
            - https://www.babelstone.co.uk/Unicode/whatisit.html
            - https://docs.python.org/3/library/unicodedata.html
        - unicode normalization
            ```
            # unicode normalization
            # remove control characters, bidirectional classes
            # cf. https://www.unicode.org/reports/tr44/#BC_Values_Table
            bidi_classes = ['', 'L', 'EN', 'ES', 'ET', 'CS', 'WS', 'ON']
            string = ''.join(c for c in string if not unicodedata.category(c).startswith('C') and unicodedata.bidirectional(c) in bidi_classes)
            # remove braille: 0x2800-0x28FF, arabic: 0x0600-0x06FF, diacritics: 0x0300-0x036F
            # cf. https://www.ssec.wisc.edu/~tomw/java/unicode.html
            string = re.sub('[\u2800-\u28FF]|[\u0600-\u06FF]|[\u0300-\u036F]', '', string)
            ```
    - fazit
        - balance von normalisierung und varianz
            - einerseits brauchen wir emoji und viele special characters
            - andererseits gibt es unsichtbare, zero-width, bidirectional, control etc. unicode characters, die man loswerden muss
        - bpe tokenization ist probabilistisch
            - bpe model muss gesampled werden, um alle pieces zu bekommen
        - die vokabulare von bpe model + embedding müssen die selbe reihenfolge haben
            - embedding file reorder + sanity check -> siehe bpemb.py
                - pieces, die nicht im embedding gefunden werden -> als erstes vektor von partial match, falls nicht -> vektor von ```<unk>```
            - nur 2 pieces nicht im embedding gefunden: ```<s>```, ```</s>```
        - der einzige wirklich test
            - vergleich output von bpemb.embed() mit values in embedding file
                - impliziert den roundtrip von string -> id -> index -> vector

