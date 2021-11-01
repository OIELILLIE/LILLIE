## Installation

Requires Python 3.6.9.

1. `pip install -r requirements.txt`
2. `python3 -m spacy download en_core_web_md`
3. Clone ClausIE to `./learning_based/pyclausie` (https://github.com/AnthonyMRios/pyclausie)
4. Install with:
`cd ./learning_based/pyclausie`
`python3 setup.py install`
5. Clone OpenIE5 to `./learning_based/OpenIE-Standalone` (https://github.com/dair-iitd/OpenIE-standalone)
6. Run OIE5 with:
`cd ./learning_based/OpenIE-standalone`
`java -Xmx16g -jar openie-assembly-5.0-SNAPSHOT.jar --httpPort 9000`
7. Download Stanford CoreNLP Server 3.9.2 to `./rule_based/parser` (https://stanfordnlp.github.io/CoreNLP/history.html)
8. Run the parser:
`java -mx6g -cp "./rule_based/parser/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 10000 -timeout 30000`
9. Run the learning-based extractor:
`python3 ./learning_based/paralleloie.py -i data/pubmedabstracts.json`
10. Run the rule-based extractor-refiner:
`python3 ./rule_based/extract_refine.py -i extracted_triples_learning.csv`
