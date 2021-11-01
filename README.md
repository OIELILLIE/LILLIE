
## LILLIE: Information Extraction and Database Integration Using Linguistics and Learning-Based Algorithms

Based on the work by Smith et al. (2021)

> Querying both structured and unstructured data via a single common query interface such as SQL or natural language has been a long standing research goal. Moreover, as methods for extracting information from unstructured data become ever more powerful, the desire to integrate the output of such extraction processes with "clean", structured data grows.  We are convinced that for successful integration into databases, such extracted information in the form of "triples" needs to be both 1) of high quality and 2) have the necessary generality to link up with varying forms of structured data. It is the combination of *both* these aspects, which heretofore have been usually treated in isolation, where our approach breaks new ground.
> 
> The cornerstone of our work is a novel, generic method for extracting open information triples from unstructured text, using a combination of *linguistics and learning-based extraction* methods, thus uniquely balancing both precision and recall. Our system called LILLIE (LInked Linguistics and Learning-Based Information Extractor)  uses *dependency tree modification rules* to refine triples from a high-recall learning-based engine, and combines them with syntactic triples from a high-precision engine to increase effectiveness. In addition, our system features several augmentations, which modify the generality and the degree of granularity of the output triples. Even though our focus is on addressing both quality and generality simultaneously, our new method substantially outperforms current state-of-the-art systems on the two widely-used CaRB and Re-OIE16 benchmark sets for information extraction.

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

