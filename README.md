# Comparing Keyword Importance Across Texts

## Description

This method identifies the most important words in a collection of documents, such as articles, speeches, or social media posts, by ranking the words for each document according to their frequency within the document and their uniqueness to the document. The specific available measures use [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), [PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information), and [Log Odds Ratio](https://en.wikipedia.org/wiki/Odds_ratio). This approach is ideal for uncovering key themes, comparing language use across texts, and tracking shifts in terminology or public discourse over time, making it a valuable tool for summarizing content or analyzing trends.

|   | TF-IDF | Log Odds Ratio | PMI |
|:-----------------|:----------------:|:----------------:|:----------------:|
| Definition | Measures the importance of a term in a document not only by frequent usage but also through its absence in other documents. | Quantifies the increase in the relative importance of a term for a document in comparison to all other documents. | Measures the association between a term and a document, indicating a dependency. |
| When to use? | Finding terms that are characteristic of a document and only used by a subset of other documents. | Finding terms that have higher relevance for a certain document. | Finding terms that are characteristic of a document and seldom used by other documents. |
| Interpretability | High scores indicate greater importance of the term within the document. | Positive values indicate association with the document. Negative values indicate low importance of the term for the document. | High scores indicate strength of association between the term and document. Low scores indicate disassociation between the term and the document. |

## Use Cases

* __Studying climate change discourse on Twitter over time:__ By extracting keywords over time (a "document" contains all tweets of a year), this method can reveal emerging terms (e.g., *carbon neutrality*), diminishing terms (e.g., *global warming*), and stable terms (e.g., *climate crisis*), offering insights into evolving public conversations and priorities.
* __Analyzing political speeches to identify shifts in rhetoric:__ By extracting keywords for specific key moments in time (a "document" contains all texts for different key moments, e.g., before different elections), this method can reveal prominent themes across different administrations or during election campaigns, providing a lens into changing political priorities and strategies.
* __Examining topical discourse in online forums:__ By extracting keywords across forums or threads (a "document" contains all texts of a forum or thread), this method can reveal specific themes in different discussions, e.g. contrasting discourse on topics like healthcare, education, or economic policies (taking a forum for each of those).
* __Studying cultural narratives in literature or media:__ By examining the output scores of specific terms (e.g., *identity*, *tradition*, *modernity*), this method can reveal how these are emphasized in different texts, indicating for example different underlying societal values, conflicts, or trends.

## Input Data

The method handles all texts, including social media posts, comments, search queries, clickstream text (e.g., website titles), forum threads, and open-text survey responses.

The method takes as input data in JSON format (one object, mapping a document name to the document content). See [data/default_corpus.json](/data/default_corpus.json) for an example. The first three documents:

```JSON
{
    "Document A": "This is the liberal solution: All text is good as well as bad. The good one has to take his own position. We are the liberal ones. Not the center nor the progressive ones.",
    "Document B": "This is the center solution: They are bad, not good, if everyone remains in his own position we are all alone which is bad. We are the center ones. Not the progressive nor the liberal ones.",
    "Document C": "This is the progressive solution: Another group's position is the problem. They don't move from their position. We are the progressive ones. Not the liberal nor the center ones."
}
```

Note: The method is intended for datasets containing at least a thousand words.

## Output Data

The method will produce a CSV in the following form, showing the score for each word (row) for each document (column):

| Words       |     Document A      |     Document B      |     Document C      |
|:------------|:-------------------:|:-------------------:|--------------------:|
| progressive | 0.24816330799414105 | 0.24816330799414105 | 1.2392023539955106  |
| ones        | 0.636647135255376   | 0.636647135255376   | 0.6276861812567451  |
| position    | 0.24816330799414105 | 0.24816330799414105 | 1.2392023539955106  |
| solution    | 0.636647135255376   | 0.636647135255376   | 0.6276861812567451  |
| center      | 0.20851385530561406 | 1.208513855305614   | 0.19955290130698336 |
| liberal     | 1.208513855305614   | 0.20851385530561406 | 0.19955290130698336 |

For reproducibility, the used configuration is stored in the [output_config/](./output_config) directory.

## Hardware Requirements

The method runs on a small virtual machine (2 x86 CPU core, 4 GB RAM, 40GB HDD).

## Environment Setup

- If not done already install Python version>=3.9, e.g.
  
  ```bash
  conda create -n env python=3.11
  ```

- Install all the packages and libraries with specific versions required to run this method:
  
  ```bash
  pip install -r requirements.txt
  ```

## How to Use

Run with the datasets and parameters as specified in the [config.json](config.json):

```bash
python keyword_extraction.py
```

You can also override the parameters using command line options, e.g.:

```bash
python keyword_extraction.py --method pmi --corpus /path/to/your_corpus.json
```

See the help for a list of available options:

```bash
python keyword_extraction.py --help
```

## Technical Details

Below are example commands demonstrating how to use the method with different configurations and parameters to extract and analyze keyword importance effectively.

__1. Pointwise Mutual Information (PMI)__ - Calculate PMI for words that appear in more than 80% of the documents:

```bash
python keyword_extraction.py --config False --method pmi --corpus /path/to/your_corpus.json --more_freq_than 80
```

Words that occur frequently across documents are prioritized.

__2. TF-IDF (Term Frequency-Inverse Document Frequency)__ - Compute importance scores based on TF-IDF, excluding words that appear in fewer than two documents:

```bash
python keyword_extraction.py --config False --method tfidf --corpus /path/to/your_corpus.json --min_df 2
```

TF-IDF highlights words that are unique to specific documents compared to those shared across all documents. With `min_df`, we specify that words should appear in at least 2 documents. Thus, we exclude all words that only appear in one document.

__3. PMI with TF-IDF__ - Combine PMI with TF-IDF scores, excluding the least frequent 20% of words. This is calculated using the weighted term frequencies from the TF-IDF matrix rather than raw term frequencies. The result is a __PMI matrix__ that incorporates the TF-IDF weighting into the PMI calculation.

```bash
python keyword_extraction.py --config False --method tfidf_pmi --corpus /path/to/your_corpus.json --less_freq_than 20
```

This approach accounts for both document-specific importance and overall word weighting.

__4. Log Odds Ratio__ - Compute Log Odds Ratio using a comparison corpus to identify word importance:

```bash
python keyword_extraction.py --config False --method log_odds --corpus /path/to/your_corpus.json --comparison_corpus /path/to/your_comparison_corpus.json 
```

A comparison corpus is required to determine word frequencies under "normal" circumstances, reducing noise and highlighting significant terms.

## Contact Details

[Stephan.Linzbach\@gesis.org](mailto:Stephan.Linzbach@gesis.org)
