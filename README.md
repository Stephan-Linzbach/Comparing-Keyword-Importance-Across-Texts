# Comparing Keyword Importance Across Texts

## Description

This method identifies and ranks the most important words in a collection of documents, such as articles, speeches, or social media posts, by analyzing their frequency and uniqueness within each document. Using measures like [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), [PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information), and [Log Odds Ratio](https://en.wikipedia.org/wiki/Odds_ratio), it highlights terms that are especially relevant to a specific document while contrasting them with others in the collection. This approach is ideal for uncovering key themes, comparing language use across texts, and tracking shifts in terminology or public discourse over time, making it a valuable tool for summarizing content or analyzing trends.

|   | TF-IDF | Log Odds Ratio | PMI |
|:-----------------|:----------------:|:----------------:|:----------------:|
|   | TF-IDF | Log Odds Ratio | PMI |
|:-----------------|:----------------:|:----------------:|:----------------:|
| Definition | Measures the importance of a term in a document not only by frequent usage but also through the absence of use in other documents. | Quantifies the increase of the relative importance of a term for a document in comparison to all other documents. | Measures the association between a term and a document, indicating a dependency. |
| When to use? | Finding terms that are characteristic of a document and only used by a subset of other documents. | Finding terms that have higher relevance for a certain document. | Finding terms that are characteristic of a document and seldom used by other documents. |
| Interpretability | High Scores: indicate greater importance of the term within the document. | Positive: indicates association with the document. Negative: indicates low importance of term for the document. | High Scores: Indicate strength of association between term and document. Low Scores: indicate disassociation between the term and the document. |

## Use Cases


Studying climate change discourse on Twitter over time. By extracting and comparing keywords, it reveals emerging terms (e.g., *carbon neutrality*), diminishing terms (e.g., *global warming*), and stable terms (e.g., *climate crisis*), offering insights into evolving public conversations and priorities.

Analyzing political speeches to identify shifts in rhetoric. Social scientists can track how key terms (e.g., *freedom*, *equality*, *security*) gain or lose prominence across different administrations or during election campaigns, providing a lens into changing political priorities and strategies.

Examining public sentiment in online forums. By comparing keyword importance across threads, researchers can uncover dominant themes, recurring concerns, or evolving opinions on topics like healthcare, education, or economic policies.

Studying cultural narratives in literature or media. Social scientists can analyze how specific terms (e.g., *identity*, *tradition*, *modernity*) are emphasized in different texts, revealing underlying societal values, conflicts, or trends over time.

## Input Data

The method handles digital behavioral data, including social media posts, comments, search queries, clickstream text (e.g., website titles), forum threads, and open-text survey responses.

The corpus data used in the script is stored in JSON format at [data/default_corpus.json](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/data/default_corpus.json) and looks something like this:

```         
{
    "Document A": "This is the liberal solution: All text is good as well as bad. The good one has to take his own position. We are the liberal ones. Not the center nor the progressive ones.",
    "Document B": "This is the center solution: They are bad, not good, if everyone remains in his own position we are all alone which is bad. We are the center ones. Not the progressive nor the liberal ones.",
    "Document C": "This is the progressive solution: Another group's position is the problem. They don't move from their position. We are the progressive ones. Not the liberal nor the center ones."
}
```

[***Note*** - The corpus should ideally be a larger text dataset to produce more meaningful results.]{.underline}

## Output Data

The method will produce a CSV in the following form:

| Words       |     Document A      |     Document B      |          Document C |
|:-----------------|:-----------------:|:-----------------:|-----------------:|
| progressive | 0.24816330799414105 | 0.24816330799414105 |  1.2392023539955106 |
| ones        |  0.636647135255376  |  0.636647135255376  |  0.6276861812567451 |
| position    | 0.24816330799414105 | 0.24816330799414105 |  1.2392023539955106 |
| solution    |  0.636647135255376  |  0.636647135255376  |  0.6276861812567451 |
| center      | 0.20851385530561406 |  1.208513855305614  | 0.19955290130698336 |
| liberal     |  1.208513855305614  | 0.20851385530561406 | 0.19955290130698336 |

Moreover, in the [output_config/](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/tree/main/output_config) you find a JSON file that saved all the used parameters for the resulting table.

```         
{
    "corpus": "/path/to/your_corpus.json",
    "comparison_corpus": "",
    "language": "english",
    "min_df": null,
    "more_freq_than": 0,
    "less_freq_than": 100,
    "method": "pmi",
    "only_words": true,
    "return_values": true
}
```

## Hardware Requirements

The method runs on a cheap virtual machine provided by cloud computing company (2 x86 CPU core, 4 GB RAM, 40GB HDD).

## Environment Setup

-   Install Python v\>=3.9 (preferably through Anaconda).

-   Download the repository with or directly copy the raw code from [keyword_extraction.py](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/keyword_extraction.py), and requirements.txt

```         
git clone https://git.gesis.org/bda/keyword_extraction.git
```

-   Install all the packages and libraries with specific versions required to run this method

```         
pip install -r requirements.txt
```

## How to Use

You can configure the parameters in the [config.json](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/config.json) file and run the script:

```         
python keyword_extraction.py
```

Alternatively, you can set the parameters directly in the command line when running the script.

To return the list of parameters you can specify, execute:
**`Command Line Options`**

```         
python keyword_extraction.py -help
```

Below is the output of the **`-help`** command, which lists all available options for the script:

```         
options:
    -h, --help            show this help message and exit
    --corpus CORPUS       A path to a json corpus in this format ./data/default_corpus.json.
    --comparison_corpus COMPARISON_CORPUS
                                                A path to a json comparison_corpus in this format ./data/default_corpus.json. You need this for the log_odd ratio.
    --config CONFIG       If you do not have a config.json in the working directory or want to set your setting with the cli tool set this var to False.
    --language LANGUAGE   Language (default: english)
    --min_df MIN_DF       Minimum document frequency (default: 1)
    --more_freq_than MORE_FREQ_THAN
                                                Frequency threshold for more frequent words (default: 0)
    --less_freq_than LESS_FREQ_THAN
                                                Frequency threshold for less frequent words (default: 1.0)
    --method METHOD       Choose a method from the list of implemented methods ['log_odds', 'tfidf', 'pmi', 'tfidf_pmi']
    --stop_words STOP_WORDS
                                                Exclude stop_words from this list ['english'].
    --only_words ONLY_WORDS
                                                Exclude numbers, urls, and everything that is not alphabetic.
    --return_values RETURN_VALUES
                                                Use this parameter if you want the associated values of the respective method to be returned.
```

It also provides explanation on the role of the parameters in altering the method behavior. Next, execute:

```         
python keyword_extraction.py --method pmi --corpus /path/to/your_corpus.json
```

## Example Commands and parameters

Below are example commands demonstrating how to use the method with different configurations and parameters to extract and analyze keyword importance effectively.

**1. Pointwise Mutual Information (PMI)** Calculate PMI for words that appear in more than 80% of the documents:

```         
python keyword_extraction.py --config False --method pmi --corpus /path/to/your_corpus.json --more_freq_than 80
```

Words that occur frequently across documents are prioritized.

**2. TF-IDF (Term Frequency-Inverse Document Frequency)** Compute importance scores based on TF-IDF, excluding words that appear in fewer than two documents:

```         
python keyword_extraction.py --config False --method tfidf --corpus /path/to/your_corpus.json --min_df 2
```

TF-IDF highlights words that are unique to specific documents compared to those shared across all documents. With 'min_df' we specify that the words should appear at least in 2 documents. Thus, we exclude all words that only appear in one document.

**3. PMI with TF-IDF** Combine PMI with TF-IDF scores, excluding the least frequent 20% of words. This is being calculated using the weighted term frequencies from the TF-IDF matrix rather than raw term frequencies. The result is a **PMI matrix** that incorporates the TF-IDF weighting into the PMI calculation.

```         
python keyword_extraction.py --config False --method tfidf_pmi --corpus /path/to/your_corpus.json --less_freq_than 20
```

This approach accounts for both document-specific importance and overall word weighting.

**4. Log Odds Ratio** Compute Log Odds Ratio using a comparison corpus to identify word importance:

```         
python keyword_extraction.py --config False --method log_odds --corpus /path/to/your_corpus.json --comparison_corpus /path/to/your_comparison_corpus.json 
```

A comparison corpus is required to determine word frequencies under "normal" circumstances, reducing noise and highlighting significant terms.

## Contact Details

[Stephan.Linzbach\@gesis.org](mailto:Stephan.Linzbach@gesis.org)
