# coding: utf-8

######## Utils repository for loading and preprocessing insurance's data for question generation ########

# This python file contains useful functions that will be used to load the data useful to generate questions
# on client's data.

# Author: Joachim Dublineau
# Zelros A.I.

# IMPORTS
import re
import json
import pandas as pd
from spacy.lang.fr.stop_words import STOP_WORDS
from collections import OrderedDict
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex



# UTILS FOR LOADING
class DataPrepModelAssess:

    def __init__(self, **kwargs):
        self.scenario_file = kwargs.get('scenario_file', None)

        # Importing dictionary from json file
        with open(self.scenario_file, 'r', encoding='utf8') as json_data:
            self.scenario_dict = json.load(json_data)

        self.df_scenario = self.get_dataframe_from_scenario_dict(self.scenario_dict)
        self.df_cleaned_scenario = self.get_cleaned_dataframe()

    def set_df_scenario(self, df_data):
        self.df_scenario = df_data
        self.df_cleaned_scenario = self.get_cleaned_dataframe()

    def parse_answers_content(self, answers_content: list) -> list:
        """
        This function joins the different extracts of an answer under the hypothesis that content is under
        the value "texts".
        INPUTS:
        - answers_content: list of dict, the answers content in json.
        OUTPUTS:
        - res_list: list of answers.
        """
        res_list = list()
        if isinstance(answers_content, list):
            for elt in answers_content:
                if isinstance(elt, dict):
                    for elt_key in elt.keys():  # 'type' / 'cases'
                        # If 'texts' key is found, we parse the content
                        if elt_key == 'texts':
                            # 'texts' value is a list of string_list or string
                            for texts in elt.get('texts', None):
                                if isinstance(texts, list):
                                    res_list.append(' '.join(texts))
                                elif isinstance(texts, str):
                                    res_list.append(texts)
                        # Else we look inside the key value, to extract a list
                        else:
                            if (elt_key != 'answers') or ((elt_key == 'answers') and (elt.get('case') == 'NA')):
                                res_list.extend(self.parse_answers_content(elt.get(elt_key, None)))

        # Removing void elements
        res_list = [x for x in res_list if x]
        return res_list

    def is_valid_elt(d):
        name = d.get('name', '')
        if d.get('details', ''):
            details = d.get('details', '')
            if details.get('tags', ''):
                tags = details.get('tags', '')
                tag = tags[0]
            else:
                tag = ''
        else:
            tag = ''
        valid = ('malltalk' not in name) and ('push_' not in name) and ('smalltalk' not in tag)
        return valid

    def get_dataframe_from_scenario_dict(self, _scenario_dict: dict = None,
                                         only_valid_intents: bool = False) -> pd.DataFrame:
        '''
        This function extracts from the dictionary the relevant data and put them
        in a pandas dataframe.
        '''

        # If we don't give a dict, it means that it is the formated_scenario
        if not _scenario_dict:
            if self.formated_scenario:
                _scenario_dict = self.formated_scenario

        intents_list = _scenario_dict.get('intents')
        rows_list = []

        for intent in intents_list:
            # Excluding non valid intent (smalltalk / push) if flag is on
            if not only_valid_intents or self.is_valid_elt(intent):
                answers_content = intent.get('details', {}).get('answers', [])
                answers_list = self.parse_answers_content(answers_content)
                row_dict = {'id': intent.get('id'),
                            'name': intent.get('name'),
                            'title': intent.get('title'),
                            'questions': intent['details']['sentences'],  # str
                            'context': str(' '.join(answers_list))}
                rows_list.append(row_dict)

        df_scenario = pd.DataFrame(rows_list, columns=["id", 'name', 'title', 'questions', 'context'])

        return df_scenario.loc[df_scenario['name'] != 'home', :].reset_index()  # WHY DID WE REMOVED 'home'?

    def get_cleaned_dataframe(self):
        '''
        Clean the previous dataframe using clean_strings for sentences and rm_BP_CE(clean_string)
        for answer
        '''
        df_cleaned_scenario = self.df_scenario.copy()
        df_cleaned_scenario['questions'] = df_cleaned_scenario['questions'].apply(lambda x: clean_strings(x))
        df_cleaned_scenario['context'] = df_cleaned_scenario['context'].apply(lambda x: rm_BP_CE(clean_string(x)))
        return df_cleaned_scenario

    def delete_intents_with_not_enough_utterances(self, nb_utterances):
        '''
        This function makes sure that the dataframe only contains intents with
        more that nb_utterances utterances.
        '''
        for row in self.df_scenario.iterrows():
            sentences = row[1]["questions"]
            if len(sentences) < nb_utterances:
                index = self.df_scenario[self.df_scenario['index'] == row[1]["index"]].index.values.astype(int)[0]
                self.df_scenario = self.df_scenario.drop(index)
        self.df_scenario = self.df_scenario.reset_index()
        self.df_cleaned_scenario = self.get_cleaned_dataframe()
        return None

def clean_string(string=None):
    if isinstance(string, str):
        # capital letter to low ones
        cleaned_string = string.lower()
        # Replacing '_' by ' '
        cleaned_string = cleaned_string.replace('_', ' ')
        # Removing double spaces
        cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
        cleaned_string = cleaned_string.strip()
        str_res = cleaned_string
    else:
        str_res = None
    return str_res


def clean_strings(strings):
    if isinstance(strings, list):
        res = [clean_string(s) for s in strings]
    else:
        res = clean_string(strings)
    return res


def rm_BP_CE(string):
    string = re.sub(r'^bp ce ', '', string)
    string = re.sub(r'^bp ', '', string)
    string = re.sub(r'^ce ', '', string)
    return string


def clean_dataframe(df_data, channel):
    """
    This function is used to remove hyperlinks and
    'Date de dernière mise à jour'
    """
    nb_rows = df_data.shape[0]
    for i in range(nb_rows):
        row = df_data.loc[i, channel]
        row = re.sub(r'<.+?>', '', row)  # remove hyperlinks
        row = re.sub(r'\*.+?\*', '', row)  # remove BP&CE
        index = row.find("Dernière mise à jour le")
        if index != -1:
            row = row[:index]
        index = row.find("Date de")
        if index != -1:
            row = row[:index]
        df_data.loc[i, channel] = row
    return df_data


def remove_useless_words(list_str, list_useless):
    '''
    This function takes removes strings in list_str if they are also in list_useless
    '''
    list_temp = []
    for i, elem in enumerate(list_str):
        for word_useless in list_useless:
            if elem[:len(word_useless)+1] == word_useless + ' ':
                list_str[i] = elem[len(word_useless)+1:]
            if elem[-len(word_useless)-1:] == ' ' + word_useless:
                list_str[i] = elem[:-len(word_useless)-1]
        if len(list_str[i]) > 2:
            list_temp.append(list_str[i])
    return list_temp


class TextRank4Keyword():
    """
    Extract keywords from text.
    """

    def __init__(self, nlp):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.nlp = nlp

    def set_stopwords(self, stopwords):
        """
        Set stop words.
        """
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """
        Store those words only in cadidate_pos.
        """
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag and with len > 1
                if len(token.text) > 1:
                    if token.pos_ in candidate_pos and token.is_stop is False:
                        if lower is True:
                            selected_words.append(token.text.lower())
                        else:
                            selected_words.append(token.text)
                    # else:
                    # print(token, token.pos_)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """
        Get all tokens.
        """
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """
        Build token_pairs from windows in sentences.
        """
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """
        Get normalized matrix.
        """
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def print_keywords(self, number=10):
        """
        Print top number keywords.
        """
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break

    def get_keywords(self, number=10):
        """
        Get top number keywords.
        """
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        keywords = []
        values = []
        for i, (key, value) in enumerate(node_weight.items()):
            keywords.append(key)
            values.append(value)
            if i > number:
                break
        return keywords, values, node_weight

    def analyze(self, text, candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """
        Main function to analyze text.
        """

        # Set stop words
        self.set_stopwords(stopwords)
        # Pare text by spaCy
        doc = self.nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


def custom_tokenizer(nlp):
    # We create our own tokenizer to avoid spliting hyphen words.
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                     suffix_search=suffix_re.search,
                     infix_finditer=infix_re.finditer,
                     token_match=None)


def preprocess_for_TextRank(text):
    text_clean = ""
    list_words = text.split(" ")
    list_replace = ['(', ')', "«", "»", "“", '...', '<', '>', '=', '+', "…", "$", '~', '@', '‘', "’", "”", "'", '"',
                    '"', '[' ']', '{', '}', '*', '/', ":", "!", ";", ".", "?", ',']
    for word in list_words:
        if len(word) > 1:
            if len(word) < 25:  # if it is too big, it might be the name of a doc, or an hyperlink
                temp = ""
                for char_ in word:
                    if char_ in list_replace:
                        temp += " " + char_ + " "
                    else:
                        temp += char_
                text_clean += temp
        else:
            text_clean += word

        text_clean += ' '
    text_clean = text_clean.replace('  ', ' ')
    return text_clean

def neighborhood_analysis(list_words, words_best, ord_dict):
    """
    This function analyses the 2 closest words from list_words at both side of each
    keywords in words_best. Based on their score given by ord_dict, it will add them to
    the keyword: for instance if 'contrat' and 'echeance' are neighbors, the new keyword will
    be 'echeance contrat'
    INPUTS:
    - list_words: list of string, words of the original text
    - words_best: list of string, best keywords
    - ord_dict: ordered dictionnary, linking words to score
    OUTPUTS:
    - words_best_bis: list of string, best new keywords
    """
    length = len(list_words)
    words_best_bis = []
    while len(words_best) > 0:
        word = words_best[0]
        try:
            j = list_words.index(word)
            score = ord_dict[word]
            score_left, score_left_supp, score_right, score_right_supp = 0, 0, 0, 0
            ok_left, ok_left_supp, ok_right, ok_right_supp = False, False, False, False

            if j > 1 and j < length - 1:
                word_left = list_words[j - 1]
                word_left_supp = list_words[j - 2]
                if word_left in ord_dict.keys(): score_left = ord_dict[word_left]
                if word_left_supp in ord_dict.keys(): score_left_supp = ord_dict[word_left_supp]
                word_right = list_words[j + 1]
                word_right_supp = list_words[j + 2]
                if word_right in ord_dict.keys(): score_right = ord_dict[word_right]
                if word_right_supp in ord_dict.keys(): score_right_supp = ord_dict[word_right_supp]
                if score_left_supp > score * 0.8:
                    ok_left_supp = True
                    ok_left = True
                    if word_left_supp in words_best: words_best.remove(word_left_supp)
                    if word_left_supp in words_best_bis: words_best_bis.remove(word_left_supp)
                if score_right_supp > score * 0.8:
                    of_right_supp = True
                    ok_right = True
                    if word_right_supp in words_best: words_best.remove(word_right_supp)
                    if word_right_supp in words_best_bis: words_best_bis.remove(word_right_supp)
                if (score_left > score * 0.8) or ok_left:
                    if len(word_left) > 2: ok_left = True
                    if word_left in words_best: words_best.remove(word_left)
                    if word_left in words_best_bis: words_best_bis.remove(word_left)
                if (score_right > score * 0.8) or ok_right:
                    if len(word_right) > 2: ok_right = True
                    if word_right in words_best: words_best.remove(word_right)
                    if word_right in words_best_bis: words_best_bis.remove(word_right)
                words_best_bis.append(
                    ok_left_supp * word_left_supp + " " + ok_left * word_left + " " + word + " " + ok_right * word_right + " " + ok_right_supp * word_right_supp)
                try:
                    words_best.remove(word)
                except:
                    pass

            elif j == 1 or j == length - 2:
                word_left = list_words[j - 1]
                if word_left in ord_dict.keys(): score_left = ord_dict[word_left]
                word_right = list_words[j + 1]
                if word_right in ord_dict.keys(): score_right = ord_dict[word_right]
                if score_left > score * 0.8:
                    if len(word_left) > 2: ok_left = True
                    if word_left in words_best: words_best.remove(word_left)
                    if word_left in words_best_bis: words_best_bis.remove(word_left)
                if score_right > score * 0.8:
                    if len(word_right) > 2: ok_right = True
                    if word_right in words_best: words_best.remove(word_right)
                    if word_right in words_best_bis: words_best_bis.remove(word_right)
                words_best_bis.append(ok_left * word_left + " " + word + " " + ok_right * word_right)
                try:
                    words_best.remove(word)
                except:
                    pass

            elif j == 0:
                if word_right in ord_dict.keys(): score_right = ord_dict[word_right]
                if score_right > score * 0.8:
                    if len(word_right) > 2: ok_right = True
                    if word_right in words_best: words_best.remove(word_right)
                    if word_right in words_best_bis: words_best_bis.remove(word_right)
                words_best_bis.append(word + " " + ok_right * word_right)
                try:
                    words_best.remove(word)
                except:
                    pass

            elif j == length - 1:
                if word_left in ord_dict.keys(): score_left = ord_dict[word_left]
                if score_left > score * 0.8:
                    if len(word_left) > 2: ok_left = True
                    if word_left in words_best: words_best.remove(word_left)
                    if word_left in words_best_bis: words_best_bis.remove(word_left)
                words_best_bis.append(ok_left * word_left + " " + word)
                try:
                    words_best.remove(word)
                except:
                    pass

        except:
            words_best_bis.append(word)
            try:
                words_best.remove(word)
            except:
                pass
    words_best_bis = check_doubles(words_best_bis)  # delete doubles
    for i, word in enumerate(words_best_bis):
        l = word.split(' ')
        if len(l[0]) <= 2: words_best_bis[i] = " ".join(l[1:])
        if len(l[-1]) <= 2: words_best_bis[i] = " ".join(l[:-1])
    return words_best_bis


def select_keywords_spacy(df_data, channel, n_keywords, nlp):
    '''
    This function update the chanel of the dataframe df_data
    by selecting the n_keywords words having the highest score
    using TextRank.
    INPUTS:
    - df_data_: pandas dataframe (n_rows, n_chanels).
    - channel: name of the channel where to extract the keywords.
    - n_keywords: int, number of keywords.
    - nlp: Spacy nlp object.
    OUTPUTS:
    - df_data_bis: copy of df_data but without the commas in sentences channel.
    '''
    df_data_bis = df_data.copy()
    answers_spans = []
    for i, row in enumerate(df_data.iterrows()):
        "Rechecker l'efficacité de ces fonctions de preprocessing."
        tr4w = TextRank4Keyword(nlp)
        row = row[-1][channel]
        row = row.replace(u'\xa0', u' ')
        row = re.sub(r'<.+?>', '', row)  # delete hyperlinks
        row = rm_BP_CE_2(row)
        row = preprocess_for_TextRank(row)
        list_words = row.split(' ')
        tr4w.analyze(row, candidate_pos=['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'], window_size=10, lower=False) #AUX
        words_best, scores_best, ord_dict = tr4w.get_keywords(n_keywords)
        words_best = remove_useless_words(words_best, STOP_WORDS)
        # ANALYSIS OF NEIGHBORHOOD
        # For the neighborhood, the score of the closest neighbors of the keywords will be checked.
        # If their score is high enough, they will be added to the keyword.
        words_best = neighborhood_analysis(list_words, words_best, ord_dict)
        words_best = clean_words_best(words_best)

        str_ = ', '.join(words_best)
        answers_spans.append(str_.lower())
    df_data_bis["keywords"] = answers_spans
    return df_data_bis


def clean_words_best(words_best):
    temp = words_best
    for i, elem in enumerate(words_best):
        temp[i] = elem.replace(',', ' ')
        temp[i] = temp[i].replace('.', ' ')
        temp[i] = temp[i].replace(';', ' ')
        temp[i] = temp[i].replace(')', ' ')
        temp[i] = temp[i].replace('(', ' ')
        temp[i] = temp[i].replace('-', ' ')
        temp[i] = temp[i].replace('  ', ' ')
    return temp


def check_doubles(list_str):
    temp = []
    list_str_bis = len(list_str) * ['']
    for i, elem in enumerate(list_str):
        words = elem.split(' ')
        for word in words:
            if word.lower() not in temp:
                temp.append(word.lower())
                list_str_bis[i] += word + ' '
        if len(list_str_bis[i]) > 0:
            if list_str_bis[i][-1] == ' ':
                list_str_bis[i] = list_str_bis[i][:-1]
    try:
        list_str_bis.remove('')
    except:
        pass
    return list_str_bis


def rm_BP_CE_2(string):
    string = string.replace("*BP&CE*", '')
    string = string.replace("*BP & CE*", '')
    string = string.replace("*BP*", '')
    string = string.replace("*CE*", '')
    return string

def separating_keywords(df, channel):
    """
    This functions separates the different keywords obtained after select_keywords_spacy.
    INPUTS:
    - df: pandas DataFrame, dataframe from which to extract the data.
    - channel: string, name of the column in df that contains the keywords.
    OUTPUTS:
    -"""
    row_list = []
    for iterrow in df.iterrows():
        row = iterrow[1]
        local_keywords = row[channel].split(', ')
        for keyword in local_keywords:
            row_list.append({"id": row["id"], "answer_span": keyword, "context": row["context"]})
    return pd.DataFrame(row_list, columns=["id", "answer_span", "context"])

