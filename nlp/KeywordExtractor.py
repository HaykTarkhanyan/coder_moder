import os
import json

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import yake
from fuzzywuzzy import fuzz


class KeywordExtractor:
    """Implement all the functions for keyword extractor models

    **List of functions** |br|
        1. get_keywords() |br| 
        2. get_intersaction() |br| 
        3. get_intersection_fuzzy() |br| 
        4. compute_and_save_keywords_for_bases() |br|
        5. load_keywords_for_bases() |br| 
        6. inference() 
        
        Static |br| 
        1. get_similarity_score() |br| 

    Parameters:
        text (str, optional): full_desc of website
        num_keywords_to_extract (int, optional): how many keywords to extract from ``text`` (default: 30) 
        num_keywords_to_take (int, optional): how many keywords to consider (default: 20)
        stoplist (list of strings, optional): words to remove from keywords(default: some corona related word + ``nltk``'s stoplists)
        ngram_size (int, optional): for ``yake`` model, what size keywords to return (default: 1)
        lemmatizer(WordNetLemmatizer, optional): preprocessor for keywrods
        intersection_method (str, optional): how to calucalate intersection between keywords (default: 'fuzzy', any other value is ordinary intersection)
        cat_dict (dict, optional): texts of the base websites
            
    
    **not passed to init()**

    Parameters:    
        
        model (yake): ``yake.KeywordExtractor(lan="en", n=ngram_size, dedupLim=0.9,
                                dedupFunc='seqm', windowsSize=1, top=num_keywords_to_extract)``
        keywords_path (str): precomputed_keywords/keywords_dict.json
        keywords_for_bases (dict): either loaded from ``keyword_path`` or computed based on ``cat_dict``
        keywords_for_input(list of string): computed by ``get_keywords()``
        num_intersections_dict(dict): eventual output(results)

    Note:
        Expects ``yake``, ``fuzzywuzzy``, ``nltk``  packages to be imported
    
    Examples:
        fill

    .. |br| raw:: html

      <br>    
    """

    def __init__(self, text=None, num_keywords_to_take=20, num_keywords_to_extract=40, 
                 stoplist=['corona', 'coronavirus', 'covid', 'epidemic', 'pandemic'], \
                 ngram_size=1, lemmatizer=WordNetLemmatizer(), intersection_method='fuzzy',
                cat_dict=None):
        self.text = text    
        
        self.num_keywords_to_extract = num_keywords_to_extract
        self.num_keywords_to_take = num_keywords_to_take
        self.stoplist = stoplist + stopwords.words('english')
        self.ngram_size = ngram_size
        self.lemmatizer = lemmatizer
        self.intersection_method=intersection_method
        self.cat_dict = cat_dict

        self.model = yake.KeywordExtractor(lan="en", n=ngram_size, dedupLim=0.9, \
                                dedupFunc='seqm', windowsSize=1, top=num_keywords_to_extract)
        
        self.keywords_path = os.path.join('./models/precomputed_keywords', 'keywords_dict.json')       
        
        self.keywords_for_bases = None
        self.keywords_for_input = None
        self.num_intersections_dict = None

        
    def get_keywords(self, text=None, update_input_keywords=True):
        """Function runs ``self.model`` on a given text (default: ``self.text``) 
        and return top ``self.num_keywords`` keywords.

        Uses yake model
        
        Note:
            1. Maybe we should also take into account the scores of keywords

            2. Currently, outputs the keywords in random order(because we remove dublicates with set())
     

        Args:
            text(str, optional): description of the website (default: ``self.text``)
            update_input_keywords (bool, optional): default is true

        Returns:
            list of strings: keywords from the text
        
        Raises:
            ValueError: if input text wasn't given

        Examples:
            >>> model_1 = KeywordExtractor(ngram_size=2)
            >>> model_1.get_keywords(text='Google is a company that company')
            ['Google', 'company']
        """

        if text is None:
            text = self.text
        if text is None:
            raise ValueError("You haven't provided the text(full_desc) neither when initalizing class, nor in function call")


        keywords_and_scores = self.model.extract_keywords(text)#[::-1] # sorted in descending order  # it seems no need to change the order """The lower the score, the more relevant the keyword is"""

        keywords = [keyword[0] for keyword in keywords_and_scores]
        keywords_lemmatized = [self.lemmatizer.lemmatize(i) for i in keywords]
        # removing all the keywords that are in stoplist
        keywords_lemmatized = [i for i in keywords_lemmatized if i not in self.stoplist]
        # get only top `self.num_keywords`
        if len(keywords_lemmatized)>self.num_keywords_to_take:
            keywords_lemmatized = keywords_lemmatized[:self.num_keywords_to_take]

        if update_input_keywords:
            self.keywords_for_input = list(set(keywords_lemmatized))
            return self.keywords_for_input
        else:
            return list(set(keywords_lemmatized))

    def get_intersaction(self, keywords_for_input=None, keywords_for_bases=None):
        """Function takes as input list of keywords(for input text) and returns 
        number of common elements(a.k.a intersection) between the list and 
        keywords for the base categories
        
        Note:
            Also updates ``self.num_intersections_dict`` variable

        Args:
            keywords_for_input (list of strings, optional): defaults to ``self.keywords_for_input`` computed by ``self.get_keywords()``        
            keywords_for_bases (dict, optional): keywords for all the base categories, default to ``self.keywords_for_bases`` 
        
        Returns:
            dict: number of common elements for each base keywords and input text keywords 

        Raises:
            ValueError: if either ``keywords_for_input`` or ``keywords_for_bases`` is unspecified
        
        
        Examples:
            >>> model_1 = KeywordExtractor(ngram_size=2)
            >>> print (model_1.get_intersaction(keywords_for_input=['ba', 'a', 'sd', 'sds'], \ 
                               keywords_for_bases={'1':['a', 'ba'], '2':['sd', 'a', 'ba']}))
            {'2': 3, '1': 2}
        """
        if keywords_for_input is None:
            keywords_for_input = self.keywords_for_input
        if keywords_for_bases is None:
            keywords_for_bases = self.keywords_for_bases
        

        if keywords_for_input is None:
            self.get_keywords()    
            keywords_for_input = self.keywords_for_input

        if keywords_for_bases is None:
            self.load_keywords_for_bases()
            keywords_for_bases = self.keywords_for_bases

        if keywords_for_input is None:
            raise ValueError("You haven't specified input text(or haven't given keywords for input, either give it when initializing model, or when calling this function")
        if keywords_for_bases is None:
            raise ValueError("You haven't specified dictionary with keywords for bases(or haven't given ``cat_dict`` to calculate it from, either give it when initializing the model, or when calling this function")


        res = {}
        for category, keywords_for_category in keywords_for_bases.items():
            inter = set(keywords_for_input).intersection(keywords_for_category)
            res[category] = len(inter)
        self.num_intersections_dict  = dict(sorted(res.items(),reverse=True, key=lambda item: item[1]))
        return self.num_intersections_dict  

    def get_intersaction_fuzzy(self, keywords_for_input=None, keywords_for_bases=None, threshold=90):
        """Function takes as input list of keywords(for input text) and returns 
        number of similar elements between the list and 
        keywords for the base categories. 
        
        Note:
            Also updates ``self.num_intersections_dict`` variable
            Here similarity between keywords is calculated by ``self.get_similarity_score()``           ``
        
        Args:
            keywords_for_input (list of strings, optional): defaults to ``self.keywords_for_input`` computed by ``self.get_keywords()``        
            keywords_for_bases (dict, optional): keywords for all the base categories, default to ``self.keywords_for_bases`` 
            threshold (float, optional): (value between 0-100) shows how similar 2 words should be to be considered the same
        
        Returns:
            dict: number of common elements for each base keywords and input text keywords 

        Raises:
            ValueError: if either ``keywords_for_input`` or ``keywords_for_bases`` is unspecified
        
        
        Examples:
            >>> model_1 = KeywordExtractor(ngram_size=2)
            >>>     print(model_1.get_intersaction_fuzzy(threshold=85,keywords_for_input=['word', 'barev'], keywords_for_bases={'1': ['world', 'bar'],  '2': ['word', 'sdf']}))
            {'2': 1, '1': 0}

        """
        if keywords_for_input is None:
            keywords_for_input = self.keywords_for_input
        if keywords_for_bases is None:
            keywords_for_bases = self.keywords_for_bases

        if keywords_for_input is None:
            self.get_keywords()    
            keywords_for_input = self.keywords_for_input

        if keywords_for_bases is None:
            self.load_keywords_for_bases()
            keywords_for_bases = self.keywords_for_bases

        if keywords_for_input is None:
            raise ValueError("You haven't specified input text(or haven't given keywords for input, either give it when initializing model, or when calling this function")
        if keywords_for_bases is None:
            raise ValueError("You haven't specified dictionary with keywords for bases(or haven't given ``cat_dict`` to calculate it from, either give it when initializing the model, or when calling this function")
   
        res = {}
        for category, keywords_for_category in keywords_for_bases.items():
            intersection_num = 0
            for category_keywords in keywords_for_category:
                
                scores = [self.get_similarity_score(category_keywords,input_keyword)  \
                                                    for input_keyword in keywords_for_input]
                if max(scores)>threshold:
                    intersection_num += 1  
            res[category] = intersection_num
        

        self.num_intersections_dict = dict(sorted(res.items(),reverse=True, key=lambda item: item[1]))
        return self.num_intersections_dict

    def compute_and_save_keywords_for_bases(self, cat_dict=None):
        """Function extracts the keywords for the base websites for each category
        and saves it to ``precomputed_keywords/keywords_dict.json``

        Note:
            Updates ``self.keywords_for_bases`` variable
        
        Args:
            cat_dict (dictionary, optional): dictionary with texts for base websites, defaults to ``self.cat_dict``
        
        Returns:
            None   

        Raises:
            ValueError: raises if you don't pass cat_dict parameter to module

        Examples:
            >>> model_1 = KeywordExtractor(ngram_size=2)
            >>> model_1.compute_and_save_keywords_for_bases(cat_dict={'a':['asd', 'fdsd'], 'b':['barev', 'dzez']})
            # saves {"a": ["fdsd", "asd fdsd", "asd"], "b": ["dzez", "barev dzez", "barev"]}  \
            # to ``precomputed_keywords/keywords_dict.json``
        """
        if cat_dict is None:
            cat_dict = self.cat_dict

        if cat_dict is None:
            raise ValueError('If there are no precomputed keywords, at least give me \
                    dictionary to compute embeddings for, please initialize the model \
                    like KeywordExtractor(cat_dict=some_dictionary, ....))')

        # join all the descriptions for category to one big string
        cat_dict = {k:' '.join(v).strip() for k,v in cat_dict.items()}   
        self.keywords_for_bases = {k: self.get_keywords(text=v, update_input_keywords=False)  for k,v in cat_dict.items()}

        with open(os.path.join('./models/precomputed_keywords', 'keywords_dict.json'), 'w') as f:
            json.dump(self.keywords_for_bases, f)

    def load_keywords_for_bases(self, keywords_path=None):
        """Function tries to load dictionary with keywords from bases from ``precomputed_keywords`` folder

        if there is no file ready, it fill generate it with ``self.compute_and_save_keywords_for_bases()`` 

        Note:
            Updates ``self.keywords_for_bases`` variable
        
        Args:
            keywords_path (str, optional): location of the file with precomputed kewords for each category 

        Returns:
            dict: dictionary with keywords for each category

        Examples:
            >>> model_1 = KeywordExtractor(ngram_size=2, cat_dict={'mek':['meker', 'erkusner'], 'panir': ['hndkahav', '2 hat']})
            >>> print (model_1.load_keywords_for_bases())
            {'mek': ['erkusner', 'meker erkusner', 'meker'], 'panir': ['hndkahav', 'hat']}
        """
        if keywords_path is None:
            keywords_path = self.keywords_path
        
        # if you provide cat_dict will overwright even precomputed keywords
        if self.cat_dict is not None:
            self.compute_and_save_keywords_for_bases(cat_dict=self.cat_dict)

        # if there are no computed embeddings, compute them and save
        elif not os.path.exists(keywords_path):
            self.compute_and_save_keywords_for_bases()

        with open(keywords_path) as f:
            self.keywords_for_bases = json.load(f) 
        
        return self.keywords_for_bases

    def inference(self, text=None):
        """Function extracts keywords for the input and returns number of instersactions
        with each category's base keywords,

        Note:
            updates ``self.num_intersections_dict`` variable

        Args:
            text (str, optional): input text(full_description of the website) 

        Returns: 
            dict: number_of_instersactions score with each category 
        
        Raises:
            ValueError: when text is not provided, neither when initalizing class nor calling ``inference``
        
        Examples: 
            >>>  model_1 = KeywordExtractor(ngram_size=2, text='panir panir oooo', cat_dict={'mek':['kargin', 'panir'], 'panir': ['hndkahav', '2 hat']})
            >>> print (model_1.inference())
            {'mek': 1, 'panir': 0}


        """
        if text is not None:
            self.text = text
        if text is None and self.text is None:
            raise ValueError("No text provided")

        # checks if functions above haven't been executed, and calls them
        if self.keywords_for_input is None:
            self.get_keywords()

 
        if self.keywords_for_bases is None:
            self.load_keywords_for_bases()

        if self.intersection_method=='fuzzy':
            self.get_intersaction_fuzzy()

        return self.num_intersections_dict

    @staticmethod
    def get_similarity_score(keyword_base, keyword_input):
        """Function computes how similar 2 words are.

        Notes:
            Equation is 0.3 * ratio + 0.3 * partial_ratio + 0.4 * token_set_ratio

        Args:
            keyword_base (str)
            keyword_input (str)
        Returns:
            float: value between 0 and 100.
        
        Examples:
            >>> model_1 = KeywordExtractor(ngram_size=2, cat_dict={'mek':['meker', 'erkusner'], 'panir': ['hndkahav', '2 hat']})
            >>> print (model_1.get_similarity_score('varsik', 'tandzik'))
            47.2
        """
        a = fuzz.ratio(keyword_input.lower(), keyword_base)
        b = fuzz.partial_ratio(keyword_input.lower(), keyword_base)
        c = fuzz.token_set_ratio(keyword_input.lower(), keyword_base)
        
        score = 0.3*a + 0.3*b + 0.4*c
        return score


# if __name__ == "__main__":
#     import doctest
#     doctest.testmod(extraglobs={'model_1': KeywordExtractor()})