import os
import torch
from sentence_transformers import SentenceTransformer, util

class EmbeddingExtractor:
    """Implement all the functions for embedding extractor models


    
    **List of functions** |br|
        1. load_model |br|
        2. load_embeddings_for_bases |br|
        3. save_embeddings |br|
        4. compute_embedding |br|
        5. inference |br|
    
    Parameters:
        model_name (str): name of the embedding extractor (all the options-> <https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0>)
        
        text (str, optional): input for the model (full_desc of the website)
        embeddings_path(str,  optional): path where embeddings of bases are saved,
                                              defaults to ``precomputed_embeddings/{model_name}``
        cat_dict(dict, optional): if there are no precomputed embedding dictionary with |br|
                                  websites' texts to compute embeddings for

    **not passed to init()**

    Parameters:      
        embeddings_dict (dict, optional): dictionary of embeddings for bases 
        embedding_for_input(tensor, optional): text2vector computed by ``compute_embedding()``
        model (SentenceTransformer, optional): loaded model defaults to ``model_name``
        results(dict) : results for the given text
        available_models (list of strings): list of available pretrained models
    
    Note:
        Expects ``sentence_transformers`` and ``torch`` packages to be imported
    
    Examples:
        >>> model_1 = EmbeddingExtractor('nli-mpnet-base-v2', 'hotel motel')
        >>> print (model_1.inference())
        {'Accomodation/Hotel, Motel': 0.375393807888031, |br|
        'Travel agencies': 0.3226759135723114, |br|
        'Real Estate': 0.2511726021766662, ....


    .. |br| raw:: html

      <br>    
    """

    def __init__(self, model_name, text=None, embeddigs_path=None, cat_dict=None):
        self.model_name = model_name
        
        self.text = text
        if embeddigs_path is None:
            self.embeddings_path = os.path.join('precomputed_embeddings', f'{self.model_name}.pt')
        else:
            self.embeddings_path = embeddigs_path
        self.cat_dict = cat_dict

        self.model = None
        self.embeddings_dict = None
        self.embedding_for_input = None
        self.result = None
        self.available_models = ['stsb-mpnet-base-v2', 'stsb-roberta-base-v2', 'stsb-distilroberta-base-v2', 'nli-mpnet-base-v2', 'stsb-roberta-large', 'nli-roberta-base-v2', 'stsb-roberta-base', 'stsb-bert-large', 'stsb-distilbert-base', 'stsb-bert-base', 'nli-distilroberta-base-v2', 'paraphrase-xlm-r-multilingual-v1', 'paraphrase-distilroberta-base-v1', 'nli-bert-large', 'nli-distilbert-base', 'nli-roberta-large', 'nli-bert-large-max-pooling', 'nli-bert-large-cls-pooling', 'nli-distilbert-base-max-pooling', 'nli-roberta-base', 'nli-bert-base-max-pooling', 'nli-bert-base', 'nli-bert-base-cls-pooling', 'average_word_embeddings_glove.6B.300d', 'average_word_embeddings_komninos', 'average_word_embeddings_levy_dependency', 'average_word_embeddings_glove.840B.300d']

        
    def load_model(self, model_name=None):
        """Function loads the specified model 
        
        Note:
            Also updates ``self.model`` variable

        Args:
            model_name (str): name of the model to load
        
        Returns:
            SentenceTransformer
        
        Raises:
            NameError: raises if ``model_name`` is not from ``self.available_models`` list 

        Examples:
            >>> model = EmbeddingExtractor('nli-mpnet-base-v2', 'some_text')
            >>> print(model.load_model())
            SentenceTransformer(
                    (0): Transformer(
                    (auto_model): MPNetModel(
                        (embeddings): MPNetEmbeddings(
                        (word_embeddings): Embedding(30527, 768, padding_idx=1)
                        .....
        """
        if model_name is None:
            model_name = self.model_name
        if model_name not in self.available_models:
            raise NameError(f'{model_name} not from {self.available_models} list')

        self.model = SentenceTransformer(model_name)
        return self.model

    def load_embeddings_for_bases(self, embeddings_path=None):
        """Function tries to load embeddings for ``self.model_name`` from ``precomputed_embeddings`` folder

        if there is no file ready, it fill generate it with ``self.save_embeddings()`` 

        Note:
            Uses ``torch`` to load the file, |br|
            Updates ``self.embeddings_dict`` variable
        
        Args:
            embeddings_path (str, optional): location of the file with precomputed embeddins for bases 

        Returns:
            dict: dictionary with embeddings for each category

        Examples:
            >>> model = EmbeddingExtractor('nli-mpnet-base-v2', 'some_text')
            >>> print(model.load_embeddings())
            {'Accomodation/Hotel, Motel': tensor([-1.2282e-01, -1.1649e-01,  5.9289e-03,  6.4346e-02,  7.5079e-02,
            7.4137e-02, -1.1155e-01,  ....

        """
        if embeddings_path is None:
            embeddings_path = self.embeddings_path
        
        # if there are no computed embeddings, compute them and save
        if not os.path.exists(embeddings_path):
            self.save_embeddings()

        self.embeddings_dict = torch.load(embeddings_path, map_location=torch.device('cpu'))
        return self.embeddings_dict

    def save_embeddings(self, cat_dict=None):
        """function saves embeddings computed by ``self.model_name`` model for 
        given bases(``cat_dict``) to ``precomputed_embeddings/{model_name}.pt`` file.
        

        Note:
            Since there are few websites for each category, it will average the embeddings
            to get one (for example) 784dimesinal vector |br|

            Uses ``torch`` to save the file, |br|
            Updates ``self.embeddings_dict`` variable
     
        Args:
            cat_dict: dictionary with base websites` texts for each category |br| 
                expects` EmbeddingExtractor(cat_dict=some_dictionary, model_name"
        
        Returns:
            None  
        
        Raises:
            ValueError: raises if you don't pass cat_dict parameter to module

        Examples:
            >>> model_1 = EmbeddingExtractor(model_name='nli-bert-base', cat_dict={'first':['asdasd', 'asd'], 'sec': ['123']})
            >>> print (model_1.save_embeddings())
            # will save nli_bert_base.pt file contaings dictionary with two keys 
            # (with corresponding embedding)) in precomputed_embeddings folder 
        """
        if self.model is None:
            self.load_model()
        if cat_dict is None:
            cat_dict = self.cat_dict

        if cat_dict is None:
            raise ValueError('If there are no precomputed embeddings, at least give me \
                    dictionary to compute embeddings for, please initialize the model \
                    like EmbeddingExtractor(cat_dict=some_dictionary, model_name....))')

        embeddings_dict = {}
        for category in cat_dict.keys():
            embeddings = []
            for text in cat_dict[category]:
                embeddings.append(self.model.encode(text, convert_to_tensor=True))
            embeddings_dict[category] = torch.mean(torch.stack(embeddings), dim=0)
        
        self.embedding_dict = embeddings_dict
        torch.save(embeddings_dict, self.embeddings_path)


    def compute_embedding(self, text=None):
        """Function extracts the vector for given input text
        
        Note:
            Updates ``self.embedding_for_input`` variable
        
        Args:
            text (str, optional): input text(full_description of the website) 
        
        Returns:
            Tensor: N dimensional representation of the text
        
        Examples:
            >>> model = EmbeddingExtractor('nli-mpnet-base-v2', 'some_text')
            >>> print(model.compute_embedding())
            [ 0.00428588  0.09967563  0.12012875  0.13440956  0.01609396 ...
            # most likely either 786d or 300d list
        """
        if text is None:
            text = self.text
        if self.model is None:
            self.load_model()

        self.embedding_for_input = self.model.encode(text)
        return self.embedding_for_input

    def inference(self, text=None):
        """Function computes similarities of the input's embedding with all the bases
        and returns it in a sorted dictionary.

        Note:
            Uses ``cosine similarity`` |br|
            May update ``self.embedding_for_input`` variable |br|
            updates ``self.results`` variable

        Args:
            text (str, optional): input text(full_description of the website) 

        Returns: 
            dict: similarities score with each category 
        
        Raises:
            ValueError: when text is not provided, neither when initalizing class nor calling ``inference``
        
        Examples: 
            >>> model = EmbeddingExtractor('nli-mpnet-base-v2', 'realest estate')
            >>> print(model.inference())
            {'Real Estate': 0.6026073694229126, |br| 
            'Construction': 0.3738774359226227, |br| ...
        """
        # checks if functions above haven't been executed, and calls them
        if text is None:
            text = self.text
        if text is None:
            raise ValueError("No text provided")

        if self.embeddings_dict is None:
            self.load_embeddings_for_bases()
        if self.embedding_for_input is None:
            self.compute_embedding(text)

        if text is not None:
            self.embedding_for_input = self.compute_embedding(text)

        results_dict = {}

        for k in self.embeddings_dict.keys():
            score = float(util.pytorch_cos_sim(self.embedding_for_input, self.embeddings_dict[k]))
            results_dict[k] = score

        results_dict = dict(sorted(results_dict.items(), key=lambda item: item[1],reverse=True))
        self.result = results_dict
        return self.result


# if __name__ == "__main__":
#     import doctest
#     doctest.testmod(extraglobs={'model': EmbeddingExtractor('nli-mpnet-v2')})