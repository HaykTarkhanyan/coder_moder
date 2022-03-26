import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
from scrapy.http import TextResponse
from requests_html import HTMLSession

# from google_trans_new import google_translator  


class MainScraper():
    """Class implements all the functions for getting website text

    
    Note:
        Class doesn't contain function for scraping fonts logos etc.

   **List of functions** |br|
        1. get_page_html() |br| 
        1. text_from_html() |br| 
        2. get_first_page() |br| 
        
        1. detect_lang()
        tag_visible

    Parameters:
        URL (str): url to scrap

    **not passed to __init__()**

    Parameters:  
        URL_root (str): Root form of the url(calucalted by ``get_root_domain()``)
        page (requests.models.Response): fill
        lang (str): language of the website
    
    .. |br| raw:: html

      <br>    
    """
    def __init__(self, URL = None):
        self.URL = URL

    def get_page_html(self,URL=None):
        """Takes as input a url and returns scraped page(response object)

        Note:
            Updates ``self.response``, ``self.status_code_200`` variables
            We have 10 second timeouts now

        Args:
            URL (str): defaults to ``self.URL``
        
        Returns:
            requests.models.Response: Entire page
        
        Examples:
            >>> MainScraper.get_page_html('http://youtube.com/')
            <Response [200]>
        """
        if URL==None:
            URL=self.URL
        #headers = {'User-Agent': 'Mozilla/5.0'}
        #response = requests.get(URL, headers=headers, timeout=(10,10))
        session = HTMLSession()
        response = session.get(URL,timeout=(10,10))#,headers=headers)
        status_code_200 = response.status_code==200
        response_ = TextResponse(url = response.url, body = response.text, encoding = "utf-8")
        lang = response_.xpath('//html/@lang').extract_first()
        return {'response':response,'response_':response_,'lang':lang,'status_code_200':status_code_200}


    @staticmethod
    def tag_visible(element):
        """Function removes all the attributes which don't contain websites text
        like style has CSS-ish stuff, which we don't need

        Args:
            element: ես սքռափինգից բան չեմ ջոգում)
        Returns:
            bool: whether tag's name is in  ['style', 'script', 'head', 'title', 'meta', '[document]'] or Comment

        """
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True
    
    
    def text_from_html(self,response):
        ''' Takes as input the url request response and returns all the text appearing in the html.
       
        Note:
            Updates ``self.full_desc``  variable
       
        Returns:
                full_desc: the full text from the main from the main page
        '''
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(self.tag_visible, texts)
        full_desc = " ".join(t.strip() for t in visible_texts)
        return full_desc 

    def get_metadata(self,response_):
        '''Takes as input the response from the url requests and returns the description and title extracted from the metadata.

        Note:
            Updates ``self.title`` and ``self.description`` variables
        Returns:
               description: the description text extracted from meta in the html translated to english if needed
               title: the title text extracted from meta in the html translated to english if needed

        '''
        
        description = response_.xpath('//meta[@name="description"]/@content').extract()
        description = ' '.join(description)
        if len(description)==0:
            description = response_.xpath('//meta[@property="og:description"]/@content').extract()
            description = ' '.join(description)
        title = response_.xpath('//head/title/text()').extract()
        title = ' '.join(title)
        if len(title)==0:
            title = response_.xpath('//meta[@property="og:title"]/@content').extract()
            title = ' '.join(title)
        # self.title = re.sub(r'[^\w\s]','',title)
        return description,title
        
    
    def get_all_text(self,URL=None):
        '''Takes as input the response from the url requests and returns the main first page text, the description and title from metadata.
        
        Note:
            Updates ``self.full_desc``, ``self.title`` and ``self.description`` variables
        Returns:
            full_desc: the full text from the main from the main page translated to english if needed
            description: the description text extracted from meta in the html translated to english if needed
            title: the title text extracted from meta in the html translated to english if needed
        '''
        if URL==None:
            URL=self.URL
        scraped = self.get_page_html(URL)
        if scraped['status_code_200']:
            full_desc = self.text_from_html(scraped['response'])
            description,title = self.get_metadata(scraped['response_'])
        else:
            full_desc = None
            description=None
            title=None   
        scraped.update({'text':full_desc,'description':description,'title':title})
        return scraped

# if __name__=='__main__':
#     import doctest 
#     # doctest.testmod(extraglobs={'MainScraper': MainScraper()})

# S = MainScraper()
# print (S.get_all_text('http://youtube.com/'))