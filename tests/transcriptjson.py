import logging

import sys
import unittest
from document_wrapper_adamllryan.doc.document import Document
from document_wrapper_adamllryan.doc.analysis import DocumentAnalysis
from document_wrapper_adamllryan.analysis.summarizer import Summarizer
import json

class TestDocumentAnalysis(unittest.TestCase):
    def setUp(self):
        # load the json 
        with open("tests/transcript.json") as f:
            self.transcript_data = json.load(f)
        self.document = DocumentAnalysis.list_to_document(self.transcript_data) 
        self.summarizer = Summarizer({
                  "model": "facebook/bart-large-cnn",
                  "token_limit": 512,
                  "max_len": 130,
                  "min_len": 30,
                  "do_sample": False
            })

    def test(self):
        logging.basicConfig( stream=sys.stderr )
        logging.getLogger( "test" ).setLevel( logging.DEBUG )
        self.summarizer.summarize(self.document.get_plain_text())
        self.assertEqual("","")





