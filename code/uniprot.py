"""get_uniprot_features.py obtains info from uniprot for the human genome and
individual uniprot ids

Author: Johanna K.S. Tiemann
Updated author Matteo Cagiada
Date of last major changes: 2022-10-09

"""

# Standard library imports
import urllib.request

# Third party imports
import pandas as pd
# from Bio import Seq,SeqRecord,SeqIO,pairwise2,SubsMat
# from Bio.SubsMat import MatrixInfo
# import numpy as np
# import yaml,csv,copy,time

def extract_uniprot_info(uniprot_id):
    """Uniprot search request
    Get info from here (https://www.uniprot.org/help/uniprotkb_column_names)
    """
    try:
        reviewed='true'
        features = [
                    #standard features:
                    'gene_primary','organism_name','length','sequence',
                    'protein_families'
        ]

        search_string = ('https://rest.uniprot.org/uniprotkb/' +
            f'search?query=reviewed:{reviewed}' +
            f'+AND+accession:{uniprot_id}' +
            '+&format=tsv&fields=accession,reviewed,id,protein_name,' +
            ','.join(features)
            )
        req2 = urllib.request.Request(search_string)
        with urllib.request.urlopen(req2) as f:
           response2 = f.read()
        result = [i.split("\t") for i in response2.decode("utf-8").split("\n")]
        #retain the same terms for the column headers as were queried to reduce confusion
        result = pd.DataFrame(data=result[1:], columns=['Accession','reviewed','id','protein_name']+features)
        return result
    except ValueError:
        reviewed='false'
        features = [
                     #standard features:
                     'gene_primary','organism_name','length','sequence',
                     'protein_families'
         ]

        search_string = ('https://rest.uniprot.org/uniprotkb/' +
            f'search?query=reviewed:{reviewed}' +
            f'+AND+accession:{uniprot_id}' +
            '+&format=tsv&fields=accession,reviewed,id,protein_name,' +
            ','.join(features)
            )
        req2 = urllib.request.Request(search_string)
        with urllib.request.urlopen(req2) as f:
           response2 = f.read()
        result = [i.split("\t") for i in response2.decode("utf-8").split("\n")]
        #retain the same terms for the column headers as were queried to reduce confusion
        result = pd.DataFrame(data=result[1:], columns=['Accession','reviewed','id','protein_name']+features)
        return result
    else:
        return 1
