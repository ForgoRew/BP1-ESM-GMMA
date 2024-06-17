""" Variant classification with ESM1b model

Author: Matteo Cagiada

Date of last major changes: 2022-10-09

For ESM1v model - Copyright (c) Facebook, Inc. and its affiliates

Please refer to LICENSE file in the online github place

Model manuscript: https://doi.org/10.1101/2021.07.09.450648

"""
# TO DO
# - implementation over 1024 residues, look weights
	
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    from Bio import SeqIO

import argparse
import pathlib
import string

import torch
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import sys,os,shutil
# sys.path.insert(2, '/projects/prism/people/bqm193/software/prism_scripts/')
from uniprot import extract_uniprot_info
from PrismData import PrismParser, VariantData
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Chain import Chain

from esm import pretrained
from esm.inverse_folding.util import load_structure, extract_coords_from_structure,CoordBatchConverter
from esm.inverse_folding.multichain_util import extract_coords_from_complex,_concatenate_coords,load_complex_coords
import esm
#from esm.data import BatchConverter

import pandas as pd
from tqdm import tqdm
from datetime import date
import itertools
from typing import List, Tuple
import numpy as np
import subprocess 
import traceback
from tqdm.contrib import tenumerate
from biotite.structure import get_residue_positions


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    parser.add_argument(
        "--structure",
        type=str,
        help="Input structure for the selected protein"
    )
    parser.add_argument(
        "--chain-id",
        type=str,
        default='A',
        help="chain selected for the input structure (default A)"
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        help="uniprot code for the protein (only for metadata)"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="sequence of the protein (if pdb is only backbone)"

    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="name of the prism file file (extention ecluded), standard: prism_{name}_ESM1v.txt",
        default=''
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default='./',
        help="output folder location (abs path), standard: run folder",
    )

    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="masked-marginals",
        choices=['masked-absolute', "masked-marginals"],
        help="possible scoring strategies for the pipeline"
    )

    parser.add_argument(
        "--frag-size",
        type=int,
        default=1023,
        help="size of fragments if sequence is too long for the model"
    )
    parser.add_argument(
        "--gpunumber",
        type=int,
        default=0,
        help="gpu [0-9]"
    )
    parser.add_argument(
        "--chains-complex",
        type=str,
        help="series of chains in the complex (provided as a list of separated characted, default A B)",
        nargs="+"
    )
    parser.add_argument('--random',
            action='store_true',
            help='')
    parser.set_defaults(random=False)

    parser.add_argument('--save-hiddens',
            action='store_true',
            help='save hidden representations of ESM-IF')

    parser.add_argument('--keep-fragments',
            action='store_true',
            help='s')
    parser.set_defaults(keep_fragments=False)
    
    parser.add_argument('--is-complex',
            action='store_true',
            help='use multichain complex reading function')
    parser.set_defaults(is_complex=False)

    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def write_to_log(logfile,string_to_add):
    if not os.path.exists(logfile):
        with open(logfile, 'w') as fp:
            pass

    with open(logfile,'a') as lf:
        lf.write(string_to_add+'\n')

def masked_absolute(mut, sequence, token_probs, alphabet,skipped_pos=0):
    wt, idx, mt = mut[0], int(mut[1:-1]) - 1, mut[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    if mt =='=':
        mt_encoded = alphabet.get_idx(wt)
    else:
        mt_encoded = alphabet.get_idx(mt)
    # no BOS here (apparently)
    score = token_probs[0,idx-skipped_pos, mt_encoded]
    return score.item()

def masked_marginals(mut, sequence, token_probs, alphabet,skipped_pos=0):
    wt, idx, mt = mut[0], int(mut[1:-1]) - 1, mut[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # no BOS here (apparently)
    score = token_probs[0,idx-skipped_pos, mt_encoded] - token_probs[0,idx-skipped_pos, wt_encoded]
    return score.item()

def generate_prism_output_parser(variant,columns,metadata_dict,output_name):

    print('>>>> printing output files  ...')
    today = date.today()
    variant_list_df,col_list = variant, columns 
    metadata={
            "version": '1.0',
            "protein": {
                "name": metadata_dict['name'],
                'uniprot':metadata_dict['uniprot'],
                'organism': metadata_dict['organism'],
                "sequence" : metadata_dict['sequence'],
                'isoform': metadata_dict['isoform']
                },
            "ESM2":{'version': 'esm2_t36_3B_UR50D',
                'manuscript': 'https://doi.org/10.1101/2022.07.20.500902'
                },
            "columns":  {i : i for i in col_list[1:]},
        }

    variant_dataset= VariantData(metadata,variant_list_df)

    parser=PrismParser()
    parser.write(output_name,variant_dataset)

def get_first_residue(structure_loc,chain):
    structure = PDBParser().get_structure('test', structure_loc)
    model = structure[0]
    chain = model[chain]
    resseqs = [residue.id[1] for residue in chain.get_residues()]
    return int(resseqs[0])-1

def run_model(coords,sequence,cmplx=False,chain_target='A',save_hiddens=False):
    
    model, alphabet = pretrained.load_model_and_alphabet('esm_if1_gvp4_t16_142M_UR50')

    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    device = next(model.parameters()).device    

    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    
    logits, extra = model.forward(coords, padding_mask, confidence, prev_output_tokens, return_all_hiddens = save_hiddens) #return_contacts=True, return_all_hiddens=save_hiddens)
   
    if save_hiddens:
        hidden_representations=extra['inner_states'][-1].cpu().detach().numpy()
        hidden_representations=np.swapaxes(hidden_representations,0,1)
        print(hidden_representations.shape)

    else:
        hidden_representations=None

    print(logits.size())
    
    logits_swapped=torch.swapaxes(logits,1,2) 
    token_probs = torch.log_softmax(logits_swapped, dim=-1)
    
    return token_probs,alphabet,hidden_representations


def score_variants(sequence,token_probs,alphabet,scoring_strategy):
    
    mutation_list=[]
    outscore=[]
    skip_pos=0

    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}
    
    for i,n in tenumerate(sequence):
        if n =='X':
            skip_pos+=1
        else:
            if scoring_strategy == 'masked-marginals':
                mutation_list.append(n+str(i+1)+"=")
                outscore.append(0)

                for j in range(1,21):
                    if alphabetAA_D_L[j]!=n:
                        mutation_list.append(n+str(i+1)+alphabetAA_D_L[j])

                        outscore.append(masked_marginals(mutation_list[-1],sequence, token_probs, alphabet,skipped_pos=skip_pos))
            else:
                
                for j in range(1,21):
                    
                    mutation_list.append(n+str(i+1)+alphabetAA_D_L[j])
                    
                    outscore.append(masked_absolute(mutation_list[-1],sequence, token_probs, alphabet,skipped_pos=skip_pos))

    return mutation_list, outscore

def create_fragments(seq_length,frag_size):

    length_half=frag_size//2
    ind_left=0
    ind_right=seq_length
    fragments_indices=[]
    check=True
    while check:
        fragments_indices.append([ind_left,ind_left+frag_size])
        fragments_indices.append([ind_right-frag_size,ind_right])
        if ind_left+frag_size > (ind_right-frag_size):
            if ind_left+frag_size - (ind_right-frag_size) <= (frag_size//2):
                fragments_indices.append([seq_length//2-frag_size//2,seq_length//2+frag_size//2])
            check=False
        else:
            ind_left+=(frag_size//2)
            ind_right-=(frag_size//2)

    return np.array(fragments_indices)

def weights_length_dependent(frame_length,saturation=True):
    smoothness=(frame_length//100)*2
    fourth_length=frame_length/4
    weights=np.ones(frame_length,dtype=float)
    for i in range(int(fourth_length)):
        weights[i]=1/(1+np.exp(-(i-fourth_length/2)/smoothness))
    for i in range(int(frame_length-fourth_length),int(frame_length)):
        weights[i]=1/(1+np.exp((i-frame_length+fourth_length/2)/smoothness))

    if saturation==True:
        return np.dot(np.ones((20,1),dtype=float),weights.reshape(1,-1))
    else:
        return weights

def create_pdb_fragment(pdb_loc,pdb_out_loc,res_range=[0,-1]):

    structure = PDBParser().get_structure('full_pdb',pdb_loc)
    res_to_change = []
    
    count_check=[0,0]
    for model in structure:
        count_check[0]+=1
        for chain in model:
            count_check[1]+=1
        
    if np.sum(count_check) !=2:
        sys.exit('for fragmentation pdb must have 1 model and ONLY the target chain')
        return None
    
    for model in structure:
        for chain in model:
            chain_id=chain.get_id()
            for residues in chain:          
                if (residues.get_id()[1] >= res_range[0]+1) and (residues.get_id()[1] < res_range[1]+1):
                    res_to_change.append(residues)
    for model in structure:
        for chain in model:
            [chain.detach_child(res.get_id()) for res in res_to_change]
    if chain_id !='B':    
        sel_chain = Chain("B")
        chain_created='B'
    else:
        sel_chain = Chain("X")
        chain_created='X'

    model.add(sel_chain)

    for res in res_to_change:
        sel_chain.add(res)
        
    io = PDBIO()
    io.set_structure(model)
    io.save(pdb_out_loc,  write_end = True, preserve_atom_numbering = False)
    
    print('Process_complete: fragment from range: ',str(res_range),' generated')

    return [chain_id,chain_created]


def main(args):
    
    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    torch.cuda.set_device(args.gpunumber)
    
    if args.structure is None: 
        sys.exit('NBeed a structure in pdb/mmcif format as input')
    

    # read structure and extract coordinates
    
    if args.is_complex:
        if args.chains_complex is None:
            all_chains = ['A','B'] ## implement later as option
        else:
            all_chains= [i for i in args.chains_complex]
        coords_structures, sequences_structure = load_complex_coords(args.structure, all_chains)
        sequence_structure=sequences_structure[args.chain_id]

        coords_structure = _concatenate_coords(coords_structures, args.chain_id)
        first_residue_idx=0 ## not implement count rreisdue here
    else:
        structure = load_structure(args.structure, args.chain_id)
        coords_structure, sequence_structure = extract_coords_from_structure(structure)

        if args.structure[-4:] == '.pdb': #doesnt work on mmcif atm

            first_residue_idx=get_first_residue(args.structure,args.chain_id)
        else:
            first_residue_idx=0

    parser_sequence= (first_residue_idx)*'X'+sequence_structure

    if args.sequence is not None:
        try:
            assert sequence_structure == args.sequence
        except AssertionError:
            if len(sequence_structure) != len(args.sequence):
                sys.exit('Sequence in pdb and sequence provided lenght do not match, run interrupted')
            else:
                
                print('Sequence in pdb and sequence provided do not match, provided sequence will overwrite pdb sequence')
                sequence_structure=args.sequence
                parser_sequence= (first_residue_idx)*'X'+sequence_structure
            
    ##fill up prism metadata 
    if args.uniprot is not None:
        uniprot_info=extract_uniprot_info(args.uniprot)
        if isinstance(uniprot_info, pd.DataFrame)==False:
            sys.error('No protein ID found in uniprot, run interrupted')
        
        sequence_uniprot= uniprot_info['sequence'][0]
        
        assert sequence_structure == sequence_uniprot
        sequence= sequence_structure
        
        metadata_parser={'name': uniprot_info['id'][0],
                'uniprot': args.uniprot,
                'organism': uniprot_info['organism_name'][0],
                'sequence': sequence,
                'isoform': 0,
                'fragmented': 'false'
                }

    else:
        sequence= parser_sequence
        metadata_parser={'name': 'unknown',
                'uniprot': 'unknown',
                'organism': 'unknown',
                'sequence': sequence,
                'isoform': 0,
                'fragmented': 'false'
                }
    
    ##check output file and format
    if args.prefix !='./':
        if not os.path.exists(os.path.realpath(args.prefix)):
            os.mkdir(os.path.realpath(args.prefix))

    if args.output_name == '':
        if args.scoring_strategy == 'masked-marginals':
            output_name=os.path.join(os.path.realpath(args.prefix),str('prism_ESM-IF_'+metadata_parser['uniprot']+'_marginals.txt'))
        else:
            output_name=os.path.join(os.path.realpath(args.prefix),str('prism_ESM-IF_'+metadata_parser['uniprot']+'_absolute.txt'))
    else:    
        output_name=os.path.join(os.path.realpath(args.prefix),str(args.output_name))

    if args.save_hiddens:
        output_name_hiddens=os.path.join(os.path.realpath(args.prefix),str('ESM-IF_'+metadata_parser['uniprot']+'_hiddens.npz'))

    print('Running model on: ', metadata_parser['sequence'], 'with length: ',len(sequence))    
    
    ## run model
    try:
        if len(sequence) < 1024:
            fragmentation_check =False
            prob_tokens,alphabet,hidden_representations = run_model(coords_structure,sequence_structure,save_hiddens=args.save_hiddens)
            mutation_list, outscore = score_variants(sequence,prob_tokens,alphabet,args.scoring_strategy)

        else:

            if args.is_complex:
                sys.exit('Complex flag no compatible for protein longer than 1023 residues')

            print('ESM-if language model max lenght hit:  starting fragmentation of the structuree')

            fragmentation_check=True

            metadata_parser['fragmented']='true'

            frag_idx=create_fragments(len(sequence),args.frag_size)

            tensor_raw=np.empty((len(sequence),20,frag_idx.shape[0]),dtype=float)
            tensor_raw[:]=np.nan

            #weights=weights_length_dependent(args.frag_size)
            weights=np.ones((1,args.frag_size),dtype=float)
            #tensor_weights=np.empty((len(sequence),20,frag_idx.shape[0]),dtype=float)
            #tensor_weights[:]=np.nan

            dir_fragments=os.path.join(os.path.realpath(args.prefix),str('fragmented_pdbs'))
            
            if not os.path.exists(os.path.realpath(dir_fragments)):
                os.mkdir(os.path.realpath(dir_fragments))

            #run_single_sequences
            for i in range(frag_idx.shape[0]):

                running_seq=sequence[frag_idx[i,0]:frag_idx[i,1]]
                frag_pdb_loc=os.path.join(dir_fragments,str('frag_pdb_'+str(frag_idx[i,0])+'_'+str(frag_idx[i,1])+'.pdb'))

                chains_frag=create_pdb_fragment(args.structure,frag_pdb_loc,res_range=[frag_idx[i,0],frag_idx[i,1]])
                
                coords_structures, sequences_structure = load_complex_coords(frag_pdb_loc, chains_frag)
                sequence_structure=sequences_structure[chains_frag[1]]               
                
                assert running_seq == sequence_structure, "Error: full sequence fragment and pdb fragment sequence don't match!"

                coords_structure = _concatenate_coords(coords_structures, chains_frag[1])
                
                prob_tokens,alphabet,_ = run_model(coords_structure,sequence_structure)
                _, outscore = score_variants(running_seq,prob_tokens,alphabet,args.scoring_strategy)
                
                results_shaped=np.array(outscore).reshape(len(running_seq),20)

                results_reweighed=results_shaped#*(weights.T)

                tensor_raw[frag_idx[i,0]:frag_idx[i,1],:,i]=results_reweighed

               # tensor_weights[frag_idx[i,0]:frag_idx[i,1],:,i]=weights.T

            #reweighted_tensor=np.nansum(tensor_raw,axis=2)/np.nansum(tensor_weights,axis=2)
            #reweighted_tensor=np.nanmean(tensor_raw,axis=2)#/np.nansum(tensor_weights,axis=2)
            reweighted_tensor=np.nanmin(tensor_raw,axis=2)#/np.nansum(tensor_weights,axis=2)

            reweighted_flatten=reweighted_tensor.flatten()

            outscore=reweighted_flatten.tolist()

            mutation_list=[]

            for i,n in enumerate(sequence):
                mutation_list.append(n+str(i+1)+"=")
                for j in range(1,21):
                    if alphabetAA_D_L[j]!=n:
                        mutation_list.append(n+str(i+1)+alphabetAA_D_L[j])
            
            if not args.keep_fragments:
                shutil.rmtree(dir_fragments) 
        ### print output
            
        df_out=pd.DataFrame({'variant':mutation_list,str(args.scoring_strategy):outscore})
        col_list=[col for col in df_out.columns]
        generate_prism_output_parser(df_out,col_list,metadata_parser,os.path.abspath(output_name))

        if args.save_hiddens and not fragmentation_check:
            np.savez_compressed(output_name_hiddens, hidden_representations=hidden_representations)

    except Exception:
        error=traceback.format_exc()
        print(error)
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
