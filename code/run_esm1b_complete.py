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
import sys,os
sys.path.insert(2, '/projects/prism/people/bqm193/software/prism_scripts/prism_bio180')
from uniprot import extract_uniprot_info
from PrismData import PrismParser, VariantData
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from datetime import date
import itertools
from typing import List, Tuple
import numpy as np
import subprocess 
import mdtraj as md
import traceback

from tqdm.contrib import tenumerate

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )

    parser.add_argument(
        "--uniprot",
        type=str,
        help="uniprot ID of the target protein",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--isoform",
        type=int,
        default=1,
        help="Isoform for the selected uniprot protein (follow uniprot order)"
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
        "--prefix-save-hidden",
        type=str,
        default='./hidden_layers_npz/',
        help="output folder location (abs path), standard: run folder",
    )
    parser.add_argument(
        "--prefix-save-entropy",
        type=str,
        default='./entropy_np/',
        help="output folder location (abs path), standard: run folder",
    )

    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="masked-marginals",
        choices=['pseudo-ppl-pos', "masked-marginals",'masked-absolute'],
        help="possible scoring strategies for the pipeline"
    )

    parser.add_argument(
        "--frag-size",
        type=int,
        default=1022,
        help="size of fragments if sequence is too long for the model"
    )
    parser.add_argument(
        "--gpunumber",
        type=int,
        default=0,
        help="gpu [0-9]"
    )
    parser.add_argument('--wt-hidden',
            action='store_true',
            help='save hidden layers representation')
    parser.set_defaults(wt_hidden=False)
     
    parser.add_argument('--contacts-only',
            action='store_true',
            help='save contacts')
    parser.set_defaults(contacts_only=False)

    parser.add_argument('--contacts-check',
            action='store_true',
            help='save contact map and check top contact with AF2 structure')
    parser.set_defaults(contacts_check=False)
    
    parser.add_argument('--entropy',
            action='store_true',
            help='save entropy')
    parser.set_defaults(entropy=False)

    # fmt: on
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def write_to_log(logfile,string_to_add):
    if not os.path.exists(logfile):
        with open(logfile, 'w') as fp:
            pass

    with open(logfile,'a') as lf:
        lf.write(string_to_add+'\n')

def masked_marginals(mut, sequence, token_probs, alphabet):
    wt, idx, mt = mut[0], int(mut[1:-1]) - 1, mut[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # add 1 for BOS
    score = token_probs[0,1+idx, mt_encoded] - token_probs[0, 1+idx, wt_encoded]
    return score.item()

def masked_absolute(mut, sequence, token_probs, alphabet):
    wt, idx, mt = mut[0], int(mut[1:-1]) - 1, mut[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    mt_encoded = alphabet.get_idx(mt)

    # add 1 for BOS
    score = token_probs[0,1+idx, mt_encoded] #- token_probs[0, 1+idx, wt_encoded]
    return score.item()

def compute_pppl_onlymut(pos, sequence, model, alphabet):
    #based on mut Keyword
    wt, idx = pos[0], int(pos[1:])-1
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    
    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()
    
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # compute probabilities at each position
    batch_tokens_masked = batch_tokens.clone()
    batch_tokens_masked[0, idx+1] = alphabet.mask_idx ## is it correct ? Shouldn't be idx,idx+1
    with torch.no_grad():
        token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
    
    return token_probs[0, idx+1, :].cpu().detach().numpy()  # vocab size

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

def evaluate_contacts(pdb_loc,len_seq,r0=8.0):

    print('eval ref map')
    pdb=md.load(pdb_loc)
    topology=pdb.topology
    chainA=topology.select('chainid 0 and protein and not resname GLY')
    chainCA=topology.select('chainid 0 and protein')

    pdb_chain0=pdb.atom_slice(chainA)
    pdb_dist,pdb_rp=md.compute_contacts(pdb_chain0,scheme='sidechain-heavy',periodic=False)

    pdb_chainCA0=pdb.atom_slice(chainCA)
    pdb_distCA,pdb_rpCA=md.compute_contacts(pdb_chainCA0,scheme='closest',periodic=False)

    cm= md.geometry.squareform(pdb_dist,pdb_rp)[0]
    cmca= md.geometry.squareform(pdb_distCA,pdb_rpCA)[0]

    cm_adj=np.empty((len_seq,len_seq),dtype=float)
    cm_adj[:]=np.nan
    chainCA_top=pdb_chainCA0.topology
    chainA_top=pdb_chain0.topology

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_adj[int(str(chainA_top.residue(i))[3:])-1,int(str(chainA_top.residue(j))[3:])-1]=cm[i,j]
    for i in range(cmca.shape[0]):
        for j in range(cmca.shape[1]):
            if str(chainCA_top.residue(i))[:3]=='GLY' or str(chainCA_top.residue(j))[:3]=='GLY':
                cm_adj[int(str(chainCA_top.residue(i))[3:])-1,int(str(chainCA_top.residue(j))[3:])-1]=cmca[i,j]

    cmap=np.zeros((cm_adj.shape[0],cm_adj.shape[1]),dtype=int)
    for i, w in np.ndenumerate(cm_adj):
        if w*10<r0:

            cmap[i[0],i[1]]=1

    return cmap

def compare_contact_maps(query_map,ref_map):
    print('compare maps')
    query_filtered=np.copy(query_map[0,:,:])

    for i, n in np.ndenumerate(query_filtered):
        if np.abs(i[0]-i[1])<6:
            query_filtered[i[0],i[1]]=0
    list_to_sort=[]
    for i, n in np.ndenumerate(query_filtered):
        if n>0:
            list_to_sort.append([i[0],i[1],n,ref_map[i[0],i[1]]])

    to_sort=np.array(list_to_sort)
    sort=to_sort[to_sort[:, 2].argsort()[::-1]]

    total_distcontacts=(sort[:,2] > 0.3).sum() / query_filtered.shape[0]
    positive=0
    total=0
    for s in sort[:query_filtered.shape[0],3]:
        if s==1:
            positive+=1
        total+=1

    precision=positive/total
    print(precision)
    return total_distcontacts,precision

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

def run_model(sequence,scoring_strategy='masked-marginals',save_entropy=False,wt_hidden=False,contact_eval=False,pdb_ref=None):
    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    if contact_eval==True and len(sequence)>1000:
        offload=True
    elif len(sequence) > 3000:
        offload=True
    else:
        offload=False

    if offload ==True:
        # init the distributed world with world_size 1
        url = "env://"
        url = "tcp://localhost:23456"
        torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)
        print('hello')
        fsdp_params = dict(
            #mixed_precision=True,
            compute_dtype=torch.float16,
            buffer_dtype=torch.float16,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
            move_params_to_cpu=True,  # enable cpu offloading
        )

        with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
            
            print('hello')
            model, alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
            batch_converter = alphabet.get_batch_converter()
            model=model.half()
            model.eval()

            # Wrap each layer in FSDP separately
            for name, child in model.named_children():
                if name == "layers":
                    for layer_name, layer in child.named_children():
                        wrapped_layer = wrap(layer)
                        setattr(child, layer_name, wrapped_layer)
            model = wrap(model)
    
    else:
        model, alphabet = pretrained.load_model_and_alphabet('esm1b_t33_650M_UR50S')
        
        model.eval()
        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
        print("Transferred model to GPU")

        batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    mutation_list=[]
    outscore=[] 
    all_entropy=[]
    wt_hidden_dict={}
    wt_contacts=[]
    ref_contacts=[]
    precision=0
    total_distcontacts=0

    if contact_eval==True:
        with torch.no_grad():
            results=model(batch_tokens.cuda(),return_contacts=True)

        wt_contacts=results['contacts'].to('cpu').detach().numpy()

    else:
        if not pdb_ref is None:

            with torch.no_grad():
                results=model(batch_tokens.cuda(),repr_layers=[12],return_contacts=True)

            wt_contacts=results['contacts'].to('cpu').detach().numpy()

            ref_contacts=evaluate_contacts(pdb_ref,len(sequence))
            total_distcontacts,precision=compare_contact_maps(wt_contacts,ref_contacts)
            print('out ',precision)

        if wt_hidden==True:
            print('wt hidden layers evaluations')
            with torch.no_grad():
                results=model(batch_tokens.cuda(),repr_layers=[i for i in range(33+1) ],need_head_weights=True)
                
            wt_logits=results['logits'].to('cpu').detach().numpy()
            wt_attentions=results['attentions'].to('cpu').detach().numpy()

            stacked_representations=[]
            for i in range(33+1):
                stacked_representations.append(results['representations'][i].to('cpu').detach().numpy())

            wt_representations=np.concatenate(stacked_representations,axis=0)
        if not scoring_strategy is None:
             all_token_probs = []
             for i in tqdm(range(batch_tokens.size(1))):
                 batch_tokens_masked = batch_tokens.clone()
                 batch_tokens_masked[0, i] = alphabet.mask_idx
                 with torch.no_grad():
                     
                     logits=model(batch_tokens_masked.cuda())["logits"]
                     token_probs = torch.log_softmax(logits, dim=-1)
                     if save_entropy:
                         entropy=(torch.softmax(logits,dim=-1)*torch.log_softmax(logits,dim=1)).sum(-1)[:,i]

                 all_token_probs.append(token_probs[:, i])  # vocab size
                 
                 if save_entropy:
                    all_entropy.append(entropy.to('cpu').detach().numpy()[0])

             token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
             
             if scoring_strategy == "masked-marginals":
                for i,n in tenumerate(sequence):
                    mutation_list.append(n+str(i+1)+"=")
                    outscore.append(0)

                    for j in range(1,21):
                        if alphabetAA_D_L[j]!=n:
                            mutation_list.append(n+str(i+1)+alphabetAA_D_L[j])
                            outscore.append(masked_marginals(mutation_list[-1],sequence, token_probs, alphabet))
             if scoring_strategy == "masked-absolute":
                for i,n in tenumerate(sequence):
                    for j in range(1,21):
                        mutation_list.append(n+str(i+1)+alphabetAA_D_L[j])
                        outscore.append(masked_absolute(mutation_list[-1],sequence, token_probs, alphabet))

        if wt_hidden==True:
            wt_hidden_dict={'wt_logits':wt_logits,'wt_attentions':wt_attentions,'wt_representations':wt_representations}

    return mutation_list, outscore, all_entropy, wt_hidden_dict,wt_contacts,ref_contacts,total_distcontacts,precision

def main(args):
    
    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    torch.cuda.set_device(args.gpunumber)
    
    if args.sequence is None and args.uniprot is None:
        sys.exit('One between flags --sequence  or --uniprot is needed as input. See documentation with --help for more information')
    elif args.sequence is not None:
        sequence=args.sequence
        if args.uniprot is None:
            metadata_parser={'name': 'unknown', 'uniprot': 'unknown', 'organism': 'unknown','sequence':args.sequence, 'isoform': 'unknown' }
        else:
            uniprot_info=extract_uniprot_info(args.uniprot)
            if isinstance(uniprot_info, pd.DataFrame)==False:
                sys.error('No protein ID found in uniprot, run interrupted')
            metadata_parser={'name': uniprot_info['id'][int(args.isoform)-1],
                'uniprot': args.uniprot,
                'organism': uniprot_info['organism_name'][int(args.isoform)-1],
                'sequence': sequence,
                'isoform': int(args.isoform)-1,
                'fragmented': 'false'
                }
    else:
        uniprot_info=extract_uniprot_info(args.uniprot)
        if isinstance(uniprot_info, pd.DataFrame)==False:
            sys.error('No protein ID found in uniprot, run interrupted')
        sequence= uniprot_info['sequence'][int(args.isoform)-1]
        metadata_parser={'name': uniprot_info['id'][int(args.isoform)-1],
                'uniprot': args.uniprot,
                'organism': uniprot_info['organism_name'][int(args.isoform)-1],
                'sequence': sequence,
                'isoform': int(args.isoform)-1,
                'fragmented': 'false'
                }
    logfile=os.path.join(os.path.realpath(os.getcwd()),'run_logfile.txt')
    
    if args.prefix !='./':
        if not os.path.exists(os.path.realpath(args.prefix)):
            os.mkdir(os.path.realpath(args.prefix))

    if args.output_name == '':
        if args.scoring_strategy == 'masked-marginals':
            output_name=os.path.join(os.path.realpath(args.prefix),str('prism_ESM1b_'+metadata_parser['uniprot']+'_marginals.txt'))
        elif args.scoring_strategy == 'masked-absolute':
            output_name=os.path.join(os.path.realpath(args.prefix),str('prism_ESM1b_'+metadata_parser['uniprot']+'_absolute.txt'))
        else: 
            output_name=os.path.join(os.path.realpath(args.prefix),str('prism_ESM1b_'+metadata_parser['uniprot']+'.txt'))

    else:    
        output_name=os.path.join(os.path.realpath(args.prefix),str(args.output_name+'.txt'))
    if args.wt_hidden == True:
        if os.path.isdir(os.path.realpath(args.prefix_save_hidden)):
            pass
        else:
            os.mkdir(os.path.realpath(args.prefix_save_hidden))
        output_name_h=os.path.join(os.path.realpath(args.prefix_save_hidden),str(metadata_parser['uniprot']+'_wt_hidden_layers')) 
    if args.entropy:
        if os.path.isdir(os.path.realpath(args.prefix_save_entropy)):
            pass
        else:
            os.mkdir(os.path.realpath(args.prefix_save_entropy))
        output_name_e=os.path.join(os.path.realpath(args.prefix_save_entropy),str(metadata_parser['uniprot']+'_entropy'))
    if args.contacts_only or args.contacts_check:
        if os.path.isdir(os.path.realpath(args.prefix)):
            pass
        else:
            os.mkdir(os.path.realpath(args.prefix))
        if args.output_name =='':
            output_name_c=os.path.join(os.path.realpath(args.prefix),str(metadata_parser['uniprot']+'_contact_maps'))
        else:
            output_name_c=os.path.join(os.path.realpath(args.prefix),str(args.output_name+'_contact_maps'))

        if args.contacts_check:
            pdb_ref=os.path.join(os.path.realpath(args.prefix),str(metadata_parser['uniprot']+'_AF2.pdb'))
            subprocess.call(['curl','-s',
                '-f',f'https://alphafold.ebi.ac.uk/files/AF-{args.uniprot}-F1-model_v4.pdb',
                '-o',pdb_ref])
    else:
        pdb_ref=None

    print('Running model on: ', metadata_parser['uniprot'], 'with length: ',len(sequence))    
    
    try:
        if len(sequence) < 1024:
            mutation_list,outscore,entropy_seq,wt_h_layers,wt_contacts,ref_contacts,total_distcontacts,precision=run_model(sequence,args.scoring_strategy,args.entropy,args.wt_hidden,args.contacts_only,pdb_ref=pdb_ref)
            fragmentation_check=False
        else:
            print('ESM1b GPU max memory hit, protein exceeds  max length, starting fragmentation of the sequence')
            
            fragmentation_check=True

            if args.wt_hidden or args.contacts_only:
                print("Extra feature such as hidden layer extraction and contact evaluation and check are not supported, output won't be produced")
            
            metadata_parser['fragmented']='true'

            frag_idx=create_fragments(len(sequence),args.frag_size)
            print(frag_idx)
            tensor_raw=np.empty((len(sequence),20,frag_idx.shape[0]),dtype=float)
            tensor_raw[:]=np.nan

            weights=weights_length_dependent(args.frag_size)
            tensor_weights=np.empty((len(sequence),20,frag_idx.shape[0]),dtype=float)
            tensor_weights[:]=np.nan
            
            #run_single_sequences
            for i in range(frag_idx.shape[0]):
                running_seq=sequence[frag_idx[i,0]:frag_idx[i,1]]

                _,results,_,_,_,_,_,_=run_model(running_seq,args.scoring_strategy,args.entropy,False,False)
        
                results_shaped=np.array(results).reshape(len(running_seq),20)

                results_reweighed=results_shaped*(weights.T)

                tensor_raw[frag_idx[i,0]:frag_idx[i,1],:,i]=results_reweighed
                    
                tensor_weights[frag_idx[i,0]:frag_idx[i,1],:,i]=weights.T

            reweighted_tensor=np.nansum(tensor_raw,axis=2)/np.nansum(tensor_weights,axis=2)
 
            reweighted_flatten=reweighted_tensor.flatten()
             
            outscore=reweighted_flatten.tolist()

            mutation_list=[]

            for i,n in enumerate(sequence):
                mutation_list.append(n+str(i+1)+"=")
                for j in range(1,21):
                    if alphabetAA_D_L[j]!=n:
                        mutation_list.append(n+str(i+1)+alphabetAA_D_L[j])
            print(sequence) 
        
        ### print output
        if args.contacts_only:
            if os.path.isdir(os.path.realpath(args.prefix)):
                pass
            else:
                os.mkdir(os.path.realpath(args.prefix))
            if args.output_name =='':
                output_name_c=os.path.join(os.path.realpath(args.prefix),str(metadata_parser['uniprot']+'_contact_maps'))
            else:
                output_name_c=os.path.join(os.path.realpath(args.prefix),str(args.output_name+'_contact_maps'))
            
            if fragmentation_check== False:
                np.savez(output_name_c,contacts=wt_contacts) 
            else:
                print('No output produced, sequence lenght exceed max allowed value (1024 aa)')

        else:
            if args.contacts_check:
                np.savez(output_name_c,contacts=wt_contacts[0,:,:],reference=ref_contacts,total_distcontacts=np.array([total_distcontacts]),precision=np.array([precision]))
                
                if precision <0.8:
                    write_to_log(logfile,'WARNING: '+args.uniprot+' evaluation completed,  contact precision ('+str(args.contacts_check)+'): '+str(precision))
                else:
                    write_to_log(logfile,args.uniprot+' evaluation completed,  contact precision ('+str(args.contacts_check)+'): '+str(precision))
            
            df_out=pd.DataFrame({'variant':mutation_list,str(args.scoring_strategy):outscore})
            col_list=[col for col in df_out.columns]

            generate_prism_output_parser(df_out,col_list,metadata_parser,os.path.abspath(output_name))

            if args.entropy and fragmentation_check==False:
                np.savez(output_name_e,entropy=entropy_seq)

            if args.wt_hidden==True and fragmentation_check==False:
                np.savez_compressed(output_name_h,logits=wt_h_layers['wt_logits'],attentions=wt_h_layers['wt_attentions'],representations=wt_h_layers['wt_representations'])
                #np.savez_compressed(output_name_h,logits=wt_h_layers['wt_logits'],representations=wt_h_layers['wt_representations'])
            if not args.uniprot is None:
                write_to_log(logfile,args.uniprot+' evaluation completed, no errors generated')
            else:
                write_to_log(logfile,'current evaluation completed, no errors generated')

    except Exception:
        error=traceback.format_exc()
        print(error)
        if not args.uniprot is None:
            write_to_log(logfile,'ERROR: '+args.uniprot+' evaluation stopped, error during the evaluations')
        else: 
            write_to_log(logfile,'ERROR: current  evaluation stopped, error during the evaluations')
        write_to_log(logfile,str(error))

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
