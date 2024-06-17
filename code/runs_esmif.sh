python3 code/run_esmif_complete.py \
--gpunumber [number] \
--prefix ESM_results/ESM-IF/v2 [output folder] \
--sequence [original sequence] \
--structure [input structure file]

# 1fb0
python3 code/run_esmif_complete.py \
--gpunumber 8 \
--prefix data/ESM_results/ESM-IF/v4_replication \
--sequence VQDVNDSSWKEFVLESEVPVMVDFWAPWCGPCKLIAPVIDELAKEYSGKIAVYKLNTDEAPGIATQYNIRSIPTVLFFKNGERKESIIGAVPKSTLTDSIEKYL \
--structure data/AF2_structures/v4/AF2_config/1FB0/single_sequence/1fb0_prediction.pdb

# Automated run
python3 code/runs_esmif.py


# GFP
python3 code/run_esmif_complete.py \
--gpunumber 8 \
--prefix data/ESM_results/ESM-IF/v4/AF2_UniProt/GFP \
--sequence MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK \
--structure data/AF2_structures/v4/AF2_UniProt/GFP.pdb

# Runs - with structure of dF106 with 5J7D as a template
# dF106
python3 code/run_esmif_complete.py \
--gpunumber 8 \
--prefix data/ESM_results/ESM-IF/v4_replication3/AF2_config/dF106/5J7D/single_sequence \
--sequence VLDVTKDHWLLYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEDKSVLIGAIPKDQLRDEILKYL \
--structure data/AF2_structures/v4/AF2_config/5J7D/single_sequence/dF106_prediction.pdb

# edF106
python3 code/run_esmif_complete.py \
--gpunumber 8 \
--prefix data/ESM_results/ESM-IF/v4 \
--sequence VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEVKSVLIGAIPKDQLRDEILKYL \
--structure data/AF2_structures/v4/AF2_config_seq=edF106_template=5J7D_MSA=single_sequence.pdb

# MM9
python3 code/run_esmif_complete.py \
--gpunumber 8 \
--prefix data/ESM_results/ESM-IF/v4 \
--sequence VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIRVFYVDINKAPEIAKKYNITTTPTVAFFHNGELKSVFTGAITKDQLRDEILKYL \
--structure data/AF2_structures/v4/AF2_config_seq=edF106_template=5J7D_MSA=single_sequence.pdb

# eMM9
python3 code/run_esmif_complete.py \
--gpunumber 8 \
--prefix data/ESM_results/ESM-IF/v4 \
--sequence VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIKVFVVDINKAPEIAKKYNITTTPTVAFFKNGELKSVFTGAISKDQLRDEILKYL \
--structure data/AF2_structures/v4/AF2_config_seq=edF106_template=5J7D_MSA=single_sequence.pdb
