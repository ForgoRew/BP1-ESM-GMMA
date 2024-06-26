# Sequences
All sequences which were used in the project or can be used for project follow up:
```fasta
>1fb0A - the one from Kristoffer's file (Active_site=[29-33])
VQDVNDSSWKEFVLESEVPVMVDFWAPWCGPCKLIAPVIDELAKEYSGKIAVYKLNTDEAPGIATQYNIRSIPTVLFFKNGERKESIIGAVPKSTLTDSIEKYL

> Thioredoxin, UNIPROT_ID=P07591 (Pos. 76-179 = 1fb0, Active_site=[104-108])
MAIENCLQLSTSASVGTVAVKSHVHHLQPSSKVNVPTFRGLKRSFPALSSSVSSSSPRQFRYSSVVCKASEAVKEVQDVNDSSWKEFVLESEVPVMVDFWAPWCGPCKLIAPVIDELAKEYSGKIAVYKLNTDEAPGIATQYNIRSIPTVLFFKNGERKESIIGAVPKSTLTDSIEKYLSP

>5j7dD (=dF106)
VLDVTKDHWLLYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEDKSVLIGAIPKDQLRDEILKYL

>1FB0A_rrx1_0006 -250.291 - the same as 5j7dD
VLDVTKDHWLLYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEDKSVLIGAIPKDQLRDEILKYL

>edF106 (dF106+L11P+D83V)
VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEVKSVLIGAIPKDQLRDEILKYL

>MM9
VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIRVFYVDINKAPEIAKKYNITTTPTVAFFHNGELKSVFTGAITKDQLRDEILKYL

>eMM9
VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIKVFVVDINKAPEIAKKYNITTTPTVAFFKNGELKSVFTGAISKDQLRDEILKYL

>2trxA
IIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANL
```

# MSA of the sequences
MSA itself
```
CLUSTAL O(1.2.4) multiple sequence alignment

dF106       VLDVTKDHWLLYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKA	60
edF106      VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKA	60
MM9         VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIRVFYVDINKA	60
eMM9        VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIKVFVVDINKA	60
1fb0        VQDVNDSSWKEFVLESEVPVMVDFWAPWCGPCKLIAPVIDELAKEYSGKIAVYKLNTDEA	60
2trxA       IIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQN	60
            : .:... :   ** ::  ::* *       .* :  :: *:*.*:.* : *  :: :: 

dF106       PEIAKKYNITTTPTVAFFKNGEDKSVLIGAIPKDQLRDEILKYL	104
edF106      PEIAKKYNITTTPTVAFFKNGEVKSVLIGAIPKDQLRDEILKYL	104
MM9         PEIAKKYNITTTPTVAFFHNGELKSVFTGAITKDQLRDEILKYL	104
eMM9        PEIAKKYNITTTPTVAFFKNGELKSVFTGAISKDQLRDEILKYL	104
1fb0        PGIATQYNIRSIPTVLFFKNGERKESIIGAVPKSTLTDSIEKYL	104
2trxA       PGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANL	104
            *  * :*.*   **: :*:***      **: *. * : :   *
```

Input for the MSA
```fasta
>1fb0
VQDVNDSSWKEFVLESEVPVMVDFWAPWCGPCKLIAPVIDELAKEYSGKIAVYKLNTDEAPGIATQYNIRSIPTVLFFKNGERKESIIGAVPKSTLTDSIEKYL
>dF106
VLDVTKDHWLLYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEDKSVLIGAIPKDQLRDEILKYL
>edF106
VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIMVFELDTNKAPEIAKKYNITTTPTVAFFKNGEVKSVLIGAIPKDQLRDEILKYL
>MM9
VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIRVFYVDINKAPEIAKKYNITTTPTVAFFHNGELKSVFTGAITKDQLRDEILKYL
>eMM9
VLDVTKDHWLPYVLLAQLPVMVLFRKDNDEEAKKVEYIVRELAQEFDGLIKVFVVDINKAPEIAKKYNITTTPTVAFFKNGELKSVFTGAISKDQLRDEILKYL
>2trxA
IIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANL
```