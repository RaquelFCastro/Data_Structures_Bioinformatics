## TODO: Include here the Python standard library modules you need.


## Part I

### T1

from matplotlib import lines


def readFASTA(filename):
    genes= {}
    with open(filename, 'r', encoding='utf-8-sig') as f:
        ID = None
        seq= []
        desc= ''
        for line in f:
            line = line.strip('\n') #not necessary, only if we had done line=f.readlines()
            if line.startswith('>'):
                if ID is not None:
                    genes[ID]= (desc, ''.join(seq))
                ID, desc = line[1:].split(maxsplit=1)
                seq=[]
            else:
                seq.append(line)
        if ID is not None:
            genes[ID]= (desc, ''.join(seq)) 
    return genes       
        
print(readFASTA('data_sequences/J02459.1.cds.fasta'))   


### T2

def seqComposition(seq, type):
    count = {}
    seq = seq.upper()
    type = type.upper()
    cat = {'DNA':['A', 'G', 'C', 'T'], 'RNA':['A', 'G', 'U', 'C'], 'PROT':["A","R","N","D","C","E","Q","G","H","I",
    "L","K","M","F","P","S","T","W","Y","V"]}
    for c in seq:
        if c in cat[type]: 
            count[c] = 1 + count.get(c, 0)

    return count

#print(seqComposition('abcvfrg', 'prot'))

def dnaGCcontent(seq):
    seq=seq.upper()
    GC=0
    list=['G','C']
    for c in seq:
        if c in list:
            GC+=1
    return GC/len(seq)*100    

print(dnaGCcontent('gcatat'),'% GC composition')


## Parte II

### T3

def transcribeDNA2RNA(dna_seq):
    dna_seq = dna_seq.upper()
    transcript= {'A':'U', 'T':'A','C':'G', 'G':'C'}
    RNA=[]
    for i in dna_seq:
        RNA+=transcript[i]
    return ''.join(RNA)

print(transcribeDNA2RNA('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG'))
    
### T4

def translateRNA2protein(seq_rna): 
    seq_rna=seq_rna.upper()
    tRNA={}
    with open('data_sequences/genetic_code.txt', 'r', encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('AAs'):
                AA = line.split('=')[1].strip()
            elif line.startswith('Base1'):
                base1 = line.split('=')[1].strip()
            elif line.startswith ('Base2'):
                base2 = line.split('=')[1].strip()
            elif line.startswith ('Base3'):
                base3 = line.split('=')[1].strip()
            
        for i in range(len(AA)):
            codon=base1[i]+base2[i]+base3[i]
            tRNA[codon]=AA[i]
        
        read=[]

        for i in range(0,len(seq_rna), 3):
            read.append(seq_rna[i:i+3])
        
        prot=[]

        for c in read:
            prot.append(tRNA.get(c,""))
    
    return ''.join(prot)

print(translateRNA2protein('AUGAUGUGA'))
    

### T5

def reverseComplement(dna_seq):
    dna_seq = dna_seq.upper()
    dict= {'A':'T', 'T':'A','C':'G', 'G':'C'}
    comp=[]
    for i in dna_seq:
        comp+=dict[i]
    return ''.join(comp[::-1])

print(reverseComplement('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCGATAG'))


## Parte III

### T6 

import re

def findMotif(dna_seq, motif):
    
    dna_seq=dna_seq.upper()
    motif=motif.upper()
    list=[]
    
    for i in re.finditer(motif,dna_seq):
        list.append(i.start())
    
    return list
            
print(findMotif('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG','GCC'))  

### T7

def mostFrequentKMotifs(dna_seq,k):
    dna_seq=dna_seq.upper()
    kmers=[]
    for c in range(len(dna_seq)-k+1):
        kmers.append(dna_seq[c:c+k])

    countkmer={}
    for i in kmers:
        countkmer[i] = 1 + countkmer.get(i,0)
    
    result= dict(sorted(countkmer.items(),key=lambda item:item[1], reverse=True))
    
    maxkmer = max(result.values())

    final_list=[]
    for kmer, count in result.items():
        if count==maxkmer:
            final_list.append((kmer,count))
    
    return final_list

print(mostFrequentKMotifs('ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCGATAG',3))

## Parte IV

### T8

def highestGC(filename):
    fasta_file = readFASTA(filename)
    
    GC_list=[]
    for ID, rest in fasta_file.items():
        GC_content=dnaGCcontent(rest[1])
        GC_list.append((ID,GC_content))
    
    return sorted(GC_list, key=lambda x:x[1], reverse=True)[0]

print(highestGC('data_sequences/J02459.1.cds.fasta'))

### T9

def compositionMatrix():
    return None

### T10

def indexSpeciesGC (filename):
    
    results_fasta = readFASTA(filename)

    IDS=set()
    for key in results_fasta:
        IDS.add(key)
    
    GC=[]
    for values in results_fasta.values():
        GC.append(values[-1])
    
    GC_content= f"{dnaGCcontent(''.join(GC))/100:.4f}"

    result={}
    result['ids']=IDS
    result['gc_medio']=GC_content

    return result  

print(indexSpeciesGC('data_sequences/J02459.1.cds.fasta'))
