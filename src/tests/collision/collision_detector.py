#!/usr/bin/env python3
import cupy as cp
import cudf
from src.hashing.hash import hashing




def detect_collision(dataframe,column): 

    # Generate a smaller set of sequences for easier management
    num_sequences = 20_000_000
    sequence_length = 31
    char_codes = cp.array([ord('A'), ord('C'), ord('G'), ord('T')])

    # Generate random indices for character codes
    random_indices = cp.random.randint(0, len(char_codes), size=(num_sequences, sequence_length))

    # Map indices to character codes
    random_char_codes = char_codes[random_indices]

    # Transfer to CPU and convert codes to characters
    random_char_codes_cpu = cp.asnumpy(random_char_codes)
    random_sequences_str = [''.join(map(chr, row)) for row in random_char_codes_cpu]

    df =cudf.DataFrame({'sequence': random_sequences_str})


    df=hashing(df)

    unique_kmers_count    = len(set(df['sequence'].to_numpy()))
    unique_hash_ids_count = len(set(df['murmur3'].to_numpy()))


    if (unique_kmers_count==unique_hash_ids_count):
        print("No collision occured!")
    
    else: 
        print("Collision occured!")
        print(f"For {unique_kmers_count} kmers, {unique_kmers_count-unique_hash_ids_count} collsion occured!")




    
    
