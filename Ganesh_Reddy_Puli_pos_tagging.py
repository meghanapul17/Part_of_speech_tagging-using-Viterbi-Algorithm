#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:59:37 2023

@author: meghanapuli
"""

import numpy as np
import matplotlib.pyplot as plt
sentence_list = []
with open('tagged_sentences.txt', 'r') as file:
    for sentence in file:
        sentence = sentence.split()
        sentence_list.append(sentence)
#print(sentence_list)        

word_tag_sequences = []

# word_tag_pairs_list is a list of word-tag pairs for each sentence
for word_tag_pairs in sentence_list:
    words = []
    tags = []
    for word_tag in word_tag_pairs:
        word, tag = word_tag.rsplit('/', 1) # word is everything before the last '/' and the tag is everything after the last '/'.
        words.append(word)
        tags.append(tag)
    word_tag_sequences.append((words, tags))
 
#print("word_tag_sequences:")
#print(word_tag_sequences)
count_req_for_words = ['family', 'guy', 'peter', 'griffin']
corresponding_counts = []

for each in count_req_for_words:
    count = 0
    for word_sequence, _ in word_tag_sequences:
        for word in word_sequence:
            if word == each:
                count += 1
    corresponding_counts.append(count)

#print(corresponding_counts)

plt.bar(count_req_for_words, corresponding_counts)
plt.ylabel('Counts')
plt.yticks(range(0, 201, 25))

for i, count in enumerate(corresponding_counts):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

plt.show()
print()

pos_tag_counts = {}

for _, tag_sequence in word_tag_sequences:
    for tag in tag_sequence:
        if tag in pos_tag_counts:
            pos_tag_counts[tag] += 1
        else:
            pos_tag_counts[tag] = 1

#print("POS tag counts:")
#print(pos_tag_counts)

exclude_tags = ["##", "$$"]
required_tag_counts = {tag: count for tag, count in pos_tag_counts.items() if tag not in exclude_tags}

plt.barh(list(required_tag_counts.keys()), list(required_tag_counts.values()))
plt.xlabel('Counts')
plt.xticks(range(0, 100001, 20000))

for i, count in enumerate(required_tag_counts.values()):
    plt.text(count + 0.1, i, str(count), va='center')

plt.show()

###############################################################
    
def part_of_speech_tagging(every_sentence):
    test_case = every_sentence
    print("\n", every_sentence)
    all_possible_words = set(test_case.split())
    all_possible_tags = set()
    
    for word_tags in word_tag_sequences:
        words, tags = word_tags
        #all_possible_words.update(words)
        all_possible_tags.update(tags)
    
    all_possible_words = list(set(test_case.split()))
    all_possible_tags = list(all_possible_tags)
    
    
    # Emission probability
    # Count the occurrences of each word-tag pair
    all_possible_word_tag_pairs = [(word, tag) for word in all_possible_words for tag in all_possible_tags]
    word_tag_counts = {}
    for word, tag in all_possible_word_tag_pairs:
        if (word, tag) not in word_tag_counts:
            word_tag_counts[(word, tag)] = 0
    
    for every_tuple in word_tag_sequences:
        for current_word, current_tag in zip(every_tuple[0], every_tuple[1]):
            if (current_word, current_tag) in word_tag_counts:
                word_tag_counts[(current_word, current_tag)] += 1
            else:
                word_tag_counts[(current_word, current_tag)] = 1
    
    #print("word_tag_counts:")
    #print(word_tag_counts)
    #print()
    
    # Calculate the emission probability for each word-tag pair
    word_tag_probs = {}
    for l in word_tag_counts:
        word_tag_probs[l] = np.log((word_tag_counts[l] + 1) / (pos_tag_counts[l[1]] + len(pos_tag_counts)))
        
    #print("word_tag_probabilities:")
    #print(word_tag_probs)
    
    # Transition probability
    # Count the occurrences of each tag pair
    tag_list = list(pos_tag_counts.keys())
    #print()
    #print("Tags list:")
    #print(tag_list)
    
    tag_pair_counts = {}
    for tag1 in tag_list:
        for tag2 in tag_list:
            tag_pair_counts[(tag1, tag2)] = 0
    
    for every_tuple in word_tag_sequences:
        tags = every_tuple[1]
        for i in range(len(tags) - 1):
            current_tag = tags[i]
            following_tag = tags[i + 1]
            if (following_tag, current_tag) in tag_pair_counts:
                tag_pair_counts[(following_tag, current_tag)] += 1
    
    #print("tag_pair_counts")
    #print(tag_pair_counts)
    
    # Calculate the transition probability for each tag pair
    tag_pair_probs = {}
    for current_tag_pair in tag_pair_counts:
        tag_pair_probs[current_tag_pair] = np.log((tag_pair_counts[current_tag_pair] + 1) / (pos_tag_counts[current_tag_pair[1]] + len(pos_tag_counts)))
    #print()
    #print("tag_pair_probabilities:")
    #print(tag_pair_probs)
    
    test_case_list = test_case.split()
    modified_test_case = test_case.split()
    modified_test_case.insert(0, "##")  # Add "##" as the 0th element
    modified_test_case.append("$$") 
    #print("\n",modified_test_case)
    M = len(test_case_list)
    #print("Total no. of words, M = ", M)
    m = list(range(M + 2))
    #print('m =', m)
    possible_tags = list(pos_tag_counts.keys())
    #print("Possible_tags: ",possible_tags)
    
    viterbi_dict = {}
    current_word = modified_test_case[1]
    #print('current word: ',current_word)
    
    # First for loop in the pseudo code
    viterbi_list = []
    for tag3 in possible_tags: 
            viterbi_dict[tag3] = (word_tag_probs.get((current_word, tag3))  + tag_pair_probs.get((tag3, '##')))
    
    viterbi_list.append(viterbi_dict)
    #print('\n',viterbi_list)  
    
    backpointer_list = []
    back_pointer_dict = {}
    for m in range(2, M + 1):
        current_word = modified_test_case[m]
        viterbi_new_dict = {}
        back_pointer_dict = {}
        for k in possible_tags:
            lst_to_find_max = []
            for k_prime in possible_tags:
                    local = (word_tag_probs.get((current_word, k)) + tag_pair_probs.get((k, k_prime)) + viterbi_list[m - 2][k_prime])
                    lst_to_find_max.append(local)
            #print(lst_to_find_max)
            #print(lst_to_find_max)
            local_high_score = max(lst_to_find_max)
            viterbi_new_dict[k] = local_high_score
            back_pointer_dict[k] = possible_tags[lst_to_find_max.index(local_high_score)]         
        #print(viterbi_new_dict)
        viterbi_list.append(viterbi_new_dict)
        backpointer_list.append(back_pointer_dict)
    #print('\nViterbi_variable_list:',viterbi_list)  
    #print('\nBack_pointer_list:',backpointer_list) 
    
    # step 7 of pseudo code
    last_word = modified_test_case[M + 1]
    #print(last_word)
    lst_to_find_max_score = []
    for k_new_prime in possible_tags:
        new_local = (word_tag_probs.get((last_word, '$$')) + tag_pair_probs.get(('$$', k_new_prime)) + viterbi_list[M - 1][k_new_prime])
        #print(new_local)
        lst_to_find_max_score.append(new_local)
    last_local_high_score = max(lst_to_find_max_score)
    #print(lst_to_find_max_score)
    #print(last_local_high_score)    
    
    predicted_tags_rev = []    
    last_predicted_tag = possible_tags[lst_to_find_max_score.index(last_local_high_score)]
    predicted_tags_rev.append(last_predicted_tag)
    #print('\nlast tag: ',predicted_tags_rev)
    
    
    for m in range(M - 1, 0, -1):
        prev_tag = predicted_tags_rev[len(predicted_tags_rev)-1]
        prev_tag = backpointer_list[m-1][prev_tag]
        predicted_tags_rev.append(prev_tag)
    #print('Predicted tags in reverse order: :', predicted_tags_rev)
    
    reversed_list = predicted_tags_rev[::-1]
    print('\n', reversed_list)

# Testing
test_sentences = ["nice !",
"good lord !",
"how are you ?",
"they can fish .",
"she is on diet .",
"we got kicked out .",
"there is no free food .",
"no need to worry about it at all !",
"how come the professor let him pass ?",
"maybe we shouldn't tell our parents that we didn't go .",
"how many people are there in the conference room ?",
"not sure if we can get there on time or not , so we should leave early .",
"speaking of that , we'll let you handle it .",
"remember : we are not here to kill time !",
"no offense but the result is not what we want . . .",
"2 examples are far from enough , you should try , at least , 3 more .",
"who would have thought we had to do remote learning for the past 2 years ? !",
"had the ball found the net , the goal would have been ruled out .",
"and the result of the analysis shows that people are currently not interested in buying their products .",
"since the final project is hard , we should ask the professor for help !"]

for every_sentence in test_sentences:
    part_of_speech_tagging(every_sentence)



