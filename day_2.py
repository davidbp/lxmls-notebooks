
# coding: utf-8

# In[1]:

cd lxmls-toolkit-student/labs/day_2


# In[2]:

get_ipython().magic(u'matplotlib inline')


# In[3]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[4]:

import sys
sys.path.append('/Users/davidbuchacaprats/Dropbox/lxmls2015/lxmls-toolkit-student')
import lxmls
import scipy
path_inside_lxmls_toolkit_student = "/Users/davidbuchacaprats/Dropbox/lxmls2015/lxmls-toolkit-student"


# In[5]:


####################################################################################################
# Ex 2.1 ###########################################################################################
####################################################################################################


# In[6]:

import lxmls.readers.simple_sequence as ssr

simple = ssr.SimpleSequence()
print simple.train
print "\n"
print simple.test


# In[7]:


for sequence in simple.train.seq_list: print sequence

for sequence in simple.train.seq_list:
    print sequence.x


# In[8]:

####################################################################################################
# Ex 2.2 ###########################################################################################
####################################################################################################


# In[9]:


import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr
simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)
print "Initial Probabilities:\n", hmm.initial_probs ,"\n"
print "Transition Probabilities:\n", hmm.transition_probs ,"\n"
print "Final Probabilities:\n", hmm.final_probs ,"\n"
print "Emission Probabilities\n", hmm.emission_probs


# In[10]:

####################################################################################################
# Ex 2.3 ###########################################################################################
####################################################################################################


# In[11]:

import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr
simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)


# In[12]:


initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])

print "initial_scores;\n", initial_scores, "\n"
print "transition_scores: \n",transition_scores, "\n"
print "final_scores:\n", final_scores, "\n"
print "emission_scores:\n", emission_scores, "\n"


# In[12]:




# In[13]:

####################################################################################################
# Ex 2.4 ###########################################################################################
####################################################################################################


# In[14]:


import numpy as np
a = np.random.rand(10)
print np.log(sum(np.exp(a)))
print np.log(sum(np.exp(10*a)))
print np.log(sum(np.exp(100*a)))
print np.log(sum(np.exp(1000*a)))

print "\n"
from lxmls.sequences.log_domain import logsum
print logsum(a)
print logsum(10*a)
print logsum(100*a)
print logsum(1000*a)


# In[15]:

####################################################################################################
# Ex 2.5 ###########################################################################################
####################################################################################################


# In[16]:


import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr
simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)
initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])


# In[17]:

log_likelihood, forward = hmm.decoder.run_forward(initial_scores, transition_scores,final_scores, emission_scores)
print "forward:\n", forward, "\n"
print '\n Log-Likelihood =', log_likelihood


# In[18]:

a =hmm.decoder.run_forward(initial_scores, transition_scores,final_scores, emission_scores)


# In[19]:

log_likelihood, backward = hmm.decoder.run_backward(initial_scores, transition_scores, final_scores, emission_scores)
print "backward :\n", backward, "\n"
print 'Log-Likelihood =', log_likelihood


## Exercise 2.6 

# In[20]:

#  Compute the node posteriors for the first training sequence 
#  (use the provided compute posteriors func- tion), and look at the output. 
#  Note that the state posteriors are a proper probability distribution 
#  (the lines of the result sum to 1).


# In[21]:

import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr
simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)
initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])
state_posteriors, transition_posteriors, log_likelihood = hmm.compute_posteriors(initial_scores, transition_scores, final_scores, emission_scores)


# In[22]:

print state_posteriors


##  Exercise 2.7 

#### Run the posterior decode on the first test sequence, and evaluate it.

# In[23]:

simple = ssr.SimpleSequence()

hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train)
initial_scores, transition_scores, final_scores, emission_scores = hmm.compute_scores(simple.train.seq_list[0])

y_pred = hmm.posterior_decode(simple.test.seq_list[0  ])
print "Prediction test 0:\n\t", y_pred, "\n"
print "Truth test 0:\n\t", simple.test.seq_list[0]


# Do the same for the second test sentence

# In[24]:

y_pred = hmm.posterior_decode(simple.test.seq_list[1])
# There are nan values in the backward and forward probabilites caused by
# not having observed tennis

print "Prediction test 1:"
print y_pred
print "Truth test 1:"
print simple.test.seq_list[1]


# What is wrong?
# 
# Note the observations for the second test sequence: the observation tennis was never seen at training time, so the probability for it will be zero (no matter what state). This will make all possible state sequences have zero probability. As seen in the previous lecture, this is a problem with generative models, which can be corrected using smoothing (among other options).
# 

# Change the train supervised method to add smoothing:
# ```
#    def train_supervised(self,sequence_list, smoothing):
# ```

# In[45]:

hmm.train_supervised(simple.train, smoothing=0.1)
y_pred = hmm.posterior_decode(simple.test.seq_list[0])
print "\n"
print "Prediction test 0 with smoothing:"
print "\t",y_pred 
print "Truth test 0:"
print "\t",simple.test.seq_list[0]
y_pred = hmm.posterior_decode(simple.test.seq_list[1])
print "\n"
print "Prediction test 1 with smoothing:"
print "\t",y_pred
print "Truth test 1:"
print "\t",simple.test.seq_list[1]


## Exercise 2.8

# Implement a method for performing Viterbi decoding
# in file sequence ```classification_decoder.py.```
# 

# In[29]:

import lxmls.sequences.hmm as hmmc
import lxmls.readers.simple_sequence as ssr
simple = ssr.SimpleSequence()
hmm = hmmc.HMM(simple.x_dict, simple.y_dict)
hmm.train_supervised(simple.train, smoothing=0.1)


# Now use ```hmm.viterbi_decode``` to predict the sequence of tags for a given sequence of visible states. 
# 
# Notice that ```hmm.viterbi_decode``` ( which can be located in ```sequences/sequence_classifier```) uses the method ```SequenceClassificationDecoder.run_viterbi``` (from the folder ```sequences/sequence_classification_decoder```).

# In[51]:

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[0])
print "Viterbi decoding Prediction test 0 with smoothing:\n\t", y_pred, "\nscore:\n\t",score

print "\nA correct implementation of Viterbi decoding Prediction test 0 with smoothing"
print "should return:"
print "\t", "walk/rainy walk/rainy shop/sunny clean/sunny "
print "score:"
print "\t",-6.02050124698


# In[40]:

print "Truth test 0:\n\t", simple.test.seq_list[0]


# In[62]:

y_pred, score = hmm.viterbi_decode(simple.test.seq_list[1])
print "Viterbi decoding Prediction test 1 with smoothing:\n\t", y_pred
print "score:"
print "\t",score

print "\nA correct implementation of Viterbi decoding Prediction test 0 with smoothing"
print "should return"
print "\t",simple.test.seq_list[1] 
print "score:"
print "\t",-11.713974074


## Part of Speech Tagging (POS)

# In[90]:

import lxmls.readers.pos_corpus as pcc
corpus = pcc.PostagCorpus()
path_to_data = path_inside_lxmls_toolkit_student + "/data/"
train_seq = corpus.read_sequence_list_conll(path_to_data + "train-02-21.conll",max_sent_len=15,max_nr_sent=1000)
test_seq = corpus.read_sequence_list_conll(path_to_data + "test-23.conll",max_sent_len=15,max_nr_sent=1000)
dev_seq = corpus.read_sequence_list_conll(path_to_data + "dev-22.conll",max_sent_len=15,max_nr_sent=1000)


# In[91]:

hmm = hmmc.HMM(corpus.word_dict, corpus.tag_dict)
hmm.train_supervised(train_seq)
hmm.print_transition_matrix()


# Look at the transition probabilities of the trained model,
# and see if they match your intuition about the English language 
# (e.g. adjectives tend to come before nouns). Each column is the previous state and row is the current state. Note the high probability of having Noun after Determinant or Adjective, or of having Verb after Nouns or Pronouns, as expected.

### Exercise 2.9

# Test the model using both posterior decoding and Viterbi decoding on
# both the train and test set, using the methods in class HMM

# In[92]:

viterbi_pred_train = hmm.viterbi_decode_corpus(train_seq) 
posterior_pred_train = hmm.posterior_decode_corpus(train_seq)
eval_viterbi_train = hmm.evaluate_corpus(train_seq, viterbi_pred_train)
eval_posterior_train = hmm.evaluate_corpus(train_seq, posterior_pred_train)
print "Train Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(eval_posterior_train,eval_viterbi_train)


# In[93]:

viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq) 
posterior_pred_test = hmm.posterior_decode_corpus(test_seq) 
eval_viterbi_test = hmm.evaluate_corpus(test_seq,viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq,posterior_pred_test) 
print "Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(
    eval_posterior_test,eval_viterbi_test)


# 
# - What do you observe? 
# 
# Remake the previous exercise but now train the HMM using smoothing.
# Try different values (0,0.1,0.01,1) and report the results on the train and 
# development set. (Use function pick best smoothing).

# In[95]:

best_smoothing = hmm.pick_best_smoothing(train_seq, dev_seq, [10,1,0.1,0])


# In[98]:

hmm.train_supervised(train_seq, smoothing=best_smoothing)
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq)
eval_viterbi_test = hmm.evaluate_corpus(test_seq, viterbi_pred_test) 
eval_posterior_test = hmm.evaluate_corpus(test_seq, posterior_pred_test)
print "Best Smoothing %f -- Test Set Accuracy: Posterior Decode %.3f, Viterbi Decode: %.3f"%(best_smoothing,eval_posterior_test,eval_viterbi_test)


#  Perform some error analysis to understand were the errors are coming from. 
# 
#  You can start by visualizing the confusion matrix (true tags vs predicted tags). 
# 
# You should get something like what is shown in Figure 2.5.

# In[101]:

import lxmls.sequences.confusion_matrix as cm
import matplotlib.pyplot as plt
confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, viterbi_pred_test, len(corpus.tag_dict), hmm.get_num_states()) 
cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict, xrange(hmm.get_num_states()), 'Confusion matrix')
plt.show()


##  Unsupervised Learning of HMMs

# > If you have made it so far you are awesome! 
# 
# > Don't worry the next couple of exercices do not require to actually code anything.

# We next address the problem of unsupervised learning. In this setting, 
# we are not given any labeled data.
# 
# All we get to see is a set of natural language sentences. 

### Exercise 2.10

# Implement the method to update the counts given the state and transition posteriors
# 
# ```
# def update_counts(self, sequence, state_posteriors, transition_posteriors):
# ```
# 
# Look at the code for EM algorithm in file ```sequences/hmm.py``` and check it for yourself.

# In[104]:

def train_EM(self, dataset, smoothing=0, num_epochs=10, evaluate=True):
    self.initialize_random()

    if evaluate:
        acc = self.evaluate_EM(dataset)
        print "Initial accuracy: %f"%(acc)

    for t in xrange(1, num_epochs):
        #E-Step
        total_log_likelihood = 0.0
        self.clear_counts(smoothing)
        for sequence in dataset.seq_list:
            # Compute scores given the observation sequence.
            initial_scores, transition_scores, final_scores, emission_scores =                 self.compute_scores(sequence)

            state_posteriors, transition_posteriors, log_likelihood =                 self.compute_posteriors(initial_scores,
                                        transition_scores,
                                        final_scores,
                                        emission_scores)
            self.update_counts(sequence, state_posteriors, transition_posteriors)
            total_log_likelihood += log_likelihood

        print "Iter: %i Log Likelihood: %f"%(t, total_log_likelihood)
        #M-Step
        self.compute_parameters()
        if evaluate:
             ### Evaluate accuracy at this iteration
            acc = self.evaluate_EM(dataset)
            print "Iter: %i Accuracy: %f"%(t,acc)


### Exercise 2.11 

# Run 20 epochs of the EM algorithm for part of speech induction:
# 

# In[105]:

hmm.train_EM(train_seq, 0.1, 20, evaluate=True)
viterbi_pred_test = hmm.viterbi_decode_corpus(test_seq)
posterior_pred_test = hmm.posterior_decode_corpus(test_seq) 
eval_viterbi_test = hmm.evaluate_corpus(test_seq, viterbi_pred_test)
eval_posterior_test = hmm.evaluate_corpus(test_seq, posterior_pred_test)


# In[106]:

confusion_matrix = cm.build_confusion_matrix(test_seq.seq_list, viterbi_pred_test, len(corpus.tag_dict), hmm.get_num_states())

cm.plot_confusion_bar_graph(confusion_matrix, corpus.tag_dict, xrange(hmm.get_num_states()), 'Confusion matrix')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



