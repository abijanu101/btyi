import streamlit as st

@st.fragment           
def render():
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("Dataset and Pretrained Embeddings")
    st.markdown('''\
    - I chose XSUM for my training data
    - I found out there's something called GloVe and I wanted to use it
    - I checked their website out, there were a fair few errors
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("GloVe Embeddings")
    st.markdown('''\
    - I downloaded one of the larger embeddings and tried to parse it with pandas, it didn't work
    - I read a snippet with open() and learned it was space separated and didn't want to write my own parser
    - I learned about ```torchtext.vocab.GloVe``` and eventually caved
    - I didn't know what stoi and itos stood for and I confused them with ```atoi``` and ```itoa``` from C and neglected them, then I spent hours trying to find stoi and itos
    - I decided torchtext was a piece of garbage and that I needed to just read it as a pandas dataframe with sep=' '
    - This took ages because the file is huge, but also because there are certain tokens WITH spaces in them asw
    - I wrote my own manual parser, it took about 40-50s to parse the file everytime while torchtext took about 20-30s
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("OOP-based Redesign")
    st.markdown('''\
    - A day had passed and the code was growing a little unmanageable, so I decided to make a GloVe handling class
    - When debugging, one of the LLMs I was using to help make sense of things used the terms ```stoi``` and ```itos``` and suddenly I realized how stupid I am.
    - I switched out my own parser for the torchtext.vocab.GloVe class, so now my class was essentially a wrapper that added Control Tokens
    - I chose <s> and </s> as my start of signal and end of signal control tokens, but then I found that GloVe actually has those tokens already for strikethrough markdown.
    - I replaced them with the symbols <START> and <END>
    - I then made even more classes and lowkey had a lot of fun OOP maxxing, most of my other big projects have been functionally written 
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("Datasets and Collation")
    st.markdown('''\
    - Another day had nearly come to an end and I now realized all that stuff I read about on Day 2 might actually be needed
    - It was
    - I wrote my first collate_fn and made use of a DataLoader and quite liked it
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("Seq2Seq Implementation")
    st.markdown('''\
    - I skipped a day and so on Day 6, I actually started implementation of the Encoder, Decoder, and the Seq2Seq classes
    - My nephews were over, I was sick, and so I got overstimmed and stopped working early
    - I still got the major implementation work done, on Sunday should be good for training
    - I skipped working on Sunday because of family commitments :P
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("Overflow into Week 2 Day 1")
    st.markdown('''\
    - I realized now the first task of week 2 was lowkey stupid
    - It was another seq2seq model, this time for Machine Translation
    - So, even though I hadn't yet made a monolingual model, I was supposed to make a multilingual one 
    - I also felt that Bi-LSTMs specifically were a bit silly here asw, what does bidirectionality even mean when you are autoregressing
    - I just thought if anything, I'd get one model done and have it perform well
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("GPU Acceleration")
    st.markdown('''\
    - I had woken up early and was very sleepy so early work was slow on this day, plus the desires vs reality was really hitting
    - I spent like 3 hours of debugging to get GPU acceleration working properly
    - Unlike Day 2, GPU acceleration ACTUALLY made a really big difference this time and that was pretty cool
''')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("Evaluation Metrics")
    st.markdown('''\
    - I first tried to use torchtext's inbuilt BLEU score, but that for some reason expected same length sequences and I would have to add padding.
    - I then switched to the ```evaluate``` library and then realized it needed text
    - I didn't want to actually decode from index sequences into text but eventually I caved
    - All of a sudden Windows Smart App controls started blocking my python imports
    - Apparently my reinstall of ```evaluate``` wasnt signed properly? I was too tired to care so I just tried reinstalling my entire environment
    - Didn't change anything, only wasted more time
    - Finally, I just switched to sacrebleu so I could finish the day with an actual training cycle done.
''')


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("Training")
    st.markdown('''\
    - I ran into a lot of Out of Memory errors for my poor old 8.0GiB GPU
    - I ran the model with a smaller vocabulary first because I knew that the 2M vocab + the softmax prediction layer would be really problematic for scaling
    - The loss was going down pretty quick but when I interrupted and ran a .predict() on an arbitray string, I lowkey would get <START> <UNK> <UNK> <UNK> ... for everything
    - The vocabulary was small enough that this was a viable loss-minimizing strategy
    - I shrunk the context vector dimensionality and expanded the vocabulary because I wanted to see both extremes first and then slowly narrow down in a Binary Search fashion
    - This still got me lots of OOM errors
    - I shrunk the context vector more than I was comfortable 512 -> 4 just so I could see SOMETHING different and eventually also set the vocab to 250k instead of the full 2M
    - I then quickly put my context vector size ack up to a fairly small 128d
    - This time though the Loss didnt go below 1 as quickly, it was definitely learning in a healthy pattern going down one unit every 1000 or so batches
    - This time when I ran it on the lyrics for heylog - 'carnage', it just started spamming <START> <START> <START> ... until it hit the max summary limit.

    The clock had struck 7pm and I had to now leave my PC for the day.
''')


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader("What even bro")
    st.markdown('''\
        I have had quite some time to mull over the failures of the summarizer and I have since restructured my plan.
    
        In fact, tommorrow, on Day 9 of btyi, I have planned some measures that you can check out in the corresponding entry.
''')
    st.caption('Dated: 06/23/26', text_alignment='right')
