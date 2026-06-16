import streamlit as st

@st.fragment           
def render():
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    st.subheader('Reading Session')
    st.text('''
I started the day by actually reading the Docs and taking notes on physical register for once. \
This was quite fun and informative and I want to log some of the core things I learned, stuff I want to \
apply on later days, and just cool tidbits I picked up. 
            
The core services of Tensorflow are discussed as follows:
''' 
    )

    with st.expander('**Tensors**'):
        st.markdown('''\
PyTorch looked at ```np.ndarrays``` and thought, "these are goated, but I wish they were even faster." They thus made their \
GPU-acceleratable replacement for numpy arrays: ```torch.tensor```. 
                    
It's really cool because not only does torch allow 1200 different operations on tensors, it also allows \
for memory operations like **page-locking host memory (*'Memory Pinning'*)** and explicitly sending objects to and from different devices \
with the ```.to(device)``` method.
                    
Another cool tidbit I picked up was that since ```np.ndarray``` and ```torch.tensor```, have very similar memory footprints \
there may be cases where type conversion is done with minimal memory overhead.
'''
        )

    with st.expander('**Datasets**'):
        st.markdown('''\
The ```torch.utils.data``` module comes with a few key classes, ```Dataset```, ```Sampler``` and ```DataLoader```.

---                  
##### Dataset Classes
```Dataset``` is an abstract class intended to act as an implementable interface for the rest of PyTorch's GPU utilization services. \
Usually, we talk about datasets in terms of tables or *'map indexed'* containers. Not only does this class support that, but also subclassed containers \
with only an ```__iter__()``` method.

Thus, the ```Dataset``` class comes with some utility subclasses, the most notable of which in the docs is ```IterableDataset```; \
this abstract class only really acts as a declaration that you shouldn't expect defined ```__getitem__()``` and ```__len()__``` functions. \
This stuff really isn't all that practically useful for my use case and only really applies to really large datasets, but its good to know.

---

##### Sampling Classes
Then, there comes the ```Sampler``` class. This basically acts as an additional layer to help whatever your concrete subclass was for ```Dataset```. \
It's only job is to control the order in which samples are drawn from the dataset. It is by all means an optional step, but it does have useful \
subclasses like ```RandomSampler``` and ```WeightedRandomSampler```.

---

##### The DataLoader Class
```DataLoader``` is PyTorch's intended interface for facilitating batch-drawing samples from a ```Dataset```. \
Unlike ```Dataset```, this class is a concrete one rather than an abstract class. Here's the main services it provides
- support for multithreaded parallel batch processing
- sampler integration
- a collation function (kind of like a preprocessing operation)
- support for memory pinning 

'''
        )

    with st.expander('**Deep Learning and AutoGrad**'):
        st.markdown('''\
The most notable things, but also the stuff I spent the least amount of time reading about in this session \
were ```torch.nn``` and ```torch.autograd```.
                    
---
           
#### torch.nn
When reading through the docs, I was a little surprised to note there were entire classes dedicated to RNNs, GRUs, LSTMs, etc. \
This made me want to stop reading and go back to reading about ```torch.utils.data``` because that was more interesting to me at the time. \

                
---

#### torch.autograd
This is definitely the most negligent on my part. I didn't really understand any of this in my reading session and only learnt about it later into the day.
'''
        )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader('Environment Setup')
    st.markdown('''\
I had grown a little tired of reading the docs so I wanted something hands-on, but as soon as I ran ```torch.cuda.is_available()```, I got \
a  ```False```. Now this was a bit confusing, but then when I compared with my previous project on the same system, I quickly realized that \
the last time I had done this sort of work, I used pip and a venv, this time: I was using conda.
                
To cut 3 hours of my life into a few sentences, let's just say there were two issues:
                
1. The ```torch``` and ```pytorch-cuda``` libraries were not speaking to each other. For this, after trying various versions, I just used pip in my conda ```environment.yml``` file.
1. My NVIDIA GeForce GTX 1080 has a Compute Capability (CC) of magnitude 6.1, while the least CC support in newer versions of ```torch``` was for CC >= 7.5
                
Eventually, after 2 hours of downloading the wrong versions of this ~1.3 GiB package, I lucked out and found a workable environment.
''')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader('A Simple DNN')
    st.markdown('''\
Since the humidity was real bad and I had a bit of a headache, I opted for something easy to call today done. I decided on a very simple 10x32x64x1 DNN \
to train for $f(x) = sum(x) > 5$.
                
The goal here was really only to read and understand every line of the training loop, the overall dataflow, and make sure GPU acceleration was \
working fine. So, I only trained on 100 samples but for 10,000 epochs. Not really a thought-on decision, just the first few values I wrote out.

The loss was basically imperceptible and it worked, which I mean of course it should - even a single logistic regression unit should be able to, \
but it was never really about the model, it was about PyTorch.
'''
    )
    st.text('So, what did I learn?', text_alignment='center', width='stretch')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.markdown('#### Memory and Performance')
    st.markdown('''\
1. I learned that ```.to(device)``` would become **a very common operation** as that was how you explicitly moved stuff to the GPU.
1. It was working quite well on the GPU, but when out of curiosity, I ran it on the CPU. Expecting a rapid decline in speed like in my prior \
projects with deep learning, I was surprised to find that it seemed to be just as quick!

This led me to benchmark the two of them, 
''')
    st.markdown('''\
and it turned out the **CPU actually _beats_ the GPU** by about 7.6 seconds.
''', text_alignment='center'
    )
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.markdown('#### CPU > GPU?') 
    st.markdown('''\
That doesn't sound quite right.
                
Well, if we look at the problem in question, there is a clear parallelization bottle neck: the 10,000 iteration sequential loop.\
Additionally, the batches were already quite small (x100 10d samples). And most notable, the **memory overhead** of moving everything from CPU to GPU \
is also worth noting.

It's an interesting find, but since this same GPU has definitely out-paced the CPU before, I believe this is just memory overhead catching up to me because the scope is too small.
'''
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.markdown('#### torch.autograd')
    st.markdown('''\
One thing that really stood out to me in the example code in the docs for the training loop was the lack of explicit parameter sharing.
```cpp
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

I was honestly a bit confused at first as to how the data flow was working, there was no X = X.new() line and everything felt implicit. \
This is what eventually led me to the following video.
''')
    st.video('https://www.youtube.com/watch?v=r1bquDz5GGA')
    st.markdown('''\
I had watched half of this video before, but stopped because the voice sounded kinda lame and even patronizing at times. \
But lowkey, after having built with these tools, I found the explanation for the DAG and AutoGrad that really cemented some of the dataflow for me.

Mainly, I learned that ```torch.optim.Adam(model.parameters())``` gave the optimizer a reference to the parameters along with their gradient info. \
It was pretty helpful, especially because the headache of mine had grown worse and I wanted time away from my desk.
                
I learned also about how you can do this stuff without the supervision of ```torch.nn.module``` with the ```requires_grad``` flag in the ```torch.tensor``` constructor.
Overall, a pretty good video.
'''
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.markdown('#### Ideas for the Future')
    st.markdown('''
To close this off, I will just mention an idea that came to me regarding neural net training. 
''')

    st.caption('Dated: 06/17/26', text_alignment='right')
