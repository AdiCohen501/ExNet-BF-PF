# ExNet-BF-PF: Explainable DNN-based Beamformer with Postfilter

<div align="center">

[Paper]() |
[Demo Page](https://exnet-bf-pf.github.io/) |
[Abstract](#Abstract) |
[Pretrained Models](#pretrained-models) |
[Inference](#inference) 

</div>

![](ExNet_BF_PF_Net.PNG)

## Authors
Adi Cohen, Daniel Wong, Jung-Suk Lee and Sharon Gannot

## Abstract
This paper introduces an explainable DNN-based beamformer with a postfilter (ExNet-BF+PF) for 
multichannel signal processing. Our approach combines the U-Net network with a beamformer 
structure to address this problem. The method involves a two-stage processing pipeline. In the
first stage, time-invariant weights are applied to construct a multichannel spatial filter, namely
a beamformer. In the second stage, a time-varying single-channel post-filter is applied at the 
beamformer output. Additionally, we incorporate an attention mechanism inspired by its successful 
application in noisy and reverberant environments to improve speech enhancement further.
Furthermore, our study fills a gap in the existing literature by conducting a thorough spatial 
analysis of the network's performance. Specifically, we examine how the network utilizes spatial 
information during processing. This analysis yields valuable insights into the network's 
functionality, thereby enhancing our understanding of its overall performance.
Experimental results demonstrate that our approach is not only straightforward to train but also 
yields superior results, obviating the necessity for prior knowledge of the speaker's activity.

## Demo Page
The [demo page](https://exnet-bf-pf.github.io/) presents audio test samples along with their spectrograms, showcasing a comparison between our proposed method, "ExNet-BF+PF", and other competing methods. 
