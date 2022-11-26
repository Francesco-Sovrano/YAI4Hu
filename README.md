# YAI4Hu
Explanatory AI for Humans (YAI4Hu, for short) is a novel pipeline of AI algorithms for the generation of user-centred explanations through the extraction of an explanatory space, intended as all the possible explanations (about something to be explained) reachable by a user through an explanatory process, via a pre-defined set of actions, i.e. open question answering and overviewing. This pipeline is meant to organise the information contained in non-structured documents written in natural language (e.g. web pages, pdf, etc.), allowing efficient information clustering, according to a pre-defined set of archetypal questions (e.g., why, what, how, who, etc.).

More details about this code can be found in the following papers:
> - ["Explanatory artificial intelligence (YAI): human-centered explanations of explainable AI and complex data"](https://doi.org/10.1007/s10618-022-00872-x)
> - ["Generating User-Centred Explanations via Illocutionary Question Answering: From Philosophy to Interfaces"](https://doi.org/10.1145/3519265)

To evaluate our algorithm, we ran a user-study (whose results are available at [user_study](user_study)) to compare the usability of the explanations generated through our novel pipeline against classical, one-size-fits- all, static XAI-based explanatory systems. The experiment consisted of explaining to more than 60 unique participants a credit approval system (based on a simple Artificial Neural Network and on CEM) and a heart disease predictor (based on XGBoost and TreeShap) in different ways, with different degrees of illocutionary power and different mechanisms for the user to ask their own questions explicitly.

In particular, we compare three different explanatory approaches:
- The 1st approach (Normal XAI-based Explainer; NXE for short) is just the output of a XAI.
- The 2nd explanatory approach (Overwhelming Static Explainer; OSE for short) adds to NXE further information (i.e., documentation), dumping on the user complex amounts of information, without any re-elaboration or explicit attempt to answer (implicit or not) questions.
- The 3rd approach (How-Why Narrator; HWN for short) re-elaborates the documentation of OSE organising it in interactive and simplified explanatory overviews that can be given on-demand to the user. The overviews of HWN focus on “how” and “why” archetypal questions, not allowing users to ask their own questions.
- The 4th approach (YAI4Hu) is an extension of HWN designed to have a much greater illocutionary power, answering also to implicit “what” questions and many other questions. Differently from the other systems, YAI4Hu empowers the users with the ability to ask their own questions via a special input box. 

These 4 explanatory tools were designed so that comparing their usability scores would indirectly allow us to isolate and measure the effects of illocution, implicit and explicit question answering, in the generation of user-centred explanations.
Regardless of the tool for explaining that we adopt, or the direction we take to produce explanation, with this software we prove that the usability (as per ISO 9241-210) of an explanatory process can be affected by its illocutionary power and goal-orientedness.

## Usage and Installation
This project has been tested on Debian 9 and macOS Mojave 10.14 with Python 3.7.9. 

The file system of this repository is organised as follows:
- In the [user_study](user_study) directory it is possible to find the results of the user studies discussed in the 2 aforementioned papers.
- The [code](code) contains the code of 8 different explainers: 4 for the Heart Disease Predictor [code/Heart Disease Prediction](code/Heart Disease Prediction) and 4 for the Credit Approval System [code/Credit Approval](code/Credit Approval). Within each directory it is possible to find a "setup.sh" script to install the software. To run an explainer, execute the following command ```./setup.sh port_num``` where port_num is the number of the port you want the explainer to available on. Once you have ran the "server.sh" script, you access the explainer through your browser at http://localhost:port_num .

**N.B.** Before being able to run the setup.sh scripts you have to install: virtualenv, python3-dev, python3-pip and make. 

## Software Details
YAI4Hu is a fully automatic explanatory tool, relying on pre-existing documentation about an AI system (i.e., generated by a XAI or manually created) to extract a special knowledge graph out of it for efficient answer retrieval. So that an explainee can ask questions about the content of the documentation or explore it by means of Aspect Overviewing. More specifically, Open Question Answering is implemented with an answer retrieval system, i.e., the system described in Sovrano et al. (2020a). Furthermore, also Aspect Overviewing is implemented with an answer retrieval system whose questions though are not asked by the explainee but are indeed instances of archetypal questions about the aspect to overview. So that the explainee can specify which aspect to overview and then get an explanation about it in the form of answers to a set of pre-defined archetypal questions (e.g., why is this aspect/concept important, what is this aspect/concept, etc.).
In YAI4Hu, through Aspect Overviewing, a user can navigate the whole explanatory space reaching explanations for every identified aspect of the explanandum.

In fact, every sentence presented to the user is annotated (as in Sovrano and Vitali 2021a) so that users can select which aspect to overview by clicking on the annotated syntagms. Annotated syntagms are clearly visible because they have a unique style that makes them easy to recognize.

After clicking on an annotation, a modal opens showing a card with the most relevant information about the aspect (see Fig. 3). This is in accordance with the relevance heuristic.
The most relevant information shown in a card is:
– A short description of the aspect (if available): abstract and type.
– The list of aspects taxonomically connected.
– A list o archetypal questions and their respective answers ordered by estimated
pertinence. Each piece of answer consists of an information unit and its summary.
All the information shown inside the modal is annotated as well. This means (for example) that clicking on the taxonomical type of the aspect, the user can open a new card (in a new tab) displaying relevant information about the type, thus being able to explore the explanatory space according to the abstraction policy.
On the other hand, the simplicity policy is ensured by the “More” and “Less” buttons (that allow to increase/decrease the level of detail of information) and by the fact that not all the words in the explanantia are linked to an overview despite being nodes of the explanatory space.

## Citation
This code is free. So, if you use this code anywhere, please cite:
- Francesco Sovrano and Fabio Vitali. Explanatory artificial intelligence (YAI): human-centered explanations of explainable AI and complex data. Data Min Knowl Disc (2022). https://doi.org/10.1007/s10618-022-00872-x
```
- Francesco Sovrano and Fabio Vitali. 2022. Generating User-Centred Explanations via Illocutionary Question Answering: From Philosophy to Interfaces. ACM Trans. Interact. Intell. Syst. 12, 4, Article 26 (December 2022), 32 pages. https://doi.org/10.1145/3519265
```

BitTeX citations:
- ```
@article{sovrano2022yai,
	Abstract = {In this paper we introduce a new class of software tools engaged in delivering successful explanations of complex processes on top of basic Explainable AI (XAI) software systems. These tools, that we call cumulatively Explanatory AI (YAI) systems, enhance the quality of the basic output of a XAI by adopting a user-centred approach to explanation that can cater to the individual needs of the explainees with measurable improvements in usability. Our approach is based on Achinstein's theory of explanations, where explaining is an illocutionary (i.e., broad yet pertinent and deliberate) act of pragmatically answering a question. Accordingly, user-centrality enters in the equation by considering that the overall amount of information generated by answering all questions can rapidly become overwhelming and that individual users may perceive the need to explore just a few of them. In this paper, we give the theoretical foundations of YAI, formally defining a user-centred explanatory tool and the space of all possible explanations, or explanatory space, generated by it. To this end, we frame the explanatory space as an hypergraph of knowledge and we identify a set of heuristics and properties that can help approximating a decomposition of it into a tree-like representation for efficient and user-centred explanation retrieval. Finally, we provide some old and new empirical results to support our theory, showing that explanations are more than textual or visual presentations of the sole information provided by a XAI.},
	Author = {Sovrano, Francesco and Vitali, Fabio},
	Da = {2022/10/10},
	Date-Added = {2022-11-26 10:23:36 +0000},
	Date-Modified = {2022-11-26 10:23:36 +0000},
	Doi = {10.1007/s10618-022-00872-x},
	Id = {Sovrano2022},
	Isbn = {1573-756X},
	Journal = {Data Mining and Knowledge Discovery},
	Title = {Explanatory artificial intelligence (YAI): human-centered explanations of explainable AI and complex data},
	Ty = {JOUR},
	Url = {https://doi.org/10.1007/s10618-022-00872-x},
	Year = {2022},
	Bdsk-Url-1 = {https://doi.org/10.1007/s10618-022-00872-x}
}
```
- ```@article{sovrano2022generating,
author = {Sovrano, Francesco and Vitali, Fabio},
title = {Generating User-Centred Explanations via Illocutionary Question Answering: From Philosophy to Interfaces},
year = {2022},
issue_date = {December 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {12},
number = {4},
issn = {2160-6455},
url = {https://doi.org/10.1145/3519265},
doi = {10.1145/3519265},
abstract = {We propose a new method for generating explanations with Artificial Intelligence (AI) and a tool to test its expressive power within a user interface. In order to bridge the gap between philosophy and human-computer interfaces, we show a new approach for the generation of interactive explanations based on a sophisticated pipeline of AI algorithms for structuring natural language documents into knowledge graphs, answering questions effectively and satisfactorily. With this work, we aim to prove that the philosophical theory of explanations presented by Achinstein can be actually adapted for being implemented into a concrete software application, as an interactive and illocutionary process of answering questions. Specifically, our contribution is an approach to frame illocution in a computer-friendly way, to achieve user-centrality with statistical question answering. Indeed, we frame the illocution of an explanatory process as that mechanism responsible for anticipating the needs of the explainee in the form of unposed, implicit, archetypal questions, hence improving the user-centrality of the underlying explanatory process. Therefore, we hypothesise that if an explanatory process is an illocutionary act of providing content-giving answers to questions, and illocution is as we defined it, the more explicit and implicit questions can be answered by an explanatory tool, the more usable (as per ISO 9241-210) its explanations. We tested our hypothesis with a user-study involving more than 60 participants, on two XAI-based systems, one for credit approval (finance) and one for heart disease prediction (healthcare). The results showed that increasing the illocutionary power of an explanatory tool can produce statistically significant improvements (hence with a P value lower than .05) on effectiveness. This, combined with a visible alignment between the increments in effectiveness and satisfaction, suggests that our understanding of illocution can be correct, giving evidence in favour of our theory.},
journal = {ACM Trans. Interact. Intell. Syst.},
month = {nov},
articleno = {26},
numpages = {32},
keywords = {explanatory artificial intelligence (YAI), Methods for explanations, education and learning-related technologies}
}```

Thank you!

## Support
For any problem or question please contact me at `cesco.sovrano@gmail.com`
