
 Intro
---------------------------------

These are the TempEval-3 platinum corpus TimeML annotations.
They consist of twenty unicode documents annotated with TimeML 
events, timexes and tlinks.


 Contents
---------------------------------
 te3-platinum/              This directory
 te3-platinum/README        This file
 te3-platinum/*.tml         The TE3 Platinum temporal annotations
 te3-platinum/src           Source annotations from annotators
 te3-platinum/src/tipsem    Annotations from TIPSem, used to support 
                             some annotators
 te3-platinum/tools         Tools used for evaluating TE3 results


 Copyright
---------------------------------
(c) 2013 TempEval-3 Co-ordinators

Annotations licensed as Creative Commons Attribution 3.0 Unported


 Credits
---------------------------------
Annotators and contributors, in alphabetical order, to whom we
are extraordinarily grateful.

James Allen
Leon Derczynski
Hector Llorens
Polina Malamud
James Pustejovsky
Sophie Pustejovsky
Seraphina Pyle
Naushad UzZaman
Marc Verhagen
Zachary Yocum


 References
---------------------------------
Naushad UzZaman, Hector Llorens, Leon Derczynski, Marc Verhagen, James 
Allen and James Pustejovsky. 2013. SemEval-2013 Task 1: TempEval-3: 
Evaluating Time Expressions, Events, and Temporal Relations. Proceedings 
of the 7th International Workshop on Semantic Evaluation (SemEval 2013),
in conjunction with the Second Joint Conference on Lexical and 
Computational Semantics (*SEM 2013).

ACL Data and Code Repository reference ADCR2013T001.


 Creation
---------------------------------
Twenty documents were distributed among annotators such that each entire document
was annotated by two annotators. TIPSem was sometimes used by some annotators
for pre-annotation processing. Annotators used a diverse range of techniques
and tools to assist them in their tasks. We used the TimeML annotation 
guidelines v1.2.1 and the TIDES 2005 standard and guidelines. The results were 
collated and adjudicated.


Adjudication procedure:
- byte align the source to correct errors during annotation
- load into gate annotation diff
- read the documents and sanity check entity annotations, looking for 
  missing ones and removing generics
- do event merge
- do timex merge
- check all DCT links (if dct = a day, avoid "today X said_e1" 
  "e1 before dct")
- check all timex links and make sure every timex is linked, if possible
- check paragraphs, finding main events linked to dct and check relation
  between subevents and main events
- validate with Hector's TimeML-validator
- load into CAVaT and check for consistency, using inconsistency discovery
  points as starts on search for wrong orderings, amending/removing tlinks 
  along the way
  
  
  
  
  
  
  
  
  
  
  This directory TBAQ-cleaned contains cleaned and improved AQUAINT and TimeBank corpus. We updated these corpora with the following changes. 

Common changes: i. Cleaned formatting for all files. All the files are in same format. Easy to review/read, ii. Made all files XML and TimeML schema compatible, iii. Some missing events and temporal expressions are added. 

AQUAINT changes: i. Added event-DCT temporal relations

TimeBank changes: i. Events are borrowed from the TempEval-2 corpus, ii. Temporal relations are borrowed from actual TimeBank corpus, which contains a full set of TimeML temporal relations. iii. Along with our correction, also added temporal expressions correction suggestion from Kolomiyets et al. (2011) (total additional 10 temporal expressions from them).


- re-validate
- commit

